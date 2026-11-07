#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = ["fire", "rich", "huggingface-hub", "soundfile", "numpy"]
# ///
# this_file: src/vexy_audioflexy/tools/tts/tools/tts-inflect-gguf.py
"""Inflect v2 GGUF wrapper (remixerdec/Inflect-Micro-v2-GGUF & remixerdec/Inflect-Nano-v2-GGUF).
Backend: gguf (InflectNanoTTS C++/GGML runtime).
Languages: en. Sample rate: 24 kHz mono. Modes: auto. Capabilities: lightweight_tts.
"""

from pathlib import Path
import json
import os
import shutil
import subprocess
import sys

import fire
from _paths import runtime_path

# Fallback HF_HOME if default ~/.cache/huggingface is a broken symlink
hf_cache = Path.home() / ".cache" / "huggingface"
if not os.environ.get("HF_HOME") and (hf_cache.is_symlink() and not hf_cache.exists()):
    os.environ["HF_HOME"] = str(Path.home() / ".cache" / "hf_local")

KNOWN_MODELS = {
    "micro": "remixerdec/Inflect-Micro-v2-GGUF",
    "nano": "remixerdec/Inflect-Nano-v2-GGUF",
}

ROOT = Path(__file__).resolve().parent.parent
_BUILD_ROOT = runtime_path('InflectNanoTTS.cpp/build/macos-arm64')
# CMake writes the binary one level deeper than the build directory it is given,
# so accept either layout rather than depending on which one this checkout used.
DEFAULT_BINARY = next(
    (c for c in (_BUILD_ROOT / "inflect-nano", _BUILD_ROOT / "macos-arm64" / "inflect-nano") if c.exists()),
    _BUILD_ROOT / "inflect-nano",
)


def _get_model_dir(repo: str) -> Path:
    from huggingface_hub import snapshot_download

    snapshot_path = snapshot_download(repo, token=False)
    return Path(snapshot_path)


def prep(force: bool = False, model: str = "micro") -> None:
    """Download and cache all model weights/assets without synthesizing. Idempotent."""
    from rich.console import Console

    console = Console()
    repo = KNOWN_MODELS.get(model, model)
    console.print(f"[cyan]Downloading/verifying {repo}…[/cyan]")
    model_dir = _get_model_dir(repo)
    console.print(f"[green][prep] ready:[/green] {repo} ({model_dir})")


def say(
    text: str,
    output: str = "output.wav",
    model: str = "micro",
    quant: str = "f16",
    speed: float = 1.0,
    variation: float = 0.667,
    seed: int = 42,
    binary: str | None = None,
    ref: str | None = None,
    play: bool = False,
    force: bool = False,
) -> None:
    """Generate speech using Inflect v2 GGUF (Micro or Nano).

    Args:
        text: Text to synthesize.
        output: Output WAV path.
        model: "micro" or "nano" or exact HF repo.
        quant: Quantization variant ("f16", "q8_0", "q4_0").
        speed: Speed multiplier.
        variation: Noise variation scale.
        seed: Random seed.
        binary: Path to inflect-nano binary.
        ref: Optional reference WAV (ignored by VITS preset).
        play: Play output WAV via afplay.
        force: Overwrite existing output WAV.
    """
    from rich.console import Console

    console = Console()
    out_path = Path(output)
    if not force and out_path.exists() and out_path.stat().st_size > 0:
        console.print(f"[yellow]Skipped:[/yellow] {output} already exists (use --force)")
        return

    repo = KNOWN_MODELS.get(model, model)
    model_dir = _get_model_dir(repo)

    # Locate binary
    bin_path = Path(binary) if binary else DEFAULT_BINARY
    if not bin_path.exists():
        _found = shutil.which("inflect-nano")
        bin_path = Path(_found) if _found else bin_path

    if not bin_path.exists():
        console.print(
            f"[red]InflectNanoTTS binary not found at {bin_path}. Build it via cmake in local/InflectNanoTTS.cpp.[/red]"
        )
        raise SystemExit(1)

    # Find GGUF weight and lexicon
    gguf_files = list(model_dir.glob(f"*{quant}*.gguf")) or list(model_dir.glob("*.gguf"))
    if not gguf_files:
        console.print(f"[red]No GGUF file found in {model_dir}[/red]")
        raise SystemExit(1)

    gguf_path = gguf_files[0]
    lexicon_path = model_dir / "lexicon.bin"

    cmd = [
        str(bin_path),
        "--v2-model",
        str(gguf_path),
        "-t",
        text,
        "-o",
        str(out_path),
        "--speed",
        str(speed),
        "--variation",
        str(variation),
        "--seed",
        str(seed),
    ]
    if lexicon_path.exists():
        cmd.extend(["--v2-lexicon", str(lexicon_path)])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    console.print(f"[dim]Repo: {repo}[/dim]")
    console.print(f"[dim]GGUF: {gguf_path.name}[/dim]")
    subprocess.run(cmd, check=True)

    if not out_path.exists() or out_path.stat().st_size == 0:
        console.print(f"[red]Binary did not produce valid WAV output: {out_path}[/red]")
        raise SystemExit(1)

    console.print(f"[green]Saved:[/green] {out_path.resolve()} (24000 Hz)")
    if play:
        subprocess.run(["afplay", str(out_path)], check=True)


def info() -> None:
    """Print one-line JSON metadata for this engine."""
    print(
        json.dumps(
            {
                "name": "Inflect v2 GGUF (Micro & Nano)",
                "repo": list(KNOWN_MODELS.values()),
                "backend": "gguf",
                "languages": ["en"],
                "sample_rate_hz": 24000,
                "modes": ["auto"],
                "capabilities": ["lightweight_tts"],
            }
        )
    )


if __name__ == "__main__":
    fire.Fire({"prep": prep, "say": say, "info": info})
