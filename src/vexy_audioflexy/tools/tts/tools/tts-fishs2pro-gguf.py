#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = ["fire", "rich", "huggingface-hub", "soundfile", "numpy"]
# ///
# this_file: src/vexy_audioflexy/tools/tts/tools/tts-fishs2pro-gguf.py
"""Fish Audio S2 Pro GGUF wrapper (rodrigomt/s2-pro-gguf).
Backend: gguf (s2.cpp pure C++/GGML DualAR engine).
Languages: 80+ (en, pl, zh, ja, ko, es, fr, de, it, etc.). Sample rate: 44.1 kHz.
Modes: auto, ivc (instant voice cloning from reference WAV).
Capabilities: voice_cloning, emotion_tags.
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

MODEL_REPO = "rodrigomt/s2-pro-gguf"
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BINARY = runtime_path('s2.cpp/build/s2')


def _get_model_dir(quant: str = "q4_k_m") -> Path:
    from huggingface_hub import snapshot_download

    snapshot_path = snapshot_download(
        MODEL_REPO,
        token=False,
        allow_patterns=[f"*{quant}*.gguf", "tokenizer.json"],
    )
    return Path(snapshot_path)


def prep(force: bool = False) -> None:
    """Download and cache model weights/assets without synthesizing. Idempotent."""
    from rich.console import Console

    console = Console()
    console.print(f"[cyan]Downloading/verifying {MODEL_REPO}…[/cyan]")
    model_dir = _get_model_dir()
    console.print(f"[green][prep] ready:[/green] {MODEL_REPO} ({model_dir})")


def say(
    text: str,
    output: str = "output.wav",
    quant: str = "q4_k_m",
    ref: str | None = None,
    ref_text: str | None = None,
    binary: str | None = None,
    play: bool = False,
    force: bool = False,
) -> None:
    """Generate speech using Fish Audio S2 Pro GGUF.

    Args:
        text: Text to synthesize.
        output: Output WAV path.
        quant: Quantization variant ("f16", "q8_0", "q6_k", "q5_k_m", "q4_k_m", "q3_k", "q2_k").
        ref: Optional reference WAV for voice cloning.
        ref_text: Optional reference text transcript.
        binary: Path to s2 binary executable.
        play: Play output WAV via afplay.
        force: Overwrite output WAV if it exists.
    """
    from rich.console import Console

    console = Console()
    out_path = Path(output)
    if not force and out_path.exists() and out_path.stat().st_size > 0:
        console.print(f"[yellow]Skipped:[/yellow] {output} already exists (use --force)")
        return

    model_dir = _get_model_dir(quant)
    bin_path = Path(binary) if binary else DEFAULT_BINARY
    if not bin_path.exists():
        _found = shutil.which("s2")
        bin_path = Path(_found) if _found else bin_path

    if not bin_path.exists():
        console.print(f"[red]s2.cpp binary not found at {bin_path}. Build it in local/s2.cpp.[/red]")
        raise SystemExit(1)

    gguf_files = list(model_dir.glob(f"*{quant}*.gguf")) or list(model_dir.glob("*.gguf"))
    if not gguf_files:
        console.print(f"[red]No GGUF model file found in {model_dir}[/red]")
        raise SystemExit(1)

    gguf_path = gguf_files[0]
    tokenizer_path = model_dir / "tokenizer.json"
    if not tokenizer_path.exists():
        tokenizer_path = runtime_path('s2.cpp/tokenizer.json')

    cmd = [
        str(bin_path),
        "-m",
        str(gguf_path),
        "--text",
        text,
        "-o",
        str(out_path),
    ]
    if tokenizer_path.exists():
        cmd.extend(["-t", str(tokenizer_path)])

    if ref:
        ref_path = Path(ref).expanduser()
        if not ref_path.exists():
            console.print(f"[red]Reference audio not found:[/red] {ref}")
            raise SystemExit(1)
        cmd.extend(["--ref-audio", str(ref_path)])
        if ref_text:
            cmd.extend(["--ref-text", ref_text])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    console.print(f"[dim]Repo: {MODEL_REPO}[/dim]")
    console.print(f"[dim]GGUF: {gguf_path.name}[/dim]")

    subprocess.run(cmd, check=True)

    if not out_path.exists() or out_path.stat().st_size == 0:
        console.print(f"[red]s2 binary did not create non-empty output WAV: {out_path}[/red]")
        raise SystemExit(1)

    console.print(f"[green]Saved:[/green] {out_path.resolve()} (44100 Hz)")
    if play:
        subprocess.run(["afplay", str(out_path)], check=True)


def info() -> None:
    """Print one-line JSON metadata for this engine."""
    print(
        json.dumps(
            {
                "name": "Fish Audio S2 Pro GGUF",
                "repo": MODEL_REPO,
                "backend": "gguf",
                "languages": ["en", "pl", "zh", "ja", "ko", "es", "fr", "de", "it"],
                "sample_rate_hz": 44100,
                "modes": ["auto", "ivc"],
                "capabilities": ["voice_cloning", "emotion_tags"],
            }
        )
    )


if __name__ == "__main__":
    fire.Fire({"prep": prep, "say": say, "info": info})
