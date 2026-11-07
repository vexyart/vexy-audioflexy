#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = ["fire", "rich", "huggingface-hub", "soundfile", "numpy"]
# ///
# this_file: src/vexy_audioflexy/tools/tts/tools/tts-cosyvoice3-gguf.py
"""CosyVoice3 0.5B GGUF wrapper (cstr/cosyvoice3-0.5b-2512-GGUF).
Backend: gguf (via CrispASR --backend cosyvoice3-tts).
Languages: 9 (zh, en, ja, ko, yue, fr, de, es, pt, it, ru). Sample rate: 24 kHz.
Modes: auto, ivc (zero-shot voice cloning).
Capabilities: voice_cloning, streaming_tts, multilingual.
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

MODEL_REPO = "cstr/cosyvoice3-0.5b-2512-GGUF"
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CRISPASR = runtime_path('CrispASR/build/bin/crispasr')


def _get_model_dir() -> Path:
    from huggingface_hub import snapshot_download

    snapshot_path = snapshot_download(MODEL_REPO, token=False)
    return Path(snapshot_path)


def prep(force: bool = False) -> None:
    """Download and cache all model weights/assets without synthesizing. Idempotent."""
    from rich.console import Console

    console = Console()
    console.print(f"[cyan]Downloading/verifying {MODEL_REPO}…[/cyan]")
    model_dir = _get_model_dir()
    console.print(f"[green][prep] ready:[/green] {MODEL_REPO} ({model_dir})")


def say(
    text: str,
    output: str = "output.wav",
    ref: str | None = None,
    ref_text: str | None = None,
    language: str = "en",
    binary: str | None = None,
    play: bool = False,
    force: bool = False,
) -> None:
    """Generate speech using CosyVoice3 0.5B GGUF via CrispASR.

    Args:
        text: Text to synthesize.
        output: Output WAV path.
        ref: Optional reference WAV for voice cloning.
        ref_text: Optional reference text transcript.
        language: Language code hint.
        binary: Path to crispasr binary executable.
        play: Play output WAV via afplay.
        force: Overwrite existing output WAV.
    """
    from rich.console import Console

    console = Console()
    out_path = Path(output)
    if not force and out_path.exists() and out_path.stat().st_size > 0:
        console.print(f"[yellow]Skipped:[/yellow] {output} already exists (use --force)")
        return

    model_dir = _get_model_dir()
    bin_path = Path(binary) if binary else DEFAULT_CRISPASR
    if not bin_path.exists():
        _found = shutil.which("crispasr") or shutil.which("crispasr-cli")
        bin_path = Path(_found) if _found else bin_path

    if not bin_path.exists():
        console.print(
            f"[red]CrispASR binary not found at {bin_path}. Build it in local/CrispASR.[/red]"
        )
        raise SystemExit(1)

    llm_gguf = list(model_dir.glob("*llm*.gguf"))[0]
    flow_gguf = list(model_dir.glob("*flow*.gguf"))[0]
    hift_gguf = list(model_dir.glob("*hift*.gguf"))[0]

    cmd = [
        str(bin_path),
        "--backend",
        "cosyvoice3-tts",
        "-m",
        str(llm_gguf),
        "--flow-model",
        str(flow_gguf),
        "--hift-model",
        str(hift_gguf),
        "--tts",
        text,
        "--tts-output",
        str(out_path),
    ]

    if ref:
        ref_path = Path(ref).expanduser()
        if not ref_path.exists():
            console.print(f"[red]Reference audio not found:[/red] {ref}")
            raise SystemExit(1)
        cmd.extend(["--voice", str(ref_path)])
        if ref_text:
            cmd.extend(["--prompt-text", ref_text])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    console.print(f"[dim]Repo: {MODEL_REPO}[/dim]")
    console.print(f"[dim]LLM: {llm_gguf.name}[/dim]")
    subprocess.run(cmd, check=True)

    if not out_path.exists() or out_path.stat().st_size == 0:
        console.print(f"[red]CrispASR did not produce valid output WAV: {out_path}[/red]")
        raise SystemExit(1)

    console.print(f"[green]Saved:[/green] {out_path.resolve()} (24000 Hz)")
    if play:
        subprocess.run(["afplay", str(out_path)], check=True)


def info() -> None:
    """Print one-line JSON metadata for this engine."""
    print(
        json.dumps(
            {
                "name": "CosyVoice3 0.5B GGUF",
                "repo": MODEL_REPO,
                "backend": "gguf",
                "languages": ["en", "pl", "zh", "ja", "ko", "yue", "fr", "de", "es", "pt", "it", "ru"],
                "sample_rate_hz": 24000,
                "modes": ["auto", "ivc"],
                "capabilities": ["voice_cloning", "streaming_tts", "multilingual"],
            }
        )
    )


if __name__ == "__main__":
    fire.Fire({"prep": prep, "say": say, "info": info})
