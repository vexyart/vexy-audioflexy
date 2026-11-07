#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = ["fire", "rich", "mlx-audio @ git+https://github.com/Blaizzy/mlx-audio.git", "soundfile", "numpy", "huggingface-hub"]
# ///
# this_file: src/vexy_audioflexy/tools/tts/tools/tts-kimi-mlx.py
"""Kimi-Audio 7B TTS via mlx-audio on Apple Silicon. MoonshotAI Kimi-Audio 7B MLX (bf16, ~22.6 GB); backend: mlx; languages: en, zh; mode: auto (preset voice, no cloning); sample rate: model-reported; no emotion/instruct tags."""

from pathlib import Path
from typing import Any
import importlib

VERSION = "1.0.0"
REPO = "mlx-community/kimi-audio-7b"


def prep(force: bool = False):
    """Download + cache all model weights/assets WITHOUT synthesizing. Idempotent."""
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn

    console = Console()
    load_model = getattr(importlib.import_module("mlx_audio.tts.utils"), "load_model")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"Downloading/caching {REPO}...", total=None)
        load_model(REPO)
        progress.update(task, description="Done.")

    print(f"[prep] ready: {REPO}")


def say(
    text: str,
    output: str = "output.wav",
    play: bool = False,
    force: bool = False,
):
    """Generate speech using Kimi-Audio 7B (MLX).

    Large multimodal audio model (22.6 GB). English and Chinese.

    Args:
        text: Text to synthesize.
        output: Output WAV file path.
        play: Play audio via afplay instead of saving (macOS).
        force: Overwrite output file if it exists.
    """
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn

    console = Console()

    if (
        not play
        and not force
        and Path(output).exists()
        and Path(output).stat().st_size > 0
    ):
        console.print(
            f"[yellow]Skipped:[/yellow] {output} already exists (use --force to overwrite)"
        )
        return

    import numpy as np
    import soundfile as sf

    load_model = getattr(importlib.import_module("mlx_audio.tts.utils"), "load_model")

    console.print(f"[dim]Model: {REPO}[/dim]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading Kimi-Audio 7B MLX...", total=None)
        tts_model: Any = load_model(REPO)
        progress.update(task, description="Generating speech...")
        results = list(
            tts_model.generate(text=text)
        )
        progress.update(task, description="Done.")

    if not results:
        console.print("[red]Error: model returned no audio.[/red]")
        raise SystemExit(1)

    audio_chunks = [np.array(r.audio) for r in results]
    audio = (
        np.concatenate(audio_chunks, axis=0)
        if len(audio_chunks) > 1
        else audio_chunks[0]
    )
    sample_rate = int(results[0].sample_rate)

    if play:
        import subprocess
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, audio, sample_rate)
            tmp_path = tmp.name

        console.print(
            f"[cyan]Playing audio ({len(audio) / sample_rate:.1f}s)...[/cyan]"
        )
        subprocess.run(["afplay", tmp_path], check=True)
        Path(tmp_path).unlink(missing_ok=True)
        console.print("[green]Done.[/green]")
    else:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_path), audio, sample_rate)
        console.print(f"[green]Saved:[/green] {out_path.resolve()} ({sample_rate} Hz)")


def info():
    """Print one-line JSON: name, repo, backend, languages, sample_rate_hz, modes, capabilities."""
    import json

    print(json.dumps({
        "name": "kimi-audio-7b",
        "repo": REPO,
        "backend": "mlx",
        "languages": ["en", "zh"],
        "sample_rate_hz": None,
        "modes": ["auto"],
        "capabilities": ["preset-voice"],
    }))


if __name__ == "__main__":
    import fire
    fire.Fire({"prep": prep, "say": say, "info": info})
