#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = ["fire", "rich", "mlx-audio @ git+https://github.com/Blaizzy/mlx-audio.git", "soundfile", "numpy"]
# ///
# this_file: src/vexy_audioflexy/tools/tts/tools/tts-supertonic-mlx.py
"""Supertonic-2 TTS via mlx-audio on Apple Silicon. Backend: MLX (mlx-audio); 44100 Hz output; languages: en, ko, es, pt, fr; mode: auto; preset voices: M1-M5 (male), F1-F5 (female); no voice cloning."""

from pathlib import Path
from typing import Any
import importlib
import json

import fire

VERSION = "1.0.0"
MODEL = "mlx-community/supertonic-2"
VALID_VOICES = ["M1", "M2", "M3", "M4", "M5", "F1", "F2", "F3", "F4", "F5"]


def prep(force: bool = False):
    """Download + cache all model weights/assets WITHOUT synthesizing. Idempotent."""
    load_model = getattr(importlib.import_module("mlx_audio.tts.utils"), "load_model")
    load_model(MODEL)
    print(f"[prep] ready: {MODEL}")


def say(
    text: str,
    output: str = "output.wav",
    voice: str = "M1",
    play: bool = False,
    force: bool = False,
):
    """Generate speech using Supertonic-2 (MLX).

    10 built-in voice styles (M1-M5 male, F1-F5 female). 44.1kHz output.

    Args:
        text: Text to synthesize.
        output: Output WAV file path (44.1kHz).
        voice: Voice style: M1-M5 (male), F1-F5 (female).
        play: Play audio via afplay instead of saving (macOS).
        force: Overwrite output file if it exists.
    """
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn

    console = Console()

    if voice not in VALID_VOICES:
        console.print(
            f"[red]Error:[/red] Invalid voice. Choose from {', '.join(VALID_VOICES)}"
        )
        raise SystemExit(1)

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

    console.print(f"[dim]Model: {MODEL}[/dim]")
    console.print(f"[dim]Voice: {voice}[/dim]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading Supertonic-2 MLX...", total=None)
        tts_model: Any = load_model(MODEL)
        progress.update(task, description="Generating speech...")
        results = list(
            tts_model.generate(text, voice=voice)
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
    print(json.dumps({
        "name": "Supertonic 2 (MLX)",
        "repo": MODEL,
        "backend": "mlx",
        "languages": ["en", "ko", "es", "pt", "fr"],
        "sample_rate_hz": 44100,
        "modes": ["auto"],
        "capabilities": ["preset_voices"],
    }))


if __name__ == "__main__":
    fire.Fire({"prep": prep, "say": say, "info": info})
