#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = ["fire", "rich", "mlx-audio", "soundfile", "numpy", "huggingface-hub"]
# ///
# this_file: src/vexy_audioflexy/tools/tts/tools/tts-chatterbox-turbo.py
"""Chatterbox Turbo TTS via MLX on Apple Silicon (mlx-community/chatterbox-turbo-8bit).

Backend: mlx-audio (8-bit quantised MLX). Languages: en, es, fr, de, it, pt, pl, hi, zh, ja, ko.
Sample rate: 24 kHz mono. Modes: ivc (instant voice cloning via ref audio).
Paralinguistic tags in text: [laugh], [cough], [chuckle], [sigh], etc.
"""

from pathlib import Path

REPO = "mlx-community/chatterbox-turbo-8bit"
VERSION = "2.0.0"


def prep(force: bool = False):
    """Download + cache all model weights without synthesising audio. Idempotent."""
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from mlx_audio.tts.utils import load_model

    console = Console()
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task(f"Downloading/caching {REPO}...", total=None)
        load_model(REPO)
    print(f"[prep] ready: {REPO}")


def say(
    text: str,
    ref: str | None = None,
    output: str = "output.wav",
    lang: str = "",
    play: bool = False,
    force: bool = False,
):
    """Generate speech using Chatterbox Turbo TTS (MLX) with optional voice cloning.

    Supports paralinguistic tags in text: [laugh], [cough], [chuckle], [sigh], etc.
    Example: "That's hilarious [laugh] I can't believe it."

    Args:
        text: Text to synthesize (supports [laugh], [cough], [chuckle] tags).
        ref: Reference audio WAV for voice cloning (optional).
        lang: Language code passed to the model as `lang_code`. Leaving it unset
            makes the multilingual checkpoint guess from the text, which can
            produce incorrect non-English pronunciation.
        output: Output file path (default: output.wav, 24kHz mono).
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

    console.print(f"[dim]Model: {REPO}[/dim]")
    if ref:
        console.print(f"[dim]Reference: {ref}[/dim]")

    from mlx_audio.tts.utils import load_model

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading Chatterbox Turbo (MLX)...", total=None)
        model = load_model(REPO)
        progress.update(task, description="Generating speech...")
        gen_kwargs = {"text": text, "ref_audio": ref if ref else None}
        if lang:
            gen_kwargs["lang_code"] = lang
        results = list(model.generate(**gen_kwargs))
        progress.update(task, description="Done.")

    # Concatenate chunks if text was split
    audio_chunks = [np.array(r.audio) for r in results]
    audio = (
        np.concatenate(audio_chunks, axis=0)
        if len(audio_chunks) > 1
        else audio_chunks[0]
    )
    sample_rate = results[0].sample_rate  # 24000

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
    """Print one-line JSON with model metadata."""
    import json

    print(json.dumps({
        "name": "Chatterbox Turbo",
        "repo": REPO,
        "backend": "mlx",
        "languages": ["en", "es", "fr", "de", "it", "pt", "pl", "hi", "zh", "ja", "ko"],
        "sample_rate_hz": 24000,
        "modes": ["ivc"],
        "capabilities": ["voice_cloning", "paralinguistic_tags"],
    }))


if __name__ == "__main__":
    import fire
    fire.Fire({"prep": prep, "say": say, "info": info})
