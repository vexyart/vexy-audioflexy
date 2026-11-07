#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = ["fire", "rich", "mlx-audio @ git+https://github.com/Blaizzy/mlx-audio.git", "soundfile", "numpy", "huggingface-hub", "g2p_en"]
# ///
# this_file: src/vexy_audioflexy/tools/tts/tools/tts-melotts-mlx.py
"""MeloTTS English via mlx-audio on Apple Silicon. VITS2-based; 44.1 kHz; English only (en); auto mode (no voice cloning). v1: 5 accent variants (EN-US, EN-BR, EN_INDIA, EN-AU, EN-Default); v3: single improved voice (EN-Newest)."""

from pathlib import Path
from typing import Any
import importlib

VERSION = "3.0.0"
DEFAULT_MODEL = "mlx-community/MeloTTS-English-MLX"

KNOWN_MODELS = {
    "v1": "mlx-community/MeloTTS-English-MLX",
    "v3": "mlx-community/MeloTTS-English-v3-MLX",
}

SPEAKERS_BY_MODEL = {
    "mlx-community/MeloTTS-English-MLX": [
        "EN-US",
        "EN-BR",
        "EN_INDIA",
        "EN-AU",
        "EN-Default",
    ],
    "mlx-community/MeloTTS-English-v3-MLX": ["EN-Newest"],
}


def prep(force: bool = False):
    """Download + cache all model weights/assets WITHOUT synthesizing. Idempotent."""
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn

    console = Console()

    # Download both known model variants
    for label, repo in KNOWN_MODELS.items():
        console.print(f"[dim]Preloading {label}: {repo}[/dim]")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(f"Downloading {repo}...", total=None)
            load_model = getattr(
                importlib.import_module("mlx_audio.tts.utils"), "load_model"
            )
            load_model(repo)
            progress.update(task, description="Done.")
        console.print(f"[green][prep] ready: {repo}[/green]")


def say(
    text: str,
    output: str = "output.wav",
    speaker: str = "EN-US",
    model: str = DEFAULT_MODEL,
    play: bool = False,
    force: bool = False,
):
    """Generate speech using MeloTTS English (MLX).

    VITS2-based English TTS. 44.1kHz output.
    v1: 5 accent variants (EN-US, EN-BR, EN_INDIA, EN-AU, EN-Default).
    v3: single improved voice (EN-Newest).

    Args:
        text: Text to synthesize.
        output: Output WAV file path (44.1kHz).
        speaker: Speaker accent (depends on model variant).
        model: Model variant: "v1", "v3", or full HF model ID.
        play: Play audio via afplay instead of saving (macOS).
        force: Overwrite output file if it exists.
    """
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn

    console = Console()

    resolved_model = KNOWN_MODELS.get(model, model)
    valid_speakers = SPEAKERS_BY_MODEL.get(resolved_model, [])

    if valid_speakers and speaker not in valid_speakers:
        console.print(
            f"[red]Error:[/red] Invalid speaker for {resolved_model}. Choose from {', '.join(valid_speakers)}"
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

    console.print(f"[dim]Model: {resolved_model}[/dim]")
    console.print(f"[dim]Speaker: {speaker}[/dim]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading MeloTTS MLX...", total=None)
        tts_model: Any = load_model(resolved_model)
        progress.update(task, description="Generating speech...")
        results = list(tts_model.generate(text=text, speaker=speaker))
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

    print(
        json.dumps(
            {
                "name": "MeloTTS English MLX",
                "repo": DEFAULT_MODEL,
                "backend": "mlx",
                "languages": ["en"],
                "sample_rate_hz": 44100,
                "modes": ["auto"],
                "capabilities": ["preset_voices"],
            }
        )
    )


if __name__ == "__main__":
    import fire

    fire.Fire({"prep": prep, "say": say, "info": info})
