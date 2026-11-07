#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = ["fire", "rich", "qwen-tts", "soundfile", "torch", "transformers==4.57.3", "numpy", "huggingface-hub"]
# ///
# this_file: src/vexy_audioflexy/tools/tts/tools/tts-celune-mlx.py
"""Celune 1.7B expressive TTS family via qwen-tts on Apple Silicon (MPS/CPU). Backend: PyTorch + Qwen3TTSModel. Languages: English. Sample rate: model-determined. Modes: auto (preset variant). Variants: Upbeat, Energetic, Calm, Neutral. No voice cloning."""

from pathlib import Path
import importlib

VERSION = "2.0.0"
VALID_VARIANTS = ["Upbeat", "Energetic", "Calm", "Neutral"]
REPO_PREFIX = "lunahr/Celune-1.7B"


def prep(force: bool = False):
    """Download + cache all model weights/assets WITHOUT synthesizing. Idempotent."""
    import importlib
    from rich.console import Console

    console = Console()
    for variant in VALID_VARIANTS:
        repo = f"{REPO_PREFIX}-{variant}"
        console.print(f"[dim]Downloading {repo}...[/dim]")
        import torch
        Qwen3TTSModel = getattr(importlib.import_module("qwen_tts"), "Qwen3TTSModel")
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        dtype = torch.bfloat16 if device != "cpu" else torch.float32
        Qwen3TTSModel.from_pretrained(repo, device_map=device, dtype=dtype)
        print(f"[prep] ready: {repo}")


def say(
    text: str,
    variant: str = "Neutral",
    language: str = "English",
    output: str = "output.wav",
    play: bool = False,
    force: bool = False,
):
    """Synthesize speech with Celune 1.7B. Variant must be one of Upbeat, Energetic, Calm, Neutral."""
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn

    console = Console()

    if variant not in VALID_VARIANTS:
        console.print(
            f"[red]Error:[/red] Invalid variant. Choose from {', '.join(VALID_VARIANTS)}"
        )
        raise SystemExit(1)

    model_id = f"{REPO_PREFIX}-{variant}"

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

    import soundfile as sf
    import torch

    Qwen3TTSModel = getattr(importlib.import_module("qwen_tts"), "Qwen3TTSModel")

    console.print(f"[dim]Model: {model_id}[/dim]")
    console.print(f"[dim]Language: {language}[/dim]")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.bfloat16 if device != "cpu" else torch.float32
    console.print(f"[dim]Device: {device}[/dim]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"Loading Celune {variant}...", total=None)
        model = Qwen3TTSModel.from_pretrained(model_id, device_map=device, dtype=dtype)
        progress.update(task, description="Generating speech...")
        wavs, sample_rate = model.generate_custom_voice(
            text=text,
            language=language,
            speaker="celune",
            non_streaming_mode=True,
        )
        progress.update(task, description="Done.")

    if not wavs:
        console.print("[red]Error: model returned no audio.[/red]")
        raise SystemExit(1)

    import numpy as np

    audio_chunks = [np.asarray(wav) for wav in wavs]
    audio = (
        np.concatenate(audio_chunks, axis=0)
        if len(audio_chunks) > 1
        else audio_chunks[0]
    )

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
        "name": "Celune",
        "repo": [f"{REPO_PREFIX}-{v}" for v in VALID_VARIANTS],
        "backend": "pytorch-qwen-tts",
        "languages": ["en"],
        "sample_rate_hz": "model-determined",
        "modes": ["auto"],
        "capabilities": ["preset-variants", "expressive"],
    }))


if __name__ == "__main__":
    import fire
    fire.Fire({"prep": prep, "say": say, "info": info})
