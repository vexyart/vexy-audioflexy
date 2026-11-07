#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = ["fire", "rich", "mlx-audio @ git+https://github.com/Blaizzy/mlx-audio.git", "soundfile", "numpy"]
# ///
# this_file: src/vexy_audioflexy/tools/tts/tools/tts-voxcpm2-mlx.py
"""VoxCPM2 2B multilingual TTS via mlx-audio on Apple Silicon (mlx-community/VoxCPM2-8bit). Backend: mlx (bf16/8bit variants); languages: en + 29 more; sample rate: 48 kHz; modes: auto, ivc; capabilities: voice_cloning."""

from pathlib import Path
from typing import Any
import importlib

import fire

VERSION = "1.0.0"
DEFAULT_MODEL = "mlx-community/VoxCPM2-8bit"

KNOWN_MODELS = {
    "bf16": "mlx-community/VoxCPM2-bf16",
    "8bit": "mlx-community/VoxCPM2-8bit",
    "bf16-aufklarer": "aufklarer/VoxCPM2-MLX-bf16",
}


def prep(force: bool = False):
    """Download + cache VoxCPM2 MLX model weights WITHOUT synthesizing. Idempotent."""
    import importlib
    from rich.console import Console

    console = Console()
    load_model = getattr(importlib.import_module("mlx_audio.tts.utils"), "load_model")
    with console.status(f"[cyan]Downloading {DEFAULT_MODEL}…"):
        load_model(DEFAULT_MODEL)
    print(f"[prep] ready: {DEFAULT_MODEL}")


def say(
    text: str,
    ref: str | None = None,
    ref_text: str | None = None,
    output: str = "output.wav",
    model: str = DEFAULT_MODEL,
    play: bool = False,
    force: bool = False,
):
    """Generate speech using VoxCPM2 (MLX) with optional voice cloning.

    Multilingual (30 languages) TTS via MLX on Apple Silicon.
    48kHz studio-quality output. Voice cloning via reference audio.

    Args:
        text: Text to synthesize.
        ref: Reference audio WAV for voice cloning (optional).
        ref_text: Transcript of reference audio (required with --ref).
        output: Output file path (default: output.wav, 48kHz mono).
        model: Model variant: "bf16", "8bit", or full HF model ID.
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

    if ref:
        ref_path = Path(ref)
        if not ref_path.exists():
            console.print(f"[red]Reference audio not found: {ref}[/red]")
            raise SystemExit(1)
        if not ref_text:
            console.print(
                "[yellow]Note:[/yellow] No --ref-text provided; using Controllable Cloning mode "
                "(lower fidelity than Ultimate Cloning)."
            )

    import numpy as np
    import soundfile as sf

    # Monkeypatch AutoTokenizer.from_pretrained to fall back to LlamaTokenizer for VoxCPM2
    import transformers
    original_from_pretrained = transformers.AutoTokenizer.from_pretrained

    def patched_from_pretrained(pretrained_model_name_or_path, *args, **kwargs):
        if "VoxCPM2" in str(pretrained_model_name_or_path):
            from transformers import LlamaTokenizer
            return LlamaTokenizer.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        return original_from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

    transformers.AutoTokenizer.from_pretrained = patched_from_pretrained

    load_model = getattr(importlib.import_module("mlx_audio.tts.utils"), "load_model")

    resolved_model = KNOWN_MODELS.get(model, model)
    console.print(f"[dim]Model: {resolved_model}[/dim]")
    if ref:
        console.print(f"[dim]Reference: {ref}[/dim]")
        console.print(f"[dim]Reference text: {ref_text}[/dim]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading VoxCPM2 MLX...", total=None)
        tts_model: Any = load_model(resolved_model)
        progress.update(task, description="Generating speech...")
        results = list(
            tts_model.generate(
                text=text,
                ref_audio=ref if ref else None,
                ref_text=ref_text or None,
            )
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
        "name": "VoxCPM2 MLX",
        "repo": DEFAULT_MODEL,
        "backend": "mlx",
        "languages": ["en", "zh", "ja", "ko", "fr", "de", "es", "it", "pt", "ru",
                      "ar", "hi", "nl", "pl", "sv", "tr", "vi", "th", "id", "ms",
                      "cs", "ro", "hu", "fi", "da", "no", "el", "uk", "he", "bg"],
        "sample_rate_hz": 48000,
        "modes": ["auto", "ivc"],
        "capabilities": ["voice_cloning"],
    }))


if __name__ == "__main__":
    fire.Fire({"prep": prep, "say": say, "info": info})
