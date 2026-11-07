#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = ["fire", "rich", "mlx-audio @ git+https://github.com/Blaizzy/mlx-audio.git", "soundfile", "numpy"]
# ///
# this_file: src/vexy_audioflexy/tools/tts/tools/tts-voxcpm-mlx.py
"""VoxCPM1.5 0.6B bilingual TTS via mlx-audio on Apple Silicon (mlx-community/VoxCPM1.5-fp16). Backend: mlx (fp16/8bit/6bit/5bit/4bit variants); languages: en, zh; sample rate: 44.1 kHz; modes: auto, ivc; capabilities: voice_cloning."""

from pathlib import Path
from typing import Any
import importlib

import fire

VERSION = "1.0.0"
DEFAULT_MODEL = "mlx-community/VoxCPM1.5-fp16"

KNOWN_MODELS = {
    "fp16": "mlx-community/VoxCPM1.5-fp16",
    "8bit": "mlx-community/VoxCPM1.5-8bit",
    "6bit": "mlx-community/VoxCPM1.5-6bit",
    "5bit": "mlx-community/VoxCPM1.5-5bit",
    "4bit": "mlx-community/VoxCPM1.5-4bit",
}


# VoxCPM's MLX attention is effectively O(n^2) over the input; a single ~500-char
# paragraph tries to allocate tens of GB and aborts past Metal's buffer limit.
# Synthesizing sentence-sized chunks and concatenating keeps memory bounded.
_MAX_CHUNK_CHARS = 160


def _chunk_text(text: str, max_chars: int = _MAX_CHUNK_CHARS) -> list[str]:
    """Split *text* into sentence-grouped chunks no longer than *max_chars*.

    Splits on en/zh sentence terminators (.!?。！？…) and newlines, then greedily
    packs sentences into chunks. A single over-long sentence becomes its own chunk
    (the model still handles it; this only caps the common multi-sentence case).
    """
    import re

    sentences = [
        s.strip()
        for s in re.split(r"(?<=[.!?。！？…])\s+|\n+", text.strip())
        if s.strip()
    ]
    if not sentences:
        return [text.strip()] if text.strip() else []

    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if not current:
            current = sentence
        elif len(current) + 1 + len(sentence) <= max_chars:
            current = f"{current} {sentence}"
        else:
            chunks.append(current)
            current = sentence
    if current:
        chunks.append(current)
    return chunks


def prep(force: bool = False):
    """Download + cache VoxCPM1.5 MLX model weights WITHOUT synthesizing. Idempotent."""
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
    """Generate speech using VoxCPM1.5 (MLX) with optional voice cloning.

    Bilingual (en+zh) TTS via MLX on Apple Silicon.
    44.1kHz output. Voice cloning via reference audio.

    Args:
        text: Text to synthesize.
        ref: Reference audio WAV for voice cloning (optional).
        ref_text: Transcript of reference audio (improves cloning quality).
        output: Output file path (default: output.wav, 44.1kHz mono).
        model: Model variant: "fp16", "8bit", "6bit", "5bit", "4bit", or full HF model ID.
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
                "[yellow]Warning:[/yellow] --ref-text not provided; voice cloning quality may be reduced. "
                "Pass the transcript of the reference audio via --ref-text for best results."
            )

    import numpy as np
    import soundfile as sf

    load_model = getattr(importlib.import_module("mlx_audio.tts.utils"), "load_model")

    resolved_model = KNOWN_MODELS.get(model, model)
    console.print(f"[dim]Model: {resolved_model}[/dim]")
    if ref:
        console.print(f"[dim]Reference: {ref}[/dim]")
        if ref_text:
            console.print(f"[dim]Reference text: {ref_text}[/dim]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading VoxCPM1.5 MLX...", total=None)
        tts_model: Any = load_model(resolved_model)
        progress.update(task, description="Generating speech...")
        ref_array = None
        if ref:
            load_audio = getattr(
                importlib.import_module("mlx_audio.utils"), "load_audio"
            )
            try:
                sr = tts_model.sample_rate
            except AttributeError:
                sr = 16000
            ref_array = load_audio(ref, sample_rate=sr)
        text_chunks = _chunk_text(text)
        results: list[Any] = []
        for idx, chunk in enumerate(text_chunks, 1):
            progress.update(
                task,
                description=f"Generating speech ({idx}/{len(text_chunks)})...",
            )
            results.extend(
                tts_model.generate(
                    text=chunk,
                    ref_audio=ref_array,
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

    print(
        json.dumps(
            {
                "name": "VoxCPM1.5 MLX",
                "repo": DEFAULT_MODEL,
                "backend": "mlx",
                "languages": ["en", "zh"],
                "sample_rate_hz": 44100,
                "modes": ["auto", "ivc"],
                "capabilities": ["voice_cloning"],
            }
        )
    )


if __name__ == "__main__":
    fire.Fire({"prep": prep, "say": say, "info": info})
