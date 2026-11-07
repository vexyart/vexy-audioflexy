#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = ["fire", "rich", "mlx-audio @ git+https://github.com/Blaizzy/mlx-audio.git", "soundfile", "numpy", "huggingface-hub"]
# ///
# this_file: src/vexy_audioflexy/tools/tts/tools/tts-longcat-mlx.py
"""Meituan LongCat AudioDiT TTS via mlx-audio. Backend: MLX (Apple Silicon); English only; 24 kHz output; ivc (instruction-based voice cloning via ref audio + ref text); diffusion-transformer architecture (AudioDiT); guidance methods: cfg (no ref) or apg (with ref); 1B and 3.5B variants in bf16/4bit/8bit/mxfp4/mxfp8 quantizations."""

from pathlib import Path
from typing import Any
import importlib

VERSION = "2.0.0"
DEFAULT_MODEL = "mlx-community/LongCat-AudioDiT-3.5B-mxfp8"
REPO = DEFAULT_MODEL

KNOWN_MODELS = {
    "1b-bf16": "mlx-community/LongCat-AudioDiT-1B-bf16",
    "1b-4bit": "mlx-community/LongCat-AudioDiT-1B-4bit",
    "1b-8bit": "mlx-community/LongCat-AudioDiT-1B-8bit",
    "1b-mxfp4": "mlx-community/LongCat-AudioDiT-1B-mxfp4",
    "1b-mxfp8": "mlx-community/LongCat-AudioDiT-1B-mxfp8",
    "3.5b-bf16": "mlx-community/LongCat-AudioDiT-3.5B-bf16",
    "3.5b-4bit": "mlx-community/LongCat-AudioDiT-3.5B-4bit",
    "3.5b-8bit": "mlx-community/LongCat-AudioDiT-3.5B-8bit",
    "3.5b-mxfp4": "mlx-community/LongCat-AudioDiT-3.5B-mxfp4",
    "3.5b-mxfp8": "mlx-community/LongCat-AudioDiT-3.5B-mxfp8",
}


def prep(force: bool = False):
    """Download + cache all model weights/assets WITHOUT synthesizing. Idempotent."""
    load_model = getattr(importlib.import_module("mlx_audio.tts.utils"), "load_model")
    load_model(REPO)
    print(f"[prep] ready: {REPO}")


def say(
    text: str,
    ref: str | None = None,
    ref_text: str | None = None,
    output: str = "output.wav",
    model: str = DEFAULT_MODEL,
    steps: int = 16,
    cfg_strength: float = 4.0,
    guidance_method: str | None = None,
    seed: int = 1024,
    play: bool = False,
    force: bool = False,
):
    """Synthesize speech with LongCat AudioDiT. Supports voice cloning via --ref + --ref-text."""
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
            console.print("[red]Error:[/red] --ref-text is required when using --ref")
            raise SystemExit(1)

    import numpy as np
    import soundfile as sf

    load_model = getattr(importlib.import_module("mlx_audio.tts.utils"), "load_model")
    load_audio = getattr(importlib.import_module("mlx_audio.utils"), "load_audio")

    resolved_model = KNOWN_MODELS.get(model, model)
    console.print(f"[dim]Model: {resolved_model}[/dim]")
    if ref:
        console.print(f"[dim]Reference: {ref}[/dim]")
        console.print(f"[dim]Reference text: {ref_text}[/dim]")

    effective_guidance_method = guidance_method or ("apg" if ref else "cfg")
    reference_audio = load_audio(ref, sample_rate=24000) if ref else None

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading LongCat MLX...", total=None)
        tts_model: Any = load_model(resolved_model)

        progress.update(task, description="Generating speech...")
        generate = getattr(tts_model, "generate")
        results = list(
            generate(
                text=text,
                ref_audio=reference_audio,
                ref_text=ref_text or None,
                guidance_method=effective_guidance_method,
                cfg_strength=cfg_strength,
                steps=steps,
                seed=seed,
            )
        )
        progress.update(task, description="Done.")

    if not results:
        console.print("[red]Error: model returned no audio.[/red]")
        raise SystemExit(1)

    audio_chunks = [np.array(result.audio) for result in results]
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
        "name": "LongCat AudioDiT",
        "repo": REPO,
        "backend": "mlx",
        "languages": ["en"],
        "sample_rate_hz": 24000,
        "modes": ["ivc"],
        "capabilities": ["voice_cloning", "cfg_guidance", "apg_guidance", "diffusion_transformer"],
    }))


if __name__ == "__main__":
    import fire
    fire.Fire({"prep": prep, "say": say, "info": info})
