#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = ["fire", "rich", "mlx-audio @ git+https://github.com/Blaizzy/mlx-audio.git", "soundfile", "numpy", "huggingface-hub"]
# ///
# this_file: src/vexy_audioflexy/tools/tts/tools/tts-fishs2pro-mlx.py
"""Fish Audio S2 Pro TTS via mlx-audio on Apple Silicon. Backend: MLX (dual-AR 5B model). Languages: 77 (en, zh, ja, ko, es, pt, ar, ru, fr, de, +67 more). Sample rate: 44100 Hz. Modes: ivc (instant voice cloning via ref audio). Capabilities: 15k+ inline prosody/emotion tags (e.g. [laugh], [sigh], [excited]), voice cloning from short WAV clip. Variants: bf16 (11 GB), 8bit (6.7 GB), 8bit-v2 (alt 6.7 GB)."""

from pathlib import Path
from typing import Any
import importlib

VERSION = "1.1.0"
DEFAULT_MODEL = "mlx-community/fish-audio-s2-pro-bf16"

KNOWN_MODELS = {
    "bf16": "mlx-community/fish-audio-s2-pro-bf16",
    "8bit": "mlx-community/fish-audio-s2-pro-8bit",
    "8bit-legacy": "mlx-community/fishaudio-s2-pro-8bit-mlx",
    "8bit-v2": "vimdhayak/fish-audio-s2-pro-8bit-mlx",
}


def prep(force: bool = False, model: str = DEFAULT_MODEL):
    """Download + cache all model weights/assets WITHOUT synthesizing. Idempotent."""
    import importlib as _importlib
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn

    console = Console()
    resolved_model = KNOWN_MODELS.get(model, model)

    load_model = getattr(_importlib.import_module("mlx_audio.tts.utils"), "load_model")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"Downloading/caching {resolved_model}...", total=None)
        load_model(resolved_model)
        progress.update(task, description="Done.")

    print(f"[prep] ready: {resolved_model}")


def say(
    text: str,
    ref: str | None = None,
    ref_text: str | None = None,
    output: str = "output.wav",
    model: str = DEFAULT_MODEL,
    play: bool = False,
    force: bool = False,
):
    """Generate speech using Fish Audio S2 Pro (MLX) with optional voice cloning.

    Dual-autoregressive 5B TTS supporting 77 languages with inline prosody/emotion
    tags. Voice cloning via reference audio. Runs on Apple Silicon via mlx-audio.

    Args:
        text: Text to synthesize (supports [tag] inline prosody/emotion).
        ref: Reference audio WAV for voice cloning (optional).
        ref_text: Transcript of reference audio (optional, improves cloning).
        output: Output file path (default: output.wav).
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

    import numpy as np
    import soundfile as sf
    from mlx_audio.utils import load_audio

    load_model = getattr(importlib.import_module("mlx_audio.tts.utils"), "load_model")

    resolved_model = KNOWN_MODELS.get(model, model)
    console.print(f"[dim]Model: {resolved_model}[/dim]")
    if ref:
        console.print(f"[dim]Reference: {ref}[/dim]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading Fish S2 Pro MLX...", total=None)
        tts_model: Any = load_model(resolved_model)
        progress.update(task, description="Generating speech...")

        gen_kwargs: dict[str, Any] = {"text": text}
        if ref:
            # Resample ref audio to model's expected sample rate (44100 Hz for
            # Fish S2 Pro). Passing 24 kHz audio without resampling causes the
            # codec to interpret it as 44.1 kHz, producing helium/squeaky output.
            model_sr = getattr(tts_model, "sample_rate", 44100)
            try:
                ref_audio_mx = load_audio(ref, sample_rate=model_sr)
                gen_kwargs["ref_audio"] = ref_audio_mx
            except Exception as exc:
                console.print(
                    f"[yellow]Voice cloning failed ({exc}), generating with default voice[/yellow]"
                )
        if ref_text:
            gen_kwargs["ref_text"] = ref_text

        results = list(tts_model.generate(**gen_kwargs))
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
    console.print(f"[dim]Debug: model.sample_rate={getattr(tts_model, 'sample_rate', 'unknown')}, results[0].sample_rate={int(results[0].sample_rate)}, audio shape={audio.shape if hasattr(audio, 'shape') else len(audio)}, computed duration={len(audio) / int(results[0].sample_rate):.2f}s[/dim]")

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
        "name": "Fish Audio S2 Pro (MLX)",
        "repo": DEFAULT_MODEL,
        "backend": "mlx",
        "languages": ["en", "zh", "ja", "ko", "es", "pt", "ar", "ru", "fr", "de", "+67 more"],
        "sample_rate_hz": 44100,
        "modes": ["ivc"],
        "capabilities": ["emotion_tags", "voice_cloning"],
    }))


if __name__ == "__main__":
    import fire
    fire.Fire({"prep": prep, "say": say, "info": info})
