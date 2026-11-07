#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = ["fire", "rich", "mlx-audio @ git+https://github.com/Blaizzy/mlx-audio.git", "soundfile", "numpy", "torchaudio", "transformers", "torch", "huggingface-hub"]
# ///
# this_file: src/vexy_audioflexy/tools/tts/tools/tts-omnivoice-mlx.py
"""OmniVoice zero-shot TTS via mlx-audio on Apple Silicon. Backend: MLX (Qwen3-0.6B + iterative masked diffusion); 600+ languages; 24 kHz mono output; ivc mode (zero-shot voice cloning with optional transcript); variants: bf16 (1.37 GB), quantized (806 MB), 8-bit, q8-g64."""

from pathlib import Path
from typing import Any
import importlib

VERSION = "1.1.0"
DEFAULT_MODEL = "theoracleguy/OmniVoice-bf16"

KNOWN_MODELS = {
    "bf16": "theoracleguy/OmniVoice-bf16",
    "quantized": "theoracleguy/OmniVoice",
    "8bit-commandeaw": "commandeaw/OmniVoice-MLX-8bit",
    "bf16-commandeaw": "commandeaw/OmniVoice-MLX-bf16",
    "q8-g64-lightsofapollo": "lightsofapollo/omnivoice-mlx-q8-g64",
}


def prep(force: bool = False):
    """Download + cache all model weights/assets WITHOUT synthesizing. Idempotent."""
    import huggingface_hub
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn

    console = Console()
    resolved_model = DEFAULT_MODEL

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"Downloading {resolved_model}...", total=None)
        load_model = getattr(importlib.import_module("mlx_audio.tts.utils"), "load_model")
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
    """Generate speech using OmniVoice (MLX) with optional voice cloning.

    Zero-shot voice cloning TTS supporting 600+ languages via MLX on Apple Silicon.
    Based on k2-fsa/OmniVoice, converted to MLX format. 24kHz mono output.

    Args:
        text: Text to synthesize.
        ref: Reference audio WAV for voice cloning (optional).
        ref_text: Transcript of reference audio (optional, improves cloning).
        output: Output file path (default: output.wav, 24kHz mono).
        model: Model variant: "bf16", "quantized", or full HF model ID.
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
        task = progress.add_task("Loading OmniVoice MLX...", total=None)
        tts_model: Any = load_model(resolved_model)
        progress.update(task, description="Generating speech...")

        gen_kwargs: dict[str, Any] = {"text": text}
        if ref:
            model_sr = getattr(tts_model, "sample_rate", 24000)
            try:
                ref_audio_mx = load_audio(ref, sample_rate=model_sr)
            except Exception as exc:
                # Falling back to the default voice here would return a clip that
                # sounds fine and is not the requested speaker, which no exit code
                # or listening pass would catch.
                console.print(f"[red]Could not load reference audio {ref}: {exc}[/red]")
                raise SystemExit(1) from exc
            gen_kwargs["ref_audio"] = ref_audio_mx
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
        "name": "OmniVoice MLX",
        "repo": DEFAULT_MODEL,
        "backend": "mlx",
        "languages": ["en", "zh", "ja", "ko", "de", "fr", "es", "pt", "it", "pl", "nl", "tr", "ru", "ar", "600+"],
        "sample_rate_hz": 24000,
        "modes": ["ivc"],
        "capabilities": ["voice_cloning"],
    }))


if __name__ == "__main__":
    import fire
    fire.Fire({"prep": prep, "say": say, "info": info})
