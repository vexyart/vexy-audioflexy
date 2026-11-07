#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = ["fire", "rich", "mlx-audio @ git+https://github.com/Blaizzy/mlx-audio.git", "soundfile", "numpy", "spacy", "misaki[en]", "huggingface-hub"]
# ///
# this_file: src/vexy_audioflexy/tools/tts/tools/tts-kokoro-bf16-mlx.py
"""Kokoro-82M TTS (MLX bf16, Apple Silicon). Backend: mlx-audio on mlx-community/Kokoro-82M-bf16.
Languages: en (American/British). Sample rate: 24 kHz mono. Modes: auto (preset voices),
ivc (voice cloning). Preset voices: af_heart, am_adam, af_sarah, am_michael, bf_emma,
bf_isabella, bm_lewis, bm_sophie. Lightweight 82M-param model; fast local English TTS."""

from pathlib import Path
import importlib

REPO = "mlx-community/Kokoro-82M-bf16"


def prep(force: bool = False):
    """Download + cache all model weights/assets WITHOUT synthesizing. Idempotent."""
    load_model = getattr(importlib.import_module("mlx_audio.tts.utils"), "load_model")
    load_model(REPO)
    print(f"[prep] ready: {REPO}")


def say(
    text: str,
    output: str = "output.wav",
    voice: str = "am_adam",
    speed: float = 1.0,
    lang_code: str = "a",
    play: bool = False,
    force: bool = False,
):
    """Generate speech using Kokoro-82M via MLX (24kHz output).

    Args:
        text: Text to synthesize.
        output: Output WAV file path (24kHz mono).
        voice: Voice preset. Options: af_heart, am_adam, af_sarah,
               am_michael, bf_emma, bf_isabella, bm_lewis, bm_sophie.
        speed: Speed multiplier (0.5-2.0).
        lang_code: Language code. a=American EN, b=British EN.
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
    console.print(f"[dim]Voice: {voice} | Speed: {speed} | Lang: {lang_code}[/dim]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading Kokoro MLX...", total=None)
        tts_model = load_model(REPO)

        progress.update(task, description="Generating speech...")
        chunks = []
        for result in tts_model.generate(
            text=text, voice=voice, speed=speed, lang_code=lang_code
        ):
            chunks.append(np.array(result.audio, copy=False))
        progress.update(task, description="Done.")

    if not chunks:
        console.print("[red]Error: model returned no audio.[/red]")
        raise SystemExit(1)

    audio = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
    sample_rate = 24000

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
    """Print one-line JSON: name, repo, backend, languages, sample_rate_hz, modes, capabilities."""
    import json

    print(json.dumps({
        "name": "Kokoro 82M",
        "repo": REPO,
        "backend": "mlx",
        "languages": ["en"],
        "sample_rate_hz": 24000,
        "modes": ["auto", "ivc"],
        "capabilities": ["voice_cloning", "preset_voices"],
    }))


if __name__ == "__main__":
    import fire
    fire.Fire({"prep": prep, "say": say, "info": info})
