#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = ["fire", "rich", "mlx-audio>=0.3.1", "mlx-lm>=0.22", "soundfile", "numpy"]
# [tool.uv]
# prerelease = "allow"
# ///
# this_file: src/vexy_audioflexy/tools/tts/tools/tts-qwen3-voicedesign-mlx.py
"""Qwen3 TTS VoiceDesign via MLX on Apple Silicon. Text-to-speech with voice description for custom voice synthesis."""

from pathlib import Path

VERSION = "1.0.0"
REPO = "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16"
MODEL = REPO


def prep(force: bool = False):
    """Download + cache all model weights/assets WITHOUT synthesizing. Idempotent."""
    from mlx_audio.tts.utils import load_model
    load_model(REPO)
    print(f"[prep] ready: {REPO}")


def say(
    text: str,
    output: str = "output.wav",
    instruct: str = "A young male speaker with a clear, natural voice.",
    language: str = "English",
    play: bool = False,
    force: bool = False,
):
    """Generate speech using Qwen3 TTS VoiceDesign (MLX).

    Uses text description of desired voice characteristics instead of reference audio.

    Args:
        text: Text to synthesize.
        output: Output file path (default: output.wav, 24kHz mono).
        instruct: Voice description text (e.g. "A male speaker with a deep, warm voice.").
        language: Language (default: English). Options: English, chinese, etc.
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

    console.print(f"[dim]Model: {MODEL}[/dim]")
    console.print(f"[dim]Language: {language}[/dim]")
    console.print(f"[dim]Voice description: {instruct}[/dim]")

    from mlx_audio.tts.utils import load_model

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading Qwen3 TTS VoiceDesign (MLX)...", total=None)
        model = load_model(MODEL)
        progress.update(task, description="Generating speech...")

        results = list(
            model.generate_voice_design(
                text=text,
                instruct=instruct,
                language=language,
            )
        )
        progress.update(task, description="Done.")

    # Concatenate chunks if text was split
    audio_chunks = [np.array(r.audio) for r in results]
    audio = (
        np.concatenate(audio_chunks, axis=0)
        if len(audio_chunks) > 1
        else audio_chunks[0]
    )
    sample_rate = results[0].sample_rate

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
        "name": "Qwen3-TTS VoiceDesign MLX",
        "repo": REPO,
        "backend": "mlx",
        "languages": ["en", "zh"],
        "sample_rate_hz": 24000,
        "modes": ["voice_design"],
        "capabilities": ["voice_design", "instruct"],
    }))


if __name__ == "__main__":
    import fire
    fire.Fire({"prep": prep, "say": say, "info": info})
