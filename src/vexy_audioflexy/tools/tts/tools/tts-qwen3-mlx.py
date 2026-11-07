#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = ["fire", "rich", "mlx-audio>=0.3.1", "mlx-lm>=0.22", "soundfile", "numpy"]
# [tool.uv]
# prerelease = "allow"
# ///
# this_file: src/vexy_audioflexy/tools/tts/tools/tts-qwen3-mlx.py
"""Qwen3 TTS via MLX on Apple Silicon. Backend: mlx-audio. Two models: CustomVoice (preset speakers: Vivian/Serena/Uncle_Fu/Dylan/Eric/Ryan/Aiden) and Base (voice cloning via --ref). Languages: en, zh, ja, ko, fr, de, es, pt, ar. Sample rate: 24 kHz. Modes: preset (CustomVoice) and ivc/voice-cloning (Base). Supports style instruct text."""

from pathlib import Path

VERSION = "1.2.0"
REPO = "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16"
MODEL_CUSTOM = "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16"
MODEL_BASE = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16"
SHORTCUTS = {
    "holler": "sentiuminc/holler-0.6b",
}


def prep(force: bool = False):
    """Download + cache all model weights/assets WITHOUT synthesizing. Idempotent."""
    from mlx_audio.tts.utils import load_model
    for repo in [MODEL_CUSTOM, MODEL_BASE]:
        load_model(repo)
    print(f"[prep] ready: {REPO}")


def say(
    text: str,
    output: str = "output.wav",
    voice: str = "Aiden",
    language: str = "auto",
    instruct: str = "",
    ref: str | None = None,
    ref_text: str | None = None,
    play: bool = False,
    force: bool = False,
    model: str | None = None,
):  # was: main
    """Generate speech using Qwen3 TTS CustomVoice (MLX).

    Two modes:
      1. Preset speaker (default): Use --voice to pick a speaker.
      2. Voice cloning: Provide --ref (audio WAV) and --ref-text (transcript).

    Preset speakers: Vivian, Serena, Uncle_Fu, Dylan, Eric, Ryan, Aiden.
    English speakers: Ryan, Aiden. Chinese speakers: Vivian, Serena, Uncle_Fu, Dylan, Eric.

    Args:
        text: Text to synthesize.
        output: Output file path (default: output.wav, 24kHz mono).
        voice: Speaker name (default: Aiden). Options: Vivian, Serena, Uncle_Fu, Dylan, Eric, Ryan, Aiden.
        language: Language (default: auto). Options: auto, English, chinese.
        instruct: Optional style instruction (e.g. "Very happy and excited.").
        ref: Path to reference audio WAV for voice cloning (overrides --voice).
        ref_text: Transcript of the reference audio (recommended with --ref).
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

    use_voice_cloning = ref is not None

    if use_voice_cloning:
        ref_path = Path(ref)
        if not ref_path.exists():
            console.print(f"[red]Error: reference audio not found: {ref}[/red]")
            raise SystemExit(1)
        model_id = model if model else MODEL_BASE
        if model_id in SHORTCUTS:
            model_id = SHORTCUTS[model_id]
        console.print(f"[dim]Model: {model_id} (voice cloning)[/dim]")
        console.print(f"[dim]Reference: {ref}[/dim]")
        if ref_text:
            console.print(f"[dim]Reference text: {ref_text}[/dim]")
    else:
        model_id = model if model else MODEL_CUSTOM
        if model_id in SHORTCUTS:
            model_id = SHORTCUTS[model_id]
        console.print(f"[dim]Model: {model_id}[/dim]")
        console.print(f"[dim]Voice: {voice} ({language})[/dim]")
        if instruct:
            console.print(f"[dim]Instruct: {instruct}[/dim]")

    from mlx_audio.tts.utils import load_model

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading Qwen3 TTS (MLX)...", total=None)
        model = load_model(model_id)
        progress.update(task, description="Generating speech...")

        if use_voice_cloning:
            results = list(
                model.generate(
                    text=text,
                    ref_audio=ref,
                    ref_text=ref_text or "",
                    lang_code=language if language != "auto" else "auto",
                )
            )
        else:
            results = list(
                model.generate_custom_voice(
                    text=text,
                    speaker=voice,
                    language=language,
                    instruct=instruct if instruct else None,
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
    """Print one-line JSON: name, repo, backend, languages, sample_rate_hz, modes, capabilities."""
    import json
    print(json.dumps({
        "name": "Qwen3-TTS MLX",
        "repo": REPO,
        "backend": "mlx",
        "languages": ["en", "zh", "ja", "ko", "fr", "de", "es", "pt", "ar"],
        "sample_rate_hz": 24000,
        "modes": ["ivc", "preset"],
        "capabilities": ["voice_cloning", "custom_voice", "instruct_style"],
    }))


if __name__ == "__main__":
    import fire
    fire.Fire({"prep": prep, "say": say, "info": info})
