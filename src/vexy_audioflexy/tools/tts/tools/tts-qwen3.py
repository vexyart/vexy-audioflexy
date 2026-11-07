#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "fire",
#   "rich",
#   "qwen-tts",
#   "soundfile",
#   "torch",
#   "transformers==4.57.3",
# ]
# ///
# this_file: src/vexy_audioflexy/tools/tts/tools/tts-qwen3.py
"""Qwen3-TTS multilingual TTS. Pins transformers==4.57.3 — may conflict with other packages."""

from pathlib import Path

VERSION = "1.1.0"
REPO = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
MODEL_BASE = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
MODEL_CUSTOM = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
MODEL_DESIGN = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"


def prep(force: bool = False):
    """Download + cache all model weights/assets WITHOUT synthesizing. Idempotent."""
    import torch
    from qwen_tts import Qwen3TTSModel
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    for repo in [MODEL_BASE, MODEL_CUSTOM]:
        Qwen3TTSModel.from_pretrained(repo, device_map=device, dtype=torch.bfloat16)
    print(f"[prep] ready: {REPO}")


def say(  # was: main
    text: str,
    output: str = "output.wav",
    language: str = "English",
    speaker: str = "Vivian",
    model: str = "",
    play: bool = False,
    speed: str = "normal",
    instruct: str = "",
    ref: str = "",
    ref_text: str = "",
    force: bool = False,
):
    """Generate speech using Qwen3-TTS multilingual model.

    Supports voice cloning (Base model), CustomVoice, and VoiceDesign variants.
    Pins transformers==4.57.3 which may conflict with other installed packages.

    Args:
        text: Text to synthesize.
        output: Output WAV file path.
        language: Language name (e.g. English, Chinese, French).
        speaker: Speaker name for CustomVoice (e.g. Vivian, Ethan).
        model: Model repo ID. Defaults to MODEL_BASE when --ref is given, MODEL_CUSTOM otherwise.
        play: Play audio via afplay instead of saving (macOS).
        speed: Speech speed preset: slow, normal, fast. Generates instruct text unless instruct is set.
        instruct: Raw style instruction overriding speed preset (e.g. 'Speak cheerfully and quickly.').
        ref: Path to reference audio WAV for voice cloning (requires Base model).
        ref_text: Transcript of reference audio. Empty string = auto-transcribe.
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

    import torch
    import soundfile as sf
    from qwen_tts import Qwen3TTSModel

    # Determine effective model: Base for cloning, CustomVoice for preset speakers
    cloning = bool(ref)
    if not model:
        model = MODEL_BASE if cloning else MODEL_CUSTOM

    # Build instruct string: explicit --instruct overrides --speed preset
    SPEED_INSTRUCTS = {
        "slow": "Speak slowly and clearly.",
        "normal": "Speak at a natural, conversational pace.",
        "fast": "Speak quickly and naturally.",
    }
    effective_instruct = (
        instruct if instruct else SPEED_INSTRUCTS.get(speed, SPEED_INSTRUCTS["normal"])
    )

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    console.print(f"[dim]Device: {device}[/dim]")
    console.print(f"[dim]Model: {model}[/dim]")
    if cloning:
        console.print(f"[dim]Mode: voice clone | ref: {ref}[/dim]")
    else:
        console.print(f"[dim]Language: {language} | Speaker: {speaker}[/dim]")
        console.print(f"[dim]Instruct: {effective_instruct}[/dim]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        t = progress.add_task("Loading model...", total=None)
        tts_model = Qwen3TTSModel.from_pretrained(
            model, device_map=device, dtype=torch.bfloat16
        )
        progress.update(t, description="Generating audio...")
        if cloning:
            wavs, sr = tts_model.generate_voice_clone(
                text=text,
                ref_audio=ref,
                ref_text=ref_text,
                language=language,
            )
        else:
            wavs, sr = tts_model.generate_custom_voice(
                text=text,
                language=language,
                speaker=speaker,
                instruct=effective_instruct,
            )

    if play:
        import tempfile, subprocess

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp = f.name
        sf.write(tmp, wavs[0], sr)
        subprocess.run(["afplay", tmp], check=True)
        Path(tmp).unlink(missing_ok=True)
        console.print("[green]Done.[/green]")
    else:
        sf.write(output, wavs[0], sr)
        console.print(f"[green]Saved:[/green] {output} ({sr} Hz)")


def info():
    """Print one-line JSON: name, repo, backend, languages, sample_rate_hz, modes, capabilities."""
    import json
    print(json.dumps({
        "name": "Qwen3-TTS",
        "repo": REPO,
        "backend": "torch",
        "languages": ["en", "zh", "fr", "de", "es", "pt", "it", "ja", "ko", "ar"],
        "sample_rate_hz": 24000,
        "modes": ["ivc", "preset"],
        "capabilities": ["voice_cloning", "multilingual", "custom_voice"],
    }))


if __name__ == "__main__":
    import fire
    fire.Fire({"prep": prep, "say": say, "info": info})
