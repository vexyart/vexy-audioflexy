#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = ["fire", "rich", "txtai[pipeline]", "soundfile", "nltk"]
# ///
# this_file: src/vexy_audioflexy/tools/tts/tools/tts-kokoro-fp16-onnx.py
"""Kokoro-82M TTS via ONNX (NeuML/kokoro-fp16-onnx). Backend: onnx/txtai, languages: en/ja/ko/zh/fr, 24 kHz mono, modes: auto/ivc, capabilities: voice_cloning. 54 voice presets; CPU inference, no GPU required."""

from pathlib import Path

VERSION = "1.0.0"
MODEL = "NeuML/kokoro-fp16-onnx"

VOICES = [
    "af",
    "af_sarah",
    "af_heart",
    "af_nicole",
    "af_sky",
    "am_adam",
    "am_michael",
    "bf_emma",
    "bf_isabella",
    "bm_lewis",
    "bm_sophie",
]


def prep(force: bool = False):
    """Download + cache all model weights WITHOUT synthesizing. Idempotent."""
    from rich.console import Console
    import nltk
    from txtai.pipeline import TextToSpeech

    console = Console()
    with console.status(f"[cyan]Downloading/caching model [bold]{MODEL}[/bold]…"):
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
        nltk.download("averaged_perceptron_tagger_eng", quiet=True)
        TextToSpeech(MODEL)
    print(f"[prep] ready: {MODEL}")


def say(
    text: str,
    output: str = "output.wav",
    voice: str = "af_heart",
    speed: float = 1.0,
    play: bool = False,
    force: bool = False,
):
    """Generate speech using Kokoro-82M ONNX (CPU inference, 24kHz output).

    Args:
        text: Text to synthesize.
        output: Output WAV file path (24kHz mono).
        voice: Voice preset. Common choices: af_heart, am_adam, bf_emma,
               bm_lewis. 54 presets available.
        speed: Speed multiplier (0.5–2.0).
        play: Play audio via sounddevice instead of saving to file.
        force: Overwrite output file if it exists.
    """
    from rich.console import Console
    from rich.panel import Panel

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

    console.print(
        Panel.fit(
            f"[bold cyan]Kokoro-82M TTS (ONNX)[/bold cyan] v{VERSION}\n"
            f"voice=[yellow]{voice}[/yellow]  "
            f"speed=[yellow]{speed}[/yellow]",
            border_style="cyan",
        )
    )

    with console.status(f"[cyan]Loading model [bold]{MODEL}[/bold]…"):
        import nltk

        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
        nltk.download("averaged_perceptron_tagger_eng", quiet=True)
        from txtai.pipeline import TextToSpeech

        tts = TextToSpeech(MODEL)

    console.print(f"[green]Model loaded.[/green] Synthesising {len(text)} chars…")

    with console.status("[cyan]Generating audio…"):
        speech, rate = tts(text, speaker=voice)

    if speech is None or len(speech) == 0:
        console.print("[red]Error: model returned no audio.[/red]")
        raise SystemExit(1)

    sample_rate = 24000

    if play:
        import sounddevice as sd

        duration = len(speech) / sample_rate
        console.print(f"[cyan]Playing {duration:.1f}s of audio…[/cyan]")
        sd.play(speech, samplerate=sample_rate)
        sd.wait()
        console.print("[green]Done.[/green]")
    else:
        import soundfile as sf

        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        sf.write(str(out_path), speech, sample_rate)

        duration = len(speech) / sample_rate
        size_kb = out_path.stat().st_size / 1024
        console.print(
            f"[green]Saved[/green] [bold]{out_path}[/bold] "
            f"({duration:.1f}s, {size_kb:.1f} KB, {sample_rate} Hz)"
        )


def info():
    """Print one-line JSON with model metadata."""
    import json

    print(json.dumps({
        "name": "Kokoro 82M ONNX fp16",
        "repo": MODEL,
        "backend": "onnx",
        "languages": ["en", "ja", "ko", "zh", "fr"],
        "sample_rate_hz": 24000,
        "modes": ["auto", "ivc"],
        "capabilities": ["voice_cloning"],
    }))


if __name__ == "__main__":
    import fire
    fire.Fire({"prep": prep, "say": say, "info": info})
