#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = ["fire", "inflect", "kanjize", "mlx", "numpy", "rich", "soundfile", "sudachidict-core", "sudachipy", "zonos-mlx @ git+https://github.com/xocialize/zonos-mlx.git"]
# ///
# this_file: src/vexy_audioflexy/tools/tts/tools/tts-zonos-mlx.py
"""Zonos v0.1 Transformer bf16 TTS via MLX on Apple Silicon. Backend: zonos-mlx on mlx-community/Zonos-v0.1-transformer-bf16. Languages: en. Sample rate: 44.1 kHz. Modes: ivc (instant voice cloning from reference WAV). Capabilities: voice_cloning, emotion_control (8-slot emotion vector: happiness/sadness/disgust/fear/surprise/anger/other/neutral), speaking_rate, pitch_std, fmax conditioning."""

from pathlib import Path
import re

VERSION = "1.0.0"
MODEL_REPO = "mlx-community/Zonos-v0.1-transformer-bf16"
REFERENCE_SAMPLE_RATE = 16000


def _emotion_vector(
    happiness: float,
    sadness: float,
    disgust: float,
    fear: float,
    surprise: float,
    anger: float,
    other: float,
    neutral: float,
) -> list[float]:
    return [happiness, sadness, disgust, fear, surprise, anger, other, neutral]


def _validate_ref(ref: str | None) -> None:
    if not ref:
        return
    ref_path = Path(ref).expanduser()
    if not ref_path.exists() or not ref_path.is_file():
        raise SystemExit(f"Reference audio not found: {ref}")


def _resample_mono(audio, source_rate: int, target_rate: int):
    import numpy as np

    if source_rate == target_rate:
        return audio.astype("float32", copy=False)
    if len(audio) == 0:
        return audio.astype("float32", copy=False)
    duration = len(audio) / float(source_rate)
    target_len = max(1, int(round(duration * target_rate)))
    source_x = np.linspace(0.0, duration, num=len(audio), endpoint=False)
    target_x = np.linspace(0.0, duration, num=target_len, endpoint=False)
    return np.interp(target_x, source_x, audio).astype("float32")


def _split_text(text: str, max_chars: int) -> list[str]:
    clean = " ".join(text.split())
    if len(clean) <= max_chars:
        return [clean]
    pieces = re.split(r"(?<=[.!?])\s+", clean)
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if not piece:
            continue
        if len(piece) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(piece[i : i + max_chars] for i in range(0, len(piece), max_chars))
            continue
        candidate = f"{current} {piece}".strip()
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = piece
    if current:
        chunks.append(current)
    return chunks


def prep(force: bool = False):
    """Download + cache all model weights/assets WITHOUT synthesizing. Idempotent."""
    from zonos_mlx.pipeline_mlx import ZonosPipeline
    ZonosPipeline.from_pretrained_mlx(MODEL_REPO)
    print(f"[prep] ready: {MODEL_REPO}")


def say(
    text: str,
    output: str = "output.wav",
    ref: str | None = None,
    speaking_rate: float = 15.0,
    fmax: float | None = None,
    pitch_std: float | None = None,
    language: str = "en-us",
    happiness: float = 0.0,
    sadness: float = 0.0,
    disgust: float = 0.0,
    fear: float = 0.0,
    surprise: float = 0.0,
    anger: float = 0.0,
    other: float = 0.1,
    neutral: float = 0.9,
    max_chars: int = 220,
    play: bool = False,
    force: bool = False,
) -> None:
    """Generate speech with the official Zonos MLX pipeline.

    Args:
        text: Text to synthesize.
        output: Output WAV path.
        ref: Optional reference audio for speaker embedding.
        speaking_rate: Zonos speaking-rate conditioning value.
        fmax: Optional fmax conditioning value.
        pitch_std: Optional pitch standard deviation conditioning value.
        language: Language code passed to the pipeline when supported.
        happiness: Emotion slot 0.
        sadness: Emotion slot 1.
        disgust: Emotion slot 2.
        fear: Emotion slot 3.
        surprise: Emotion slot 4.
        anger: Emotion slot 5.
        other: Emotion slot 6.
        neutral: Emotion slot 7.
        max_chars: Split long prompts into chunks no longer than this many chars.
        play: Play with afplay instead of keeping output.
        force: Overwrite output when it exists.
    """
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn

    console = Console()
    out_path = Path(output)
    if not play and not force and out_path.exists() and out_path.stat().st_size > 0:
        console.print(
            f"[yellow]Skipped:[/yellow] {output} already exists (use --force to overwrite)"
        )
        return
    _validate_ref(ref)

    try:
        import mlx.core as mx
        import numpy as np
        import soundfile as sf
        from zonos_mlx.pipeline_mlx import ZonosPipeline
    except ImportError as exc:
        raise SystemExit(
            "Zonos MLX dependencies are missing. Install this script with uv, or run:\n"
            "  uv pip install 'zonos-mlx @ git+https://github.com/xocialize/zonos-mlx.git' mlx soundfile inflect kanjize sudachipy sudachidict-core\n"
            f"Import error: {exc}"
        ) from exc

    console.print(f"[dim]Model: {MODEL_REPO}[/dim]")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading Zonos MLX...", total=None)
        pipeline = ZonosPipeline.from_pretrained_mlx(MODEL_REPO)
        speaker = None
        if ref:
            progress.update(task, description="Encoding speaker reference...")
            ref_audio, ref_sr = sf.read(ref, dtype="float32", always_2d=False)
            if getattr(ref_audio, "ndim", 1) > 1:
                ref_audio = ref_audio.mean(axis=1)
            ref_audio = _resample_mono(ref_audio, int(ref_sr), REFERENCE_SAMPLE_RATE)
            speaker = pipeline.make_speaker_embedding(mx.array(ref_audio))
        progress.update(task, description="Generating speech...")
        kwargs = {
            "speaker": speaker,
            "speaking_rate": speaking_rate,
            "emotion": _emotion_vector(
                happiness, sadness, disgust, fear, surprise, anger, other, neutral
            ),
        }
        if fmax is not None:
            kwargs["fmax"] = fmax
        if pitch_std is not None:
            kwargs["pitch_std"] = pitch_std
        if language:
            kwargs["language"] = language
        chunks = _split_text(text, max_chars=max_chars)
        audio_parts = []
        for index, chunk in enumerate(chunks, start=1):
            progress.update(
                task,
                description=f"Generating speech chunk {index}/{len(chunks)}...",
            )
            try:
                wav = pipeline.generate(chunk, **kwargs)
            except TypeError:
                kwargs.pop("language", None)
                wav = pipeline.generate(chunk, **kwargs)
            part = mx.array(wav)
            while getattr(part, "ndim", 1) > 1:
                part = part[0]
            audio_parts.append(part)
        progress.update(task, description="Done.")

    audio_np = np.concatenate([np.array(part) for part in audio_parts])
    sample_rate = int(pipeline.sampling_rate)

    if play:
        import subprocess
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, audio_np, sample_rate)
            tmp_path = Path(tmp.name)
        subprocess.run(["afplay", str(tmp_path)], check=True)
        tmp_path.unlink(missing_ok=True)
        console.print("[green]Done.[/green]")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), audio_np, sample_rate)
    console.print(f"[green]Saved:[/green] {out_path.resolve()} ({sample_rate} Hz)")


def info():
    """Print one-line JSON: name, repo, backend, languages, sample_rate_hz, modes, capabilities."""
    import json
    print(json.dumps({
        "name": "Zonos v0.1 Transformer bf16 MLX",
        "repo": MODEL_REPO,
        "backend": "mlx",
        "languages": ["en"],
        "sample_rate_hz": 44100,
        "modes": ["ivc"],
        "capabilities": ["voice_cloning", "emotion_control", "speaking_rate", "pitch_std", "fmax"],
    }))


if __name__ == "__main__":
    import fire
    fire.Fire({"prep": prep, "say": say, "info": info})
