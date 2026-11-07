#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = ["fire", "rich", "soundfile", "numpy", "mlx-audio>=0.5.0", "huggingface-hub"]
# ///
# this_file: src/vexy_audioflexy/tools/asr/tools/asr-granite-nar-mlx.py
"""IBM Granite Speech 4.1 2B (non-autoregressive) on MLX via mlx-audio.

`mlx_audio.stt.utils.MODEL_REMAPPING` maps the HF `model_type: "granite_speech_nar"`
to its own `granite_speech_nar` model module, whose `Model.generate(audio) -> STTOutput`
is the whole API. bf16 only (no quantized conversion published), batch size 1,
no streaming, macOS 14+.
"""

from __future__ import annotations

import json
import time
import wave
from pathlib import Path

VERSION = "1.0.0"
DEFAULT_REPO = "mlx-community/granite-speech-4.1-2b-nar-mlx"


def prep(repo: str = DEFAULT_REPO) -> None:
    """Download and cache the MLX weights. Idempotent.

    Args:
        repo: HuggingFace repo id of the MLX conversion.
    """
    from huggingface_hub import snapshot_download

    print(f"[prep] ready: {repo} -> {snapshot_download(repo)}")


def _duration(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as wf:
            return wf.getnframes() / wf.getframerate()
    except Exception:
        return 0.0



TARGET_RATE = 16000


def _ensure_16k(src: Path, tmpdir: str) -> Path:
    """Return *src* if it is already 16 kHz mono, else an ffmpeg-converted copy.

    The Granite MLX checkpoint rejects anything else outright — "audio must be
    16000 Hz; got 24000" — and the speaker dataset is 24 kHz throughout, so every
    clip needs converting. The audio.cpp wrapper does the same for the same reason.
    """
    import shutil
    import subprocess
    import wave

    try:
        with wave.open(str(src), "rb") as w:
            if w.getframerate() == TARGET_RATE and w.getnchannels() == 1:
                return src
    except (OSError, wave.Error):
        pass  # not a plain PCM WAV; let ffmpeg deal with it

    if shutil.which("ffmpeg") is None:
        raise SystemExit(
            f"{src} is not {TARGET_RATE} Hz mono and ffmpeg is not installed to convert it."
        )
    dst = Path(tmpdir) / f"{src.stem}.16k.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
         "-ar", str(TARGET_RATE), "-ac", "1", str(dst)],
        check=True,
    )
    return dst


def transcribe(
    audio: str,
    repo: str = DEFAULT_REPO,
    text_out: str = "",
    lang: str = "",
    force: bool = False,
) -> None:
    """Transcribe one audio file.

    Args:
        audio: Input audio file.
        repo: HuggingFace repo id of the MLX conversion.
        text_out: Transcript output path (default: *audio* with a `.txt` suffix).
        lang: Ignored — this checkpoint is English-only.
        force: Overwrite an existing transcript file.
    """
    from rich.console import Console

    console = Console()
    src = Path(audio)
    if not src.exists():
        raise SystemExit(f"audio not found: {src}")
    out_path = Path(text_out) if text_out else src.with_suffix(".txt")
    if not force and out_path.exists() and out_path.stat().st_size > 0:
        console.print(f"[yellow]Skipped:[/yellow] {out_path} already exists")
        return

    import tempfile

    from mlx_audio.stt.utils import load_model

    console.print(f"[cyan]granite-nar-mlx[/cyan] v{VERSION} repo=[yellow]{repo}[/yellow]")
    model = load_model(repo)
    with tempfile.TemporaryDirectory() as tmpdir:
        feed = _ensure_16k(src, tmpdir)
        t0 = time.monotonic()
        result = model.generate(str(feed))
        wall = time.monotonic() - t0

    transcript = (getattr(result, "text", None) or str(result)).strip()
    if not transcript:
        console.print("[red]empty transcript[/red]")
        raise SystemExit(1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(transcript + "\n", encoding="utf-8")
    audio_s = _duration(src)
    console.print(f"[green]Transcript[/green] {out_path.resolve()}\n  {transcript}")
    print("asr.result=" + json.dumps(
        {
            "text": transcript,
            "text_out": str(out_path),
            "wall_s": round(wall, 4),
            "audio_duration_s": round(audio_s, 4),
            "rtf": round(wall / audio_s, 6) if audio_s else 0.0,
            "repo": repo,
        },
        ensure_ascii=False, separators=(",", ":"),
    ))


def info(repo: str = DEFAULT_REPO) -> None:
    """Print one-line JSON with model metadata."""
    print(json.dumps(
        {"name": repo.split("/")[-1], "repo": repo, "stack": "mlx",
         "runtime": "mlx-audio", "quant": "bf16", "tasks": ["asr"],
         "languages": ["en"]},
        separators=(",", ":"),
    ))


if __name__ == "__main__":
    import fire

    fire.Fire({"prep": prep, "transcribe": transcribe, "info": info})
