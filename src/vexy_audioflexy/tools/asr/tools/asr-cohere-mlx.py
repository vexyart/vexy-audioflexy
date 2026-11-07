#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.13"
# dependencies = ["fire", "rich", "mlx-speech", "huggingface-hub", "soundfile", "numpy"]
# ///
# this_file: src/vexy_audioflexy/tools/asr/tools/asr-cohere-mlx.py
"""Cohere Transcribe (03-2026) 8-bit on MLX via the `mlx-speech` package.

Two things about this model differ from every other wrapper here:

* `mlx-speech` requires Python >= 3.13, hence the raised floor in the PEP 723 header.
* The weights live under a `mlx-int8/` subfolder of the HF repo, not at its root.

The model card's snippet calls `CohereAsrModel.from_path(...)`, which does not exist;
the real constructor is `from_dir(model_dir)`. mlx-audio also remaps a `cohere_asr`
model type, but its weight-key compatibility with *this* conversion is unverified,
so the documented `mlx-speech` path is the one used here.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

VERSION = "1.0.0"
DEFAULT_REPO = "mlx-community/cohere-transcribe-03-2026-mlx-8bit"
WEIGHTS_SUBDIR = "mlx-int8"


def _snapshot(repo: str) -> Path:
    from huggingface_hub import snapshot_download

    local = snapshot_download(repo, allow_patterns=[f"{WEIGHTS_SUBDIR}/*"])
    return Path(local) / WEIGHTS_SUBDIR


def prep(repo: str = DEFAULT_REPO) -> None:
    """Download and cache the `mlx-int8/` weights subfolder. Idempotent.

    Args:
        repo: HuggingFace repo id of the MLX conversion.
    """
    print(f"[prep] ready: {repo} -> {_snapshot(repo)}")


TARGET_RATE = 16000


def transcribe(
    audio: str,
    repo: str = DEFAULT_REPO,
    text_out: str = "",
    lang: str = "en",
    force: bool = False,
) -> None:
    """Transcribe one audio file.

    Args:
        audio: Input audio file.
        repo: HuggingFace repo id of the MLX conversion.
        text_out: Transcript output path (default: *audio* with a `.txt` suffix).
        lang: Language code passed to `transcribe(language=...)`.
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

    import soundfile as sf
    from mlx_speech.generation import CohereAsrModel

    console.print(f"[cyan]cohere-asr-mlx[/cyan] v{VERSION} repo=[yellow]{repo}[/yellow]")
    model = CohereAsrModel.from_dir(str(_snapshot(repo)))
    samples, rate = sf.read(str(src), dtype="float32", always_2d=False)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    if rate != TARGET_RATE:
        # CohereAsr refuses anything but 16 kHz and the speaker dataset is 24 kHz
        # throughout, so resample rather than reject every clip. Linear
        # interpolation is enough for a speech-recognition feed.
        import numpy as np

        n = int(round(len(samples) * TARGET_RATE / rate))
        samples = np.interp(
            np.linspace(0.0, len(samples) - 1, n, dtype="float64"),
            np.arange(len(samples), dtype="float64"),
            samples,
        ).astype("float32")
        rate = TARGET_RATE
    t0 = time.monotonic()
    result = model.transcribe(samples, sample_rate=rate, language=lang or "en")
    wall = time.monotonic() - t0

    transcript = (getattr(result, "text", None) or str(result)).strip()
    if not transcript:
        console.print("[red]empty transcript[/red]")
        raise SystemExit(1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(transcript + "\n", encoding="utf-8")
    audio_s = len(samples) / rate if rate else 0.0
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
         "runtime": "mlx-speech", "quant": "8bit", "tasks": ["asr"]},
        separators=(",", ":"),
    ))


if __name__ == "__main__":
    import fire

    fire.Fire({"prep": prep, "transcribe": transcribe, "info": info})
