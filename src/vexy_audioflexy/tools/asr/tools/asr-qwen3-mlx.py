#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = ["fire", "rich", "mlx-audio>=0.5.0", "huggingface-hub"]
# ///
# this_file: src/vexy_audioflexy/tools/asr/tools/asr-qwen3-mlx.py
"""Qwen3-ASR 1.7B on MLX via mlx-audio.

mlx-audio registers `qwen3_asr` as a native STT model module, so the HF
`model_type: "qwen3_asr"` conversions load through `mlx_audio.stt.utils.load_model`
with no bespoke code. Both the bf16 and 8bit mlx-community conversions use this
same path; pick one with `--repo`.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

VERSION = "1.0.0"
DEFAULT_REPO = "mlx-community/Qwen3-ASR-1.7B-bf16"


def prep(repo: str = DEFAULT_REPO) -> None:
    """Download and cache the MLX weights. Idempotent.

    Args:
        repo: HuggingFace repo id of the MLX conversion.
    """
    from huggingface_hub import snapshot_download

    path = snapshot_download(repo)
    print(f"[prep] ready: {repo} -> {path}")


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
        lang: Language hint, forwarded to the model when it accepts one.
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

    from mlx_audio.stt.utils import load_model

    console.print(f"[cyan]qwen3-asr-mlx[/cyan] v{VERSION} repo=[yellow]{repo}[/yellow]")
    model = load_model(repo)
    kwargs = {"language": lang} if lang else {}
    t0 = time.monotonic()
    result = model.generate(str(src), **kwargs)
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


def _duration(path: Path) -> float:
    import wave

    try:
        with wave.open(str(path), "rb") as wf:
            return wf.getnframes() / wf.getframerate()
    except Exception:
        return 0.0


def info(repo: str = DEFAULT_REPO) -> None:
    """Print one-line JSON with model metadata."""
    print(json.dumps(
        {
            "name": repo.split("/")[-1],
            "repo": repo,
            "stack": "mlx",
            "runtime": "mlx-audio",
            "quant": "8bit" if "8bit" in repo else "bf16",
            "tasks": ["asr"],
        },
        separators=(",", ":"),
    ))


if __name__ == "__main__":
    import fire

    fire.Fire({"prep": prep, "transcribe": transcribe, "info": info})
