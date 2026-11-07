#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12,<3.15"
# dependencies = [
#   "fire", "rich", "mlx==0.32.0", "transformers==4.57.6",
#   "librosa", "safetensors", "numpy", "huggingface-hub[hf-xet]",
# ]
# ///
# this_file: src/vexy_audioflexy/tools/asr/tools/asr-ark-mlx.py
"""Ark-ASR on MLX, using the runtime vendored inside the model repo itself.

Ark-ASR is `model_type: "arkasr"`, which neither mlx-audio, mlx-lm nor Transformers
AutoModel registers — the repo README says so outright. What the repo *does* ship is
a complete standalone package under `src/ark_asr_mlx/`, which is not published to
PyPI. So `prep` snapshots the repo and `transcribe` puts that `src/` on `sys.path`;
the package's own dependencies (pinned `mlx==0.32.0` and `transformers==4.57.6`,
taken from its pyproject) are declared in the PEP 723 header above.
"""

from __future__ import annotations

import json
import sys
import time
import wave
from pathlib import Path

VERSION = "1.0.0"
DEFAULT_REPO = "leope/ark-asr-0.6B-mlx"


def _snapshot(repo: str) -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo))


def prep(repo: str = DEFAULT_REPO) -> None:
    """Download the model repo, which also carries its own runtime source.

    Args:
        repo: HuggingFace repo id (`leope/ark-asr-0.6B-mlx` or `-3B-`).
    """
    local = _snapshot(repo)
    src = local / "src"
    if not (src / "ark_asr_mlx").is_dir():
        raise SystemExit(
            f"{repo} does not contain the expected src/ark_asr_mlx/ runtime — "
            f"the repo layout changed and this wrapper needs updating."
        )
    print(f"[prep] ready: {repo} -> {local}")


def _duration(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as wf:
            return wf.getnframes() / wf.getframerate()
    except Exception:
        return 0.0


def transcribe(
    audio: str,
    repo: str = DEFAULT_REPO,
    text_out: str = "",
    lang: str = "",
    max_new_tokens: int = 256,
    force: bool = False,
) -> None:
    """Transcribe one audio file.

    Args:
        audio: Input audio file.
        repo: HuggingFace repo id.
        text_out: Transcript output path (default: *audio* with a `.txt` suffix).
        lang: Ignored — the vendored API takes no language argument.
        max_new_tokens: Decoder token budget.
        force: Overwrite an existing transcript file.
    """
    from rich.console import Console

    console = Console()
    src_audio = Path(audio)
    if not src_audio.exists():
        raise SystemExit(f"audio not found: {src_audio}")
    out_path = Path(text_out) if text_out else src_audio.with_suffix(".txt")
    if not force and out_path.exists() and out_path.stat().st_size > 0:
        console.print(f"[yellow]Skipped:[/yellow] {out_path} already exists")
        return

    local = _snapshot(repo)
    sys.path.insert(0, str(local / "src"))
    from ark_asr_mlx import ArkASR

    console.print(f"[cyan]ark-asr-mlx[/cyan] v{VERSION} repo=[yellow]{repo}[/yellow]")
    model = ArkASR.from_pretrained(str(local))
    t0 = time.monotonic()
    result = model.transcribe(str(src_audio), max_new_tokens=max_new_tokens)
    wall = time.monotonic() - t0

    transcript = (getattr(result, "text", None) or str(result)).strip()
    if not transcript:
        console.print("[red]empty transcript[/red]")
        raise SystemExit(1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(transcript + "\n", encoding="utf-8")
    audio_s = _duration(src_audio)
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
         "runtime": "ark-asr-mlx (vendored in repo)", "quant": "bf16",
         "tasks": ["asr"]},
        separators=(",", ":"),
    ))


if __name__ == "__main__":
    import fire

    fire.Fire({"prep": prep, "transcribe": transcribe, "info": info})
