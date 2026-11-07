#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = ["fire>=0.7,<0.8", "mlx-whisper==0.4.3", "huggingface-hub"]
# ///
# this_file: src/vexy_audioflexy/tools/asr/tools/asr-mlxwhisper.py
"""Pinned MLX Whisper recognition with context and native word-timing receipts."""

import hashlib
import itertools
import json
import math
import subprocess
import time
from pathlib import Path

DEFAULT_REPO = "mlx-community/whisper-large-v3-turbo"
DEFAULT_REVISION = "a4aaeec0636e6fef84abdcbe3544cb2bf7e9f6fb"


def prep(repo: str = DEFAULT_REPO, revision: str = DEFAULT_REVISION) -> str:
    """Resolve a pinned model snapshot; reuse the standard Hugging Face cache."""
    from huggingface_hub import snapshot_download

    if not revision:
        raise ValueError("A model revision is required")
    return snapshot_download(repo, revision=revision)


def duration(source: Path) -> float:
    """Probe actual media duration before accepting model word timestamps."""
    value = float(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "csv=p=0",
                str(source),
            ],
            text=True,
            timeout=30,
        ).strip()
    )
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Require positive finite audio duration")
    return value


def context(prompt: str, prompt_file: str) -> str:
    """Forward optional vocabulary lines; free-form hints remain supported."""
    text = prompt
    if prompt_file:
        path = Path(prompt_file)
        if path.suffix.lower() not in {".md", ".txt"}:
            raise ValueError("prompt_file must be UTF-8 Markdown or text")
        content = path.read_text(encoding="utf-8").strip()
        if not content:
            raise ValueError("prompt_file is empty")
        text = "\n".join(filter(None, (prompt, content)))
    vocabulary = [
        term.strip().rstrip(".")
        for line in text.splitlines()
        if line.startswith(("Names:", "Terms:"))
        for term in line.split(":", 1)[1].split(",")
        if term.strip()
    ]
    return ", ".join(vocabulary) or text


def words_from(result: dict, audio_duration: float) -> list[dict]:
    """Accept native timings only; never manufacture word times from segment bounds."""
    words = [
        {"text": w["word"].strip(), "start": float(w["start"]), "end": float(w["end"])}
        for segment in result["segments"]
        for w in segment.get("words", [])
        if w["word"].strip()
    ]
    if not result["text"].strip() or not words:
        raise ValueError("Whisper returned no transcript with native word timestamps")
    if any(
        not math.isfinite(w["start"] + w["end"])
        or not 0 <= w["start"] <= w["end"] <= audio_duration + 0.1
        for w in words
    ):
        raise ValueError("Native word timings exceed the source timeline")
    if any(a["start"] > b["start"] for a, b in itertools.pairwise(words)):
        raise ValueError("Native word timestamps are out of order")
    return words


def transcribe(
    audio: str,
    text_out: str = "",
    lang: str = "en",
    prompt: str = "",
    prompt_file: str = "",
    repo: str = DEFAULT_REPO,
    revision: str = DEFAULT_REVISION,
    force: bool = False,
    verbose: bool = False,
) -> None:
    """Recognize timed words; validation precedes replacement of existing output."""
    source = Path(audio).resolve()
    output = Path(text_out).resolve() if text_out else source.with_suffix(".txt")
    receipt = output.with_suffix(".native.json")
    if not source.is_file() or source in {output, receipt}:
        raise ValueError("Require existing media and distinct output paths")
    if not force and (output.exists() or receipt.exists()):
        raise FileExistsError("Output exists; use --force to regenerate")
    hints = context(prompt, prompt_file)
    audio_duration = duration(source)
    import mlx_whisper

    start = time.monotonic()
    snapshot = prep(repo, revision)
    resolved = time.monotonic()
    raw = mlx_whisper.transcribe(
        str(source),
        path_or_hf_repo=snapshot,
        initial_prompt=hints or None,
        language=lang or None,
        word_timestamps=True,
        condition_on_previous_text=False,
        hallucination_silence_threshold=2.0,
        verbose=True if verbose else None,
    )
    words = words_from(raw, audio_duration)
    wall = time.monotonic() - resolved
    data = {
        "text": raw["text"].strip(),
        "words": words,
        "raw": raw,
        "text_out": str(output),
        "repo": repo,
        "revision": revision,
        "snapshot": snapshot,
        "wall_s": wall,
        "preparation_s": resolved - start,
        "audio_duration_s": audio_duration,
        "rtf": wall / audio_duration,
        "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "prompt_sha256": hashlib.sha256(hints.encode()).hexdigest(),
        "timing_contract": "native_word_estimates",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    output.write_text(data["text"] + "\n", encoding="utf-8")
    print(
        "asr.result="
        + json.dumps({k: v for k, v in data.items() if k != "raw"}, ensure_ascii=False)
    )


def info() -> None:
    """Describe the pinned route without loading inference libraries."""
    print(
        json.dumps(
            {
                "repo": DEFAULT_REPO,
                "revision": DEFAULT_REVISION,
                "stack": "mlx",
                "tasks": ["asr"],
                "word_timestamps": True,
            }
        )
    )


if __name__ == "__main__":
    import fire

    fire.Fire({"prep": prep, "transcribe": transcribe, "info": info})
