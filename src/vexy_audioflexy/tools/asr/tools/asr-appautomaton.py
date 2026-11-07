#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.13"
# dependencies = ["fire", "huggingface-hub", "soundfile", "mlx-speech @ git+https://github.com/appautomaton/mlx-speech@4577ad28e6bef2ae2a11b8fa76a6cdb23bb2a9cc"]
# ///
# this_file: src/vexy_audioflexy/tools/asr/tools/asr-appautomaton.py
"""Exact App Automaton Qwen3-ASR int8 through its own pinned MLX runtime."""
from __future__ import annotations

import json
import subprocess
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = "appautomaton/qwen3-asr-1.7b-int8-mlx"
REVISION = "488cf63af6acf68448b0a8d2e7f88e1d6bb1b9fb"


def language(lang: str) -> str | None:
    """This runtime's published validated scope is English/Chinese/automatic."""
    hints = {"": None, "en": "English", "English": "English", "zh": "Chinese", "Chinese": "Chinese"}
    if lang not in hints:
        raise SystemExit("This conversion/runtime supports en/zh language hints")
    return hints[lang]


def text_result(result: object) -> str:
    """Reject empty native text; never turn a Python object representation into ASR."""
    text = getattr(result, "text", None)
    if not isinstance(text, str) or not text.strip():
        raise SystemExit("Native ASR returned no transcript")
    return text.strip()


def prep(repo: str = REPO) -> str:
    """Pin the exact int8 export and use the workspace cache."""
    from huggingface_hub import snapshot_download

    if repo != REPO:
        raise SystemExit(f"Expected exact repository {REPO}")
    path = snapshot_download(REPO, revision=REVISION, cache_dir=None,
                             allow_patterns=["*.json", "*.safetensors", "*.txt"])
    print(f"[prep] ready: {REPO} -> {path}")
    return path


def transcribe(audio: str, text_out: str = "", repo: str = REPO, lang: str = "",
               force: bool = False, timeout: int = 900) -> None:
    """Write a transcript with synchronized inference/load timing; greedy decode."""
    import mlx.core as mx
    import soundfile as sf
    from mlx_speech.asr import load

    hint = language(lang)
    source = Path(audio).resolve()
    output = Path(text_out).resolve() if text_out else source.with_suffix(".txt")
    if not source.is_file() or output == source or timeout <= 0:
        raise SystemExit("Require audio, distinct transcript path and positive timeout")
    if output.exists() and not force:
        raise SystemExit(f"Transcript exists: {output}; use --force")
    path = prep(repo)
    start = time.monotonic()
    model = load(path)
    mx.synchronize()
    load_s = time.monotonic() - start
    with tempfile.TemporaryDirectory() as temporary:
        converted = Path(temporary) / "input.wav"
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(source), "-ar", "16000", "-ac", "1",
                        str(converted)], check=True, timeout=timeout)
        duration = sf.info(converted).duration
        if duration <= 0:
            raise SystemExit("Input audio is empty")
        start = time.monotonic()
        result = model.generate(str(converted), language=hint, max_new_tokens=512)
        mx.synchronize()
        wall = time.monotonic() - start
    text = text_result(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text + "\n")
    receipt = {"text": text, "text_out": str(output), "wall_s": wall,
        "load_s": load_s, "audio_duration_s": duration, "rtf": wall / duration,
        "repo": REPO, "weights_revision": REVISION, "language": result.language}
    output.with_suffix(".native.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
    print("asr.result=" + json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    import fire

    fire.Fire({"prep": prep, "transcribe": transcribe})
