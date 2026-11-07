#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = ["fire>=0.7,<0.8", "mlx-audio==0.5.7", "huggingface-hub", "soundfile", "sentencepiece>=0.2", "platformdirs"]
# ///
# this_file: src/vexy_audioflexy/tools/asr/tools/asr-mlx301.py
"""Requested MLX conversions, pinned separately from their base-model families."""

import dataclasses
import hashlib
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVISIONS = {
    "majentik/Qwen3-ASR-1.7B-MLX-8bit": "7f4c20a5a0d0dd5c60cd14cb2bde9cf02c09efc4",
    "littoralai/cohere-transcribe-mlx-fp16": "6a66668d9b9b26646fce93d70412ee33188b17e0",
    "divydeep/granite-speech-4.1-2b-mlx-8bit": "8b45a92464686505ee42658cdbbd1946c2762b5b",
}
DEFAULT_REPO = "littoralai/cohere-transcribe-mlx-fp16"


def prep(repo: str = DEFAULT_REPO) -> str:
    """Download the exact named conversion; never replace it with a similar export."""
    from huggingface_hub import snapshot_download

    if repo not in REVISIONS:
        raise ValueError("Unknown issue 301 conversion")
    return snapshot_download(
        repo,
        revision=REVISIONS[repo],
        cache_dir=None,
        allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model", "*.md"],
    )


def generation_options(repo: str, language: str, prompt: str) -> dict:
    """Prevent silent prompt ignoring and Granite's language-driven translation mode."""
    if repo not in REVISIONS:
        raise ValueError("Unknown issue 301 conversion")
    if repo.startswith("littoralai/"):
        if prompt:
            raise ValueError(
                "This Cohere runtime accepts language/punctuation, not custom context"
            )
        return {"language": language or "en", "punctuation": True, "max_tokens": 4096}
    if repo.startswith("divydeep/"):
        instruction = (
            "Transcribe the speech with proper punctuation and capitalization."
        )
        return {
            "prompt": instruction + ("\n" + prompt if prompt else ""),
            "max_tokens": 4096,
        }
    return {
        "language": {"en": "English", "pl": "Polish"}.get(language, language or None),
        "system_prompt": prompt or None,
        "max_tokens": 4096,
    }


def text_result(result: object) -> str:
    """A printable model object is not a recognized transcript."""
    text = getattr(result, "text", None)
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Model returned no recognized text")
    return text.strip()


def transcribe(
    audio: str,
    text_out: str = "",
    repo: str = DEFAULT_REPO,
    lang: str = "en",
    prompt: str = "",
    prompt_file: str = "",
    force: bool = False,
    verbose: bool = False,
) -> None:
    """Retain raw output and distinguish model chunk intervals from aligned words."""
    source = Path(audio).resolve()
    output = Path(text_out).resolve() if text_out else source.with_suffix(".txt")
    receipt = output.with_suffix(".native.json")
    if not source.is_file() or source in {output, receipt}:
        raise ValueError("Require existing audio and distinct output paths")
    if not force and (output.exists() or receipt.exists()):
        raise FileExistsError("Output exists; use --force")
    if prompt_file:
        content = Path(prompt_file).read_text(encoding="utf-8").strip()
        if not content:
            raise ValueError("prompt_file is empty")
        prompt = "\n".join(filter(None, (prompt, content)))
    options = generation_options(repo, lang, prompt)
    import mlx.core as mx
    import soundfile as sf
    from mlx_audio.stt.utils import load_model

    duration = sf.info(source).duration
    if duration <= 0:
        raise ValueError("Input audio is empty")
    snapshot = prep(repo)
    runtime = snapshot
    if repo.startswith("majentik/"):
        from qwen301_export import prepare

        runtime = prepare(snapshot)
    start = time.monotonic()
    model = load_model(runtime, strict=True)
    mx.synchronize()
    loaded = time.monotonic()
    result = model.generate(str(source), **options, verbose=verbose)
    mx.synchronize()
    wall = time.monotonic() - loaded
    text = text_result(result)
    raw = (
        dataclasses.asdict(result)
        if dataclasses.is_dataclass(result)
        else {"text": text, "segments": getattr(result, "segments", None)}
    )
    data = {
        "text": text,
        "repo": repo,
        "revision": REVISIONS[repo],
        "snapshot": snapshot,
        "runtime_snapshot": runtime,
        "text_out": str(output),
        "load_s": loaded - start,
        "wall_s": wall,
        "audio_duration_s": duration,
        "rtf": wall / duration,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "timing_contract": "chunk_intervals_only_no_word_alignment",
        "raw": raw,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    output.write_text(text + "\n", encoding="utf-8")
    print(
        "asr.result="
        + json.dumps({k: v for k, v in data.items() if k != "raw"}, ensure_ascii=False)
    )


def info() -> None:
    """List exact conversion identities without loading a model."""
    print(json.dumps(REVISIONS, indent=2))


if __name__ == "__main__":
    import fire

    fire.Fire({"prep": prep, "transcribe": transcribe, "info": info})
