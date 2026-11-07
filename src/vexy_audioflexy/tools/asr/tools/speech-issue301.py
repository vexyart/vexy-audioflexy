#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = ["fire>=0.7,<0.8", "rich", "soundfile", "numpy", "platformdirs"]
# ///
# this_file: src/vexy_audioflexy/tools/asr/tools/speech-issue301.py
"""Consumer entry point for qualified workspace recognition and native audio tasks."""

import json
import re
import runpy
import tempfile
from pathlib import Path

from _audioflexy_paths import models_root

ROOT = Path(__file__).resolve().parents[1]


def retain_text(text: str, aligned: list[dict]) -> list[dict]:
    """Keep ASR spelling/punctuation while rejecting aligner omissions or rewrites."""
    original = text.split()
    normalize = lambda s: re.sub(r"[\W_]+", "", s.casefold())
    if len(original) != len(aligned) or any(
        normalize(word) != normalize(item["text"])
        for word, item in zip(original, aligned)
    ):
        raise ValueError("Forced alignment changed recognized lexical words")
    return [
        {**item, "text": word} for word, item in zip(original, aligned, strict=True)
    ]


def native_model(family: str) -> Path:
    """Resolve the declared exact package or the documented local Whistle conversion."""
    models = models_root() / "audio.cpp-issue301"
    if family == "whistle_asr":
        return models / "Whistle-GGUF/whistle-f32.gguf"
    catalog = json.loads((ROOT / "catalogs/issue301-audiocpp.json").read_text())[
        "models"
    ]
    item = next((r for r in catalog if r["family"] == family), None)
    if not item or not item["package"]:
        raise ValueError(f"Unknown requested native family: {family}")
    spec = json.loads((ROOT / f"model_specs/issue301/{family}.json").read_text())
    package = next(p for p in spec["packages"] if p["id"] == item["package"])
    root = models / package["target_directory"]
    if len(package["files"]) == 1 and package["files"][0].endswith(".gguf"):
        relative = package["files"][0].removeprefix(
            package.get("strip_prefix", "") + "/"
        )
        return root / relative
    return root


def task(
    audio: str,
    output: str,
    family: str,
    text: str = "",
    language: str = "",
    timeout: int = 1800,
    mode: str = "offline",
    backend: str = "metal",
) -> dict:
    """Run the validated native task adapter; results preserve their task-specific type."""
    adapter = runpy.run_path(str(Path(__file__).with_name("audio-task-audiocpp.py")))
    adapter["run"](
        audio,
        output,
        family,
        str(native_model(family)),
        text=text,
        language=language,
        timeout=timeout,
        mode=mode,
        backend=backend,
    )
    return json.loads((Path(output) / "report.json").read_text())


def recognize(
    audio: str,
    output: str,
    model: str = "mlx-whisper-large-v3-turbo",
    language: str = "en",
    aligner: str = "qwen3_forced_aligner",
    prompt: str = "",
    timeout: int = 1800,
) -> dict:
    """Recognize through isolated adapters; use forced alignment when native words are absent."""
    destination = Path(output).resolve()
    if destination.exists():
        raise FileExistsError("Consumer result exists; choose a new output path")
    runner = runpy.run_path(str(ROOT / "asr_all.py"))
    entry = next((m for m in runner["MODELS"] if m["name"] == model), None)
    if not entry:
        raise ValueError(f"Unknown ASR model: {model}")
    entry = dict(entry)
    if prompt:
        if entry["script"] not in {"tools/asr-mlxwhisper.py", "tools/asr-mlx301.py"}:
            raise ValueError(
                "This ASR adapter does not expose custom recognition context"
            )
        arguments = entry["args_fn"]
        entry["args_fn"] = lambda *args: arguments(*args) + ["--prompt", prompt]
    with tempfile.TemporaryDirectory() as temporary:
        folder = Path(temporary)
        success, wall, error, result = runner["run_model"](
            entry,
            Path(audio).resolve(),
            folder / "transcript.txt",
            language,
            timeout,
            False,
        )
        if not success:
            raise RuntimeError(error)
        native_receipt = folder / "transcript.native.json"
        if native_receipt.exists():
            result = json.loads(native_receipt.read_text())
        text = result.get("text") or (folder / "transcript.txt").read_text().strip()
        if not text or not any(c.isalnum() for c in text):
            raise ValueError("Recognition produced no lexical transcript")
        words = result.get("words")
        alignment = {"native_word_estimates": True}
        if not words or any(w["end"] <= w["start"] for w in words):
            language_name = {"en": "English", "pl": "Polish"}.get(language, language)
            if aligner == "mms_forced_aligner":
                language_name = language
            alignment = task(
                audio, str(folder / "alignment"), aligner, text, language_name, timeout
            )
            words = retain_text(text, alignment["result"]["words"])
            alignment["native"] = json.loads((folder / "alignment/native.json").read_text())
        data = {
            "words": words,
            "raw": {
                "language": language,
                "recognition": result,
                "alignment": alignment,
            },
            "wall_s": wall,
        }
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    return data


if __name__ == "__main__":
    import fire

    fire.Fire({"recognize": recognize, "task": task})
