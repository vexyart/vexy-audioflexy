#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = ["fire", "huggingface-hub", "platformdirs"]
# ///
# this_file: src/vexy_audioflexy/tools/asr/tools/asr-openasr.py
"""OpenASR native .oasr packs using its pinned published model registry.

OPENASR_ROOT selects a built checkout. Native CLI output stays unaltered.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import time
import wave
from pathlib import Path

from _audioflexy_paths import runtime_root, data_root

ROOT = Path(__file__).resolve().parents[1]
ENGINE_ROOT = Path(os.environ.get("OPENASR_ROOT", runtime_root("openasr")))
PACKAGES = {entry["package"]: entry for entry in json.loads((ROOT / "catalogs/openasr.json").read_text())["models"]}
BINARY = ENGINE_ROOT / "target/release/openasr"



def resolve(package: str) -> dict:
    """Reject unknown packages before downloading or launching a subprocess."""
    if package not in PACKAGES:
        raise SystemExit(f"Unknown OpenASR package: {package}; available: {list(PACKAGES)}")
    return PACKAGES[package]


def prep(package: str = "moonshine-tiny") -> str:
    """Fetch the requested conversion rather than substituting the base checkpoint."""
    from huggingface_hub import hf_hub_download

    spec = resolve(package)
    path = hf_hub_download(repo_id=spec["repo"], filename=spec["filename"], revision=spec["revision"])
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != spec["sha256"]:
        raise SystemExit("Checkpoint checksum differs from OpenASR model index")
    print(f"[prep] ready: {package} -> {path}")
    return path


def command(model: str, audio: str, output: str, lang: str, backend: str, family: str = "whisper") -> list[str]:
    """Use a pinned pack path; the native runtime chooses its execution target."""
    if backend != "auto":
        raise SystemExit("OpenASR execution target is configured by its native preferences; use auto")
    argv = [str(BINARY), "transcribe", audio, "--model-pack", model,
            "--offline", "--format", "text", "--output", output]
    return argv + (["--language", lang] if lang and family == "whisper" else [])


def transcribe(audio: str, package: str = "moonshine-tiny", text_out: str = "",
               lang: str = "", backend: str = "auto", timeout: int = 900,
               force: bool = False, verbose: bool = False) -> None:
    """Run one clip and emit wall time, duration, RTF and the unaltered transcript."""
    spec = resolve(package)
    src = Path(audio).resolve()
    if not src.is_file():
        raise SystemExit(f"Audio not found: {src}")
    if lang and lang not in spec["langs"] and lang not in {item.split("-")[0] for item in spec["langs"]}:
        raise SystemExit(f"{package} does not declare language {lang}")
    if timeout <= 0:
        raise SystemExit("timeout must be positive")
    if backend != "auto":
        raise SystemExit("OpenASR uses native execution preferences; backend must be auto")
    if not BINARY.is_file():
        raise SystemExit(f"Build OpenASR at {ENGINE_ROOT}; see https://github.com/QuintinShaw/openasr")
    out = Path(text_out).resolve() if text_out else src.with_suffix(".txt")
    if out == src:
        raise SystemExit("Transcript output must differ from audio input")
    if out.exists() and not force:
        raise SystemExit(f"Transcript exists: {out}; use --force")
    model = prep(package)
    with tempfile.TemporaryDirectory() as tmp:
        wav, base = Path(tmp) / "input.wav", Path(tmp) / "transcript.txt"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
                        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(wav)],
                       check=True, timeout=timeout)
        with wave.open(str(wav), "rb") as wf:
            duration = wf.getnframes() / wf.getframerate()
        if duration <= 0:
            raise SystemExit("Audio is empty")
        env = os.environ.copy()
        env.setdefault("OPENASR_HOME", str(data_root() / "openasr"))
        start = time.monotonic()
        result = subprocess.run(command(model, str(wav), str(base), lang, backend, spec["family"]),
                                capture_output=True, text=True, timeout=timeout, cwd=ENGINE_ROOT, env=env, check=False)
        wall = time.monotonic() - start
        if verbose or result.returncode:
            print(result.stdout + result.stderr)
        if result.returncode:
            raise SystemExit(result.returncode)
        text = base.read_text(encoding="utf-8").strip()
        if not text:
            raise SystemExit("OpenASR returned an empty transcript")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    print("asr.result=" + json.dumps({"text": text, "wall_s": wall,
          "audio_duration_s": duration, "rtf": wall / duration, "package": package,
          "backend": backend}, ensure_ascii=False))


if __name__ == "__main__":
    import fire

    fire.Fire({"prep": prep, "transcribe": transcribe, "packages": lambda: PACKAGES})
