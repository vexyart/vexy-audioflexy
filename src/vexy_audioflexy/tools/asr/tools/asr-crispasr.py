#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = ["fire", "huggingface-hub", "platformdirs"]
# ///
# this_file: src/vexy_audioflexy/tools/asr/tools/asr-crispasr.py
"""CrispASR transcription for the exact Phonon 2 and Parakeet Redux GGUF exports.

Uses the sibling's native CrispASR build; CRISPASR_ROOT selects another checkout.
Its output-file argument is a basename: CrispASR appends .txt itself.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
import wave
from pathlib import Path

from _audioflexy_paths import runtime_root

ROOT = Path(__file__).resolve().parents[1]
ENGINE_ROOT = Path(os.environ.get("CRISPASR_ROOT", runtime_root("CrispASR")))
PACKAGES = {
    "phonon2": {"repo": "cstr/phonon2-GGUF", "filename": "phonon2-q8_0.gguf", "langs": ["en"]},
    "parakeet-redux": {"repo": "cstr/parakeet-redux-GGUF", "filename": "parakeet-redux-q8_0.gguf", "langs": ["en", "pl"]},
}


def resolve(package: str) -> dict:
    """Reject unknown packages before downloading or launching a subprocess."""
    if package not in PACKAGES:
        raise SystemExit(f"Unknown CrispASR package: {package}; available: {list(PACKAGES)}")
    return PACKAGES[package]


def prep(package: str = "phonon2") -> str:
    """Fetch the requested conversion rather than substituting the base checkpoint."""
    from huggingface_hub import hf_hub_download

    spec = resolve(package)
    path = hf_hub_download(repo_id=spec["repo"], filename=spec["filename"])
    print(f"[prep] ready: {package} -> {path}")
    return path


def command(model: str, audio: str, basename: str, lang: str, backend: str) -> list[str]:
    """Build argv using the compiled CLI's backend and transcript-file switches."""
    return [str(ENGINE_ROOT / "build/bin/crispasr"), "--backend", "parakeet",
            "--gpu-backend", backend, "-m", model, "-f", audio, "-l", lang or "auto",
            "--output-txt", "--output-file", basename, "--no-timestamps"]


def transcribe(audio: str, package: str = "phonon2", text_out: str = "",
               lang: str = "", backend: str = "metal", timeout: int = 900,
               force: bool = False, verbose: bool = False) -> None:
    """Run one clip and emit wall time, duration, RTF and the unaltered transcript."""
    spec = resolve(package)
    src = Path(audio).resolve()
    if not src.is_file():
        raise SystemExit(f"Audio not found: {src}")
    if lang and lang not in spec["langs"]:
        raise SystemExit(f"{package} does not declare language {lang}")
    if timeout <= 0:
        raise SystemExit("timeout must be positive")
    if not (ENGINE_ROOT / "build/bin/crispasr").is_file():
        raise SystemExit(f"Build CrispASR at {ENGINE_ROOT}; see https://github.com/CrispStrobe/CrispASR")
    out = Path(text_out).resolve() if text_out else src.with_suffix(".txt")
    if out == src:
        raise SystemExit("Transcript output must differ from audio input")
    if out.exists() and not force:
        raise SystemExit(f"Transcript exists: {out}; use --force")
    model = prep(package)
    with tempfile.TemporaryDirectory() as tmp:
        wav, base = Path(tmp) / "input.wav", Path(tmp) / "transcript"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
                        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(wav)],
                       check=True, timeout=timeout)
        with wave.open(str(wav), "rb") as wf:
            duration = wf.getnframes() / wf.getframerate()
        if duration <= 0:
            raise SystemExit("Audio is empty")
        start = time.monotonic()
        result = subprocess.run(command(model, str(wav), str(base), lang, backend),
                                capture_output=True, text=True, timeout=timeout, cwd=ENGINE_ROOT, check=False)
        wall = time.monotonic() - start
        if verbose or result.returncode:
            print(result.stdout + result.stderr)
        if result.returncode:
            raise SystemExit(result.returncode)
        text = base.with_suffix(".txt").read_text(encoding="utf-8").strip()
        if not text:
            raise SystemExit("CrispASR returned an empty transcript")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    print("asr.result=" + json.dumps({"text": text, "wall_s": wall,
          "audio_duration_s": duration, "rtf": wall / duration, "package": package,
          "backend": backend}, ensure_ascii=False))


if __name__ == "__main__":
    import fire

    fire.Fire({"prep": prep, "transcribe": transcribe, "packages": lambda: PACKAGES})
