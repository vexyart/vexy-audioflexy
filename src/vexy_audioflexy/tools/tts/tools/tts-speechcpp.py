#!/usr/bin/env -S uv run -s
# /// script
# requires-python = ">=3.11"
# dependencies = ["fire"]
# ///
# this_file: src/vexy_audioflexy/tools/tts/tools/tts-speechcpp.py
"""Fish Audio Drama 3 Q6_K through speech.cpp >=0.8.8.

Uses speech.cpp's own resumable, SHA-256 checked model download. SPEECHCPP_BIN
selects the executable; SPEECH_MODEL_DIR relocates its native model cache.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path
from _paths import runtime_path

ROOT = Path(__file__).resolve().parents[1]
MODEL = "drama-3:q6_k"


def binary() -> str:
    """Use an explicit binary, the local release, or the installed command."""
    local = runtime_path('speech.cpp-0.8.11/speech')
    path = os.environ.get("SPEECHCPP_BIN") or (
        str(local) if local.is_file() else shutil.which("speech")
    )
    if not path:
        raise SystemExit(
            "Install speech.cpp >=0.8.8: https://github.com/nyosegawa/speech.cpp/blob/main/docs/install.md (or set SPEECHCPP_BIN)"
        )
    return path


def prep() -> None:
    """Download only sakasegawa/drama-3-GGUF's Q6_K file (29.9 GB)."""
    subprocess.run([binary(), "pull", MODEL], check=True, timeout=14400)


def say(
    text: str,
    output: str = "output.wav",
    ref: str = "",
    ref_text: str = "",
    lang: str = "",
    device: str = "auto",
    seed: int = 0,
    force: bool = False,
) -> None:
    """Speak text/tags, optionally cloning a WAV with its required transcript.

    Reference encoding uses the CPU, as recommended upstream. A failed or
    truncated synthesis preserves any earlier output. Language is inferred from
    text; --lang asks the engine to check its supported language list.
    """
    if not text.strip():
        raise SystemExit("Text must not be empty")
    if bool(ref) != bool(ref_text.strip()) or (ref and not Path(ref).is_file()):
        raise SystemExit(
            "Cloning requires an existing --ref WAV and nonempty --ref-text transcript"
        )
    out = Path(output).resolve()
    if ref and out == Path(ref).resolve():
        raise SystemExit("Output must differ from the reference recording")
    if out.exists() and not force:
        print(f"Skipped: {out} already exists (use --force)")
        return
    engine = binary()
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".speechcpp-", dir=out.parent) as work:
        wav_path = Path(work) / "speech.wav"
        cmd = [
            engine,
            "tts",
            MODEL,
            "--device",
            device,
            "--seed",
            str(seed),
            "-o",
            str(wav_path),
        ]
        if ref:
            voice = Path(work) / "reference.voice.gguf"
            subprocess.run(
                [
                    engine,
                    "voice",
                    MODEL,
                    str(Path(ref).resolve()),
                    str(voice),
                    "--transcript",
                    ref_text,
                    "--device",
                    "cpu",
                ],
                check=True,
                timeout=3600,
            )
            cmd += ["--add-voice", f"reference={voice}", "--voice", "reference"]
        else:
            cmd += ["--voice", "none"]
        if lang:
            cmd += ["--language", lang]
        subprocess.run(cmd + ["--", text], check=True, timeout=3600)
        with wave.open(str(wav_path), "rb") as wav:
            if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (
                1,
                2,
                44100,
            ):
                raise ValueError("Expected mono 16-bit 44.1 kHz Drama 3 output")
            frames = wav.readframes(wav.getnframes())
            if not frames or len(frames) != wav.getnframes() * 2 or not any(frames):
                raise ValueError("Drama 3 returned empty, truncated or silent audio")
        os.replace(wav_path, out)
    print(f"Saved: {out}")


def info() -> None:
    """Print the selected Q6_K model's native metadata as JSON."""
    subprocess.run([binary(), "info", MODEL, "--json"], check=True, timeout=14400)


if __name__ == "__main__":
    import fire

    fire.Fire({"prep": prep, "say": say, "info": info})
