#!/usr/bin/env -S uv run -s
# /// script
# dependencies = ["fire", "huggingface-hub", "numpy", "soundfile"]
# ///
# this_file: src/vexy_audioflexy/tools/tts/tools/tts-smallgguf.py
"""Exact requested KittenTTS Mini0.8 and Kokoro82M GGUF through native audio.cpp."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from _paths import runtime_path

ROOT = Path(__file__).resolve().parents[1]
REPO = "audio-cpp/audio.cpp-gguf"
REVISION = "351dbab8d8534675ee29440bb402e348b09e55e2"
COMMIT = "2892ed3e94b6ccbdacc111fa2896aaae0a5dc8fc"
PACKAGES = {
    "kitten_tts_mini_0_8_orig": ("kitten_tts", "KittenTTS-GGUF/kitten-tts-mini-0.8-orig.gguf", "8f015d32f30cab7b9eebdebb244b0124cfd855ffe543daa3c6967e496fd24336"),
    "kokoro_82m_bf16": ("kokoro_tts", "Kokoro-82M-GGUF/kokoro-82m-bf16.gguf", "51d10c63557d21f70ef5d607e91c11023cf408dbc3af24c283b18106778e1e67"),
    "kokoro_82m_q8_0": ("kokoro_tts", "Kokoro-82M-GGUF/kokoro-82m-q8_0.gguf", "5d800fd204029302c10313daeafdb31c875c7c29ae31974d0d156cc7f512d1d0"),
}
VOICES = {"en-us": "af_heart", "en-gb": "bf_emma", "es": "ef_dora", "fr": "ff_siwis", "hi": "hf_alpha",
    "it": "if_sara", "ja": "jf_alpha", "pt-br": "pf_dora", "zh": "zf_xiaobei"}
PREFIXES = {"en-us": ("af", "am"), "en-gb": ("bf", "bm"), "es": ("ef", "em"), "fr": ("ff",),
    "hi": ("hf", "hm"), "it": ("if", "im"), "ja": ("jf", "jm"), "pt-br": ("pf", "pm"), "zh": ("zf", "zm")}


def resolve(package: str, lang: str, voice_id: str) -> tuple[str, str]:
    """Reject unsupported combinations before fetching weights; native validates voice IDs."""
    if package not in PACKAGES:
        raise SystemExit("Require exact registered small TTS package")
    if package.startswith("kitten_"):
        voice = voice_id or "Leo"
        if lang != "en" or voice not in {"Bella", "Jasper", "Luna", "Bruno", "Rosie", "Hugo", "Kiki", "Leo"}:
            raise SystemExit("Kitten supports English and its eight built-in voices")
        return lang, voice
    language = {"en": "en-us", "fr-fr": "fr"}.get(lang, lang)
    if language not in VOICES:
        raise SystemExit("Unsupported Kokoro language")
    voice = voice_id or VOICES[language]
    if voice.split("_")[0] not in PREFIXES[language]:
        raise SystemExit("Kokoro voice and language must match")
    return language, voice


def prep(package: str = "kitten_tts_mini_0_8_orig", force: bool = False) -> str:
    """Verify exact published weight bytes and stage a regular named GGUF."""
    from huggingface_hub import hf_hub_download

    if package not in PACKAGES:
        raise SystemExit("Require exact registered small TTS package")
    _, filename, expected = PACKAGES[package]
    downloaded = Path(hf_hub_download(REPO, filename, revision=REVISION, force_download=force))
    if hashlib.sha256(downloaded.read_bytes()).hexdigest() != expected:
        raise SystemExit("Pinned TTS weight checksum mismatch")
    path = runtime_path('models/issue201-smalltts') / Path(filename).name
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        shutil.copy2(downloaded, path)
    return str(path)


def inspect_audio(path: Path) -> dict:
    """Require actual finite non-silent native24kHz mono audio."""
    import numpy as np
    import soundfile as sf

    signal, rate = sf.read(path, always_2d=True)
    if rate != 24000 or signal.shape[1] != 1 or not len(signal) or not np.isfinite(signal).all() or not np.any(signal):
        raise ValueError("Require finite nonempty non-silent mono24kHz native TTS output")
    return {"frames": len(signal), "sample_rate": rate, "channels": 1, "duration_s": len(signal) / rate,
        "rms": float(np.sqrt(np.mean(signal ** 2))), "peak": float(np.max(np.abs(signal))),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def say(text: str, output: str = "output.wav", package: str = "kitten_tts_mini_0_8_orig", lang: str = "en",
        voice_id: str = "", backend: str = "cpu", seed: int = 42, timeout: int = 600, force: bool = False) -> None:
    """Synthesize atomically and retain native log and complete provenance beside the WAV."""
    language, voice = resolve(package, lang, voice_id)
    if not text.strip() or backend not in {"cpu", "metal"} or seed < 0 or timeout <= 0:
        raise SystemExit("Require text, CPU/Metal, nonnegative seed and positive timeout")
    out = Path(output).resolve()
    if out.exists() and not force:
        raise SystemExit("Output exists; use --force")
    engine = Path(os.environ.get("AUDIOCPP_ISSUE201_ROOT", str(runtime_path('audio.cpp'))))
    binary = engine / "build/macos-metal-release/bin/audiocpp_cli"
    if not binary.is_file() or subprocess.check_output(["git", "-C", str(engine), "rev-parse", "HEAD"], text=True).strip() != COMMIT:
        raise SystemExit("Require pinned native audio.cpp build")
    weights = prep(package)
    family, _, weight_hash = PACKAGES[package]
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent) as temporary:
        stage = Path(temporary) / "speech.wav"
        argv = [str(binary), "--task", "tts", "--family", family, "--model", weights, "--backend", backend,
            "--text", text, "--voice-id", voice, "--seed", str(seed), "--out", str(stage),
            "--out-format", "float32", "--log"]
        if family == "kokoro_tts":
            argv += ["--language", language]
        started = time.monotonic()
        native = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
        wall = time.monotonic() - started
        log = native.stdout + native.stderr
        out.with_suffix(".native.log").write_text(log)
        if native.returncode:
            raise SystemExit(f"Native TTS failed ({native.returncode}); see {out.with_suffix('.native.log')}")
        audio = inspect_audio(stage)
        timing = re.findall(r"session\.wall_ms\s+(\S+)", log)
        session_s = float(timing[0]) / 1000 if len(timing) == 1 else None
        if session_s is not None and (not math.isfinite(session_s) or session_s <= 0):
            raise ValueError("Native TTS session timing must be finite and positive")
        report = {"package": package, "family": family, "repo": REPO, "revision": REVISION, "weights_sha256": weight_hash,
            "runtime_commit": COMMIT, "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
            "text": text, "language": language, "voice_id": voice, "backend": backend, "seed": seed,
            "cli_wall_s": wall, "cli_rtf": wall / audio["duration_s"], "session_s": session_s,
            "session_rtf": session_s / audio["duration_s"] if session_s else None,
            "timing_scope": "CLI includes load/I/O; session excludes them; no physical per-operator hardware trace", "audio": audio}
        stage.replace(out)
        out.with_suffix(".report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("tts.result=" + json.dumps(report))


if __name__ == "__main__":
    import fire

    fire.Fire({"prep": prep, "say": say})
