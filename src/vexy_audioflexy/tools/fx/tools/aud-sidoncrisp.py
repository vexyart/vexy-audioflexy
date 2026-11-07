#!/usr/bin/env -S uv run -s
# /// script
# dependencies = ["platformdirs>=4", "fire", "huggingface-hub", "numpy", "soundfile"]
# ///
# this_file: src/vexy_audioflexy/tools/fx/tools/aud-sidoncrisp.py
"""Exact cstr Sidon quant ladder through the unmodified native CrispASR CLI."""
from __future__ import annotations


import os
from pathlib import Path
from platformdirs import user_cache_path, user_data_path
RUNTIME_ROOT = Path(os.environ.get("VAFX_AUDIO_RUNTIME_ROOT", user_data_path("vexy-audioflexy") / "audio" )).expanduser()
CACHE_ROOT = Path(os.environ.get("VAFX_AUDIO_CACHE_ROOT", user_cache_path("vexy-audioflexy") / "audio")).expanduser()
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = "cstr/Sidon-GGUF"
REVISION = "c4003fa18e69df2b7baa5aa6c7d32c2d6b47da5b"
COMMIT = "c23ba1c284761d873da8769fa92703d7f71a5eed"
BINARY_SHA256 = "90d14b7f4f7adc05ee7e5787e5428e4ac1279edea6f0b3128362bfd4e5eb29f9"
PACKAGES = {'f16': ('sidon-v0.1-f16.gguf', '7e27282a09be71e2f37e87731a074b291a2cab0d3f1de4a8f8eb48421587e912'), 'q8_0': ('sidon-v0.1-q8_0.gguf', '8de6498eba9f1ee53371614cee2dcb8db8a7411cc5a43444cb912a456e023ade'), 'q6_k': ('sidon-v0.1-q6_k.gguf', 'f48a0407760037fd8e9aa0ae4150135bcd92a30a684b5921e5fffc4b8d1b94de'), 'q4_k': ('sidon-v0.1-q4_k.gguf', '1bb2ab2c42a1050a69187ef20a705d2c68a310b5cc6ba0b5efdd79d9c8c7e71e')}



def sha(path: Path) -> str:
    """Hash large packages without reading them into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prep(package: str = "f16", force: bool = False) -> str:
    """Return a checksum-verified immutable Hub cache path, without duplicating weights."""
    if package not in PACKAGES:
        raise SystemExit("Require exact registered cstr Sidon package")
    from huggingface_hub import hf_hub_download

    filename, expected = PACKAGES[package]
    path = Path(hf_hub_download(REPO, filename, revision=REVISION, force_download=force))
    if sha(path) != expected:
        raise SystemExit("Pinned cstr Sidon weight checksum mismatch")
    return str(path)


def output_frames(frames: int, rate: int) -> int:
    """Native HQ polyphase first rounds up to16k; Sidon then emits three samples per input."""
    return ((frames * 16000 + rate - 1) // rate) * 3


def resume_ready(audio: str, output: str, package: str, backend: str = "metal") -> bool:
    """Reuse only an unchanged source/output pair from this exact native conversion."""
    source, target = Path(audio), Path(output)
    try:
        receipt = json.loads(target.with_suffix(".report.json").read_text())
        engine = Path(os.environ.get("CRISPASR_ISSUE201_ROOT", str(RUNTIME_ROOT / "engines/CrispASR")))
        return (receipt["repo"] == REPO and receipt["revision"] == REVISION and receipt["package"] == package
            and receipt["weights_sha256"] == PACKAGES[package][1] and receipt["backend"] == backend
            and receipt["runtime_commit"] == COMMIT and receipt["binary_sha256"] == BINARY_SHA256
            and sha(engine / "build/bin/crispasr") == BINARY_SHA256 and receipt["input_sha256"] == sha(source)
            and receipt["audio"]["sha256"] == sha(target))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def inspect_audio(path: Path, expected: int) -> dict:
    """Require the native complete48k mono result; retain silence residuals as measured."""
    import numpy as np
    import soundfile as sf

    x, rate = sf.read(path, always_2d=True)
    if rate != 48000 or x.shape != (expected, 1) or not len(x) or not np.isfinite(x).all():
        raise ValueError("Invalid native48k mono output or duration")
    return {"frames": len(x), "sample_rate": rate, "channels": 1, "duration_s": len(x) / rate,
        "rms": float(np.sqrt(np.mean(x ** 2))), "peak": float(np.max(np.abs(x))), "sha256": sha(path)}


def run(audio: str, output: str, package: str = "f16", backend: str = "metal", timeout: int = 900,
        force: bool = False, task: str = "s2s", family: str = "sidon") -> None:
    """Publish checked native WAV and a receipt; failed inference preserves any prior WAV."""
    import numpy as np
    import soundfile as sf

    if package not in PACKAGES or backend not in {"cpu", "metal"} or timeout <= 0 or task != "s2s" or family != "sidon":
        raise SystemExit("Require registered Sidon package, CPU/Metal, s2s and positive timeout")
    source, target = Path(audio).resolve(), Path(output).resolve()
    if not source.is_file() or source == target or target.suffix.lower() != ".wav" or (target.exists() and not force):
        raise SystemExit("Require source and distinct unused WAV output; --force overwrites")
    samples, rate = sf.read(source, dtype="float32", always_2d=True)
    if source.suffix.lower() not in {".wav", ".flac"} or not len(samples) or samples.shape[1] not in {1, 2} or not np.isfinite(samples).all():
        raise SystemExit("Require nonempty finite mono/stereo WAV/FLAC")
    engine = Path(os.environ.get("CRISPASR_ISSUE201_ROOT", str(RUNTIME_ROOT / "engines/CrispASR")))
    binary = engine / "build/bin/crispasr"
    if not binary.is_file() or sha(binary) != BINARY_SHA256 or subprocess.check_output(
        ["git", "-C", str(engine), "rev-parse", "HEAD"], text=True).strip() != COMMIT:
        raise SystemExit("Require pinned unmodified CrispASR native build")
    weights = prep(package)
    target.parent.mkdir(parents=True, exist_ok=True)
    options = {"CRISPASR_HQ_RESAMPLE": "1", "CRISPASR_SIDON_LOOKAHEAD": "1",
        "CRISPASR_SIDON_DECODER_CHUNK_FRAMES": "512", "CRISPASR_SIDON_MAX_FRAMES": "3000"}
    env = os.environ.copy()
    env.update(options)
    env.pop("CRISPASR_SIDON_DUMP_HANDOFF", None)
    with tempfile.TemporaryDirectory(dir=target.parent) as temporary:
        stage = Path(temporary) / "restored.wav"
        argv = [str(binary), "--s2s", "-m", weights, "-f", str(source), "--s2s-output", str(stage),
            "--gpu-backend", backend, "--verbose"] + (["--no-gpu"] if backend == "cpu" else [])
        started = time.monotonic()
        native = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False, env=env)
        wall = time.monotonic() - started
        log = native.stdout + native.stderr
        target.with_suffix(".native.log").write_text(log)
        if native.returncode:
            raise SystemExit(f"Native Sidon failed ({native.returncode}); see {target.with_suffix('.native.log')}")
        result = inspect_audio(stage, output_frames(len(samples), rate))
        timings = re.findall(r"sidon: timings .* total=([\d.]+) ms", log)
        session = float(timings[0]) / 1000 if len(timings) == 1 else None
        if session is None or not math.isfinite(session) or session <= 0:
            raise ValueError("Missing or invalid native Sidon timing")
        report = {"package": package, "family": family, "repo": REPO, "revision": REVISION,
            "weights_sha256": PACKAGES[package][1], "binary_sha256": sha(binary), "runtime_commit": COMMIT,
            "backend": backend, "input_sha256": sha(source), "input_frames": len(samples), "input_rate": rate,
            "input_channels": samples.shape[1], "native_options": options, "marking": "Native CLI defaults retained",
            "session_s": session, "session_rtf": session / (len(samples) / rate), "cli_wall_s": wall,
            "cli_rtf": wall / (len(samples) / rate), "audio": result,
            "timing_scope": "CLI includes loading/I/O/marking; native total excludes loading; no operator trace"}
        stage.replace(target)
        target.with_suffix(".report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("aud.result=" + json.dumps(report))


if __name__ == "__main__":
    import fire

    fire.Fire({"prep": prep, "run": run})
