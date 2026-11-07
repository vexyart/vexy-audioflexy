#!/usr/bin/env -S uv run
# /// script
# dependencies = ["platformdirs>=4", "fire", "huggingface-hub", "numpy", "soundfile"]
# ///
# this_file: src/vexy_audioflexy/tools/fx/tools/aud-speechlens.py
"""Exact faraday RE-USE conversion through the pinned Swift/Metal SpeechLens CLI."""
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
import subprocess
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = "faraday/re-use-mlx"
REVISION = "f4432e3e5101762bc30550b8e0031c7c8c0c4489"
COMMIT = "f6c640fce1d265e8b4b7564ec81dff1f08d67200"
SHA256 = "d1158502eaf39d0b11d097177160ce3804454653c5d14d17921b6c274ca53237"
SIZE = 38_583_628
PACKAGE = "faraday-re-use-mlx"


def check_package(package: str) -> None:
    """Reject a substitute conversion or task before invoking the runtime."""
    if package != PACKAGE:
        raise SystemExit(f"Unknown exact SpeechLens package: {package}")


def prep(package: str = PACKAGE, force: bool = False) -> str:
    """Pin and verify the requested artifact before MLX allocation."""
    from huggingface_hub import snapshot_download

    check_package(package)
    snapshot = Path(snapshot_download(REPO, revision=REVISION, force_download=force))
    weights = snapshot / "model_mlx.safetensors"
    if weights.stat().st_size != SIZE or hashlib.sha256(weights.read_bytes()).hexdigest() != SHA256:
        raise SystemExit("Requested faraday weight size/checksum mismatch")
    print(f"[prep] ready: {REPO}@{REVISION}")
    return str(weights)


def processing_seconds(report: dict) -> float:
    """Native pipeline time excludes weight loading but includes media I/O."""
    seconds = report.get("processingDurationSeconds")
    if report.get("schemaVersion") != 2 or report.get("outcome") != "completed":
        raise SystemExit("SpeechLens native operation did not complete under schema 2")
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds <= 0:
        raise SystemExit("SpeechLens pipeline timing must be finite and positive")
    return float(seconds)


def verify_waveform(source: Path, output: Path) -> dict:
    """Native-rate enhancement must preserve every frame and channel."""
    import numpy as np
    import soundfile as sf

    before, after = sf.info(source), sf.info(output)
    if before.frames <= 0 or (before.frames, before.samplerate, before.channels) != (after.frames, after.samplerate, after.channels):
        raise SystemExit("SpeechLens changed frame count, channel count or sample rate")
    samples, _ = sf.read(output, dtype="float32", always_2d=True)
    if not np.isfinite(samples).all():
        raise SystemExit("SpeechLens returned nonfinite output samples")
    return {"frames": after.frames, "sample_rate": after.samplerate, "channels": after.channels,
            "duration_s": after.duration, "rms": float(np.sqrt(np.mean(samples ** 2))),
            "peak": float(np.max(np.abs(samples)))}


def run(audio: str, output: str, package: str = PACKAGE, task: str = "denoise",
        mode: str = "strict", force: bool = False, timeout: int = 900) -> None:
    """Run full native processing, retain diagnostics/logs and atomically publish WAV."""
    import soundfile as sf

    check_package(package)
    source, destination = Path(audio).resolve(), Path(output).resolve()
    if task not in {"denoise", "dn"} or mode not in {"standard", "strict"}:
        raise SystemExit("Require denoise task and standard or strict execution mode")
    if not source.is_file() or source == destination or timeout <= 0 or destination.suffix.lower() != ".wav":
        raise SystemExit("Require source audio, distinct WAV output and positive timeout")
    if destination.exists() and not force:
        raise SystemExit("Output exists; use --force")
    info = sf.info(source)
    if info.frames <= 0 or not 8000 <= info.samplerate <= 48000:
        raise SystemExit("Require nonempty audio at 8–48 kHz")
    engine = RUNTIME_ROOT / "engines/SpeechLens"
    binary = Path(os.environ.get("SPEECHLENS_BIN", str(engine / ".build/release/speechlens-cli")))
    if not binary.is_file() or not binary.with_name("mlx.metallib").is_file():
        raise SystemExit("Build SpeechLens Release CLI and package mlx.metallib beside the executable")
    commit = subprocess.check_output(["git", "-C", str(engine), "rev-parse", "HEAD"], text=True).strip()
    if commit != COMMIT:
        raise SystemExit("SpeechLens checkout differs from the verified runtime")
    weights = prep(package)
    destination.parent.mkdir(parents=True, exist_ok=True)
    diagnostic = destination.with_suffix(".native.json")
    environment = os.environ.copy()
    environment.setdefault("SPEECHLENS_FFMPEG", "/opt/homebrew/bin/ffmpeg")
    environment.setdefault("SPEECHLENS_FFPROBE", "/opt/homebrew/bin/ffprobe")
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        native_output = Path(temporary) / "enhanced.wav"
        command = [str(binary), "--weights", weights, "--input", str(source), "--output", str(native_output),
                   "--diagnostics-out", str(diagnostic)]
        if mode == "strict":
            command.append("--strict")
        start = time.monotonic()
        native = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=timeout, check=False)
        wall = time.monotonic() - start
        destination.with_suffix(".native.log").write_text(native.stdout + native.stderr)
        native.check_returncode()
        pipeline = processing_seconds(json.loads(diagnostic.read_text()))
        stats = verify_waveform(source, native_output)
        native_output.replace(destination)
    receipt = {"repo": REPO, "revision": REVISION, "weights_sha256": SHA256, "runtime_commit": COMMIT,
        "mode": mode, "chunk_seconds": 5.0, "overlap_portion": .05,
        "cli_wall_s": wall, "pipeline_s": pipeline, "pipeline_rtf": pipeline / info.duration,
        "pipeline_scope": "includes media decode/encode and file operations, excludes weight loading",
        "cli_rtf": wall / info.duration, "audio": str(source), "output": str(destination),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "metallib_sha256": hashlib.sha256(binary.with_name("mlx.metallib").read_bytes()).hexdigest(), **stats}
    destination.with_suffix(".result.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print("aud.result=" + json.dumps(receipt))


if __name__ == "__main__":
    import fire

    fire.Fire({"prep": prep, "run": run})
