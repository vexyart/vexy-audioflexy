#!/usr/bin/env -S uv run -s
# /// script
# dependencies = ["platformdirs>=4", "fire", "huggingface-hub", "numpy", "soundfile"]
# ///
# this_file: src/vexy_audioflexy/tools/fx/tools/aud-restoration.py
"""Pinned Sidon restoration and UniverSR audio/speech super-resolution at48kHz."""
from __future__ import annotations


import os
from pathlib import Path
import runpy
model_path = runpy.run_path(str(Path(__file__).with_name("_audioflexy_paths.py")))["model_path"]
from platformdirs import user_cache_path, user_data_path
RUNTIME_ROOT = Path(os.environ.get("VAFX_AUDIO_RUNTIME_ROOT", user_data_path("vexy-audioflexy") / "audio" )).expanduser()
CACHE_ROOT = Path(os.environ.get("VAFX_AUDIO_CACHE_ROOT", user_cache_path("vexy-audioflexy") / "audio")).expanduser()
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

ROOT = Path(__file__).resolve().parents[1]
REPO = "audio-cpp/audio.cpp-gguf"
REVISION = "351dbab8d8534675ee29440bb402e348b09e55e2"
COMMIT = "2892ed3e94b6ccbdacc111fa2896aaae0a5dc8fc"
PACKAGES = {
    "sidon_f32": ("sidon", "Sidon-GGUF/sidon-v0.1-f32.gguf", "5af689683a8142764dd69c956a87455c34f8b685cce90ac4bcb722037bc8e360"),
    "universr_audio_orig": ("universr", "UniverSR-GGUF/universr-audio-orig.gguf", "86d2980546effb2255fff79d813f5d4ccd09171d151f3f103c6c44be1e490a1e"),
    "universr_speech_orig": ("universr", "UniverSR-GGUF/universr-speech-orig.gguf", "0d8dc11bcbe9a1fa515071ff7fd5bf040b3487ddfecd05826ab6f11bc92aa740"),
}


def prep(package: str = "sidon_f32", force: bool = False) -> str:
    """Verify the requested conversion and retain a regular named GGUF file."""
    from huggingface_hub import hf_hub_download

    if package not in PACKAGES:
        raise SystemExit("Require exact registered restoration package")
    _, filename, expected = PACKAGES[package]
    downloaded = Path(hf_hub_download(REPO, filename, revision=REVISION, force_download=force))
    if hashlib.sha256(downloaded.read_bytes()).hexdigest() != expected:
        raise SystemExit("Pinned restoration weight checksum mismatch")
    target = model_path("issue201-restoration") / Path(filename).name
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest() != expected:
        shutil.copy2(downloaded, target)
    return str(target)


def request_options(package: str, rate: int, bandwidth: int, steps: int, seed: int) -> list[str]:
    """Bandwidth describes the source; it is never guessed from an upsampled file."""
    if package not in PACKAGES or steps <= 0 or seed < 0:
        raise SystemExit("Require registered package, positive steps and nonnegative seed")
    if package == "sidon_f32":
        if bandwidth:
            raise SystemExit("Sidon has no input-bandwidth request option")
        return []
    effective = bandwidth or rate
    if effective not in {8000, 12000, 16000, 24000}:
        raise SystemExit("UniverSR requires effective bandwidth rate8000/12000/16000/24000")
    return ["--request-option", f"input_sample_rate={effective}", "--request-option", f"num_inference_steps={steps}",
        "--request-option", f"seed={seed}"]


def inspect_audio(path: Path, expected_frames: int, family: str) -> dict:
    """Preserve native cropping: UniverSR ISTFT can omit less than one512-sample hop."""
    import numpy as np
    import soundfile as sf

    signal, rate = sf.read(path, always_2d=True)
    missing = expected_frames - len(signal)
    allowed = 511 if family == "universr" else 0
    if rate != 48000 or signal.shape[1] != 1 or not len(signal) or not np.isfinite(signal).all() or not 0 <= missing <= allowed:
        raise ValueError("Invalid native48k mono output or unexpected duration loss")
    return {"frames": len(signal), "sample_rate": rate, "channels": 1, "duration_s": len(signal) / rate,
        "missing_tail_samples": missing, "rms": float(np.sqrt(np.mean(signal ** 2))),
        "peak": float(np.max(np.abs(signal))), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def expected_output_frames(frames: int, rate: int, family: str) -> int:
    """Sidon truncates duration; UniverSR's Torch-compatible resampler rounds up."""
    return (frames * 48000 + (rate - 1 if family == "universr" else 0)) // rate


def run(audio: str, output: str, package: str = "sidon_f32", backend: str = "metal", input_sample_rate: int = 0,
        steps: int = 4, seed: int = 42, timeout: int = 1800, force: bool = False, task: str = "s2s", family: str = "") -> None:
    """Run the upstream offline contract, publishing checked output and native provenance."""
    import numpy as np
    import soundfile as sf

    if package not in PACKAGES or backend not in {"cpu", "metal"} or timeout <= 0 or task != "s2s":
        raise SystemExit("Require exact package, CPU/Metal, positive timeout and s2s task")
    native_family, _, weight_hash = PACKAGES[package]
    if family and family != native_family:
        raise SystemExit("Family must match the exact requested package")
    source, target = Path(audio).resolve(), Path(output).resolve()
    if not source.is_file() or source == target or (target.exists() and not force):
        raise SystemExit("Require source and distinct unused output; --force overwrites")
    samples, rate = sf.read(source, dtype="float32", always_2d=True)
    if not len(samples) or samples.shape[1] not in {1, 2} or not np.isfinite(samples).all():
        raise SystemExit("Require nonempty finite mono/stereo source")
    if native_family == "sidon" and not np.any(samples):
        raise SystemExit("Sidon peak normalization requires non-silent input")
    options = request_options(package, rate, input_sample_rate, steps, seed)
    engine = Path(os.environ.get("AUDIOCPP_ISSUE201_ROOT", str(RUNTIME_ROOT / "engines/audio.cpp")))
    binary = engine / "build/macos-metal-release/bin/audiocpp_cli"
    if not binary.is_file() or subprocess.check_output(["git", "-C", str(engine), "rev-parse", "HEAD"], text=True).strip() != COMMIT:
        raise SystemExit("Require pinned native audio.cpp build")
    weights = prep(package)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent) as temporary:
        stage = Path(temporary) / "restored.wav"
        argv = [str(binary), "--task", "s2s", "--family", native_family, "--model", weights, "--backend", backend,
            "--audio", str(source), "--out", str(stage), "--out-format", "float32", "--log"] + options
        started = time.monotonic()
        native = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
        wall = time.monotonic() - started
        log = native.stdout + native.stderr
        target.with_suffix(".native.log").write_text(log)
        if native.returncode:
            raise SystemExit(f"Native restoration failed ({native.returncode}); see {target.with_suffix('.native.log')}")
        result = inspect_audio(stage, expected_output_frames(len(samples), rate, native_family), native_family)
        timing = re.findall(r"session\.wall_ms\s+(\S+)", log)
        session_s = float(timing[0]) / 1000 if len(timing) == 1 else None
        if session_s is not None and (not math.isfinite(session_s) or session_s <= 0):
            raise ValueError("Native restoration timing must be finite and positive")
        report = {"package": package, "family": native_family, "repo": REPO, "revision": REVISION,
            "weights_sha256": weight_hash, "runtime_commit": COMMIT, "backend": backend,
            "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(), "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "input": str(source), "input_frames": len(samples), "input_rate": rate, "input_channels": samples.shape[1],
            "input_sample_rate": (input_sample_rate or rate) if native_family == "universr" else None,
            "steps": steps if native_family == "universr" else None, "seed": seed if native_family == "universr" else None,
            "session_s": session_s, "session_rtf": session_s / (len(samples) / rate) if session_s else None,
            "cli_wall_s": wall, "cli_rtf": wall / (len(samples) / rate), "audio": result,
            "timing_scope": "CLI includes model load/I/O; native session excludes them; no per-operator hardware trace"}
        stage.replace(target)
        target.with_suffix(".report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("aud.result=" + json.dumps(report))


if __name__ == "__main__":
    import fire

    fire.Fire({"prep": prep, "run": run})
