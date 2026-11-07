#!/usr/bin/env -S uv run -s
# /// script
# dependencies = ["platformdirs>=4", "fire", "huggingface-hub", "numpy", "soundfile"]
# ///
# this_file: src/vexy_audioflexy/tools/fx/tools/aud-detection.py
"""Exact PulseVAD/Smart Turn GGUF inference with JSON results and provenance."""
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
    "pulsevad_2_1k_f32": ("pulsevad", "vad", "PulseVAD-GGUF/pulsevad-2.1k-f32.gguf", "0f7e6e897d1d380489e09ffaffc07b1f8396a2d1dac9ba3664512c816c5908bd"),
    "pulsevad_81k_f32": ("pulsevad", "vad", "PulseVAD-GGUF/pulsevad-81k-f32.gguf", "574482cd225645646b9474d21a5a05eb6351efc8193d7eedb7ab241696146bec"),
    "smart_turn_v3_2_f32": ("smart_turn", "turn", "Smart-Turn-v3-GGUF/smart-turn-v3.2-f32.gguf", "dbd3fde8c39271b201c950d43f027318a3ec650292e40cecc37a29f82abc1e88"),
}


def prep(package: str = "pulsevad_2_1k_f32", force: bool = False) -> str:
    """Pin the exact GGUF and stage a regular filename accepted by native loaders."""
    from huggingface_hub import hf_hub_download

    if package not in PACKAGES:
        raise SystemExit(f"Unknown detection package: {package}")
    _, _, filename, expected = PACKAGES[package]
    source = Path(hf_hub_download(REPO, filename, revision=REVISION, force_download=force))
    if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
        raise SystemExit("Exact detection weight checksum mismatch")
    destination = model_path("issue201-detection") / Path(filename).name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.is_file() or hashlib.sha256(destination.read_bytes()).hexdigest() != expected:
        shutil.copy2(source, destination)
    return str(destination)


def parse_result(text: str, family: str, task: str, frames: int, threshold: float) -> tuple[dict, float]:
    """An empty VAD result is valid only after the native offline task completed."""
    for field in [f"family={family}", f"task={task}", "mode=offline"]:
        if field not in text.splitlines():
            raise ValueError("Native task identity/completion missing")
    timing = re.findall(r"session\.wall_ms\s+(\S+)", text)
    if len(timing) != 1:
        raise ValueError("Native session timing missing or ambiguous")
    seconds = float(timing[0]) / 1000
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("Native session timing must be positive and finite")
    key = "speech_segments=" if task == "vad" else "custom_schema_output="
    values = [line[len(key):] for line in text.splitlines() if line.startswith(key)]
    if len(values) > 1 or (task == "turn" and not values):
        raise ValueError("Native detection result missing or ambiguous")
    payload = json.loads(values[0]) if values else []
    if task == "turn":
        if not isinstance(payload, dict) or payload.get("schema") != "smart_turn.v1":
            raise ValueError("Unsupported turn schema")
        result = payload.get("data", {})
        if not isinstance(result, dict):
            raise ValueError("Turn data must contain a decision object")
        probability = result.get("probability")
        if isinstance(probability, bool) or not isinstance(probability, (int, float)) or not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Turn probability must be finite in [0,1]")
        if type(result.get("complete")) is not bool or result["complete"] != (probability > threshold):
            raise ValueError("Turn decision violates strict threshold")
        return result, seconds
    if not isinstance(payload, list):
        raise TypeError("Speech segments must be a list")
    end = 0
    for segment in payload:
        if not isinstance(segment, dict):
            raise TypeError("Speech segment must be an object")
        start, stop = segment.get("start_sample"), segment.get("end_sample")
        if type(start) is not int or type(stop) is not int or not end <= start < stop <= frames:
            raise ValueError("Speech spans must be ordered, disjoint and inside the timeline")
        end = stop
    return {"segments": payload}, seconds


def run(audio: str, output: str, package: str = "pulsevad_2_1k_f32", backend: str = "cpu",
        threshold: float = .5, force: bool = False, timeout: int = 120) -> None:
    """Analyze native mono16k audio and atomically publish checked JSON, not WAV."""
    import numpy as np
    import soundfile as sf

    if package not in PACKAGES or backend not in {"cpu", "metal"} or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise SystemExit("Require registered package, CPU/Metal backend and finite threshold in [0,1]")
    source, destination = Path(audio).resolve(), Path(output).resolve()
    if not source.is_file() or source == destination or destination.suffix.lower() != ".json" or timeout <= 0:
        raise SystemExit("Require source audio, distinct JSON output and positive timeout")
    if destination.exists() and not force:
        raise SystemExit("Output exists; use --force")
    samples, rate = sf.read(source, dtype="float32", always_2d=True)
    if rate != 16000 or samples.shape[1] != 1 or not len(samples) or not np.isfinite(samples).all():
        raise SystemExit("Require nonempty finite mono16k audio; resample explicitly before running")
    engine = Path(os.environ.get("AUDIOCPP_ISSUE201_ROOT", str(RUNTIME_ROOT / "engines/audio.cpp")))
    binary = engine / "build/macos-metal-release/bin/audiocpp_cli"
    if not binary.is_file() or subprocess.check_output(["git", "-C", str(engine), "rev-parse", "HEAD"], text=True).strip() != COMMIT:
        raise SystemExit("Require verified pinned audio.cpp native binary")
    weights = Path(prep(package))
    family, task, _, expected = PACKAGES[package]
    command = [str(binary), "--task", task, "--family", family, "--model", str(weights),
        "--backend", backend, "--audio", str(source), "--request-option", f"threshold={threshold}", "--log"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    native = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    wall = time.monotonic() - started
    log = native.stdout + native.stderr
    destination.with_suffix(".native.log").write_text(log)
    native.check_returncode()
    result, processing = parse_result(log, family, task, len(samples), threshold)
    buffers = re.findall(r"\S+\.buffer_name\s+(\S+)", log)
    receipt = {"package": package, "family": family, "task": task, "repo": REPO, "revision": REVISION,
        "weights_sha256": expected, "runtime_commit": COMMIT,
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(), "backend": backend,
        "observed_weight_buffers": buffers, "threshold": threshold, "result": result,
        "frames": len(samples), "sample_rate": rate, "duration_s": len(samples) / rate,
        "session_s": processing, "session_rtf": processing / (len(samples) / rate),
        "cli_wall_s": wall, "cli_rtf": wall / (len(samples) / rate),
        "timing_scope": "native session includes frontend/inference/decision; excludes model load and file I/O",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "source": str(source)}
    with tempfile.NamedTemporaryFile(mode="w", dir=destination.parent, delete=False) as temporary:
        json.dump(receipt, temporary, indent=2)
        temporary.write("\n")
    Path(temporary.name).replace(destination)
    print("aud.result=" + json.dumps(receipt))


if __name__ == "__main__":
    import fire

    fire.Fire({"prep": prep, "run": run})
