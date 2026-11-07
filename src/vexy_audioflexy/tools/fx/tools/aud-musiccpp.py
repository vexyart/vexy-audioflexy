#!/usr/bin/env -S uv run -s
# /// script
# dependencies = ["platformdirs>=4", "fire", "huggingface-hub", "numpy", "soundfile"]
# ///
# this_file: src/vexy_audioflexy/tools/fx/tools/aud-musiccpp.py
"""Pinned native Apollo restoration and HTDemucs six-stem GGUF separation."""
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
REVISION = "351dbab8d8534675ee29440bb402e348b09e55e2"
COMMIT = "2892ed3e94b6ccbdacc111fa2896aaae0a5dc8fc"
REPO = "audio-cpp/audio.cpp-gguf"
STEMS = ("drums", "bass", "other", "vocals", "guitar", "piano")
CPU_BINARY_SHA256 = "4d614bfd068bcc8a63ae161994b5afa9684bdbe772d980b988584a7acd3d5785"
CPU_PATCH_SHA256 = "42350ccd3c9cc975850412067ff60d2aadf380ee71a1a5abfe53232194b1c247"
PACKAGES = {
    "apollo_orig": ("apollo", "Apollo-GGUF/apollo-orig.gguf", "41ee9eb6c5c265f126c1baaa3aca94d506dd604f4750f6da46d3cec00310df00"),
    "htdemucs_6stems_f16": ("htdemucs_6stems", "HTDemucs-6stems-GGUF/htdemucs-6stems-f16.gguf", "1db9bd58af6995d8caab209c99f5a284212e5bd0654fbfd7e50ce1d5ac140ba9"),
    "htdemucs_6stems_q8_0": ("htdemucs_6stems", "HTDemucs-6stems-GGUF/htdemucs-6stems-q8_0.gguf", "71f3d65681a25c28fbf6278fee579f810312eae47d3b26eb04bcd77c77934c0d"),
}


def prep(package: str = "apollo_orig", force: bool = False) -> str:
    """Verify pinned artifact bytes and preserve a regular GGUF filename."""
    from huggingface_hub import hf_hub_download

    if package not in PACKAGES:
        raise SystemExit("Require exact registered music package")
    _, filename, expected = PACKAGES[package]
    downloaded = Path(hf_hub_download(REPO, filename, revision=REVISION, force_download=force))
    if hashlib.sha256(downloaded.read_bytes()).hexdigest() != expected:
        raise SystemExit("Pinned music weight checksum mismatch")
    destination = model_path("issue201-music") / Path(filename).name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.is_file() or hashlib.sha256(destination.read_bytes()).hexdigest() != expected:
        shutil.copy2(downloaded, destination)
    return str(destination)


def inspect_audio(path: Path, frames: int, channels: int) -> tuple[dict, object]:
    """Restoration and each separated stem must retain every frame/channel."""
    import numpy as np
    import soundfile as sf

    samples, rate = sf.read(path, dtype="float32", always_2d=True)
    if rate != 44100 or samples.shape != (frames, channels) or not np.isfinite(samples).all():
        raise ValueError(f"Invalid music output geometry or samples: {path.name}")
    return {"file": path.name, "frames": frames, "channels": channels, "sample_rate": rate,
        "rms": float(np.sqrt(np.mean(samples ** 2))), "peak": float(np.max(np.abs(samples))),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}, samples


def native_runtime(package: str, backend: str) -> tuple[Path, str | None]:
    """Admit HTDemucs CPU only through the binary qualified after the batch fix."""
    repaired = package.startswith("htdemucs_6stems_") and backend == "cpu"
    if repaired:
        engine = Path(os.environ.get("AUDIOCPP_HTDEMUCS_FIXED_ROOT", str(RUNTIME_ROOT / "engines/audio.cpp-demucs-diagnostic")))
        binary = engine / "build/diagnostic/bin/audiocpp_cli"
        patch = ROOT / "patches/ggml-conv-transpose2d-batch.patch"
        if not binary.is_file() or not patch.is_file() or hashlib.sha256(binary.read_bytes()).hexdigest() != CPU_BINARY_SHA256 or hashlib.sha256(patch.read_bytes()).hexdigest() != CPU_PATCH_SHA256:
            raise SystemExit("Require the qualified repaired HTDemucs CPU build; original CPU outputs are unqualified. See issues/201-htdemucs-cpu-regression.md")
    else:
        engine = Path(os.environ.get("AUDIOCPP_ISSUE201_ROOT", str(RUNTIME_ROOT / "engines/audio.cpp")))
        binary = engine / "build/macos-metal-release/bin/audiocpp_cli"
    if not binary.is_file() or subprocess.check_output(["git", "-C", str(engine), "rev-parse", "HEAD"], text=True).strip() != COMMIT:
        raise SystemExit("Require the pinned native audio.cpp build")
    return binary, CPU_PATCH_SHA256 if repaired else None


def run(audio: str, output: str, package: str = "apollo_orig", backend: str = "metal",
        force: bool = False, timeout: int = 600) -> None:
    """Publish a checked output directory containing WAVs, native log and report."""
    import numpy as np
    import soundfile as sf

    if package not in PACKAGES or backend not in {"cpu", "metal"} or timeout <= 0:
        raise SystemExit("Require registered music package, CPU/Metal and positive timeout")
    binary, repair = native_runtime(package, backend)
    source, destination = Path(audio).resolve(), Path(output).resolve()
    if not source.is_file() or source == destination or destination in source.parents:
        raise SystemExit("Require source outside output directory")
    if destination.exists() and not force:
        raise SystemExit("Output exists; use --force")
    samples, rate = sf.read(source, dtype="float32", always_2d=True)
    if rate != 44100 or samples.shape[1] not in {1, 2} or not len(samples) or not np.isfinite(samples).all():
        raise SystemExit("Require nonempty finite mono/stereo44.1kHz audio; convert explicitly")
    weights = Path(prep(package))
    family, _, expected = PACKAGES[package]
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        stage = Path(temporary)
        command = [str(binary), "--task", "s2s" if family == "apollo" else "sep", "--family", family,
            "--model", str(weights), "--backend", backend, "--audio", str(source), "--log", "--out-format", "float32"]
        command += ["--out", str(stage / "restored.wav")] if family == "apollo" else ["--out-dir", str(stage)]
        started = time.monotonic()
        native = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
        wall = time.monotonic() - started
        log = native.stdout + native.stderr
        destination.with_suffix(".native.log").write_text(log)
        native.check_returncode()
        files = [stage / "restored.wav"] if family == "apollo" else [stage / f"{name}.wav" for name in STEMS]
        records, signals = [], []
        for path in files:
            record, signal = inspect_audio(path, len(samples), samples.shape[1])
            records.append(record); signals.append(signal)
        timing = re.findall(r"session\.wall_ms\s+(\S+)", log)
        session_s = float(timing[0]) / 1000 if len(timing) == 1 else None
        if session_s is not None and (not math.isfinite(session_s) or session_s <= 0):
            raise ValueError("Native music session timing must be positive and finite")
        reconstruction = np.sum(signals, axis=0) if family != "apollo" else None
        reconstruction_sdr = float(10 * np.log10(np.sum(samples ** 2) / max(float(np.sum((samples - reconstruction) ** 2)), 1e-20))) if reconstruction is not None and np.any(samples) else None
        report = {"package": package, "family": family, "repo": REPO, "revision": REVISION,
            "weights_sha256": expected, "runtime_commit": COMMIT, "backend": backend,
            "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
            "cpu_repair_patch_sha256": repair,
            "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "input": str(source),
            "cli_wall_s": wall, "cli_rtf": wall / (len(samples) / rate), "session_s": session_s,
            "session_rtf": session_s / (len(samples) / rate) if session_s else None,
            "timing_scope": "CLI includes model load/I/O; native session excludes them; null when no native total is emitted",
            "audio_duration_s": len(samples) / rate, "outputs": records, "reconstruction_sdr_db": reconstruction_sdr,
            "separation_quality_note": "No ground-truth stems supplied" if family != "apollo" else None}
        (stage / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        (stage / "engine.log").write_text(log)
        destination.mkdir(parents=True, exist_ok=True)
        for path in stage.iterdir():
            path.replace(destination / path.name)
    print("aud.result=" + json.dumps(report))


if __name__ == "__main__":
    import fire

    fire.Fire({"prep": prep, "run": run})
