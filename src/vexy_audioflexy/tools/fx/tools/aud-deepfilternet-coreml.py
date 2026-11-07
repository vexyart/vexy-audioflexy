#!/usr/bin/env -S uv run -s
# /// script
# dependencies = ["platformdirs>=4", "fire", "huggingface-hub", "numpy", "soundfile"]
# ///
# this_file: src/vexy_audioflexy/tools/fx/tools/aud-deepfilternet-coreml.py
"""Exact DF3 Core ML graph through the original Swift streaming/flush runtime."""
from __future__ import annotations


import os
from pathlib import Path
from platformdirs import user_cache_path, user_data_path
RUNTIME_ROOT = Path(os.environ.get("VAFX_AUDIO_RUNTIME_ROOT", user_data_path("vexy-audioflexy") / "audio" )).expanduser()
CACHE_ROOT = Path(os.environ.get("VAFX_AUDIO_CACHE_ROOT", user_cache_path("vexy-audioflexy") / "audio")).expanduser()
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATALOG = json.loads((ROOT / "catalogs/deepfilternet-coreml.json").read_text())
COMPUTE = {"cpu": "cpuOnly", "gpu": "cpuAndGPU", "all": "all"}


def sha(path: Path) -> str:
    """Tie runtime, asset and output reuse to exact bytes."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prep(package: str = "df3_fp16", force: bool = False) -> str:
    """Materialize exact Hub assets: Core ML compilation needs regular package files."""
    if package != CATALOG["package"]:
        raise SystemExit("Require exact registered DF3 FP16 package")
    target = CACHE_ROOT / "deepfilternet-coreml-materialized"
    if not target.exists() or force:
        from huggingface_hub import snapshot_download

        source = Path(snapshot_download(CATALOG["repo"], revision=CATALOG["revision"],
            force_download=force))
        for name, digest in CATALOG["files"].items():
            if sha(source / name) != digest:
                raise SystemExit("Pinned DF3 source checksum mismatch")
        with tempfile.TemporaryDirectory(dir=target.parent) as temporary:
            stage = Path(temporary) / "assets"
            shutil.copytree(source, stage, symlinks=False)
            for name, digest in CATALOG["files"].items():
                if sha(stage / name) != digest:
                    raise SystemExit("Materialized DF3 checksum mismatch")
            if target.exists():
                shutil.rmtree(target)
            stage.replace(target)
    for name, digest in CATALOG["files"].items():
        if (target / name).is_symlink() or sha(target / name) != digest:
            raise SystemExit("Pinned materialized DF3 checksum mismatch")
    return str(target)


def inspect_audio(path: Path, input_frames: int, input_rate: int) -> dict:
    """Preserve native AVFoundation rounding and partial-hop trimming without repair."""
    import numpy as np
    import soundfile as sf

    x, rate = sf.read(path, always_2d=True)
    expected = input_frames * 48000 // input_rate
    tolerance = 0 if input_rate == 48000 else 1
    if rate != 48000 or x.shape[1] != 1 or not len(x) or not np.isfinite(x).all() or not 0 <= len(x) - expected <= tolerance:
        raise ValueError("Invalid native48k mono output or duration")
    return {"frames": len(x), "sample_rate": rate, "channels": 1, "duration_s": len(x) / rate,
            "resampling_rounding_samples": len(x) - expected, "rms": float(np.sqrt(np.mean(x ** 2))), "sha256": sha(path)}


def resume_ready(audio: str, output: str, package: str = "df3_fp16", backend: str = "cpu") -> bool:
    """Reuse only unchanged source/output/native binary and all pinned support assets."""
    try:
        p = Path(output)
        r = json.loads(p.with_suffix(".report.json").read_text())
        engine = Path(os.environ.get("DF3_COREML_ISSUE201_ROOT", str(RUNTIME_ROOT / "engines/DeepFilterNet-mlx")))
        models = CACHE_ROOT / "deepfilternet-coreml-materialized"
        return (package == r["package"] == CATALOG["package"] and backend == r["backend"]
            and r["repo"] == CATALOG["repo"] and r["revision"] == CATALOG["revision"]
            and r["runtime_commit"] == CATALOG["runtime_commit"] and r["model_files_sha256"] == CATALOG["files"]
            and r["binary_sha256"] == sha(engine / CATALOG["binary_relative"]) == CATALOG["binary_sha256"]
            and r["input_sha256"] == sha(Path(audio)) and r["audio"]["sha256"] == sha(p)
            and all(not (models / name).is_symlink() and sha(models / name) == digest for name, digest in CATALOG["files"].items()))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def run(audio: str, output: str, package: str = "df3_fp16", backend: str = "cpu", task: str = "denoise",
        family: str = "deepfilternet3", timeout: int = 900, force: bool = False) -> None:
    """Native CLI performs a measured pass and a separate saved-output pass; both flush."""
    import numpy as np
    import soundfile as sf

    if package != CATALOG["package"] or backend not in COMPUTE or task != "denoise" or family != "deepfilternet3" or timeout <= 0:
        raise SystemExit("Require registered DF3 FP16 package, denoise and positive timeout")
    src, dst = Path(audio).resolve(), Path(output).resolve()
    if not src.is_file() or dst == src or dst.suffix.lower() != ".wav" or (dst.exists() and not force):
        raise SystemExit("Require source and distinct unused WAV output; --force overwrites")
    x, rate = sf.read(src, dtype="float32", always_2d=True)
    if not len(x) or x.shape[1] not in {1, 2} or not np.isfinite(x).all():
        raise SystemExit("Require finite nonempty mono/stereo input")
    engine = Path(os.environ.get("DF3_COREML_ISSUE201_ROOT", str(RUNTIME_ROOT / "engines/DeepFilterNet-mlx")))
    binary = engine / CATALOG["binary_relative"]
    if not binary.is_file() or sha(binary) != CATALOG["binary_sha256"] or subprocess.check_output(
        ["git", "-C", str(engine), "rev-parse", "HEAD"], text=True).strip() != CATALOG["runtime_commit"]:
        raise SystemExit("Require pinned original DeepFilterNet Swift release build")
    models = prep(package)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dst.parent) as temporary:
        target = Path(temporary)
        argv = [str(binary), str(src), "--model", models, "--coreml-model-dir", models,
            "--engines", "coreml", "--coreml-compute-units", COMPUTE[backend], "--skip-offline",
            "--warmup", "0", "--runs", "1", "--save-output", "--output-dir", str(target)]
        started = time.monotonic()
        native = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
        wall = time.monotonic() - started
        dst.with_suffix(".native.log").write_text(native.stdout + native.stderr)
        if native.returncode:
            raise SystemExit(f"Native DF3 failed ({native.returncode}); see {dst.with_suffix('.native.log')}")
        stage = target / "output_deepfilternet3_core_ml_batch1.wav"
        if not stage.is_file():
            raise ValueError("Native DF3 exited without audio; inspect native log")
        result = inspect_audio(stage, len(x), rate)
        files = list(target.glob("benchmark_*.json"))
        if len(files) != 1:
            raise ValueError("Missing native DF3 benchmark receipt")
        suite = json.loads(files[0].read_text())
        if len(suite["streaming_results"]) != 1 or suite["streaming_results"][0]["engine"] != "coreml_ane":
            raise ValueError("Missing successful Core ML measurement")
        timing = suite["streaming_results"][0]
        session = timing["total_streaming_time_seconds"]
        report = {"package": package, "backend": backend, "compute_units": COMPUTE[backend],
            "repo": CATALOG["repo"], "revision": CATALOG["revision"], "model_files_sha256": CATALOG["files"],
            "runtime_commit": CATALOG["runtime_commit"], "binary_sha256": sha(binary), "input_sha256": sha(src),
            "input_frames": len(x), "input_rate": rate, "session_s": session,
            "session_rtf": session / suite["audio_length_seconds"], "cli_wall_s": wall, "cli_rtf": wall / (len(x) / rate),
            "algorithmic_delay_samples": 1440, "audio": result, "native_benchmark": suite,
            "timing_scope": "Original measured pass includes full hops/flush but omits final partial hop; saved-output pass includes partial hop/flush. CLI includes both passes/loading/resampling/writer. Core ML units authorize placement; MLX is also used in model preparation."}
        stage.replace(dst)
        dst.with_suffix(".report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("aud.result=" + json.dumps({k: v for k, v in report.items() if k != "native_benchmark"}))


if __name__ == "__main__":
    import fire

    fire.Fire({"prep": prep, "run": run})
