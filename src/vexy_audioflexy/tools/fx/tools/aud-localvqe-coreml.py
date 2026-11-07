#!/usr/bin/env -S uv run -s
# /// script
# dependencies = ["platformdirs>=4", "fire", "huggingface-hub", "numpy", "soundfile"]
# ///
# this_file: src/vexy_audioflexy/tools/fx/tools/aud-localvqe-coreml.py
"""Exact four FluidInference Core ML variants through the original streaming CLI."""
from __future__ import annotations


import os
from pathlib import Path
from platformdirs import user_cache_path, user_data_path
RUNTIME_ROOT = Path(os.environ.get("VAFX_AUDIO_RUNTIME_ROOT", user_data_path("vexy-audioflexy") / "audio" )).expanduser()
CACHE_ROOT = Path(os.environ.get("VAFX_AUDIO_CACHE_ROOT", user_cache_path("vexy-audioflexy") / "audio")).expanduser()
import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATALOG = json.loads((ROOT / "catalogs/localvqe-coreml.json").read_text())


def sha(path: Path) -> str:
    """Hash unchanged native models without duplicating their cache files."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prep(package: str = "v13_256ms", force: bool = False) -> str:
    """Download one exact compiled bundle and verify every required file."""
    if package not in CATALOG["packages"]:
        raise SystemExit("Require exact registered Core ML LocalVQE package")
    from huggingface_hub import snapshot_download

    info = CATALOG["packages"][package]
    root = Path(snapshot_download(CATALOG["repo"], revision=CATALOG["revision"],
        allow_patterns=[info["bundle"] + "/**"], force_download=force))
    for name, expected in info["files"].items():
        if sha(root / info["bundle"] / name) != expected:
            raise SystemExit("Pinned Core ML bundle checksum mismatch")
    return str(root)


def inspect_audio(path: Path, input_frames: int, input_rate: int) -> dict:
    """Keep native sample count; allow only one sample of resampler rounding."""
    import numpy as np
    import soundfile as sf

    x, rate = sf.read(path, always_2d=True)
    expected = input_frames * 16000 // input_rate
    tolerance = 0 if input_rate == 16000 else 1
    if rate != 16000 or x.shape[1] != 1 or not len(x) or not np.isfinite(x).all() or not 0 <= len(x) - expected <= tolerance:
        raise ValueError("Invalid native16k mono output or duration")
    return {"frames": len(x), "sample_rate": rate, "channels": 1, "duration_s": len(x) / rate,
        "resampling_rounding_samples": len(x) - expected, "rms": float(np.sqrt(np.mean(x ** 2))), "sha256": sha(path)}


def resume_ready(audio: str, output: str, package: str, reference: str = "", backend: str = "cpu") -> bool:
    """Reuse only the unchanged source/reference/output and exact loaded model bytes."""
    try:
        p = Path(output)
        receipt = json.loads(p.with_suffix(".report.json").read_text())
        info = CATALOG["packages"][package]
        engine = Path(os.environ.get("FLUIDAUDIO_ISSUE201_ROOT", str(RUNTIME_ROOT / "engines/FluidAudio")))
        from huggingface_hub.constants import HF_HUB_CACHE
        models = Path(HF_HUB_CACHE) / f"models--FluidInference--localvqe-coreml/snapshots/{CATALOG['revision']}"
        return (receipt["repo"] == CATALOG["repo"] and receipt["revision"] == CATALOG["revision"]
            and receipt["package"] == package and receipt["backend"] == backend
            and receipt["runtime_commit"] == CATALOG["runtime_commit"] and receipt["model_files_sha256"] == info["files"]
            and receipt["binary_sha256"] == sha(engine / ".build/release/fluidaudiocli") == CATALOG["binary_sha256"]
            and receipt["input_sha256"] == sha(Path(audio)) and receipt["audio"]["sha256"] == sha(p)
            and receipt["reference_sha256"] == (sha(Path(reference)) if reference else None)
            and all(sha(models / info["bundle"] / name) == digest for name, digest in info["files"].items()))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def run(audio: str, output: str, package: str = "v13_256ms", backend: str = "cpu", task: str = "denoise",
        reference: str = "", voice_ref: str = "", family: str = "localvqe", timeout: int = 900, force: bool = False) -> None:
    """Use the original state/flush API; a failed inference preserves any prior WAV."""
    import numpy as np
    import soundfile as sf

    if package not in CATALOG["packages"] or backend not in {"cpu", "gpu"} or task not in {"denoise", "aec"} or family != "localvqe" or timeout <= 0:
        raise SystemExit("Require registered package, CPU/GPU, denoise/AEC and positive timeout")
    if reference and voice_ref and Path(reference).resolve() != Path(voice_ref).resolve():
        raise SystemExit("Conflicting far-end reference arguments")
    far = Path(reference or voice_ref).resolve() if reference or voice_ref else None
    src, dst = Path(audio).resolve(), Path(output).resolve()
    if not src.is_file() or dst in {src, far} or dst.suffix.lower() != ".wav" or (dst.exists() and not force):
        raise SystemExit("Require source and distinct unused WAV output; --force overwrites")
    x, rate = sf.read(src, dtype="float32", always_2d=True)
    if not len(x) or x.shape[1] not in {1, 2} or not np.isfinite(x).all():
        raise SystemExit("Require finite nonempty mono/stereo source")
    if task == "aec" and not far:
        raise SystemExit("AEC requires the synchronized loudspeaker reference")
    if far:
        y, far_rate = sf.read(far, dtype="float32", always_2d=True)
        if len(y) * rate != len(x) * far_rate or y.shape[1] not in {1, 2} or not np.isfinite(y).all():
            raise SystemExit("Require equal-duration finite synchronized far-end reference")
    engine = Path(os.environ.get("FLUIDAUDIO_ISSUE201_ROOT", str(RUNTIME_ROOT / "engines/FluidAudio")))
    binary = engine / ".build/release/fluidaudiocli"
    if not binary.is_file() or sha(binary) != CATALOG["binary_sha256"] or subprocess.check_output(
        ["git", "-C", str(engine), "rev-parse", "HEAD"], text=True).strip() != CATALOG["runtime_commit"]:
        raise SystemExit("Require pinned unmodified FluidAudio release build")
    model_dir = prep(package)
    info = CATALOG["packages"][package]
    dst.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dst.parent) as temporary:
        stage = Path(temporary) / "enhanced.wav"
        argv = [str(binary), "enhance", str(src), "--model-dir", model_dir, "--variant", info["variant"],
            "--chunk", info["chunk"], "--compute-units", backend, "--streaming", "--buffer-samples", "256",
            "--output", str(stage)] + (["--reference", str(far)] if far else [])
        start = time.monotonic()
        native = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
        wall = time.monotonic() - start
        log = native.stdout + native.stderr
        dst.with_suffix(".native.log").write_text(log)
        if native.returncode:
            raise SystemExit(f"Native LocalVQE failed ({native.returncode}); see {dst.with_suffix('.native.log')}")
        result = inspect_audio(stage, len(x), rate)
        timing = re.search(r"Enhanced .* in ([\d.]+)s wall", log)
        if not timing:
            raise ValueError("Missing native LocalVQE timing")
        session = float(timing[1])
        report = {"package": package, "variant": info["variant"], "chunk": info["chunk"], "backend": backend,
            "task": task, "repo": CATALOG["repo"], "revision": CATALOG["revision"], "model_files_sha256": info["files"],
            "runtime_commit": CATALOG["runtime_commit"], "binary_sha256": sha(binary), "input_sha256": sha(src),
            "reference_sha256": sha(far) if far else None, "input_frames": len(x), "input_rate": rate,
            "session_s": session, "session_rtf": session / (len(x) / rate), "cli_wall_s": wall,
            "cli_rtf": wall / (len(x) / rate), "audio": result,
            "timing_scope": "Native stream/flush rounded to1ms; CLI includes loading/resampling/writer; gpu means CPU+GPU request"}
        stage.replace(dst)
        dst.with_suffix(".report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("aud.result=" + json.dumps(report))


if __name__ == "__main__":
    import fire

    fire.Fire({"prep": prep, "run": run})
