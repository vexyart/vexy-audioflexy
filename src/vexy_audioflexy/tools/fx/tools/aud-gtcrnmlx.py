#!/usr/bin/env -S uv run -s
# /// script
# dependencies = ["platformdirs>=4", "fire", "huggingface-hub", "httpx", "rich", "onnxruntime==1.30.0", "onnxruntime-ep-mlx==0.29.6", "onnx", "safetensors", "numpy", "soundfile", "scipy"]
# ///
# this_file: src/vexy_audioflexy/tools/fx/tools/aud-gtcrnmlx.py
"""Exact agentable GTCRN tensors via a profiled hybrid MLX/CPU ONNX runtime."""
from __future__ import annotations


import os
from pathlib import Path
import runpy
model_path = runpy.run_path(str(Path(__file__).with_name("_audioflexy_paths.py")))["model_path"]
from platformdirs import user_cache_path, user_data_path
RUNTIME_ROOT = Path(os.environ.get("VAFX_AUDIO_RUNTIME_ROOT", user_data_path("vexy-audioflexy") / "audio" )).expanduser()
CACHE_ROOT = Path(os.environ.get("VAFX_AUDIO_CACHE_ROOT", user_cache_path("vexy-audioflexy") / "audio")).expanduser()
import hashlib
import importlib.util
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf
from gtcrn_audio import analyze, synthesize
from gtcrn_layout import bind

ROOT = Path(__file__).resolve().parents[1]
REPO = "agentable/gtcrn-simple-mlx"
REVISION = "95fa4d44f891841f3d7867feff866a7788c50c63"
PACKAGE = "agentable-gtcrn-simple-mlx"
METADATA_SHA = "070f320defc15d4e1781b0c5b9225d9f91031ed5f14f675efa497f244e640615"


def sherpa_root() -> Path:
    """Match the sibling downloader's explicit override and configured store."""
    override = os.environ.get("SHERPA_ONNX_MODELS_ROOT")
    return Path(override).expanduser() if override else model_path("sherpa-onnx")


def prep(package: str = PACKAGE, force: bool = False) -> str:
    """Download the exact conversion and verify every learned tensor binding."""
    from huggingface_hub import snapshot_download

    if package != PACKAGE:
        raise SystemExit(f"Unknown exact GTCRN conversion: {package}")
    snapshot = Path(snapshot_download(REPO, revision=REVISION, force_download=force))
    metadata = snapshot / "metadata.json"
    if hashlib.sha256(metadata.read_bytes()).hexdigest() != METADATA_SHA:
        raise SystemExit("Pinned GTCRN protocol checksum mismatch")
    graph = sherpa_root() / "_enh/gtcrn_simple.onnx"
    if not graph.is_file():
        spec = importlib.util.spec_from_file_location("sherpa_download", ROOT / "tools/aud-sherpaonnx.py")
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        helper._download(helper.PACKAGES["gtcrn"]["url"], graph)
    bind(graph, snapshot / "model.safetensors")
    return str(snapshot)


def run(audio: str, output: str, package: str = PACKAGE, task: str = "denoise",
        force: bool = False, backend: str = "mlx") -> None:
    """Process bounded full files; profile actual fallback and publish checked PCM."""
    import onnxruntime as ort
    import onnxruntime_ep_mlx as ep
    from scipy.signal import resample_poly

    source, destination = Path(audio).resolve(), Path(output).resolve()
    if task not in {"dn", "denoise"} or backend not in {"mlx", "cpu"}:
        raise SystemExit("Require denoise task and mlx or cpu backend")
    if not source.is_file() or source == destination or destination.suffix.lower() != ".wav":
        raise SystemExit("Require existing audio and distinct WAV destination")
    if destination.exists() and not force:
        raise SystemExit("Output exists; use --force")
    samples, rate = sf.read(source, dtype="float32", always_2d=True)
    if not samples.size or not np.isfinite(samples).all() or not 8000 <= rate <= 48000:
        raise SystemExit("Require nonempty finite audio at 8–48 kHz")
    if len(samples) / rate > 300:
        raise SystemExit("GTCRN published maximum duration is 300 seconds")
    samples = samples.mean(axis=1)
    divisor = math.gcd(rate, 16000)
    if rate != 16000:
        samples = resample_poly(samples, 16000 // divisor, rate // divisor).astype(np.float32)
    if len(samples) > 4_800_000:
        raise SystemExit("GTCRN published maximum is 300 seconds at 16 kHz")
    snapshot = Path(prep(package))
    model, provenance = bind(sherpa_root() / "_enh/gtcrn_simple.onnx", snapshot / "model.safetensors")
    metadata = json.loads((snapshot / "metadata.json").read_text())
    destination.parent.mkdir(parents=True, exist_ok=True)
    options = ort.SessionOptions()
    options.enable_profiling = True
    options.profile_file_prefix = str(destination.with_suffix(".profile"))
    if backend == "mlx":
        ep.register_execution_provider_library()
    providers = ["MLXExecutionProvider", "CPUExecutionProvider"] if backend == "mlx" else ["CPUExecutionProvider"]
    load_start = time.monotonic()
    session = ort.InferenceSession(model, sess_options=options, providers=providers)
    load_seconds = time.monotonic() - load_start
    start = time.monotonic()
    spectra = analyze(samples)
    state = {item["input_name"]: np.zeros(item["shape"], np.float32) for item in metadata["caches"]}
    enhanced = []
    for spectrum in spectra:
        frame = np.stack((spectrum.real, spectrum.imag), axis=-1)[None, :, None, :]
        result = session.run(None, {"mix": frame, **state})
        if any(not np.isfinite(array).all() for array in result):
            raise RuntimeError("GTCRN returned nonfinite output or recurrent state")
        enhanced.append(result[0][0, :, 0, 0] + 1j * result[0][0, :, 0, 1])
        state = {item["input_name"]: result[i + 1] for i, item in enumerate(metadata["caches"])}
    waveform = synthesize(np.stack(enhanced), len(samples))
    seconds = time.monotonic() - start
    profile = Path(session.end_profiling())
    events = json.loads(profile.read_text())
    placements = Counter(event.get("args", {}).get("provider") for event in events
                         if event.get("cat") == "Node" and event.get("args", {}).get("provider"))
    if backend == "mlx" and not placements.get("MLXExecutionProvider"):
        raise RuntimeError("Requested MLX provider executed no graph regions")
    temporary = destination.with_suffix(".pending.wav")
    sf.write(temporary, waveform, 16000, subtype="FLOAT")
    temporary.replace(destination)
    receipt = {"repo": REPO, "revision": REVISION, **provenance, "metadata_sha256": METADATA_SHA,
        "onnxruntime": ort.__version__, "mlx_provider": ep.version(), "backend": backend,
        "placements": dict(placements), "profile": str(profile), "load_s": load_seconds,
        "processing_s": seconds, "processing_rtf": seconds / (len(samples) / 16000),
        "timing_scope": "STFT, sequential frame inference and ISTFT; excludes model load, resampling and WAV I/O",
        "resampler": "scipy.resample_poly (not publisher Sherpa resampler)" if rate != 16000 else "none",
        "frontend": "periodic sqrt-Hann, centered reflection, weighted overlap-add; offline full-file host",
        "frames": len(waveform), "sample_rate": 16000, "input_sample_rate": rate,
        "rms": float(np.sqrt(np.mean(waveform ** 2))), "peak": float(np.max(np.abs(waveform))),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(destination.read_bytes()).hexdigest()}
    destination.with_suffix(".result.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print("aud.result=" + json.dumps(receipt))


if __name__ == "__main__":
    import fire

    fire.Fire({"prep": prep, "run": run})
