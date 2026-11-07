#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = ["platformdirs>=4", "fire", "numpy", "soundfile", "speechbrain==1.0.3", "torch==2.5.1", "torchaudio==2.5.1", "huggingface-hub<1"]
# ///
# this_file: src/vexy_audioflexy/tools/fx/tools/aud-speechbrain.py
"""MetricGAN+ speech enhancement through SpeechBrain's spectral-mask interface.

The model requires 16 kHz mono tensors. Input conversion uses ffmpeg and inference
runs on the CPU; output retains the complete resampled duration.
"""
from __future__ import annotations


import os
from pathlib import Path
import runpy
model_path = runpy.run_path(str(Path(__file__).with_name("_audioflexy_paths.py")))["model_path"]
from platformdirs import user_cache_path, user_data_path
RUNTIME_ROOT = Path(os.environ.get("VAFX_AUDIO_RUNTIME_ROOT", user_data_path("vexy-audioflexy") / "audio" )).expanduser()
CACHE_ROOT = Path(os.environ.get("VAFX_AUDIO_CACHE_ROOT", user_cache_path("vexy-audioflexy") / "audio")).expanduser()
import json
import subprocess
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "metricgan-plus-voicebank"
REPO = "speechbrain/metricgan-plus-voicebank"


def check_package(package: str) -> None:
    """Keep package addressing explicit instead of accepting arbitrary YAML sources."""
    if package != PACKAGE:
        raise SystemExit(f"Unknown SpeechBrain package: {package}; available: {PACKAGE}")


def load():
    """Load the documented spectral-mask checkpoint in an isolated CPU environment."""
    from speechbrain.inference.enhancement import SpectralMaskEnhancement

    return SpectralMaskEnhancement.from_hparams(
        source=REPO, savedir=str(model_path("speechbrain") / PACKAGE),
        run_opts={"device": "cpu"},
    )


def prep(package: str = PACKAGE, force: bool = False) -> None:
    """Fetch the checkpoint and instantiate the model before reporting ready."""
    check_package(package)
    load()
    print(f"[prep] ready: {REPO}")


def run(audio: str, output: str = "output.wav", package: str = PACKAGE,
        task: str = "denoise", force: bool = False) -> None:
    """Enhance a full recording and report inference time and output measurements."""
    check_package(package)
    if task != "denoise":
        raise SystemExit("MetricGAN+ supports task=denoise")
    src, out = Path(audio).resolve(), Path(output).resolve()
    if not src.is_file():
        raise SystemExit(f"Audio not found: {src}")
    if src == out:
        raise SystemExit("Output must differ from input")
    if out.exists() and not force:
        raise SystemExit(f"Output exists: {out}; use --force")
    import numpy as np
    import soundfile as sf
    import torch

    model = load()
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "input.wav"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
                        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(wav)],
                       check=True, timeout=120)
        samples, rate = sf.read(wav, dtype="float32")
        if not samples.size:
            raise SystemExit("Audio is empty")
        start = time.monotonic()
        with torch.inference_mode():
            enhanced = model.enhance_batch(torch.from_numpy(samples).unsqueeze(0),
                                          lengths=torch.tensor([1.0])).squeeze(0).cpu().numpy()
        wall = time.monotonic() - start
    if enhanced.shape != samples.shape or not np.isfinite(enhanced).all():
        raise SystemExit("MetricGAN+ returned invalid samples or changed duration")
    if not np.any(enhanced):
        raise SystemExit("MetricGAN+ returned silence")
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out, enhanced, rate, subtype="PCM_16")
    duration = len(enhanced) / rate
    print("aud.result=" + json.dumps({"output": str(out), "wall_s": wall,
          "audio_duration_s": duration, "rtf": wall / duration, "sample_rate": rate,
          "rms": float(np.sqrt(np.mean(enhanced ** 2))), "package": package}))


if __name__ == "__main__":
    import fire

    fire.Fire({"prep": prep, "run": run, "packages": lambda: {PACKAGE: {"repo": REPO, "op": "dn"}}})
