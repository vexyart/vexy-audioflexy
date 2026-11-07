#!/usr/bin/env -S uv run -s
# /// script
# dependencies = ["platformdirs>=4", "fire", "numpy", "soundfile", "scipy"]
# ///
# this_file: src/vexy_audioflexy/tools/fx/tools/make_detection_fixture.py
"""Reproduce coarse speech-envelope and exact-zero silence detection fixtures."""

import os
from pathlib import Path
from platformdirs import user_cache_path, user_data_path
RUNTIME_ROOT = Path(os.environ.get("VAFX_AUDIO_RUNTIME_ROOT", user_data_path("vexy-audioflexy") / "audio" )).expanduser()
CACHE_ROOT = Path(os.environ.get("VAFX_AUDIO_CACHE_ROOT", user_cache_path("vexy-audioflexy") / "audio")).expanduser()
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import scipy
import soundfile as sf
from scipy.signal import resample_poly


def main(source: str, output: str) -> None:
    """Insert known silence without pretending the full speech clip has no pauses."""
    original, destination = Path(source).resolve(), Path(output).resolve()
    samples, rate = sf.read(original, dtype="float32", always_2d=True)
    samples = samples.mean(axis=1)
    if not len(samples) or not np.isfinite(samples).all() or rate <= 0:
        raise SystemExit("Require nonempty finite source audio")
    divisor = math.gcd(rate, 16000)
    samples = resample_poly(samples, 16000 // divisor, rate // divisor).astype(np.float32) if rate != 16000 else samples
    destination.mkdir(parents=True, exist_ok=True)
    cases = {"speech": samples, "cut": samples[:len(samples) // 2], "silence": np.zeros(48000, np.float32),
        "padded": np.concatenate((np.zeros(16000), samples, np.zeros(32000)))}
    fixtures = []
    for name, waveform in cases.items():
        path = destination / f"{name}.wav"
        sf.write(path, waveform, 16000, subtype="PCM_16")
        intervals = [] if name == "silence" else [[16000, 16000 + len(samples)]] if name == "padded" else [[0, len(waveform)]]
        fixtures.append({"name": name, "path": str(path), "frames": len(waveform),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "coarse_speech_envelope": intervals})
    manifest = {"source": str(original), "source_sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
        "versions": {"numpy": np.__version__, "scipy": scipy.__version__, "soundfile": sf.__version__},
        "resampling": "SciPy polyphase to mono16kHz, PCM16 output", "fixtures": fixtures,
        "label_scope": "Inserted silence is exact; speech envelopes include natural pauses, not frame-level speech labels. No ground-truth turn-completion labels."}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Prepared {len(fixtures)} fixtures and source provenance")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
