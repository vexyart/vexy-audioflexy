#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = ["platformdirs>=4", "fire", "huggingface-hub", "numpy", "soundfile"]
# ///
# this_file: src/vexy_audioflexy/tools/fx/tools/aud-localvqe.py
"""LocalVQE v1.3 noise/echo removal through ServeurpersoCom/s2s.cpp.

The native hop harness accepts mic/reference float streams at 16 kHz. One hop
of latency is flushed and removed; every original input sample is retained.
"""
from __future__ import annotations


import os
from pathlib import Path
from platformdirs import user_cache_path, user_data_path
RUNTIME_ROOT = Path(os.environ.get("VAFX_AUDIO_RUNTIME_ROOT", user_data_path("vexy-audioflexy") / "audio" )).expanduser()
CACHE_ROOT = Path(os.environ.get("VAFX_AUDIO_CACHE_ROOT", user_cache_path("vexy-audioflexy") / "audio")).expanduser()
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ENGINE = Path(os.environ.get("S2SCPP_ROOT", RUNTIME_ROOT / "engines/s2s.cpp"))
REPO = "Serveurperso/LocalVQE-GGUF"
REVISION = "a96f128b0fd4a499e5c415cdaef9a4cde5d65a65"
PACKAGE = "localvqe-v1.3-f32"
HOP = 256
RATE = 16000


def prep(package: str = PACKAGE) -> str:
    """Use this port's exact layout rather than LocalAI's incompatible GGUF."""
    from huggingface_hub import hf_hub_download

    if package != PACKAGE:
        raise SystemExit(f"Unknown package: {package}; expected {PACKAGE}")
    return hf_hub_download(REPO, "localvqe-v1.3-F32.gguf", revision=REVISION)


def prepare_streams(mic: np.ndarray, far: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Validate synchronized mono arrays, round up to hops and append a flush hop."""
    import numpy as np

    if mic.ndim != 1 or far.ndim != 1 or not len(mic) or len(mic) != len(far):
        raise SystemExit("Mic and reference must be nonempty synchronized mono streams")
    if not np.isfinite(mic).all() or not np.isfinite(far).all():
        raise SystemExit("Mic and reference must contain only finite samples")
    size = ((len(mic) + HOP - 1) // HOP + 1) * HOP
    return tuple(np.pad(signal.astype(np.float32), (0, size - len(signal))) for signal in (mic, far))


def align_output(raw: np.ndarray, length: int) -> np.ndarray:
    """Remove the native one-hop latency without truncating the input's last hop."""
    import numpy as np

    if raw.ndim != 1 or len(raw) < length + HOP or not np.isfinite(raw).all():
        raise SystemExit("LocalVQE returned truncated or nonfinite audio")
    return raw[HOP:HOP + length]


def run(audio: str, output: str, package: str = PACKAGE, task: str = "denoise",
        reference: str = "", backend: str = "MTL0", timeout: int = 900,
        force: bool = False, verbose: bool = False) -> None:
    """Run enhancement, with far-end speaker reference for acoustic echo removal."""
    import numpy as np
    import soundfile as sf

    source, destination = Path(audio).resolve(), Path(output).resolve()
    far_source = Path(reference).resolve() if reference else None
    if task not in {"denoise", "aec"} or package != PACKAGE or timeout <= 0:
        raise SystemExit("Require localvqe-v1.3-f32, task denoise/aec and positive timeout")
    if not source.is_file() or (far_source and not far_source.is_file()):
        raise SystemExit("Microphone audio and any far-end reference must exist")
    if task == "aec" and not far_source:
        raise SystemExit("AEC requires far-end loudspeaker reference audio")
    if destination in {source, far_source}:
        raise SystemExit("Output must differ from microphone and reference audio")
    if destination.exists() and not force:
        raise SystemExit(f"Output exists: {destination}; use --force")
    binary = ENGINE / "build/test-localvqe"
    if not binary.is_file():
        raise SystemExit(f"Build s2s.cpp with CMake at {ENGINE}")
    model = prep(package)
    with tempfile.TemporaryDirectory() as temporary:
        tmp = Path(temporary)
        for name, path in (("mic", source), ("far", far_source)):
            if path:
                subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-ar", str(RATE),
                                "-ac", "1", "-c:a", "pcm_f32le", str(tmp / f"{name}.wav")],
                               check=True, timeout=timeout)
        mic, _ = sf.read(tmp / "mic.wav", dtype="float32")
        far = sf.read(tmp / "far.wav", dtype="float32")[0] if far_source else np.zeros_like(mic)
        padded_mic, padded_far = prepare_streams(mic, far)
        padded_mic.tofile(tmp / "mic.f32")
        padded_far.tofile(tmp / "far.f32")
        env = os.environ.copy()
        env["GGML_BACKEND"] = backend
        start = time.monotonic()
        result = subprocess.run([str(binary), model, str(tmp / "mic.f32"), str(tmp / "far.f32"),
                                 str(tmp / "output.f32")], capture_output=True, text=True,
                                env=env, cwd=ENGINE, timeout=timeout, check=False)
        wall = time.monotonic() - start
        if verbose or result.returncode:
            print(result.stdout + result.stderr)
        if result.returncode:
            raise SystemExit(result.returncode)
        signal = align_output(np.fromfile(tmp / "output.f32", dtype=np.float32), len(mic))
        wav = tmp / "output.wav"
        sf.write(wav, signal, RATE, subtype="PCM_16")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(wav.read_bytes())
    duration = len(signal) / RATE
    print("aud.result=" + json.dumps({"output": str(destination), "wall_s": wall,
          "audio_duration_s": duration, "rtf": wall / duration, "backend": backend,
          "latency_compensated_samples": HOP, "weights_revision": REVISION}))


if __name__ == "__main__":
    import fire

    fire.Fire({"prep": prep, "run": run})
