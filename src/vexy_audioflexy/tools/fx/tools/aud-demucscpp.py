#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = ["platformdirs>=4", "fire", "huggingface-hub", "numpy", "soundfile"]
# ///
# this_file: src/vexy_audioflexy/tools/fx/tools/aud-demucscpp.py
"""Separate music into named stems with sevagh/demucs.cpp; retain every stem."""
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
ENGINE = Path(os.environ.get("DEMUCSCPP_ROOT", RUNTIME_ROOT / "engines/demucs.cpp"))
REPO = "Retrobear/demucs.cpp"
REVISION = "8f58ac0491bbea657275bcd3e38af1ab3a27bfc9"
STEMS = ("drums", "bass", "other", "vocals", "guitar", "piano")


def resolve(package: str) -> tuple[str, tuple[str, ...]]:
    """The four and six source v4 checkpoints use distinct upstream weights."""
    if package not in {"htdemucs-4s", "htdemucs-6s"}:
        raise SystemExit("package must be htdemucs-4s or htdemucs-6s")
    return f"ggml-model-{package}-f16.bin", STEMS[:4] if package.endswith("4s") else STEMS


def prep(package: str = "htdemucs-4s") -> str:
    """Download the pinned original ggml checkpoint, not an audio.cpp GGUF."""
    from huggingface_hub import hf_hub_download

    filename, _ = resolve(package)
    return hf_hub_download(REPO, filename, repo_type="dataset", revision=REVISION)


def inspect_stems(directory: Path, stems: tuple[str, ...], frames: int) -> tuple[list[dict], object]:
    """Require finite stereo stems of the input length; measure reconstruction."""
    import numpy as np
    import soundfile as sf

    records, signals = [], []
    for index, name in enumerate(stems):
        path = directory / f"target_{index}_{name}.wav"
        signal, rate = sf.read(path, dtype="float32", always_2d=True)
        if rate != 44100 or signal.shape != (frames, 2) or not np.isfinite(signal).all():
            raise SystemExit(f"Invalid stem geometry or samples: {path.name}")
        records.append({"stem": name, "file": path.name, "frames": frames, "sample_rate": rate,
                        "rms": float(np.sqrt(np.mean(signal ** 2))),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        signals.append(signal)
    if not any(item["rms"] > 0 for item in records):
        raise SystemExit("All separated stems are silent")
    return records, np.sum(signals, axis=0)


def run(audio: str, output: str, package: str = "htdemucs-4s", timeout: int = 1800,
        force: bool = False, verbose: bool = False) -> None:
    """Write a directory of stems plus provenance and timing in report.json.

    SDR of the summed stems measures reconstruction only; separation quality
    requires independently labelled ground-truth stems and is not claimed here.
    """
    import numpy as np
    import soundfile as sf

    _, stems = resolve(package)
    source, destination = Path(audio).resolve(), Path(output).resolve()
    if not source.is_file() or timeout <= 0:
        raise SystemExit("Require existing input audio and a positive timeout")
    if destination == source or destination in source.parents:
        raise SystemExit("Output directory must not contain input audio")
    if destination.exists() and not force:
        raise SystemExit(f"Output exists: {destination}; use --force")
    binary = ENGINE / "build/demucs.cpp.main"
    if not binary.is_file():
        raise SystemExit(f"Build demucs.cpp at {ENGINE} with CMake")
    model = prep(package)
    with tempfile.TemporaryDirectory() as temporary:
        tmp = Path(temporary)
        converted, outputs = tmp / "input.wav", tmp / "stems"
        outputs.mkdir()
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(source), "-ar", "44100", "-ac", "2",
                        "-c:a", "pcm_f32le", str(converted)], check=True, timeout=timeout)
        signal, rate = sf.read(converted, dtype="float32", always_2d=True)
        if not len(signal) or not np.isfinite(signal).all() or not np.any(signal):
            raise SystemExit("Input is empty, nonfinite or silent")
        start = time.monotonic()
        result = subprocess.run([str(binary), model, str(converted), str(outputs)],
                                capture_output=True, text=True, timeout=timeout, check=False)
        wall = time.monotonic() - start
        (outputs / "engine.log").write_text(result.stdout + result.stderr)
        if verbose or result.returncode:
            print(result.stdout + result.stderr)
        if result.returncode:
            raise SystemExit(result.returncode)
        records, reconstructed = inspect_stems(outputs, stems, len(signal))
        error_power = float(np.sum((signal - reconstructed) ** 2))
        reconstruction_sdr = float(10 * np.log10(float(np.sum(signal ** 2)) / max(error_power, 1e-30)))
        report = {"package": package, "weights_repo": REPO, "weights_revision": REVISION,
                  "input": str(source), "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                  "wall_s": wall, "audio_duration_s": len(signal) / rate,
                  "rtf": wall / (len(signal) / rate), "stems": records,
                  "reconstruction_sdr_db": reconstruction_sdr,
                  "separation_sdr_db": None, "separation_quality_note": "No ground-truth stems supplied"}
        (outputs / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(outputs, destination)
    print("aud.result=" + json.dumps(report))


if __name__ == "__main__":
    import fire

    fire.Fire({"prep": prep, "run": run})
