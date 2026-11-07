#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = ["fire", "huggingface-hub", "mlx-audio==0.5.7", "soundfile", "platformdirs"]
# ///
# this_file: src/vexy_audioflexy/tools/asr/tools/diarize-mlx.py
"""Exact MLX Nemotron 3 eight-bit conversion, bounded AOSC and incremental turns.

API source: Blaizzy/mlx-audio mlx_audio/vad/models/nemotron_diarization.
The 'offline' preset uses 30.4s input context with AOSC, not unrestricted attention.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "mlx-nemotron3-8bit"
REPO = "mlx-community/Nemotron-3-Diarization-8bit"
REVISION = "dd8b8ce3d69a80540a7e50150e2beed1c34fee37"


def prep(package: str = PACKAGE) -> str:
    """Fetch this exact quantized conversion rather than the card's unquantized example."""
    from huggingface_hub import snapshot_download

    if package != PACKAGE:
        raise SystemExit(f"Expected {PACKAGE}")
    return snapshot_download(REPO, revision=REVISION, allow_patterns=["*.json", "*.safetensors"])


def run(audio: str, output: str, package: str = PACKAGE, backend: str = "metal",
        mode: str = "offline", timeout: int = 900, force: bool = False,
        verbose: bool = False) -> None:
    """Write the same structured report as native adapters; batch process sets deadline."""
    import mlx.core as mx
    import soundfile as sf
    from mlx_audio.vad import load

    source, destination = Path(audio).resolve(), Path(output).resolve()
    if not source.is_file() or timeout <= 0 or mode not in {"offline", "streaming"} or backend != "metal":
        raise SystemExit("Require audio, positive timeout, offline/streaming mode and Metal backend")
    if destination == source or destination in source.parents:
        raise SystemExit("Output must not contain source audio")
    if destination.exists() and not force:
        raise SystemExit(f"Output exists: {destination}; use --force")
    model_path = Path(prep(package))
    start = time.monotonic()
    model = load(str(model_path), strict=True)
    model.set_streaming_config("offline" if mode == "offline" else "low")
    mx.synchronize()
    load_s = time.monotonic() - start
    with tempfile.TemporaryDirectory() as temporary:
        converted = Path(temporary) / "input.wav"
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(source), "-ar", "16000", "-ac", "1",
                        str(converted)], timeout=timeout, check=True)
        duration = sf.info(converted).duration
        if duration <= 0:
            raise SystemExit("Input audio is empty")
        start = time.monotonic()
        segments, events = [], []
        first_result_s = None
        results = model.generate_stream(str(converted)) if mode == "streaming" else [model.generate(str(converted))]
        for result in results:
            mx.eval(result.speaker_probs)
            if first_result_s is None:
                first_result_s = time.monotonic() - start
            rows = [{"start": float(s.start), "end": float(s.end), "speaker": int(s.speaker)} for s in result.segments]
            segments.extend(rows)
            events.append({"elapsed_s": time.monotonic() - start, "segments": rows})
        mx.synchronize()
        wall = time.monotonic() - start
    spec = importlib.util.spec_from_file_location("native_diar", ROOT / "tools/diarize-native.py")
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    turns = native.normalize({"segments": segments}, "nemo", duration)
    report = {"package": package, "weights_repo": REPO, "weights_revision": REVISION,
              "weights_files": [{"file": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                                for p in sorted(model_path.glob("*")) if p.is_file()],
              "input": str(source), "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "duration_s": duration, "wall_s": wall, "load_s": load_s, "rtf": wall / duration,
              "first_result_s": first_result_s, "mode": mode, "backend": backend,
              "latency_profile": "offline" if mode == "offline" else "low", "turns": turns,
              "speakers": sorted({row["speaker"] for row in turns}), "runtime": "mlx-audio==0.5.7"}
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "native.json").write_text(json.dumps(events, indent=2) + "\n")
    (destination / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if verbose:
        print(json.dumps(events, indent=2))
    print("diar.result=" + json.dumps(report))


if __name__ == "__main__":
    import fire

    fire.Fire({"prep": prep, "run": run})
