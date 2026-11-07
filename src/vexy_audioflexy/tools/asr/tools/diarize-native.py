#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = ["fire", "huggingface-hub", "soundfile", "platformdirs"]
# ///
# this_file: src/vexy_audioflexy/tools/asr/tools/diarize-native.py
"""Speaker turns through native audio.cpp/NeMo; retain raw and normalized JSON."""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from _audioflexy_paths import runtime_root

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "catalogs/diarization.json"


def resolve(package: str) -> dict:
    """Require an exact conversion identity, not just a shared base model name."""
    for item in json.loads(CATALOG.read_text())["models"]:
        if item["package"] == package:
            if item["engine"] not in {"nemo", "audiocpp"}:
                raise SystemExit("Use diarize-mlx.py for MLX packages")
            return item
    raise SystemExit(f"Unknown diarization package: {package}; see {CATALOG}")


def prep(package: str = "audiocpp-nemotron3-q8_0") -> str:
    """Fetch pinned conversion; validate SHA256 where the upstream index gives it."""
    from huggingface_hub import hf_hub_download

    item = resolve(package)
    path = hf_hub_download(item["repo"], item["filename"], revision=item["revision"])
    if item.get("sha256") and hashlib.sha256(Path(path).read_bytes()).hexdigest() != item["sha256"]:
        raise SystemExit(f"Weight hash mismatch: {path}")
    return path


def normalize(raw: object, engine: str, duration: float) -> list[dict]:
    """Parse engine-specific offsets into seconds; permit overlaps and empty output."""
    rows = raw if engine == "audiocpp" else raw.get("segments") if isinstance(raw, dict) else None
    if not isinstance(rows, list):
        raise SystemExit("Native diarization output must contain a segment array")
    turns = []
    for row in rows:
        if not isinstance(row, dict):
            raise SystemExit("Native diarization segment must be an object")
        start_key, end_key, speaker_key = ("start_sample", "end_sample", "speaker_id") if engine == "audiocpp" else ("start", "end", "speaker")
        start, end, speaker = row.get(start_key), row.get(end_key), row.get(speaker_key)
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in (start, end)):
            raise SystemExit("Turn timestamps must be finite numbers")
        if engine == "audiocpp":
            if not isinstance(start, int) or not isinstance(end, int):
                raise SystemExit("Native sample offsets must be integers")
            start, end = start / 16000, end / 16000
        if not 0 <= start < end <= duration + .001 or isinstance(speaker, bool) or not isinstance(speaker, (str, int)) or str(speaker) == "":
            raise SystemExit("Invalid speaker identity or out-of-range turn")
        turns.append({"start": float(start), "end": float(end), "speaker": str(speaker)})
    return sorted(turns, key=lambda row: (row["start"], row["end"], row["speaker"]))


def command(item: dict, model: str, audio: Path, raw: Path, backend: str, mode: str) -> tuple[Path, list[str]]:
    """Use inspected upstream CLI contracts; standalone streaming is faster than realtime."""
    if item["engine"] == "audiocpp":
        engine = Path(os.environ.get("AUDIOCPP_ISSUE201_ROOT", runtime_root("audio.cpp")))
        args = [str(engine / "build/macos-metal-release/bin/audiocpp_cli"), "--task", "diar",
                "--family", item["family"], "--model", model, "--backend", backend,
                "--audio", str(audio), "--turns-out", str(raw), "--mode", mode]
        if mode == "streaming":
            args.extend(["--session-option", "nemotron_3_diar.latency_profile=low"])
    else:
        engine = Path(os.environ.get("NEMOSPEECHCPP_ROOT", runtime_root("NeMo-Speech.cpp")))
        args = [str(engine / "build/metal-asr/bin/nemo-speech"), "diarize", str(audio),
                "--model", model, "--device", backend, "--format", "json", "--output", str(raw)]
        if mode == "offline":
            args.append("--offline")
    return engine, args


def run(audio: str, output: str, package: str = "audiocpp-nemotron3-q8_0",
        backend: str = "metal", mode: str = "offline", timeout: int = 900,
        force: bool = False, verbose: bool = False) -> None:
    """Create turns/report/engine.log directory without altering source audio."""
    import soundfile as sf

    item = resolve(package)
    source, destination = Path(audio).resolve(), Path(output).resolve()
    if not source.is_file() or timeout <= 0 or mode not in {"offline", "streaming"}:
        raise SystemExit("Require existing audio, positive timeout and offline/streaming mode")
    if destination == source or destination in source.parents:
        raise SystemExit("Output directory must not contain source audio")
    if destination.exists() and not force:
        raise SystemExit(f"Output exists: {destination}; use --force")
    model = prep(package)
    with tempfile.TemporaryDirectory() as temporary:
        tmp = Path(temporary)
        converted, raw = tmp / "input.wav", tmp / "native.json"
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(source), "-ar", "16000", "-ac", "1",
                        "-c:a", "pcm_s16le", str(converted)], check=True, timeout=timeout)
        info = sf.info(converted)
        if not info.frames:
            raise SystemExit("Input audio is empty")
        # HF's cache may return an extensionless blob. audio.cpp resolves links
        # before choosing its reader by extension, so a named real file is needed.
        named_model = tmp / item["filename"]
        shutil.copyfile(model, named_model)
        engine, args = command(item, str(named_model), converted, raw, backend, mode)
        start = time.monotonic()
        result = subprocess.run(args, cwd=engine, capture_output=True, text=True, timeout=timeout, check=False)
        wall = time.monotonic() - start
        log = result.stdout + result.stderr
        if verbose or result.returncode:
            print(log)
        if result.returncode:
            raise SystemExit(result.returncode)
        # audio.cpp intentionally omits the turns file when there are no turns.
        native = json.loads(raw.read_text()) if raw.exists() else [] if item["engine"] == "audiocpp" else None
        turns = normalize(native, item["engine"], info.duration)
        report = {"package": package, "weights_repo": item["repo"], "weights_revision": item["revision"],
                  "weights_sha256": hashlib.sha256(Path(model).read_bytes()).hexdigest(),
                  "input": str(source), "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                  "duration_s": info.duration, "wall_s": wall, "rtf": wall / info.duration,
                  "mode": mode, "backend": backend, "turns": turns,
                  "latency_profile": "low" if item["engine"] == "audiocpp" and mode == "streaming" else "upstream-default",
                  "speakers": sorted({row["speaker"] for row in turns})}
        receipts = tmp / "receipts"
        receipts.mkdir()
        (receipts / "native.json").write_text(json.dumps(native, indent=2) + "\n")
        (receipts / "engine.log").write_text(log)
        (receipts / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(receipts, destination)
    print("diar.result=" + json.dumps(report))


if __name__ == "__main__":
    import fire

    fire.Fire({"prep": prep, "run": run})
