#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = ["fire>=0.7,<0.8", "soundfile>=0.13", "numpy>=2", "platformdirs"]
# ///
# this_file: src/vexy_audioflexy/tools/asr/tools/audio-task-audiocpp.py
"""Native alignment, speaker turns, turn completion and AudioSR with typed receipts."""

import hashlib
import json
import math
import os
import runpy
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from _audioflexy_paths import runtime_root

ROOT = Path(__file__).resolve().parents[1]
FORMATS = runpy.run_path(str(Path(__file__).with_name("audio_task_contracts.py")))
turn_result = FORMATS["turn_result"]
alignment_words = FORMATS["alignment_words"]
result_for = FORMATS["result_for"]
command = FORMATS["command"]
TASKS = FORMATS["TASKS"]


def run(
    audio: str,
    output: str,
    family: str,
    model: str,
    backend: str = "metal",
    text: str = "",
    language: str = "",
    mode: str = "offline",
    request_options: str = "",
    timeout: float = 900,
    engine_root: str = "",
    force: bool = False,
    verbose: bool = False,
) -> None:
    """Stage native output, verify its contract and retain model/input provenance."""
    source, destination, weights = (
        Path(audio).resolve(),
        Path(output).resolve(),
        Path(model).resolve(),
    )
    if family not in TASKS or not source.is_file() or not weights.exists():
        raise ValueError(
            "Require a requested family, existing audio and exact model path"
        )
    if (
        not math.isfinite(timeout)
        or timeout <= 0
        or mode not in {"offline", "streaming"}
    ):
        raise ValueError("Require positive finite timeout and supported mode")
    if destination == source or destination in source.parents or destination == weights:
        raise ValueError("Output directory must be distinct from source/model assets")
    if destination.exists() and not force:
        raise FileExistsError("Output exists; use --force")
    task = TASKS[family]
    if task == "align" and not text.strip():
        raise ValueError("Alignment requires a nonempty transcript")
    engine = Path(
        engine_root
        or os.getenv(
            "AUDIOCPP_ISSUE301_ROOT", str(runtime_root("audio.cpp-issue301"))
        )
    ).resolve()
    binary = engine / "build/macos-metal-release/bin/audiocpp_cli"
    import soundfile as sf

    with tempfile.TemporaryDirectory() as temporary:
        stage = Path(temporary)
        feed = source
        if task != "s2s":
            feed = stage / "input.wav"
            subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-i",
                    str(source),
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    "-c:a",
                    "pcm_s16le",
                    str(feed),
                ],
                check=True,
                timeout=timeout,
            )
        duration = sf.info(feed).duration
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("Input audio is empty")
        raw = stage / ("enhanced.wav" if task == "s2s" else "native.json")
        args = command(
            binary,
            family,
            weights,
            feed,
            raw,
            backend,
            text,
            language,
            mode,
            request_options,
        )
        start = time.monotonic()
        native = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=engine,
            check=False,
        )
        wall = time.monotonic() - start
        log = native.stdout + native.stderr
        if verbose or native.returncode:
            print(log)
        native.check_returncode()
        data = result_for(task, raw, native.stdout, duration)
        receipt = {
            "family": family,
            "task": task,
            "backend": backend,
            "mode": mode,
            "model": str(weights),
            "input": str(source),
            "duration_s": duration,
            "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
            "model_sha256": hashlib.sha256(weights.read_bytes()).hexdigest()
            if weights.is_file()
            else None,
            "wall_s": wall,
            "rtf": wall / duration,
            "result": data,
        }
        (stage / "report.json").write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2) + "\n"
        )
        (stage / "engine.log").write_text(log)
        if feed != source:
            feed.unlink()
        shutil.copytree(stage, destination, dirs_exist_ok=force)
    print("audio_task.result=" + json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    import fire

    fire.Fire({"run": run})
