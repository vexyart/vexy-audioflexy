#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = ["fire", "huggingface-hub", "soundfile", "platformdirs"]
# ///
# this_file: src/vexy_audioflexy/tools/asr/tools/diarize-coreai.py
"""Pinned mlboydaisuke CoreAI conversion through its original Swift host.

The maintained CLI-only patch exports existing turns/probabilities without any
graph, frontend or speaker-cache modifications. Mac bundles only, GPU preferred.
"""
from __future__ import annotations

import array
import hashlib
import importlib.util
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
ENGINE = Path(os.environ.get("COREAI_ZOO_ROOT", runtime_root("coreai-model-zoo")))
SWIFT = ENGINE / "conversion/nemotron3_diar/swift"
COMMIT = "2d214b3d20cdd66c9a45b57e591df407b8a0147d"
REPO = "mlboydaisuke/Nemotron-3-Diarization-CoreAI"
REVISION = "8dc6258987e69fc7addadd240df0998b6ae131c0"
PACKAGE = "coreai-mlboydaisuke-nemotron3-fp16"
CACHE = None


def bundle_name(mode: str) -> str:
    """Never choose the iPhone h18p ahead-of-time graph on a Mac."""
    if mode not in {"offline", "streaming"}:
        raise SystemExit("mode must be offline or streaming")
    return f"n3d_{mode}_float16.aimodel"


def host_mode(mode: str, latency_profile: str) -> str:
    """Select each published streaming profile without changing graph geometry."""
    bundle_name(mode)
    profiles = {"low": "ll", "very_low": "vll", "ultra_low": "ull"}
    if latency_profile not in profiles:
        raise SystemExit("latency_profile must be low, very_low or ultra_low")
    return "offline" if mode == "offline" else profiles[latency_profile]


def verify_assets(directory: Path, bundle: str = "") -> None:
    """Upstream Swift checks geometry; validate published host asset hashes too."""
    for name, record in json.loads((directory / "metadata.json").read_text())["assets"].items():
        path = directory / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise SystemExit(f"CoreAI host asset hash mismatch: {name}")
    if bundle:
        digest = hashlib.sha256()
        paths = sorted(p for p in (directory / bundle).rglob("*") if p.is_file())
        if not paths:
            raise SystemExit("Missing CoreAI Mac bundle")
        for path in paths:
            relative = path.relative_to(directory / bundle).as_posix()
            digest.update(f"{relative}\0{hashlib.sha256(path.read_bytes()).hexdigest()}\n".encode())
        expected = json.loads((directory / "metadata.json").read_text())["bundles"][bundle]["tree_sha256"]
        if digest.hexdigest() != expected:
            raise SystemExit("CoreAI graph tree hash mismatch")


def prep(package: str = PACKAGE, mode: str = "offline") -> str:
    """Fetch only the Mac graph and its host assets into the workspace cache."""
    from huggingface_hub import snapshot_download

    if package != PACKAGE:
        raise SystemExit(f"Expected {PACKAGE}; this host cannot run coder543 graphs")
    bundle = bundle_name(mode)
    path = Path(snapshot_download(REPO, revision=REVISION, cache_dir=CACHE,
        allow_patterns=[bundle + "/*", "*.f32le", "metadata.json", "LICENSE", "NOTICE", "SHA256SUMS"]))
    verify_assets(path, bundle)
    return str(path)


def verify_probabilities(path: Path, frames: int) -> None:
    """Require all eight speaker probabilities per frame, finite and in [0, 1]."""
    if path.stat().st_size != frames * 8 * 4 or frames <= 0:
        raise SystemExit("CoreAI probability output geometry mismatch")
    values = array.array("f")
    values.frombytes(path.read_bytes())
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
        raise SystemExit("CoreAI returned invalid speaker probabilities")


def prepare_host(timeout: int) -> Path:
    """Apply the turn-export patch once and build the inspected Swift CLI."""
    source = SWIFT / "Sources/n3d-selftest/main.swift"
    if not source.is_file():
        raise SystemExit(f"Clone john-rocky/coreai-model-zoo at {COMMIT} into {ENGINE}")
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ENGINE, text=True, timeout=timeout).strip()
    if actual != COMMIT:
        raise SystemExit(f"CoreAI host requires commit {COMMIT}; found {actual}")
    if 'j["segments"] = turns' not in source.read_text():
        patch = ROOT / "patches/coreai-n3d-turns.patch"
        subprocess.run(["git", "apply", "--check", str(patch)], cwd=ENGINE, check=True, timeout=timeout)
        subprocess.run(["git", "apply", str(patch)], cwd=ENGINE, check=True, timeout=timeout)
    subprocess.run(["swift", "build", "-c", "release", "--package-path", str(SWIFT)],
                   capture_output=True, text=True, check=True, timeout=timeout)
    return SWIFT / ".build/release/n3d-selftest"


def run(audio: str, output: str, package: str = PACKAGE, backend: str = "metal",
        mode: str = "offline", timeout: int = 900, force: bool = False,
        verbose: bool = False, latency_profile: str = "low") -> None:
    """Retain full Swift receipt, probabilities and normalized speaker turns."""
    import soundfile as sf

    source, destination = Path(audio).resolve(), Path(output).resolve()
    selected_mode = host_mode(mode, latency_profile)
    if not source.is_file() or timeout <= 0 or backend != "metal":
        raise SystemExit("Require existing audio, positive timeout and GPU/Metal backend")
    if destination == source or destination in source.parents:
        raise SystemExit("Output must not contain input audio")
    if destination.exists() and not force:
        raise SystemExit(f"Output exists: {destination}; use --force")
    model = Path(prep(package, mode))
    binary = prepare_host(timeout)
    with tempfile.TemporaryDirectory() as temporary:
        tmp = Path(temporary)
        converted, raw, probabilities = tmp / "input.wav", tmp / "native.json", tmp / "probabilities.f32le"
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(source), "-ar", "16000", "-ac", "1",
                        "-c:a", "pcm_s16le", str(converted)], check=True, timeout=timeout)
        duration = sf.info(converted).duration
        args = [str(binary), "--assets", str(model), "--bundle", str(model / bundle_name(mode)),
                "--wav", str(converted), "--mode", selected_mode, "--unit", "gpu",
                "--json", str(raw), "--probs-f32-out", str(probabilities)]
        start = time.monotonic()
        result = subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False)
        cli_wall = time.monotonic() - start
        if verbose or result.returncode:
            print(result.stdout + result.stderr)
        if result.returncode:
            raise SystemExit(result.returncode)
        receipt = json.loads(raw.read_text())
        if receipt.get("pass") is not True:
            raise SystemExit("CoreAI host did not finish successfully")
        loop = receipt["closed_loop"]
        spec = importlib.util.spec_from_file_location("native_diar", ROOT / "tools/diarize-native.py")
        parser = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(parser)
        turns = parser.normalize(loop, "nemo", duration)
        verify_probabilities(probabilities, loop["frames"])
        report = {"package": package, "weights_repo": REPO, "weights_revision": REVISION,
                  "runtime_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ENGINE, text=True).strip(),
                  "input": str(source), "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                  "duration_s": duration, "wall_s": loop["wall_s"], "cli_wall_s": cli_wall,
                  "load_s": loop["load_s"], "rtf": loop["wall_s"] / duration, "mode": mode,
                  "latency_profile": "offline" if mode == "offline" else latency_profile, "backend": "gpu-preferred",
                  "turns": turns, "speakers": sorted({row["speaker"] for row in turns}),
                  "weights_files": [{"file": str(p.relative_to(model)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                                    for p in sorted(model.rglob("*")) if p.is_file()]}
        receipts = tmp / "receipts"
        receipts.mkdir()
        shutil.copyfile(raw, receipts / "native.json")
        shutil.copyfile(probabilities, receipts / probabilities.name)
        (receipts / "engine.log").write_text(result.stdout + result.stderr)
        (receipts / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(receipts, destination)
    print("diar.result=" + json.dumps(report))


if __name__ == "__main__":
    import fire

    fire.Fire({"prep": prep, "run": run})
