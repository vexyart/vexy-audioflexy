#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = ["fire", "huggingface-hub", "soundfile", "platformdirs"]
# ///
# this_file: src/vexy_audioflexy/tools/asr/tools/diarize-coder-coreai.py
"""Exact coder543 CoreAI fast32/fast128 graphs using the workspace Swift host.

Source graphs remain unmodified; the host reuses NemotronDiarizer's tested mel
frontend/cache. Graph geometry and policy constants come from pinned metadata.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from _audioflexy_paths import runtime_root, stage_native_host

ROOT = Path(__file__).resolve().parents[1]
SWIFT = runtime_root("coreai-diar")
REPO = "coder543/nemotron-3-diarization-coreai"
REVISION = "99f6a8a5b614b28d61bfd16190edb92eab745198"
PACKAGES = {f"coreai-coder-nemotron3-{profile}": profile for profile in ("fast32", "fast128")}
CACHE = None


def profile_for(package: str) -> str:
    """Keep the two explicit policies distinct; neither is unrestricted offline."""
    if package not in PACKAGES:
        raise SystemExit("Unknown coder543 package; choose " + ",".join(PACKAGES))
    return PACKAGES[package]


def prep(package: str = "coreai-coder-nemotron3-fast32") -> str:
    """Fetch the exact profile plus license/notice into a local workspace cache."""
    from huggingface_hub import snapshot_download

    profile = profile_for(package)
    return str(Path(snapshot_download(REPO, revision=REVISION, cache_dir=CACHE,
        allow_patterns=[profile + "/*", "LICENSE", "NOTICE"])) / profile)


def run(audio: str, output: str, package: str = "coreai-coder-nemotron3-fast32",
        backend: str = "metal", mode: str = "offline", timeout: int = 900,
        force: bool = False, verbose: bool = False) -> None:
    """Run a bounded chunk policy over the file; retains final flush and probabilities.

The current host consumes a whole file. Use offline in this consumer; the graph
policy itself is stateful streaming. Incremental PCM delivery is not claimed.
    """
    import soundfile as sf

    source, destination = Path(audio).resolve(), Path(output).resolve()
    profile = profile_for(package)
    if not source.is_file() or timeout <= 0 or mode != "offline" or backend not in {"metal", "ane"}:
        raise SystemExit("Require audio, positive timeout, mode offline and backend metal/ane")
    if destination == source or destination in source.parents:
        raise SystemExit("Output must not contain source audio")
    if destination.exists() and not force:
        raise SystemExit(f"Output exists: {destination}; use --force")
    model = Path(prep(package))
    spec = importlib.util.spec_from_file_location("coreai_shared", ROOT / "tools/diarize-coreai.py")
    shared = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shared)
    library = shared.ENGINE
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=library, text=True, timeout=timeout).strip()
    if actual != shared.COMMIT:
        raise SystemExit(f"Reused Swift primitives require commit {shared.COMMIT}; found {actual}")
    stage_native_host(ROOT / "native/coreai-diar", SWIFT)
    subprocess.run(["swift", "build", "-c", "release", "--package-path", str(SWIFT)],
                   check=True, capture_output=True, text=True, timeout=timeout,
                   env={**os.environ, "COREAI_ZOO_ROOT": str(library)})
    with tempfile.TemporaryDirectory() as temporary:
        tmp = Path(temporary)
        converted, raw, probabilities = tmp / "input.wav", tmp / "native.json", tmp / "probabilities.f32le"
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(source), "-ar", "16000", "-ac", "1",
                        "-c:a", "pcm_s16le", str(converted)], check=True, timeout=timeout)
        duration = sf.info(converted).duration
        args = [str(SWIFT / ".build/release/coder-diar"), str(model), str(converted), str(raw),
                str(probabilities), "gpu" if backend == "metal" else "ane"]
        start = time.monotonic()
        result = subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False)
        cli_wall = time.monotonic() - start
        if verbose or result.returncode:
            print(result.stdout + result.stderr)
        if result.returncode:
            raise SystemExit(result.returncode)
        native = json.loads(raw.read_text())
        shared.verify_probabilities(probabilities, native["frames"])
        parser_spec = importlib.util.spec_from_file_location("native_diar", ROOT / "tools/diarize-native.py")
        parser = importlib.util.module_from_spec(parser_spec)
        parser_spec.loader.exec_module(parser)
        turns = parser.normalize(native, "nemo", duration)
        report = {"package": package, "weights_repo": REPO, "weights_revision": REVISION,
                  "input": str(source), "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                  "duration_s": duration, "wall_s": native["wall_s"], "load_s": native["load_s"],
                  "cli_wall_s": cli_wall, "rtf": native["wall_s"] / duration,
                  "mode": "file-chunked", "latency_profile": profile, "backend": "gpu-preferred" if backend == "metal" else "ane-preferred",
                  "turns": turns, "speakers": sorted({row["speaker"] for row in turns}),
                  "host_library_commit": shared.COMMIT, "compressions": native["compressions"],
                  "weights_files": [{"file": str(path.relative_to(model)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                                    for path in sorted(model.rglob("*")) if path.is_file()]}
        receipts = tmp / "receipts"
        receipts.mkdir()
        shutil.copyfile(raw, receipts / "native.json")
        shutil.copyfile(probabilities, receipts / "probabilities.f32le")
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
