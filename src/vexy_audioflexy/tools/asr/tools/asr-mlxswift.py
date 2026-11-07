#!/usr/bin/env -S uv run
# /// script
# dependencies = ["fire", "huggingface-hub", "soundfile", "platformdirs"]
# ///
# this_file: src/vexy_audioflexy/tools/asr/tools/asr-mlxswift.py
"""Exact Parakeet Ultra/Audio8 through their distinct Xcode-built Swift runtimes."""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import tempfile
import time
from pathlib import Path

from _audioflexy_paths import runtime_root

ROOT = Path(__file__).resolve().parents[1]
REPO = "selcukkubur/parakeet-ultra-mlx"
REVISION = "b554592c50b2a48471add2daa3d46fa9f00fef5e"
EXPORTS = {
    REPO: {"revision": REVISION, "engine": "mlx-audio-swift",
           "build": "mlx-audio-swift-xcode", "commit": "8d86630ade569728aaea3dc1a29fc44e2efa719b",
           "bin_env": "MLX_SWIFT_STT_BIN", "patch": "mlx-swift-local-parakeet.patch",
           "langs": ["bg", "cs", "da", "de", "el", "en", "es", "et", "fi", "fr", "hr", "hu", "it", "lt", "lv", "mt", "nl", "pl", "pt", "ro", "ru", "sk", "sl", "sv", "uk"]},
    "selcukkubur/Audio8-ASR-Infinite-mlx": {
        "revision": "0e3f912dc09ac09dec0348e81c7eea96f19cac0e",
        "engine": "mlx-audio-swift-audio8", "build": "mlx-audio-swift-audio8-xcode",
        "commit": "9473f480658d30834a9885cd078401584109e865", "bin_env": "AUDIO8_SWIFT_STT_BIN",
        "patch": "mlx-swift-local-audio8.patch", "langs": ["en", "zh"]},
}


def resolve(repo: str) -> dict:
    """Choose exact artifact/runtime contracts before download or native execution."""
    if repo not in EXPORTS:
        raise SystemExit(f"Unknown exact Swift conversion: {repo}")
    return EXPORTS[repo]


def parse_result(result: dict) -> tuple[str, float]:
    """Use CLI processing_time; Parakeet total_time is not measured processing time."""
    text = result.get("text")
    seconds = result.get("processing_time")
    if not isinstance(text, str) or not text.strip():
        raise SystemExit("Native MLX Swift ASR returned no transcript")
    if not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds <= 0:
        raise SystemExit("Require finite positive CLI processing_time; apply the maintained patch")
    return text.strip(), float(seconds)


def validate_language(spec: dict, lang: str) -> None:
    """Audio8's trained prompt requires an explicit en/zh language token."""
    if lang and lang not in spec["langs"]:
        raise SystemExit(f"Unsupported language {lang}")
    if spec["engine"] == "mlx-audio-swift-audio8" and not lang:
        raise SystemExit("Audio8 requires --lang en or zh for the trained prefill")


def prep(repo: str = REPO) -> str:
    """Download only the exact requested export to the workspace cache."""
    from huggingface_hub import snapshot_download

    spec = resolve(repo)
    path = snapshot_download(repo, revision=spec["revision"], cache_dir=None,
                             allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.vocab"])
    print(f"[prep] ready: {repo} -> {path}")
    return path


def transcribe(audio: str, text_out: str = "", repo: str = REPO, lang: str = "",
               force: bool = False, timeout: int = 900) -> None:
    """Save native segments/logs and report processing separately from CLI wall."""
    import soundfile as sf

    spec = resolve(repo)
    engine = runtime_root(spec["engine"])
    default_bin = runtime_root(spec["build"]) / "Build/Products/Release/mlx-audio-swift-stt"
    binary = Path(os.environ.get(spec["bin_env"], default_bin))
    source = Path(audio).resolve()
    output = Path(text_out).resolve() if text_out else source.with_suffix(".txt")
    if not source.is_file() or source == output or timeout <= 0:
        raise SystemExit("Require audio, distinct transcript path and positive timeout")
    validate_language(spec, lang)
    if output.exists() and not force:
        raise SystemExit(f"Transcript exists: {output}; use --force")
    if not binary.is_file():
        raise SystemExit(f"Build mlx-audio-swift-stt with Xcode in {engine}; apply patches/{spec['patch']}")
    commit = subprocess.check_output(["git", "-C", str(engine), "rev-parse", "HEAD"], text=True).strip()
    if commit != spec["commit"]:
        raise SystemExit("MLX Swift checkout differs from the verified runtime")
    patch = ROOT / "patches" / spec["patch"]
    subprocess.run(["git", "-C", str(engine), "apply", "--reverse", "--check", str(patch)],
                   check=True, capture_output=True, timeout=30)
    snapshot = prep(repo)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary:
        wav = Path(temporary) / "input.wav"
        stem = Path(temporary) / "native"
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(source), "-ar", "16000", "-ac", "1", str(wav)],
                       check=True, timeout=timeout)
        duration = sf.info(wav).duration
        if duration <= 0:
            raise SystemExit("Input audio is empty")
        command = [str(binary), "--model", snapshot, "--audio", str(wav), "--output-path", str(stem),
                   "--format", "json", "--verbose"]
        if lang:
            command += ["--language", lang]
        start = time.monotonic()
        native = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
        wall = time.monotonic() - start
        output.with_suffix(".native.log").write_text(native.stdout + native.stderr)
        native.check_returncode()
        data = stem.with_suffix(".json").read_text()
    output.with_suffix(".native.json").write_text(data)
    text, processing = parse_result(json.loads(data))
    output.write_text(text + "\n")
    receipt = {"text": text, "text_out": str(output), "wall_s": processing,
        "cli_wall_s": wall, "audio_duration_s": duration, "rtf": processing / duration,
        "repo": repo, "weights_revision": spec["revision"], "runtime_commit": spec["commit"],
        "language": lang, "runtime_patch_sha256": hashlib.sha256(patch.read_bytes()).hexdigest(),
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest()}
    output.with_suffix(".result.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print("asr.result=" + json.dumps(receipt))


if __name__ == "__main__":
    import fire

    fire.Fire({"prep": prep, "transcribe": transcribe})
