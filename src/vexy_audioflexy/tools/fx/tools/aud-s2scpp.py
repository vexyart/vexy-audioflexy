#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = ["platformdirs>=4", "fire", "huggingface-hub"]
# ///
# this_file: src/vexy_audioflexy/tools/fx/tools/aud-s2scpp.py
"""Prepare, serve and benchmark the native s2s.cpp realtime voice pipeline."""
from __future__ import annotations


import os
from pathlib import Path
import runpy
model_path = runpy.run_path(str(Path(__file__).with_name("_audioflexy_paths.py")))["model_path"]
from platformdirs import user_cache_path, user_data_path
RUNTIME_ROOT = Path(os.environ.get("VAFX_AUDIO_RUNTIME_ROOT", user_data_path("vexy-audioflexy") / "audio" )).expanduser()
CACHE_ROOT = Path(os.environ.get("VAFX_AUDIO_CACHE_ROOT", user_cache_path("vexy-audioflexy") / "audio")).expanduser()
import array
import hashlib
import json
import math
import os
import re
import socket
import subprocess
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENGINE = Path(os.environ.get("S2SCPP_ROOT", RUNTIME_ROOT / "engines/s2s.cpp"))
CATALOG = json.loads((ROOT / "catalogs/s2scpp.json").read_text())
MODELS = model_path("s2s-models")


def prep(package: str = "s2scpp-loopback-q8") -> str:
    """Pin every stage's original conversion and share existing cached weights."""
    from huggingface_hub import hf_hub_download

    if package != CATALOG["package"]:
        raise SystemExit(f"Unknown pipeline package: {package}")
    MODELS.mkdir(parents=True, exist_ok=True)
    for model in CATALOG["models"]:
        cached = Path(hf_hub_download(model["repo"], model["filename"], revision=model["revision"])).resolve()
        target = MODELS / model["filename"]
        if (target.exists() or target.is_symlink()) and target.resolve() != cached:
            raise SystemExit(f"Existing model link points to another revision: {target}")
        if not target.exists():
            target.symlink_to(cached)
    return str(MODELS)


def prepare_capture_client() -> None:
    """Apply the maintained receipt-only client patch and rebuild its binary."""
    source = ENGINE / "tests/test-server.cpp"
    if "--audio-f32-out" not in source.read_text():
        patch = ROOT / "patches/s2scpp-client-audio.patch"
        subprocess.run(["git", "apply", "--check", str(patch)], cwd=ENGINE, check=True)
        subprocess.run(["git", "apply", str(patch)], cwd=ENGINE, check=True)
    subprocess.run(["cmake", "--build", str(ENGINE / "build"), "--target", "test-server", "-j4"],
                   check=True, capture_output=True, text=True, timeout=300)


def capture_wav(raw: Path, output: Path) -> tuple[float, str]:
    """Retain the actual received 24 kHz speech for independent ASR validation."""
    samples = array.array("f")
    samples.frombytes(raw.read_bytes())
    if not samples or not any(samples) or not all(math.isfinite(s) for s in samples):
        raise SystemExit("Received PCM is empty, silent or nonfinite")
    pcm = array.array("h", (max(-32768, min(32767, round(s * 32768))) for s in samples))
    with wave.open(str(output), "wb") as audio:
        audio.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
        audio.writeframes(pcm.tobytes())
    return len(samples) / 24000, hashlib.sha256(output.read_bytes()).hexdigest()


def server_command(models: str, port: int) -> list[str]:
    """Keep the local server on loopback; endpoint selection remains explicit."""
    if not 1 <= port <= 65535:
        raise SystemExit("port must be between 1 and 65535")
    return [str(ENGINE / "build/s2s-server"), "--models", models, "--voices", str(ENGINE / "voices"),
            "--host", "127.0.0.1", "--port", str(port)]


def serve(port: int = 18089, backend: str = "MTL0") -> None:
    """Run the native WebSocket server in the foreground after explicit prep."""
    env = os.environ.copy()
    env["GGML_BACKEND"] = backend
    subprocess.run(server_command(prep(), port), env=env, cwd=ENGINE, check=True)


def wait_ready(process: subprocess.Popen, port: int, timeout: int) -> None:
    """Bound startup and surface a failed server instead of waiting indefinitely."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise SystemExit("s2s-server exited during startup; inspect server.log")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(0.2)
    raise SystemExit("s2s-server startup deadline exceeded; inspect server.log")


def benchmark(output: str, audio: str = "", port: int = 18089, backend: str = "MTL0",
              timeout: int = 180, force: bool = False) -> None:
    """Drive real ASR-to-TTS loopback; no language-model endpoint or mock inference.

    The upstream client streams the first 5.9 seconds of its real speech fixture
    followed by 30 seconds of silence. Preserve native events and audio-duration
    receipts; this measures pipeline latency, not voice similarity.
    """
    out = Path(output).resolve()
    source = Path(audio).resolve() if audio else ENGINE / "qwentts.cpp/examples/freeman.wav"
    if not source.is_file() or timeout <= 0 or (out.exists() and not force):
        raise SystemExit("Require existing audio, positive timeout and a fresh output directory (--force to replace receipts)")
    if source == out or source in {out / name for name in ("server.log", "client.log", "report.json", "received.wav", "received.f32")}:
        raise SystemExit("Output receipts must not overwrite input audio")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", port))
    prepare_capture_client()
    models = prep()
    env = os.environ.copy()
    env["GGML_BACKEND"] = backend
    out.mkdir(parents=True, exist_ok=True)
    with (out / "server.log").open("w") as log:
        process = subprocess.Popen(server_command(models, port), env=env, cwd=ENGINE,
                                   stdout=log, stderr=subprocess.STDOUT)
        start = time.monotonic()
        try:
            wait_ready(process, port, timeout)
            startup = time.monotonic() - start
            command = [str(ENGINE / "build/test-server"), f"ws://127.0.0.1:{port}/v1/realtime",
                       str(source), "--mode", "loopback", "--audio-f32-out", str(out / "received.f32"),
                       "say:0-5.9", "pause:30"]
            result = subprocess.run(command, env=env, cwd=ENGINE, capture_output=True,
                                    text=True, timeout=timeout, check=False)
            log_text = result.stdout + result.stderr
            (out / "client.log").write_text(log_text)
            receipt = re.search(r"(\d+) events, ([\d.]+)s of audio received over ([\d.]+)s", log_text)
            first_audio = re.search(r"\[Event\]\s+([\d.]+)s\s+response.output_audio.delta", log_text)
            transcripts = re.findall(r'conversation.item.input_audio_transcription.completed[^\n]*"([^"\n]+)"', log_text)
            if result.returncode or not receipt or float(receipt[2]) <= 0 or not first_audio or not transcripts:
                raise SystemExit("Native loopback did not produce transcript and audio; inspect client.log")
            lifecycle = re.findall(r"\[Event\]\s+[\d.]+s\s+(response.created|response.done|response.cancelled|error)\b", log_text)
            if "error" in lifecycle or lifecycle.count("response.created") != lifecycle.count("response.done") + lifecycle.count("response.cancelled") or not lifecycle.count("response.done"):
                raise SystemExit("Unbalanced or failed native response lifecycle; inspect client.log")
            duration, digest = capture_wav(out / "received.f32", out / "received.wav")
            report = {"backend": backend, "startup_s": startup, "events": int(receipt[1]),
                      "received_audio_s": float(receipt[2]), "client_wall_s": float(receipt[3]),
                      "first_audio_s": float(first_audio[1]), "input_s": 5.9, "transcripts": transcripts,
                      "first_audio_after_input_s": float(first_audio[1]) - 5.9,
                      "input": str(source), "mode": "native-loopback", "model_bundle": CATALOG,
                      "saved_audio_s": duration, "saved_audio_sha256": digest,
                      "client_capture_patch": "patches/s2scpp-client-audio.patch"}
            (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report))
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == "__main__":
    import fire

    fire.Fire({"prep": prep, "serve": serve, "benchmark": benchmark})
