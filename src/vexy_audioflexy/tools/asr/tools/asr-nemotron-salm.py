#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = ["fire>=0.7,<0.8", "gguf==0.19.0"]
# ///
# this_file: src/vexy_audioflexy/tools/asr/tools/asr-nemotron-salm.py
"""Exact Nemotron 3.5 30B AD-Q5_0_XL SALM ASR with its separate F16 encoder."""

import hashlib
import json
import subprocess
import tempfile
import time
import wave
from pathlib import Path

COMMIT = "536b17a41bef62d2a67a40ed3df61df26a4389fe"
REPO = "AtomicChat/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-GGUF"
REVISION = "dbd06c5cad7a2f33c8e06be2fb3ef814dbd9aa59"
MODELS = (
    {"size": 29257398432, "sha256": "a1eef76726d252509032a84339f8300584d7590014a63139d2d008ff981e2f4f",
     "architecture": "nemotron_h_moe", "name": "NVIDIA-Nemotron-3.5-Transcribe-30B-A4B"},
    {"size": 1670410080, "sha256": "4221029c8a565c4a80c507ffa760c716e595acf663f9f4273ce69910050e7ba2",
     "architecture": "pee", "name": "NVIDIA-Nemotron-3.5-Transcribe-30B-A4B Encoder F16"},
)
# shortcut: 128 output tokens suit the qualified short-utterance run, expose larger budgets after long-file acceptance.
# The qualified 48 GB host requires one bounded SALM slot; native defaults reserve much more memory.
SALM_OPTIONS = {"max_sequences": 1, "n_ctx_per_sequence": 2048, "max_context": 2048,
                "max_new_tokens": 128, "n_batch": 512, "encoder_max_batch": 1, "mtp_draft": 0}


def validate_models(model_path: str, encoder_path: str) -> tuple[Path, Path]:
    """Verify exact published bytes and GGUF families before passing paths to native code."""
    from gguf import GGUFReader

    paths = tuple(Path(p).expanduser().resolve() for p in (model_path, encoder_path))
    for path, expected in zip(paths, MODELS, strict=True):
        if not path.is_file() or path.stat().st_size != expected["size"]:
            raise ValueError("Nemotron decoder or encoder is missing or has the wrong byte size")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected["sha256"]:
            raise ValueError("Nemotron decoder or encoder SHA256 differs from the qualified artifact")
        reader = GGUFReader(path)
        for key, value in (("general.architecture", expected["architecture"]), ("general.name", expected["name"])):
            field = reader.get_field(key)
            if field is None or field.contents() != value:
                raise ValueError(f"Nemotron model metadata mismatch: {key}")
    return paths


def runtime(engine_root: str, timeout: int) -> Path:
    """Require the pinned SALM-enabled source build rather than a plain ASR binary."""
    engine = Path(engine_root).expanduser().resolve()
    binary = engine / "build/metal-asr/bin/nemo-speech"
    cache = engine / "build/metal-asr/CMakeCache.txt"
    if not binary.is_file() or not cache.is_file() or "NEMO_SPEECH_BUILD_SPEECHLM:BOOL=ON" not in cache.read_text():
        raise ValueError("Build the pinned metal-asr runtime with NEMO_SPEECH_BUILD_SPEECHLM=ON")
    actual = subprocess.check_output(["git", "-C", str(engine), "rev-parse", "HEAD"],
                                     text=True, timeout=min(timeout, 30)).strip()
    if actual != COMMIT:
        raise ValueError("Nemotron SALM runtime commit differs from the qualified build")
    return binary


def command(binary: Path, audio: Path, model: Path, encoder: Path, output: Path, lang: str, threads: int) -> list[str]:
    """Pass encoder explicitly; no sibling weight copy or guessed model replacement."""
    argv = [str(binary), "transcribe", str(audio), "--model", str(model), "--device", "metal",
            "--language", lang, "--format", "text", "--output", str(output), "--no-warmup", "--no-batching",
            "--asr.salm.encoder_path", str(encoder), "--asr.salm.batched=false", "--asr.backend.threads", str(threads)]
    for key, value in SALM_OPTIONS.items():
        argv.extend(["--asr.salm." + key, str(value)])
    return argv


def transcribe(audio: str, model_path: str, encoder_path: str, engine_root: str,
               text_out: str = "", lang: str = "en", timeout: int = 900,
               threads: int = 4, force: bool = False, verbose: bool = False) -> None:
    """Transcribe through the qualified bounded Metal SALM runtime; no downloads occur."""
    source = Path(audio).expanduser().resolve()
    output = Path(text_out).expanduser().resolve() if text_out else source.with_suffix(".txt")
    if not source.is_file() or source == output or timeout <= 0 or threads <= 0:
        raise ValueError("Require audio, a distinct transcript path and positive timeout/threads")
    if output.exists() and not force:
        raise FileExistsError("Transcript exists; use --force")
    binary = runtime(engine_root, timeout)
    model, encoder = validate_models(model_path, encoder_path)
    with tempfile.TemporaryDirectory() as temporary:
        wav, transcript = Path(temporary) / "input.wav", Path(temporary) / "transcript.txt"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(source), "-ar", "16000",
                        "-ac", "1", "-c:a", "pcm_s16le", str(wav)], check=True, timeout=timeout)
        with wave.open(str(wav), "rb") as stream:
            duration = stream.getnframes() / stream.getframerate()
        if duration <= 0:
            raise ValueError("Input audio is empty")
        start = time.monotonic()
        result = subprocess.run(command(binary, wav, model, encoder, transcript, lang, threads),
                                capture_output=True, text=True, timeout=timeout, check=False)
        wall = time.monotonic() - start
        if verbose or result.returncode:
            print(result.stdout + result.stderr)
        if result.returncode:
            raise SystemExit(result.returncode)
        text = transcript.read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError("Nemotron SALM returned no recognized text")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    print("asr.result=" + json.dumps({"text": text, "text_out": str(output), "wall_s": wall,
          "audio_duration_s": duration, "rtf": wall / duration, "package": "nemotron-3.5-ad-q5-0-xl",
          "model_repo": REPO, "revision": REVISION, "runtime_commit": COMMIT}, ensure_ascii=False))


if __name__ == "__main__":
    import fire

    fire.Fire({"transcribe": transcribe})
