#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = ["fire", "huggingface-hub"]
# ///
# this_file: src/vexy_audioflexy/tools/tts/tools/tts-qwenttscpp.py
"""Qwen3-TTS through ServeurpersoCom/qwentts.cpp's native GGML CLI.

Target text goes to stdin; --ref-text takes a file, not transcript text.
QWENTTSCPP_ROOT selects a built checkout, default ../.external/qwentts.cpp.
"""
from __future__ import annotations

import array
import json
import os
import subprocess
import tempfile
import time
import wave
from pathlib import Path
from _paths import runtime_path

ROOT = Path(__file__).resolve().parents[1]
ENGINE_ROOT = Path(os.environ.get("QWENTTSCPP_ROOT", runtime_path('qwentts.cpp')))
REPO = "Serveurperso/Qwen3-TTS-GGUF"
REVISION = "b7ee2e8c7459c3bea99da23e3d178125a7d1713c"
PACKAGES = ("0.6b-base", "0.6b-customvoice", "1.7b-base", "1.7b-customvoice", "1.7b-voicedesign")
LANGUAGES = {"en": "English", "zh": "Chinese", "ja": "Japanese", "ko": "Korean",
             "de": "German", "fr": "French", "ru": "Russian", "pt": "Portuguese",
             "es": "Spanish", "it": "Italian", "auto": "auto"}


def resolve(package: str, quant: str) -> tuple[str, str, str]:
    """Resolve the talker, shared codec and mode before any network operation."""
    if package not in PACKAGES or quant.upper() not in {"Q8_0", "Q4_K_M", "BF16", "F32"}:
        raise SystemExit(f"Unknown package/quant: {package}/{quant}")
    size, mode = package.split("-", 1)
    return f"qwen-talker-{size}-{mode}-{quant.upper()}.gguf", f"qwen-tokenizer-12hz-{quant.upper()}.gguf", mode


def prep(package: str = "0.6b-customvoice", quant: str = "Q8_0") -> tuple[str, str]:
    """Fetch only the selected talker and codec through the shared HF cache."""
    from huggingface_hub import hf_hub_download

    talker, codec, _ = resolve(package, quant)
    paths = tuple(hf_hub_download(repo_id=REPO, filename=name, revision=REVISION) for name in (talker, codec))
    print(f"[prep] ready: {package}/{quant}")
    return paths


def command(talker: str, codec: str, output: str, lang: str, seed: int,
            mode: str, speaker: str, ref: str, ref_text_file: str, instruct: str,
            max_new: int) -> list[str]:
    """Use upstream mode-specific switches without shell interpolation."""
    argv = [str(ENGINE_ROOT / "build/qwen-tts"), "--model", talker, "--codec", codec,
            "--lang", LANGUAGES[lang], "--seed", str(seed), "--max-new", str(max_new), "-o", output]
    if mode == "customvoice":
        argv += ["--speaker", speaker]
    if ref:
        argv += ["--ref-wav", ref]
    if ref_text_file:
        argv += ["--ref-text", ref_text_file]
    if instruct:
        argv += ["--instruct", instruct]
    return argv


def validate_mode(mode: str, ref: str, ref_text: str, instruct: str, speaker: str) -> None:
    """Reject combinations the upstream ABI cannot represent."""
    if mode == "base" and (speaker or instruct):
        raise SystemExit("Base supports reference cloning; speaker and instruct require another mode")
    if mode != "base" and (ref or ref_text):
        raise SystemExit("Reference cloning requires a Base package")
    if ref_text and not ref:
        raise SystemExit("ref_text requires ref audio")
    if mode == "voicedesign" and (not instruct or speaker):
        raise SystemExit("VoiceDesign requires instruct and does not take speaker")


def say(text: str, output: str = "output.wav", package: str = "0.6b-customvoice",
        quant: str = "Q8_0", lang: str = "en", speaker: str = "", ref: str = "",
        ref_text: str = "", instruct: str = "", seed: int = 42, max_new: int = 512,
        timeout: int = 900, force: bool = False, verbose: bool = False) -> None:
    """Synthesize, validate 24 kHz PCM output, and report wall time and RTF."""
    _, _, mode = resolve(package, quant)
    validate_mode(mode, ref, ref_text, instruct, speaker)
    if lang not in LANGUAGES or not text.strip() or max_new <= 0 or timeout <= 0:
        raise SystemExit("Require nonempty text, supported language and positive limits")
    out = Path(output).resolve()
    reference = Path(ref).resolve() if ref else None
    if reference and (not reference.is_file() or reference == out):
        raise SystemExit("Reference must exist and differ from output")
    if out.exists() and not force:
        raise SystemExit(f"Output exists: {out}; use --force")
    if not (ENGINE_ROOT / "build/qwen-tts").is_file():
        raise SystemExit(f"Build qwentts.cpp at {ENGINE_ROOT} with cmake and GGML_METAL=ON")
    talker, codec = prep(package, quant)
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "speech.wav"
        transcript = Path(tmp) / "reference.txt"
        if ref_text:
            transcript.write_text(ref_text, encoding="utf-8")
        argv = command(talker, codec, str(wav), lang, seed, mode, speaker or "ryan",
                       str(reference) if reference else "", str(transcript) if ref_text else "", instruct, max_new)
        env = os.environ.copy()
        env.setdefault("GGML_BACKEND", "MTL0")
        start = time.monotonic()
        result = subprocess.run(argv, input=text + "\n", capture_output=True, text=True,
                                env=env, cwd=ENGINE_ROOT, timeout=timeout, check=False)
        wall = time.monotonic() - start
        if verbose or result.returncode:
            print(result.stdout + result.stderr)
        if result.returncode:
            raise SystemExit(result.returncode)
        with wave.open(str(wav), "rb") as audio:
            rate, channels, width = audio.getframerate(), audio.getnchannels(), audio.getsampwidth()
            samples = array.array("h", audio.readframes(audio.getnframes()))
        if (rate, channels, width) != (24000, 1, 2) or not samples or not any(samples):
            raise SystemExit("qwentts.cpp returned invalid or silent audio")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(wav.read_bytes())
    duration = len(samples) / rate
    print("tts.result=" + json.dumps({"output": str(out), "package": package, "quant": quant,
          "wall_s": wall, "audio_duration_s": duration, "rtf": wall / duration}))


def info() -> None:
    """Expose available modes and engine provenance to callers."""
    print(json.dumps({"repo": REPO, "engine": str(ENGINE_ROOT), "stack": "qwenttscpp",
                      "packages": PACKAGES, "languages": list(LANGUAGES)}))


if __name__ == "__main__":
    import fire

    fire.Fire({"prep": prep, "say": say, "info": info, "packages": lambda: PACKAGES})
