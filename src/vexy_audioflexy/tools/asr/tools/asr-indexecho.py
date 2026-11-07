#!/usr/bin/env -S uv run
# /// script
# dependencies = ["fire", "huggingface-hub", "soundfile", "platformdirs"]
# ///
# this_file: src/vexy_audioflexy/tools/asr/tools/asr-indexecho.py
"""Exact Index-Echo Chinese source transcript and separately retained translations."""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

from _audioflexy_paths import runtime_root

ROOT = Path(__file__).resolve().parents[1]
ENGINE = Path(os.environ.get("AUDIOCPP_ISSUE201_ROOT", runtime_root("audio.cpp"))).expanduser()
BINARY = ENGINE / "build/macos-metal-release/bin/audiocpp_cli"
REPO = "audio-cpp/Index-Echo-S2TT-GGUF"
REVISION = "08c2fc8e3fc87da1d848c5c75aeba1b761d2624e"
FILENAME = "index-echo-s2tt-2b-q8_0.gguf"
COMMIT = "2892ed3e94b6ccbdacc111fa2896aaae0a5dc8fc"


def parse_cues(text: str, duration: float) -> list[dict]:
    """Read the native three-line format, rejecting incomplete or reversed cues."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines or len(lines) % 3:
        raise SystemExit("Index-Echo returned incomplete three-line subtitle cues")
    cues = []
    pattern = re.compile(r"\[(\d+):(\d+(?:\.\d+)?)-(\d+):(\d+(?:\.\d+)?)\]")
    for index in range(0, len(lines), 3):
        match = pattern.fullmatch(lines[index])
        if match is None:
            raise SystemExit("Index-Echo subtitle timestamp is malformed")
        start = int(match[1]) * 60 + float(match[2])
        end = int(match[3]) * 60 + float(match[4])
        if not 0 <= start <= end <= duration + .25:
            raise SystemExit("Index-Echo cue is reversed or outside audio duration")
        cues.append({"start": start, "end": end, "source": lines[index + 1], "translation": lines[index + 2]})
    return cues


def prep(repo: str = REPO) -> str:
    """Pin the requested native Q8 export and retain a real .gguf filename."""
    from huggingface_hub import hf_hub_download

    if repo != REPO:
        raise SystemExit(f"Require exact repository {REPO}")
    source = Path(hf_hub_download(repo, FILENAME, revision=REVISION,
        cache_dir=None)).resolve()
    print(f"[prep] ready: {repo} -> {source}")
    return str(source)


def transcribe(audio: str, text_out: str = "", repo: str = REPO, lang: str = "zh",
               target: str = "en", force: bool = False, timeout: int = 900) -> None:
    """Score only Chinese source; retain translated subtitles and native timing."""
    import soundfile as sf

    if lang != "zh" or target not in ["en", "es", "ja"]:
        raise SystemExit("Index-Echo input is zh; translation target must be en/es/ja")
    source = Path(audio).resolve()
    output = Path(text_out).resolve() if text_out else source.with_suffix(".txt")
    if not source.is_file() or output == source or timeout <= 0:
        raise SystemExit("Require input audio, distinct output and positive timeout")
    if output.exists() and not force:
        raise SystemExit(f"Transcript exists: {output}; use --force")
    if not BINARY.is_file():
        raise SystemExit("Build the pinned audio.cpp Metal CLI")
    commit = subprocess.check_output(["git", "-C", str(ENGINE), "rev-parse", "HEAD"], text=True).strip()
    if commit != COMMIT:
        raise SystemExit("audio.cpp checkout differs from the verified runtime")
    model = prep(repo)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary:
        wav, subtitles = Path(temporary) / "input.wav", Path(temporary) / "subtitles.txt"
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(source), "-ar", "16000", "-ac", "1", str(wav)],
            check=True, timeout=timeout)
        duration = sf.info(wav).duration
        if duration <= 0:
            raise SystemExit("Empty input audio")
        command = [str(BINARY), "--task", "asr", "--family", "index_echo", "--model", model,
            "--backend", "metal", "--audio", str(wav), "--text-out", str(subtitles),
            "--request-option", f"target_language={target}", "--log"]
        start = time.monotonic()
        native = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
        wall = time.monotonic() - start
        output.with_suffix(".native.log").write_text(native.stdout + native.stderr)
        native.check_returncode()
        raw = subtitles.read_text()
    output.with_suffix(".subtitles.txt").write_text(raw)
    cues = parse_cues(raw, duration)
    text = "\n".join(cue["source"] for cue in cues)
    translation = "\n".join(cue["translation"] for cue in cues)
    output.write_text(text + "\n")
    output.with_suffix(".translation.txt").write_text(translation + "\n")
    receipt = {"text": text, "translation": translation, "cues": cues, "target": target,
        "wall_s": wall, "audio_duration_s": duration, "rtf": wall / duration,
        "repo": REPO, "weights_revision": REVISION, "runtime_commit": COMMIT,
        "audio_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "timing_boundary": "native CLI wall including model load"}
    output.with_suffix(".native.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
    print("asr.result=" + json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    import fire

    fire.Fire({"prep": prep, "transcribe": transcribe})
