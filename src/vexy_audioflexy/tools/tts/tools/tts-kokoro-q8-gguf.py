#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# this_file: src/vexy_audioflexy/tools/tts/tools/tts-kokoro-q8-gguf.py
"""BROKEN ENGINE — Kokoro-82M GGUF q8 (mmwillet2/Kokoro_GGUF). Requires TTS.cpp binary; not supported in this repo. Backend: gguf/TTS.cpp, languages: en/ja/ko/zh/fr, modes: auto/ivc, capabilities: voice_cloning."""

import json
import sys

REPO = "mmwillet2/Kokoro_GGUF"

_REASON = "ERROR: tts-kokoro-q8-gguf is marked BROKEN and ignored in this repo. Do not use."


def prep(force: bool = False):
    """Download + cache model weights WITHOUT synthesizing. Broken: raises SystemExit(1)."""
    print(_REASON, file=sys.stderr)
    raise SystemExit(1)


def say(
    text,
    ref=None,
    output="output.wav",
    play=False,
    force=False,
):
    """Synthesise speech. Broken: raises SystemExit(1)."""
    print(_REASON, file=sys.stderr)
    raise SystemExit(1)


def info():
    """Print one-line JSON with model metadata."""
    print(json.dumps({
        "name": "Kokoro 82M GGUF q8",
        "repo": REPO,
        "backend": "gguf",
        "languages": ["en", "ja", "ko", "zh", "fr"],
        "sample_rate_hz": 24000,
        "modes": ["auto", "ivc"],
        "capabilities": ["voice_cloning"],
    }))


if __name__ == "__main__":
    import fire
    fire.Fire({"prep": prep, "say": say, "info": info})
