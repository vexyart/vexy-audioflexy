#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# this_file: src/vexy_audioflexy/tools/tts/tools/tts-dia-gguf.py
"""BROKEN ENGINE — Dia 1.6B GGUF (mmwillet2/Dia_GGUF). Requires TTS.cpp binary; not supported in this repo. Backend: gguf/TTS.cpp, languages: en, modes: auto/ivc, capabilities: dialogue_tags, voice_cloning."""

import json
import sys

REPO = "mmwillet2/Dia_GGUF"

_REASON = "ERROR: tts-dia-gguf is marked BROKEN and ignored in this repo. Do not use."


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
        "name": "Dia 1.6B GGUF",
        "repo": REPO,
        "backend": "gguf",
        "languages": ["en"],
        "sample_rate_hz": 44100,
        "modes": ["auto", "ivc"],
        "capabilities": ["dialogue_tags", "voice_cloning"],
    }))


if __name__ == "__main__":
    import fire
    fire.Fire({"prep": prep, "say": say, "info": info})
