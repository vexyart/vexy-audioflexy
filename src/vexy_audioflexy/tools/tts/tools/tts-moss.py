#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# this_file: src/vexy_audioflexy/tools/tts/tools/tts-moss.py
"""BROKEN ENGINE — MOSS TTS. OpenMOSS Team bilingual en/zh TTS; no working runner in this repo."""

REPO = "OpenMOSS-Team/MOSS-TTS"

_REASON = "ERROR: tts-moss is marked BROKEN and ignored in this repo. Do not use."


def prep(force: bool = False):
    """Download + cache model weights. BROKEN — raises SystemExit(1)."""
    import sys
    print(_REASON, file=sys.stderr)
    raise SystemExit(1)


def say(text, ref=None, output="output.wav", play=False, force=False):
    """Synthesize speech. BROKEN — raises SystemExit(1)."""
    import sys
    print(_REASON, file=sys.stderr)
    raise SystemExit(1)


def info():
    """Print one-line JSON with model metadata."""
    import json
    print(json.dumps({
        "name": "MOSS TTS",
        "repo": REPO,
        "backend": "unknown",
        "languages": ["en", "zh"],
        "sample_rate_hz": None,
        "modes": [],
        "capabilities": [],
        "broken": True,
    }))


if __name__ == "__main__":
    import fire
    fire.Fire({"prep": prep, "say": say, "info": info})
