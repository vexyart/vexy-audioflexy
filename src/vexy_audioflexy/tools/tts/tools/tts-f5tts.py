#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# this_file: src/vexy_audioflexy/tools/tts/tools/tts-f5tts.py
"""BROKEN ENGINE — F5-TTS. Flow-matching IVC TTS (E2/F5); torch 2.5.1 MPS; en/zh; marked broken/ignored in this repo."""

REPO = "SWivid/F5-TTS"

_REASON = "ERROR: tts-f5tts is marked BROKEN and ignored in this repo. Do not use."


def prep(force: bool = False):
    """Download + cache model weights. Broken stub — always exits 1."""
    import sys
    print(_REASON, file=sys.stderr)
    raise SystemExit(1)


def say(text, ref=None, output="output.wav", play=False, force=False):
    """Synthesize speech with F5-TTS. Broken stub — always exits 1."""
    import sys
    print(_REASON, file=sys.stderr)
    raise SystemExit(1)


def info():
    """Print one-line JSON metadata for this engine."""
    import json
    print(json.dumps({
        "name": "F5-TTS",
        "repo": REPO,
        "backend": "torch",
        "languages": ["en", "zh"],
        "sample_rate_hz": 24000,
        "modes": ["ivc"],
        "capabilities": ["voice_cloning"],
    }))


if __name__ == "__main__":
    import fire
    fire.Fire({"prep": prep, "say": say, "info": info})
