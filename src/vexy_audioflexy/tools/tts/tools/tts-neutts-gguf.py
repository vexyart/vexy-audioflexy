#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# this_file: src/vexy_audioflexy/tools/tts/tools/tts-neutts-gguf.py
"""BROKEN ENGINE — NeuTTS GGUF (neuphonic/neutts-nano-q8-gguf). Marked broken/ignored in this repo. Do not use."""

REPO = "neuphonic/neutts-nano-q8-gguf"

_REASON = "ERROR: tts-neutts-gguf is marked BROKEN and ignored in this repo. Do not use."


def prep(force: bool = False):
    """Download + cache model weights — BROKEN, raises SystemExit(1)."""
    import sys
    print(_REASON, file=sys.stderr)
    raise SystemExit(1)


def say(text, ref=None, output="output.wav", play=False, force=False):
    """Synthesize speech — BROKEN, raises SystemExit(1)."""
    import sys
    print(_REASON, file=sys.stderr)
    raise SystemExit(1)


def info():
    """Print one-line JSON with model metadata."""
    import json
    print(json.dumps({
        "name": "NeuTTS Nano GGUF",
        "repo": REPO,
        "backend": "gguf",
        "languages": ["en"],
        "sample_rate_hz": 24000,
        "modes": ["ivc"],
        "capabilities": ["voice_cloning"],
        "status": "broken",
    }))


if __name__ == "__main__":
    import fire
    fire.Fire({"prep": prep, "say": say, "info": info})
