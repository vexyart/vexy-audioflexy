#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# this_file: src/vexy_audioflexy/tools/tts/tools/tts-dia.py
"""BROKEN ENGINE — Dia 1.6B (nari-labs/Dia-1.6B). Marked broken/ignored in this repo. Do not use."""

REPO = "nari-labs/Dia-1.6B-0626"

_REASON = "ERROR: tts-dia is marked BROKEN and ignored in this repo. Do not use."


def prep(force: bool = False):
    """Download + cache model weights. BROKEN — raises SystemExit(1)."""
    import sys
    print(_REASON, file=sys.stderr)
    raise SystemExit(1)


def say(text, ref=None, output="output.wav", play=False, force=False):
    """Synthesise speech with Dia. BROKEN — raises SystemExit(1)."""
    import sys
    print(_REASON, file=sys.stderr)
    raise SystemExit(1)


def info():
    """Print one-line JSON metadata for this engine."""
    import json
    print(json.dumps({
        "name": "Dia 1.6B",
        "repo": REPO,
        "backend": "pytorch",
        "languages": ["en"],
        "sample_rate_hz": 44100,
        "modes": ["auto"],
        "capabilities": ["dialogue_tags"],
    }))


if __name__ == "__main__":
    import fire
    fire.Fire({"prep": prep, "say": say, "info": info})
