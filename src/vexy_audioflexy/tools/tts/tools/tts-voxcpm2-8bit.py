#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = ["fire"]
# ///
# this_file: src/vexy_audioflexy/tools/tts/tools/tts-voxcpm2-8bit.py
"""BROKEN ENGINE — voxcpm2-8bit rejected based on subjective quality. Marked broken/ignored."""

_REASON = "ERROR: tts-voxcpm2-8bit is marked BROKEN and ignored in this repo. Do not use."

def prep(force: bool = False):
    import sys
    print(_REASON, file=sys.stderr)
    raise SystemExit(1)

def say(text, **kwargs):
    import sys
    print(_REASON, file=sys.stderr)
    raise SystemExit(1)

def info():
    import json
    print(json.dumps({"name": "voxcpm2-8bit", "repo": "rejected", "backend": "stub", "languages": ["en"], "sample_rate_hz": 22050, "modes": ["auto"], "capabilities": []}))

if __name__ == "__main__":
    import fire
    fire.Fire({"prep": prep, "say": say, "info": info})
