#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# this_file: src/vexy_audioflexy/tools/tts/tools/tts-mlxtada.py
"""BROKEN ENGINE — MLX-TADA 3B. TADA text-aligned dialogue audio model via MLX; voice cloning (ivc); English; marked broken/ignored in this repo."""

REPO = "see tools/tts-mlxtada.py"

_REASON = "ERROR: tts-mlxtada is marked BROKEN and ignored in this repo. Do not use."


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
    """Print one-line JSON metadata for this engine."""
    import json
    print(json.dumps({
        "name": "MLX-TADA 3B",
        "repo": REPO,
        "backend": "mlx",
        "languages": ["en"],
        "sample_rate_hz": None,
        "modes": ["ivc"],
        "capabilities": ["voice_cloning"],
        "status": "broken",
    }))


if __name__ == "__main__":
    import fire
    fire.Fire({"prep": prep, "say": say, "info": info})
