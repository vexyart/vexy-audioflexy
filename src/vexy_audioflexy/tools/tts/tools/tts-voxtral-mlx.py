#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# this_file: src/vexy_audioflexy/tools/tts/tools/tts-voxtral-mlx.py
"""BROKEN ENGINE — Voxtral MLX (mlx-community/Voxtral-Mini-3B-2507-4bit). Mistral Voxtral audio model TTS runner via MLX; preset voice. Backend: mlx (4bit); languages: en; modes: auto; capabilities: none. Marked broken/ignored in this repo. Do not use."""

REPO = "mlx-community/Voxtral-Mini-3B-2507-4bit"

_REASON = "ERROR: tts-voxtral-mlx is marked BROKEN and ignored in this repo. Do not use."


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
        "name": "Voxtral MLX",
        "repo": REPO,
        "backend": "mlx",
        "languages": ["en"],
        "sample_rate_hz": None,
        "modes": ["auto"],
        "capabilities": [],
    }))


if __name__ == "__main__":
    import fire
    fire.Fire({"prep": prep, "say": say, "info": info})
