#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# this_file: src/vexy_audioflexy/tools/tts/tools/tts-dia-mlx.py
"""BROKEN ENGINE — Dia 1.6B MLX. MLX backend (mlx-community/Dia-1.6B-6bit), English, auto/ivc modes, dialogue [S1]/[S2] tags. Marked broken/ignored in this repo."""

import json
import sys

REPO = "mlx-community/Dia-1.6B-6bit"
_REASON = "tts-dia-mlx is marked BROKEN and ignored in this repo. Do not use."


def prep(force: bool = False):
    """Download + cache model weights WITHOUT synthesizing. Idempotent. (BROKEN — raises SystemExit)"""
    print(f"ERROR: {_REASON}", file=sys.stderr)
    raise SystemExit(1)


def say(
    text,
    ref=None,
    output="output.wav",
    play: bool = False,
    force: bool = False,
):
    """Synthesise speech with Dia 1.6B MLX. (BROKEN — raises SystemExit)"""
    print(f"ERROR: {_REASON}", file=sys.stderr)
    raise SystemExit(1)


def info():
    """Print one-line JSON with model metadata."""
    print(json.dumps({
        "name": "Dia 1.6B MLX",
        "repo": REPO,
        "backend": "mlx",
        "languages": ["en"],
        "sample_rate_hz": 44100,
        "modes": ["auto", "ivc"],
        "capabilities": ["dialogue_tags", "voice_cloning"],
    }))


if __name__ == "__main__":
    import fire
    fire.Fire({"prep": prep, "say": say, "info": info})
