#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = ["fire"]
# ///
# this_file: src/vexy_audioflexy/tools/tts/tools/tts-ming-omni-bf16-mlx.py
"""BROKEN ENGINE — Ming-omni-tts 16.8B-A3B MoE TTS. Marked broken/ignored in this repo."""

REPO = "mlx-community/Ming-omni-tts-16.8B-A3B-bf16"

_REASON = "ERROR: tts-ming-omni-bf16-mlx is marked BROKEN and ignored in this repo. Do not use."


def prep(force: bool = False):
    """Download + cache model weights. Broken stub — always exits 1."""
    import sys
    print(_REASON, file=sys.stderr)
    raise SystemExit(1)


def say(text, ref=None, ref_text=None, output="output.wav", model=None, play=False, force=False):
    """Synthesize speech. Broken stub — always exits 1."""
    import sys
    print(_REASON, file=sys.stderr)
    raise SystemExit(1)


def info():
    """Print one-line JSON metadata for this engine."""
    import json
    print(json.dumps({
        "name": "ming-omni-bf16-mlx",
        "repo": REPO,
        "backend": "mlx",
        "languages": ["en", "zh"],
        "sample_rate_hz": 44100,
        "modes": ["ivc"],
        "capabilities": ["voice_cloning"],
    }))


if __name__ == "__main__":
    import fire
    fire.Fire({"prep": prep, "say": say, "info": info})
