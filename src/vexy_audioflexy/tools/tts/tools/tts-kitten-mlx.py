#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# this_file: src/vexy_audioflexy/tools/tts/tools/tts-kitten-mlx.py
"""BROKEN ENGINE — Kitten-TTS MLX. KittenML ultra-compact TTS (nano 14.6M / mini / micro) via mlx-audio; preset voices only, no voice cloning; English; auto mode; marked broken/ignored in this repo."""

import sys

REPO = "see tools/tts-kitten-mlx.py"  # repo unknown; engine is broken


def prep(force: bool = False):
    """Download + cache all model weights/assets WITHOUT synthesizing. Idempotent."""
    print(
        "ERROR: tts-kitten-mlx is marked BROKEN and ignored in this repo. Do not use.",
        file=sys.stderr,
    )
    raise SystemExit(1)


def say(text, ref=None, output="output.wav", play=False, force=False):
    """Synthesize speech (broken — this engine is ignored in this repo)."""
    print(
        "ERROR: tts-kitten-mlx is marked BROKEN and ignored in this repo. Do not use.",
        file=sys.stderr,
    )
    raise SystemExit(1)


def info():
    """Print one-line JSON: name, repo, backend, languages, sample_rate_hz, modes, capabilities."""
    import json

    print(
        json.dumps(
            {
                "name": "Kitten TTS",
                "repo": REPO,
                "backend": "mlx",
                "languages": ["en"],
                "sample_rate_hz": None,
                "modes": ["auto"],
                "capabilities": {
                    "voice_cloning": False,
                    "preset_voices": True,
                    "broken": True,
                },
            }
        )
    )


if __name__ == "__main__":
    import fire

    fire.Fire({"prep": prep, "say": say, "info": info})
