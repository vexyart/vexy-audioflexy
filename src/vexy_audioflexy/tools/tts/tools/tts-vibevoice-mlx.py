#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# this_file: src/vexy_audioflexy/tools/tts/tools/tts-vibevoice-mlx.py
"""BROKEN ENGINE — VibeVoice MLX (mlx-community/VibeVoice-1.5B). Backend: mlx; languages: en; modes: auto; capabilities: none. Marked broken/ignored in this repo. Do not use."""

REPO = "mlx-community/VibeVoice-1.5B"

_REASON = "ERROR: tts-vibevoice-mlx is marked BROKEN and ignored in this repo. Do not use."


def prep(force: bool = False):
    """Download + cache model weights WITHOUT synthesizing. Broken: raises SystemExit(1)."""
    import sys
    print(_REASON, file=sys.stderr)
    raise SystemExit(1)


def say(text, ref=None, output="output.wav", play=False, force=False):
    """Synthesise speech with VibeVoice MLX. Broken: raises SystemExit(1)."""
    import sys
    print(_REASON, file=sys.stderr)
    raise SystemExit(1)


def info():
    """Print one-line JSON: name, repo, backend, languages, sample_rate_hz, modes, capabilities."""
    import json
    print(json.dumps({
        "name": "VibeVoice MLX",
        "repo": REPO,
        "backend": "mlx",
        "languages": ["en"],
        "sample_rate_hz": 24000,
        "modes": ["auto"],
        "capabilities": [],
    }))


if __name__ == "__main__":
    import fire
    fire.Fire({"prep": prep, "say": say, "info": info})
