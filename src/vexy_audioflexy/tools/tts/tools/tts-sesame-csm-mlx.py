#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# this_file: src/vexy_audioflexy/tools/tts/tools/tts-sesame-csm-mlx.py
"""BROKEN ENGINE — Sesame CSM 1B MLX (mlx-community/CSM-1B-4bit). Sesame's Character Speech Model; natural conversational voice cloning from short reference audio. Backend: mlx (4bit quantization); languages: en; modes: auto, ivc; capabilities: voice_cloning. Marked broken/ignored in this repo. Do not use."""

REPO = "mlx-community/CSM-1B-4bit"

_REASON = "ERROR: tts-sesame-csm-mlx is marked BROKEN and ignored in this repo. Do not use."


def prep(force: bool = False):
    """Download + cache model weights WITHOUT synthesizing. Broken: raises SystemExit(1)."""
    import sys
    print(_REASON, file=sys.stderr)
    raise SystemExit(1)


def say(text, ref=None, output="output.wav", play=False, force=False):
    """Synthesise speech with Sesame CSM MLX. Broken: raises SystemExit(1)."""
    import sys
    print(_REASON, file=sys.stderr)
    raise SystemExit(1)


def info():
    """Print one-line JSON: name, repo, backend, languages, sample_rate_hz, modes, capabilities."""
    import json
    print(json.dumps({
        "name": "CSM 1B MLX",
        "repo": REPO,
        "backend": "mlx",
        "languages": ["en"],
        "sample_rate_hz": 24000,
        "modes": ["auto", "ivc"],
        "capabilities": ["voice_cloning"],
    }))


if __name__ == "__main__":
    import fire
    fire.Fire({"prep": prep, "say": say, "info": info})
