#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# this_file: src/vexy_audioflexy/tools/tts/tools/tts-magpie-mlx.py
"""BROKEN ENGINE — Magpie TTS Multilingual 357M (MLX 8-bit). Backend: MLX (Swift/C++ only); no Python bindings available. Repo: aufklarer/Magpie-TTS-Multilingual-357M-MLX-8bit. Languages: en, es, de, fr, it, vi, zh, hi. Sample rate: 22050 Hz. Mode: auto. No voice cloning. Preset voices: John, Sofia, Aria, Jason, Leo."""

import json

REPO = "aufklarer/Magpie-TTS-Multilingual-357M-MLX-8bit"
_REASON = "tts-magpie-mlx is marked BROKEN and ignored in this repo. The MLX 8-bit variant requires Swift/C++ only tooling with no Python bindings available. Do not use."


def prep(force: bool = False):
    """Download + cache all model weights/assets WITHOUT synthesizing. Idempotent."""
    import sys
    print(f"ERROR: {_REASON}", file=sys.stderr)
    raise SystemExit(1)


def say(text, ref=None, output="output.wav", play=False, force=False):
    """Synthesize speech. BROKEN — prints error and exits 1."""
    import sys
    print(f"ERROR: {_REASON}", file=sys.stderr)
    raise SystemExit(1)


def info():
    """Print one-line JSON: name, repo, backend, languages, sample_rate_hz, modes, capabilities."""
    print(json.dumps({
        "name": "Magpie TTS Multilingual 357M (MLX 8-bit)",
        "repo": REPO,
        "backend": "mlx",
        "languages": ["en", "es", "de", "fr", "it", "vi", "zh", "hi"],
        "sample_rate_hz": 22050,
        "modes": ["auto"],
        "capabilities": {
            "voice_cloning": False,
            "preset_voices": ["John", "Sofia", "Aria", "Jason", "Leo"],
            "quantization": "8bit",
            "status": "broken",
            "reason": "Swift/C++ only, no Python bindings"
        },
    }))


if __name__ == "__main__":
    import fire
    fire.Fire({"prep": prep, "say": say, "info": info})
