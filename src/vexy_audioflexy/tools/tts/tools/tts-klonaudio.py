#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# this_file: src/vexy_audioflexy/tools/tts/tools/tts-klonaudio.py
"""BROKEN ENGINE — KlonAudio (Roland-JAAI/klonaudio). 7B VibeVoice-based TTS with voice cloning for 23 European languages; fork of KugelAudio restoring dual-encoder architecture. Backend: python (fp32, ~18GB, trust_remote_code=True). Modes: auto, ivc. Marked broken/ignored in this repo."""

import sys

REPO = "Roland-JAAI/klonaudio"


def prep(force: bool = False):
    """Download + cache all model weights/assets WITHOUT synthesizing. Idempotent."""
    print("ERROR: tts-klonaudio is marked BROKEN and ignored in this repo. Do not use.", file=sys.stderr)
    raise SystemExit(1)


def say(text, ref=None, output="output.wav", play=False, force=False):
    """Synthesize speech. BROKEN — this engine is ignored in this repo."""
    print("ERROR: tts-klonaudio is marked BROKEN and ignored in this repo. Do not use.", file=sys.stderr)
    raise SystemExit(1)


def info():
    """Print one-line JSON with model metadata."""
    import json
    print(json.dumps({
        "name": "KlonAudio",
        "repo": REPO,
        "backend": "python",
        "languages": ["en", "de", "es", "fr", "it", "pt", "nl", "pl", "ru", "uk", "cs", "ro", "hu", "sv", "da", "fi", "no", "el", "bg", "sk", "hr", "sr", "tr"],
        "sample_rate_hz": None,
        "modes": ["auto", "ivc"],
        "capabilities": ["voice_cloning"],
        "status": "broken",
    }))


if __name__ == "__main__":
    import fire
    fire.Fire({"prep": prep, "say": say, "info": info})
