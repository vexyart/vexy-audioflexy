#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = ["fire"]
# ///
# this_file: src/vexy_audioflexy/tools/tts/tools/tts-openaudio-gguf.py
"""BROKEN ENGINE — OpenAudio GGUF (calcuis/openaudio-gguf). Marked broken/ignored in this repo.

Backend: fishaudio (GGUF dequantized via gguf-connector). Repo: calcuis/openaudio-gguf.
Languages: en, zh, ja, ko. Sample rate: unknown. Mode: ivc. Capabilities: voice_cloning.
"""

import sys

REPO = "calcuis/openaudio-gguf"
_REASON = "ERROR: tts-openaudio-gguf is marked BROKEN and ignored in this repo. Do not use."


def prep(force: bool = False):
    """Broken stub — prints error and exits."""
    print(_REASON, file=sys.stderr)
    raise SystemExit(1)


def say(text: str, output: str = "output.wav", ref: str = "", play: bool = False, force: bool = False):
    """Broken stub — prints error and exits."""
    print(_REASON, file=sys.stderr)
    raise SystemExit(1)


def info():
    """Print one-line JSON: name, repo, backend, languages, sample_rate_hz, modes, capabilities."""
    import json
    print(json.dumps({
        "name": "OpenAudio s1-mini GGUF",
        "repo": REPO,
        "backend": "fishaudio",
        "languages": ["en", "zh", "ja", "ko"],
        "sample_rate_hz": None,
        "modes": ["ivc"],
        "capabilities": ["voice_cloning"],
        "broken": True,
    }))


if __name__ == "__main__":
    import fire
    fire.Fire({"prep": prep, "say": say, "info": info})
