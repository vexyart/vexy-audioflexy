#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# this_file: src/vexy_audioflexy/tools/tts/tools/tts-parler-gguf.py
"""BROKEN ENGINE — Parler TTS GGUF (mmwillet2/Parler_TTS_GGUF). Requires TTS.cpp binary; not supported in this repo. Backend: gguf/TTS.cpp; languages: en; modes: desi (text-described voice design); capabilities: voice_design. Marked broken/ignored in this repo. Do not use."""

REPO = "mmwillet2/Parler_TTS_GGUF"

_REASON = "ERROR: tts-parler-gguf is marked BROKEN and ignored in this repo. Do not use."


def prep(force: bool = False):
    """Download + cache model weights WITHOUT synthesizing. Broken: raises SystemExit(1)."""
    import sys
    print(_REASON, file=sys.stderr)
    raise SystemExit(1)


def say(text, ref=None, output="output.wav", play=False, force=False):
    """Synthesise speech with Parler GGUF. Broken: raises SystemExit(1)."""
    import sys
    print(_REASON, file=sys.stderr)
    raise SystemExit(1)


def info():
    """Print one-line JSON: name, repo, backend, languages, sample_rate_hz, modes, capabilities."""
    import json
    print(json.dumps({
        "name": "Parler TTS GGUF",
        "repo": REPO,
        "backend": "gguf",
        "languages": ["en"],
        "sample_rate_hz": 44100,
        "modes": ["desi"],
        "capabilities": ["voice_design"],
    }))


if __name__ == "__main__":
    import fire
    fire.Fire({"prep": prep, "say": say, "info": info})
