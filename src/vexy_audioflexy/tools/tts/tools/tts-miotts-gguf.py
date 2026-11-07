#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# this_file: src/vexy_audioflexy/tools/tts/tools/tts-miotts-gguf.py
"""BROKEN ENGINE — MioTTS GGUF. Ultra-compact 0.1B multilingual TTS via MioTTS-llama.cpp; preset voices for English and Japanese (en_female, en_male, jp_female, jp_male); backend: gguf (requires pre-built miotts binary); no voice cloning; mode: auto; marked broken/ignored in this repo."""

REPO = "Aratako/MioTTS-GGUF"

_REASON = "ERROR: tts-miotts-gguf is marked BROKEN and ignored in this repo. Do not use."


def prep(force: bool = False):
    """Download + cache all model weights/assets WITHOUT synthesizing. Idempotent."""
    import sys
    print(_REASON, file=sys.stderr)
    raise SystemExit(1)


def say(text, ref=None, output="output.wav", play=False, force=False):
    """Synthesize speech (BROKEN — raises SystemExit(1))."""
    import sys
    print(_REASON, file=sys.stderr)
    raise SystemExit(1)


def info():
    """Print one-line JSON: name, repo, backend, languages, sample_rate_hz, modes, capabilities."""
    import json
    print(json.dumps({
        "name": "MioTTS GGUF",
        "repo": REPO,
        "backend": "gguf",
        "languages": ["en", "ja"],
        "sample_rate_hz": None,
        "modes": ["auto"],
        "capabilities": ["preset_voices"],
    }))


if __name__ == "__main__":
    import fire
    fire.Fire({"prep": prep, "say": say, "info": info})
