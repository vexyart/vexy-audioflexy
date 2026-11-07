#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# this_file: src/vexy_audioflexy/tools/tts/tools/tts-scenema-audio.py
"""BROKEN ENGINE — Scenema Audio (ScenemaAI/scenema-audio). Zero-shot expressive voice cloning based on LTX-2.3 audio branch; supports long-form narration. Backend: python; languages: en/de/fr/es/it/pt/ja/zh/ko/ru/ar/hi/sw; modes: ivc; capabilities: voice_cloning. Requires Gemma 3 access, LTX-2 Community License, 16GB+ VRAM NVIDIA GPU. Marked broken/ignored in this repo. Do not use."""

REPO = "ScenemaAI/scenema-audio"

_REASON = "ERROR: tts-scenema-audio is marked BROKEN and ignored in this repo. Do not use."


def prep(force: bool = False):
    """Download + cache model weights WITHOUT synthesizing. Broken: raises SystemExit(1)."""
    import sys
    print(_REASON, file=sys.stderr)
    raise SystemExit(1)


def say(text, ref=None, output="output.wav", play=False, force=False):
    """Synthesise speech with Scenema Audio. Broken: raises SystemExit(1)."""
    import sys
    print(_REASON, file=sys.stderr)
    raise SystemExit(1)


def info():
    """Print one-line JSON: name, repo, backend, languages, sample_rate_hz, modes, capabilities."""
    import json
    print(json.dumps({
        "name": "Scenema Audio",
        "repo": REPO,
        "backend": "python",
        "languages": ["en", "de", "fr", "es", "it", "pt", "ja", "zh", "ko", "ru", "ar", "hi", "sw"],
        "sample_rate_hz": None,
        "modes": ["ivc"],
        "capabilities": ["voice_cloning"],
    }))


if __name__ == "__main__":
    import fire
    fire.Fire({"prep": prep, "say": say, "info": info})
