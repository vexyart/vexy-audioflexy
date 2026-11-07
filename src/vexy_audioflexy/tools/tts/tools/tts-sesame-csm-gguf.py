#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# this_file: src/vexy_audioflexy/tools/tts/tools/tts-sesame-csm-gguf.py
"""BROKEN ENGINE — Sesame CSM 1B GGUF. Backend: gguf (llama.cpp CSM branch binary required). Languages: en. Modes: auto, ivc. Capabilities: voice_cloning. Marked broken/ignored in this repo: requires llama-tts-csm binary not available via uv."""

REPO = "ggml-org/sesame-csm-1b-GGUF"

_REASON = "tts-sesame-csm-gguf is marked BROKEN and ignored in this repo: requires llama-tts-csm binary from llama.cpp CSM branch. Do not use."


def prep(force: bool = False):
    """Download + cache model weights WITHOUT synthesizing. BROKEN — raises SystemExit(1)."""
    import sys
    print(f"ERROR: {_REASON}", file=sys.stderr)
    raise SystemExit(1)


def say(text, ref=None, output="output.wav", play=False, force=False):
    """Synthesize speech. BROKEN — raises SystemExit(1)."""
    import sys
    print(f"ERROR: {_REASON}", file=sys.stderr)
    raise SystemExit(1)


def info():
    """Print one-line JSON metadata for this engine."""
    import json
    print(json.dumps({
        "name": "CSM 1B GGUF",
        "repo": REPO,
        "backend": "gguf",
        "languages": ["en"],
        "sample_rate_hz": 24000,
        "modes": ["auto", "ivc"],
        "capabilities": ["voice_cloning"],
    }))


if __name__ == "__main__":
    import fire
    fire.Fire({"prep": prep, "say": say, "info": info})
