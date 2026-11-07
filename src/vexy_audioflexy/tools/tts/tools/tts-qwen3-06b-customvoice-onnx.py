#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = ["fire"]
# ///
# this_file: src/vexy_audioflexy/tools/tts/tools/tts-qwen3-06b-customvoice-onnx.py
"""BROKEN ENGINE — Qwen3 0.6B CustomVoice ONNX. Requires external C# app/dotnet; marked broken/ignored in this repo."""

REPO = "tonythethompson/Qwen3-TTS-12Hz-0.6B-CustomVoice-ONNX"

_REASON = "ERROR: tts-qwen3-06b-customvoice-onnx is marked BROKEN and ignored in this repo. Do not use."


from pathlib import Path


def _project_path(app_dir: str | Path = "qwen-labs-cs") -> Path:
    p = Path(app_dir).resolve()
    candidates = [
        p / "src" / "ElBruno.QwenTTS",
        p / "ElBruno.QwenTTS",
        p,
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def _model_dir_path(model_dir: str | Path = "models") -> Path:
    return Path(model_dir).resolve()


def _output_path(output: str | Path = "output.wav") -> Path:
    return Path(output).resolve()


def prep(force: bool = False):
    """Download + cache model weights. Broken stub — always exits 1."""
    import sys
    print(_REASON, file=sys.stderr)
    raise SystemExit(1)


def say(text, ref=None, output="output.wav", play=False, force=False):
    """Synthesize speech. Broken stub — always exits 1."""
    import sys
    print(_REASON, file=sys.stderr)
    raise SystemExit(1)


def info():
    """Print one-line JSON metadata for this engine."""
    import json
    print(json.dumps({
        "name": "Qwen3-TTS 0.6B CustomVoice ONNX",
        "repo": REPO,
        "backend": "onnx",
        "languages": ["en", "zh"],
        "sample_rate_hz": 24000,
        "modes": ["preset"],
        "capabilities": ["custom_voice"],
    }))


if __name__ == "__main__":
    import fire
    fire.Fire({"prep": prep, "say": say, "info": info})
