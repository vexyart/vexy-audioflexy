#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = ["fire", "rich", "huggingface-hub", "soundfile", "numpy", "onnxruntime", "phonemizer", "espeakng-loader", "num2words", "Unidecode"]
# ///
# this_file: src/vexy_audioflexy/tools/tts/tools/tts-inflect-onnx.py
"""Inflect v2 ONNX wrapper supporting Inflect-Micro-v2 and Inflect-Nano-v2 ONNX variants.
Repos:
- owensong/Inflect-Micro-v2-ONNX (Micro FP32 ONNX, 24 kHz)
- owensong/Inflect-Nano-v2-ONNX (Nano FP32 ONNX, 24 kHz)
- csukuangfj/Inflect-Nano-v2-ONNX (Nano FP32 ONNX fork, 24 kHz)
Backend: onnx runtime + VITS duration and decode sessions + espeak-ng frontend.
Languages: en. Sample rate: 24 kHz mono. Modes: auto. Capabilities: lightweight_tts.
"""

from pathlib import Path
import json
import os
import sys

import fire

# Fallback HF_HOME if default ~/.cache/huggingface is a broken symlink
hf_cache = Path.home() / ".cache" / "huggingface"
if not os.environ.get("HF_HOME") and (hf_cache.is_symlink() and not hf_cache.exists()):
    os.environ["HF_HOME"] = str(Path.home() / ".cache" / "hf_local")

KNOWN_MODELS = {
    "micro": "owensong/Inflect-Micro-v2-ONNX",
    "nano": "owensong/Inflect-Nano-v2-ONNX",
    "nano-csukuangfj": "csukuangfj/Inflect-Nano-v2-ONNX",
}


def _get_model_dir(repo: str) -> Path:
    from huggingface_hub import snapshot_download

    snapshot_path = snapshot_download(repo, token=False)
    return Path(snapshot_path)


def prep(force: bool = False, model: str = "micro") -> None:
    """Download and cache all model weights/assets without synthesizing. Idempotent."""
    from rich.console import Console

    console = Console()
    repo = KNOWN_MODELS.get(model, model)
    console.print(f"[cyan]Downloading/verifying {repo}…[/cyan]")
    model_dir = _get_model_dir(repo)
    console.print(f"[green][prep] ready:[/green] {repo} ({model_dir})")


def say(
    text: str,
    output: str = "output.wav",
    model: str = "micro",
    speed: float = 1.0,
    variation: float = 0.667,
    seed: int = 0,
    ref: str | None = None,
    play: bool = False,
    force: bool = False,
) -> None:
    """Generate speech using Inflect v2 ONNX (Micro or Nano).

    Args:
        text: Text to synthesize.
        output: Output WAV path.
        model: "micro", "nano", "nano-csukuangfj", or exact HF repo.
        speed: Speed multiplier (0.5 to 2.0).
        variation: Noise variation (0.0 to 1.0).
        seed: Random seed.
        ref: Reference WAV (accepted for compatibility, ignored by preset VITS).
        play: Play WAV after generating.
        force: Overwrite existing output WAV.
    """
    from rich.console import Console

    console = Console()
    out_path = Path(output)
    if not force and out_path.exists() and out_path.stat().st_size > 0:
        console.print(f"[yellow]Skipped:[/yellow] {output} already exists (use --force)")
        return

    repo = KNOWN_MODELS.get(model, model)
    model_dir = _get_model_dir(repo)

    # Import inference module directly from model directory
    for p in [model_dir, model_dir / "runtime", model_dir / "onnx"]:
        if str(p) not in sys.path and p.exists():
            sys.path.insert(0, str(p))

    from inference_onnx import InflectONNX, SAMPLE_RATE

    console.print(f"[dim]Repo: {repo}[/dim]")
    engine = InflectONNX(model_dir=model_dir, provider="cpu")
    engine.save(
        text,
        out_path,
        speed=speed,
        variation=variation,
        seed=seed,
    )

    console.print(f"[green]Saved:[/green] {out_path.resolve()} ({SAMPLE_RATE} Hz)")
    if play:
        import subprocess

        subprocess.run(["afplay", str(out_path)], check=True)


def info() -> None:
    """Print one-line JSON metadata for this engine."""
    print(
        json.dumps(
            {
                "name": "Inflect v2 ONNX (Micro & Nano)",
                "repo": list(KNOWN_MODELS.values()),
                "backend": "onnx",
                "languages": ["en"],
                "sample_rate_hz": 24000,
                "modes": ["auto"],
                "capabilities": ["lightweight_tts"],
            }
        )
    )


if __name__ == "__main__":
    fire.Fire({"prep": prep, "say": say, "info": info})
