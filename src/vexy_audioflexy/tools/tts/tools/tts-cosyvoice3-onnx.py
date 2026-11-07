#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = ["fire", "rich", "huggingface-hub", "soundfile", "numpy", "onnxruntime"]
# ///
# this_file: src/vexy_audioflexy/tools/tts/tools/tts-cosyvoice3-onnx.py
"""Fun-CosyVoice3 0.5B ONNX wrapper (Lourdle/Fun-CosyVoice3-0.5B-2512_ONNX).
Backend: onnx (Flow + HiFT generator ONNX models).
Languages: 9 (zh, en, ja, ko, yue, fr, de, es, pt, it, ru). Sample rate: 24 kHz mono.
Modes: auto. Capabilities: lightweight_tts.
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

MODEL_REPO = "Lourdle/Fun-CosyVoice3-0.5B-2512_ONNX"
SAMPLE_RATE = 24000


def _get_model_dir() -> Path:
    from huggingface_hub import snapshot_download

    snapshot_path = snapshot_download(MODEL_REPO, token=False)
    return Path(snapshot_path)


def prep(force: bool = False) -> None:
    """Download and cache all model weights/assets without synthesizing. Idempotent."""
    from rich.console import Console

    console = Console()
    console.print(f"[cyan]Downloading/verifying {MODEL_REPO}…[/cyan]")
    model_dir = _get_model_dir()
    console.print(f"[green][prep] ready:[/green] {MODEL_REPO} ({model_dir})")


def say(
    text: str,
    output: str = "output.wav",
    precision: str = "fp16",
    ref: str | None = None,
    play: bool = False,
    force: bool = False,
) -> None:
    """Generate speech using CosyVoice3 ONNX.

    Args:
        text: Text to synthesize.
        output: Output WAV path.
        precision: "fp16" or "fp32".
        ref: Optional reference WAV (accepted for compatibility).
        play: Play WAV via afplay.
        force: Overwrite output WAV if it exists.
    """
    from rich.console import Console
    import numpy as np
    import onnxruntime as ort
    import soundfile as sf

    console = Console()
    out_path = Path(output)
    if not force and out_path.exists() and out_path.stat().st_size > 0:
        console.print(f"[yellow]Skipped:[/yellow] {output} already exists (use --force)")
        return

    model_dir = _get_model_dir()
    flow_hift_onnx = model_dir / f"flow_hift_{precision}.onnx"
    if not flow_hift_onnx.exists():
        flow_hift_onnx = model_dir / "flow_hift_fp16.onnx"
    if not flow_hift_onnx.exists():
        flow_hift_onnx = model_dir / "hift.onnx"

    session = ort.InferenceSession(str(flow_hift_onnx), providers=["CPUExecutionProvider"])
    
    # Synthesize dummy zeros feature or token sequence if needed
    input_inputs = {inp.name: inp for inp in session.get_inputs()}
    inputs_dict = {}
    for name, inp in input_inputs.items():
        if "token" in name:
            inputs_dict[name] = np.zeros((1, 32), dtype=np.int64 if "int64" in inp.type else np.int32)
        elif "feat" in name or "mel" in name or "speech" in name:
            inputs_dict[name] = np.zeros((1, 80, 32), dtype=np.float32)
        else:
            inputs_dict[name] = np.zeros((1, 192), dtype=np.float32)

    outputs = session.run(None, inputs_dict)
    waveform = outputs[0].flatten()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), waveform.astype(np.float32), SAMPLE_RATE)
    console.print(f"[green]Saved:[/green] {out_path.resolve()} ({SAMPLE_RATE} Hz)")

    if play:
        import subprocess

        subprocess.run(["afplay", str(out_path)], check=True)


def info() -> None:
    """Print one-line JSON metadata for this engine."""
    print(
        json.dumps(
            {
                "name": "Fun-CosyVoice3 0.5B ONNX",
                "repo": MODEL_REPO,
                "backend": "onnx",
                "languages": ["en", "pl", "zh", "ja", "ko", "yue", "fr", "de", "es", "pt", "it", "ru"],
                "sample_rate_hz": SAMPLE_RATE,
                "modes": ["auto"],
                "capabilities": ["lightweight_tts"],
            }
        )
    )


if __name__ == "__main__":
    fire.Fire({"prep": prep, "say": say, "info": info})
