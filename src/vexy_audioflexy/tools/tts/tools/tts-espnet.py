#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#   "fire", "rich", "espnet", "espnet_model_zoo", "torch", "torchaudio",
#   "soundfile", "numpy", "setuptools<81",
# ]
# ///
# this_file: src/vexy_audioflexy/tools/tts/tools/tts-espnet.py
"""ESPnet2 TTS. Backend: torch/espnet, languages: en/ja/zh, 22-24 kHz mono, modes: auto, capabilities: none.

ESPnet is a research toolkit, not a model, so what this wrapper registers is a
handful of its published checkpoints: VITS, Tacotron 2 and FastSpeech 2 on
LJSpeech. They matter here as the reference recipes behind several engines this
project already runs — VITS in particular, which reaches the batch through
sherpa-onnx as Piper voices and through ESPnet as the original training recipe.

Python 3.11: ESPnet's dependency set does not resolve on 3.12 in this environment,
and `setuptools<81` is needed because parts of the stack still import
`pkg_resources`.

Checkpoints come from HuggingFace via `Text2Speech.from_pretrained`. English only
among the registered ones; there is no Polish recipe in the zoo worth benchmarking.
"""

from __future__ import annotations

import json
from pathlib import Path

VERSION = "1.0.0"

MODELS: dict[str, dict] = {
    "ljspeech-vits": {
        "id": "espnet/kan-bayashi_ljspeech_vits",
        "langs": ["en"],
        "sample_rate": 22050,
        "arch": "VITS (end-to-end)",
    },
    "ljspeech-tacotron2": {
        "id": "espnet/kan-bayashi_ljspeech_tacotron2",
        "langs": ["en"],
        "sample_rate": 22050,
        "arch": "Tacotron 2 + Griffin-Lim",
    },
    "ljspeech-fastspeech2": {
        "id": "espnet/kan-bayashi_ljspeech_fastspeech2",
        "langs": ["en"],
        "sample_rate": 22050,
        "arch": "FastSpeech 2 + Griffin-Lim",
    },
}


def _spec(model: str) -> dict:
    try:
        return MODELS[model]
    except KeyError:
        raise SystemExit(f"unknown model '{model}'. Known: {', '.join(MODELS)}") from None


def _tts(model: str, device: str):
    from espnet2.bin.tts_inference import Text2Speech

    return Text2Speech.from_pretrained(_spec(model)["id"], device=device)


def prep(model: str = "ljspeech-vits", force: bool = False) -> None:
    """Download + cache one checkpoint WITHOUT synthesizing. Idempotent.

    Args:
        model: Registered model name; `models` lists them.
        force: Unused; cached weights are never re-fetched.
    """
    spec = _spec(model)
    _tts(model, "cpu")
    print(f"[prep] ready: {model} ({spec['id']})")


def say(
    text: str,
    output: str = "output.wav",
    model: str = "ljspeech-vits",
    ref: str = "",
    lang: str = "en",
    device: str = "cpu",
    play: bool = False,
    force: bool = False,
) -> None:
    """Synthesize `text` with one ESPnet checkpoint.

    Args:
        text: Text to synthesize.
        output: Output WAV path.
        model: Registered model name.
        ref: Rejected. These are single-speaker recipes with no cloning.
        lang: Checked against the model's languages; ESPnet checkpoints are
            single-language and take no language argument.
        device: `cpu` (default) or `mps`.
        play: Play the result with afplay instead of just saving it.
        force: Overwrite an existing output file.
    """
    import time

    import numpy as np
    import soundfile as sf
    from rich.console import Console
    from rich.panel import Panel

    console = Console()
    out_path = Path(output).resolve()
    if not force and out_path.exists() and out_path.stat().st_size > 0:
        console.print(f"[yellow]Skipped:[/yellow] {output} already exists (use --force)")
        return
    spec = _spec(model)
    if ref:
        raise SystemExit(f"{model} is single-speaker and cannot clone {ref}")
    if lang not in spec["langs"]:
        raise SystemExit(f"{model} speaks {'/'.join(spec['langs'])}, not '{lang}'")

    console.print(
        Panel.fit(
            f"[bold cyan]ESPnet2[/bold cyan] v{VERSION}  model=[yellow]{model}[/yellow]\n"
            f"{spec['arch']}  device={device}",
            border_style="cyan",
        )
    )

    tts = _tts(model, device)
    t0 = time.monotonic()
    result = tts(text)
    wall = time.monotonic() - t0

    samples = np.asarray(result["wav"].cpu().numpy(), dtype="float32").reshape(-1)
    if samples.size == 0 or not np.any(samples):
        console.print(f"[red]{model} produced silence[/red]")
        raise SystemExit(1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), samples, tts.fs, subtype="PCM_16")
    duration = samples.size / tts.fs
    console.print(
        f"[green]Saved[/green] [bold]{out_path}[/bold] "
        f"({duration:.1f}s @ {tts.fs} Hz, rtf={wall / duration:.2f})"
    )
    if play:
        import subprocess

        subprocess.run(["afplay", str(out_path)], check=True)


def info(model: str = "ljspeech-vits") -> None:
    """Print one-line JSON with model metadata."""
    spec = _spec(model)
    print(json.dumps(
        {
            "name": f"espnet-{model}",
            "repo": f"espnet/espnet ({spec['id']})",
            "stack": "torch",
            "runtime": "ESPnet2 (PyTorch, CPU, Python 3.11)",
            "quant": "fp32",
            "mode": "auto",
            "voice_clone": False,
            "langs": spec["langs"],
            "sample_rate": spec["sample_rate"],
            "arch": spec["arch"],
        },
        separators=(",", ":"),
    ))


def models() -> None:
    """List the registered checkpoints."""
    for name, spec in MODELS.items():
        print(f"{name:22} {spec['arch']:28} {spec['sample_rate']} Hz  {spec['id']}")


if __name__ == "__main__":
    import fire

    fire.Fire({"prep": prep, "say": say, "info": info, "models": models})
