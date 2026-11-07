#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = ["fire", "rich", "huggingface-hub", "soundfile", "numpy", "onnxruntime", "scipy", "tokenizers"]
# ///
# this_file: src/vexy_audioflexy/tools/tts/tools/tts-audio8-onnx.py
"""Audio8 / ArkTTS ONNX wrapper — two published quantizations of the same DualAR stack.

Backend: onnx (weight-only quantized slow/fast AR transformers + FP16 neural audio codec).
Languages: 11 (en, pl, zh, es, fr, de, it, ja, ko, nl, yue). Sample rate: 44.1 kHz.
Modes: auto, ivc (instant voice cloning from reference WAV).
Capabilities: voice_cloning, zero_shot_cloning.

Variants (`--variant`):
  preview-0.6b-int4  Audio8/Audio8-TTS-Preview-0.6B-ONNX-INT4  (default)
  0.1b-int8          Audio8/audio8-TTS-0.1B-ONNX-INT8
"""

from pathlib import Path
import json
import os
import shutil
import sys
import tempfile
from typing import Any

import fire
from _paths import runtime_path

VARIANTS: dict[str, dict[str, str]] = {
    "preview-0.6b-int4": {
        "repo": "Audio8/Audio8-TTS-Preview-0.6B-ONNX-INT4",
        "precision": "int4",
        "runtime": "onnx_runtime",
    },
    "0.1b-int8": {
        "repo": "Audio8/audio8-TTS-0.1B-ONNX-INT8",
        "precision": "int8",
        "runtime": "onnx_runtime_0_1b_int8",
    },
}
DEFAULT_VARIANT = "preview-0.6b-int4"
MODEL = VARIANTS[DEFAULT_VARIANT]["repo"]  # kept for backwards compatibility
ROOT = Path(__file__).resolve().parent.parent


def _variant(name: str) -> dict[str, str]:
    try:
        return VARIANTS[name]
    except KeyError:
        raise SystemExit(
            f"unknown variant '{name}'. Choose from: {', '.join(VARIANTS)}"
        ) from None

# Fallback HF_HOME if default ~/.cache/huggingface is a broken symlink
hf_cache = Path.home() / ".cache" / "huggingface"
if not os.environ.get("HF_HOME") and (hf_cache.is_symlink() and not hf_cache.exists()):
    os.environ["HF_HOME"] = str(Path.home() / ".cache" / "hf_local")

# Each published quantization ships its own `arktts_runtime` package upstream: the
# 0.1B export is a Mamba/attention hybrid with batched KV plus SSM state tensors,
# which the 0.6B runtime cannot drive. Put the matching one on sys.path per call.
AUDIO8_CHECKOUT = runtime_path('Audio8_TTS')


def _use_runtime(variant: str) -> None:
    runtime_dir = AUDIO8_CHECKOUT / _variant(variant)["runtime"]
    if not runtime_dir.exists():
        raise SystemExit(
            f"Audio8 runtime missing: {runtime_dir}\n"
            f"Refresh it with: git -C {AUDIO8_CHECKOUT} pull"
        )
    path = str(runtime_dir)
    if path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)


def _get_model_dir(variant: str = DEFAULT_VARIANT) -> Path:
    from huggingface_hub import snapshot_download

    snapshot_path = snapshot_download(_variant(variant)["repo"], token=False)
    return Path(snapshot_path)


def prep(variant: str = DEFAULT_VARIANT, force: bool = False) -> None:
    """Download + cache all model weights/assets WITHOUT synthesizing. Idempotent."""
    from rich.console import Console

    console = Console()
    repo = _variant(variant)["repo"]
    console.print(f"[cyan]Downloading/verifying {repo}…[/cyan]")
    model_dir = _get_model_dir(variant)
    console.print(f"[green][prep] ready:[/green] {repo} ({model_dir})")


def say(
    text: str,
    output: str = "output.wav",
    variant: str = DEFAULT_VARIANT,
    ref: str | None = None,
    ref_text: str | None = None,
    language: str = "en",
    voice: str = "speaker_a",
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    seed: int = 42,
    threads: int = 4,
    play: bool = False,
    force: bool = False,
) -> None:
    """Generate speech using Audio8 TTS Preview 0.6B ONNX INT4.

    Args:
        text: Text to synthesize.
        output: Output WAV path.
        variant: Which published quantization to run (see module docstring).
        ref: Optional reference WAV for zero-shot voice cloning.
        ref_text: Transcript of reference audio.
        language: Language hint (en, pl, zh, es, fr, de, it, ja, ko, nl).
        voice: Voice profile name when ref is omitted.
        max_new_tokens: Autoregressive token limit.
        temperature: Sampling temperature.
        top_p: Nucleus sampling p.
        seed: Random seed.
        threads: ONNX Runtime CPU threads.
        play: Play WAV via afplay after generation.
        force: Overwrite output WAV if it already exists.
    """
    from rich.console import Console

    console = Console()
    out_path = Path(output)
    if not force and out_path.exists() and out_path.stat().st_size > 0:
        console.print(f"[yellow]Skipped:[/yellow] {output} already exists (use --force)")
        return

    spec = _variant(variant)
    model_dir = _get_model_dir(variant)
    _use_runtime(variant)

    from arktts_runtime.registration import VoiceRegistration
    from arktts_runtime.runtime import ArkTtsRuntime
    import numpy as np
    import soundfile as sf

    with tempfile.TemporaryDirectory() as tmpdir:
        voices_dir = Path(tmpdir) / "voices"
        voices_dir.mkdir(parents=True, exist_ok=True)

        runtime = ArkTtsRuntime(
            model_dir=model_dir,
            voices_dir=voices_dir,
            precision=spec["precision"],
            codec_precision="fp16",
            threads=threads,
        )

        selected_voice = voice
        if ref:
            ref_path = Path(ref).expanduser()
            if not ref_path.exists():
                console.print(f"[red]Reference audio not found:[/red] {ref}")
                raise SystemExit(1)
            
            ref_data = ref_path.read_bytes()
            selected_voice = f"cloned_{ref_path.stem}"
            reg_dir = model_dir / "registration" if (model_dir / "registration").exists() else model_dir
            # The two upstream runtimes name the first argument differently
            # (`model_dir` in the 0.6B package, `registration_dir` in the 0.1B one).
            reg = VoiceRegistration(
                reg_dir,
                voices_dir,
                runtime.manifest.get("model_fingerprint", ""),
            )
            text_to_use = ref_text
            if not text_to_use:
                lab_path = ref_path.with_suffix(".lab")
                if lab_path.exists():
                    text_to_use = lab_path.read_text(encoding="utf-8").strip()
            text_to_use = text_to_use or "Reference audio clip transcript."
            reg.register(
                data=ref_data,
                filename=ref_path.name,
                text=text_to_use,
                name=selected_voice,
                overwrite=True,
            )
        else:
            # Create default dummy profile if no voice exists yet
            dummy_voice_dir = voices_dir / selected_voice
            dummy_voice_dir.mkdir(parents=True, exist_ok=True)
            dummy_profile = {
                "name": selected_voice,
                "created_at": "2026-08-01T00:00:00Z",
                "semantic_length": 0,
                "reference_length": 0,
                "transcript": "",
            }
            (dummy_voice_dir / "profile.json").write_text(json.dumps(dummy_profile))

        console.print(f"[dim]Model repo: {spec["repo"]}[/dim]")
        if ref:
            console.print(f"[dim]Reference WAV: {ref}[/dim]")

        audio, codes = runtime.synthesize(
            text=text,
            voice=selected_voice,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
        )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        sample_rate = int(runtime.manifest.get("sample_rate", 44100))
        sf.write(str(out_path), audio, sample_rate)
        console.print(f"[green]Saved:[/green] {out_path.resolve()} ({sample_rate} Hz)")

        if play:
            import subprocess

            subprocess.run(["afplay", str(out_path)], check=True)


def info(variant: str = DEFAULT_VARIANT) -> None:
    """Print one-line JSON metadata for this engine."""
    spec = _variant(variant)
    print(
        json.dumps(
            {
                "name": f"Audio8 TTS ONNX ({variant})",
                "repo": spec["repo"],
                "quant": spec["precision"],
                "backend": "onnx",
                "languages": ["en", "pl", "zh", "es", "fr", "de", "it", "ja", "ko", "nl"],
                "sample_rate_hz": 44100,
                "modes": ["auto", "ivc"],
                "capabilities": ["voice_cloning", "zero_shot_cloning"],
            }
        )
    )


if __name__ == "__main__":
    fire.Fire({"prep": prep, "say": say, "info": info})
