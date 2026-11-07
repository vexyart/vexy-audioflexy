#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = ["fire", "rich", "huggingface-hub"]
# ///
# this_file: src/vexy_audioflexy/tools/tts/tools/tts-chatterbox-mtl-codeccpp.py
"""Chatterbox Multilingual T3 GGUF via codec.cpp (BricksDisplay/Chatterbox-Multilingual-TTS-GGUF).

Backend: gguf (ggml) through codec.cpp's `tts-cli`. Languages: the 23 that
ResembleAI's `t3_mtl23ls_v3` covers, including en and pl. 24 kHz mono. Mode: ivc
(voice cloning from a reference WAV plus its transcript).

This packaging is *not* the one CrispASR's `chatterbox` backend loads: it uses the
native codec_lm split, where the S3G decoder and the whole T3 LM adaptor (audio
embed table, speech head, positional embeddings, cond encoder) live in the codec
GGUF, and the T3 backbone is a plain llama-arch GGUF. Only codec.cpp binds that
layout, so this model needs its own host rather than a variant of the CrispASR
wrapper.

codec.cpp's backbone library links with GNU-ld-only flags upstream, so the local
checkout carries a small patch for Apple's linker; see BUILD below.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from _paths import runtime_path

VERSION = "1.0.0"
MODEL_REPO = "BricksDisplay/Chatterbox-Multilingual-TTS-GGUF"

ROOT = Path(__file__).resolve().parent.parent
ENGINE_ROOT = runtime_path('codec.cpp')
BIN_PATH = ENGINE_ROOT / "build/tts-cli"

CODEC_FILE = "chatterbox-mtl-codec-q8_0.gguf"
BACKBONE_FILE = "chatterbox-mtl-t3-q8_0.gguf"

BUILD = f"""\
  git clone --recursive https://github.com/mybigday/codec.cpp {ENGINE_ROOT}
  cd {ENGINE_ROOT}
  cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j --target tts-cli
On macOS the backbone link step needs the Apple-linker patch in
cmake/SetupTtsBackbone.cmake (-force_load / -exported_symbols_list instead of
--whole-archive / --version-script, plus the Accelerate and Metal frameworks)."""


def _require_engine() -> Path:
    if BIN_PATH.exists():
        return BIN_PATH
    raise SystemExit(f"codec.cpp tts-cli not built at {BIN_PATH}\nBuild it with:\n{BUILD}")


def _weights() -> tuple[Path, Path]:
    from huggingface_hub import hf_hub_download

    codec = Path(hf_hub_download(MODEL_REPO, CODEC_FILE))
    backbone = Path(hf_hub_download(MODEL_REPO, BACKBONE_FILE))
    return codec, backbone


def prep(force: bool = False) -> None:
    """Download and cache the codec and T3 backbone GGUFs. Idempotent."""
    _require_engine()
    codec, backbone = _weights()
    print(f"[prep] ready: {MODEL_REPO} ({codec.parent})")


def say(
    text: str,
    output: str = "output.wav",
    ref: str = "",
    ref_text: str = "",
    seed: int = 0,
    temperature: float = 0.8,
    cfg_weight: float = 0.5,
    max_frames: int = 0,
    play: bool = False,
    force: bool = False,
) -> None:
    """Generate speech with Chatterbox Multilingual through codec.cpp.

    Args:
        text: Text to synthesize.
        output: Output WAV path (24 kHz mono).
        ref: Reference speaker WAV; without it the model uses its baked cond.
        ref_text: Transcript of *ref*.
        seed: Fixed sampling seed; 0 leaves the engine default.
        temperature: AR sampling temperature.
        cfg_weight: Classifier-free guidance weight.
        max_frames: Cap on generated codec frames; 0 leaves the engine default.
        play: Play the result with afplay instead of reporting the path.
        force: Overwrite an existing output file.
    """
    from rich.console import Console
    from rich.panel import Panel

    console = Console()
    binary = _require_engine()
    out_path = Path(output)

    if not force and out_path.exists() and out_path.stat().st_size > 0:
        console.print(
            f"[yellow]Skipped:[/yellow] {output} already exists (use --force to overwrite)"
        )
        return

    codec, backbone = _weights()
    cmd = [
        str(binary), "synthesize",
        "--model", str(codec),
        "--backbone", str(backbone),
        "--text", text,
        "--output", str(out_path),
        "--temp", str(temperature),
        "--cfg-weight", str(cfg_weight),
    ]
    if ref:
        ref_path = Path(ref).expanduser()
        if not ref_path.is_file():
            raise SystemExit(f"Reference audio not found: {ref}")
        cmd += ["--ref-audio", str(ref_path)]
        if ref_text:
            cmd += ["--ref-text", ref_text]
    if seed:
        cmd += ["--seed", str(seed)]
    if max_frames:
        cmd += ["--max-frames", str(max_frames)]

    console.print(
        Panel.fit(
            f"[bold cyan]Chatterbox Multilingual (codec.cpp)[/bold cyan] v{VERSION}\n"
            f"repo=[yellow]{MODEL_REPO}[/yellow]",
            border_style="cyan",
        )
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ENGINE_ROOT))
    if result.returncode != 0 or not out_path.exists():
        noise = ("ggml_metal", "ggml_backend", "~llama_context", "llama_")
        lines = [
            ln
            for ln in (result.stdout + "\n" + result.stderr).splitlines()
            if ln.strip() and not ln.startswith(noise)
        ]
        console.print("[red]tts-cli failed[/red]")
        console.print("\n".join(lines[-12:]))
        raise SystemExit(1)

    if play:
        subprocess.run(["afplay", str(out_path)], check=True)
    else:
        console.print(f"[green]Saved[/green] [bold]{out_path.resolve()}[/bold]")


def info() -> None:
    """Print one-line JSON with model metadata."""
    print(json.dumps({
        "name": "chatterbox-mtl-codeccpp",
        "repo": MODEL_REPO,
        "backend": "codeccpp",
        "quant": "q8_0",
        "languages": [
            "ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi", "it",
            "ja", "ko", "ms", "nl", "no", "pl", "pt", "ru", "sv", "sw", "tr", "zh",
        ],
        "sample_rate_hz": 24000,
        "modes": ["ivc"],
        "capabilities": ["voice_cloning", "multilingual"],
    }, separators=(",", ":")))


if __name__ == "__main__":
    import fire

    fire.Fire({"prep": prep, "say": say, "info": info})
