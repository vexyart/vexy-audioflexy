#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = ["fire", "rich", "huggingface-hub"]
# ///
# this_file: src/vexy_audioflexy/tools/tts/tools/tts-soniqo-voxcpm2-onnx.py
"""Soniqo VoxCPM2 ONNX wrapper (soniqo/VoxCPM2-ONNX). Backend: onnx via speech-core C++ helper binary; languages: en; sample rate: 48 kHz; modes: ivc (instant voice cloning from reference WAV); capabilities: voice_cloning. Requires pre-built speech-core ONNX helper binary and local model checkout."""

from pathlib import Path
import os
import shutil
import subprocess

import fire
from _paths import runtime_path, bind_snapshot

VERSION = "1.0.0"
MODEL = "soniqo/VoxCPM2-ONNX"
ENV_BIN = "VOXCPM2_ONNX_BIN"
ENV_MODEL_DIR = "VOXCPM2_ONNX_MODEL_DIR"
DEFAULT_BIN = str(runtime_path("speech-core/build-onnx/voxcpm2_onnx_synth"))
DEFAULT_MODEL_DIR = str(runtime_path("voxcpm2-soniqo-onnx"))


def _skip_existing(output: str, force: bool, play: bool) -> bool:
    out_path = Path(output)
    return not play and not force and out_path.exists() and out_path.stat().st_size > 0


def _existing_binary(path_or_name: str) -> str | None:
    path = Path(path_or_name).expanduser()
    if path.exists() and path.is_file():
        return str(path)
    return shutil.which(path_or_name)


def _build_host_command(
    binary: str,
    model_dir: str,
    text: str,
    output: str,
    ref: str | None,
    instruct: str | None,
    max_steps: int,
    language: str,
) -> list[str]:
    return [
        binary,
        model_dir,
        ref or "none",
        text,
        output,
        instruct or "",
        str(max_steps),
        language,
    ]


def _setup_message(reason: str) -> str:
    return (
        f"{reason}\n"
        "Build the exact ONNX speech-core helper first:\n"
        "  cmake -S .external/speech-core -B .external/speech-core/build-onnx "
        "-DSPEECH_CORE_WITH_ONNX=ON -DORT_DIR=/opt/homebrew/opt/onnxruntime "
        "-DCMAKE_CXX_FLAGS=-I/opt/homebrew/opt/onnxruntime/include/onnxruntime\n"
        "  cmake --build .external/speech-core/build-onnx -j\n"
        "  c++ -std=c++17 -O2 "
        "-I.external/speech-core/include "
        "-I/opt/homebrew/opt/onnxruntime/include/onnxruntime "
        "tools/voxcpm2_onnx_synth.cpp "
        ".external/speech-core/build-onnx/libspeech_core_models.a "
        ".external/speech-core/build-onnx/libspeech_core.a "
        "-L/opt/homebrew/opt/onnxruntime/lib -lonnxruntime "
        "-Wl,-rpath,/opt/homebrew/opt/onnxruntime/lib "
        "-o .external/speech-core/build-onnx/voxcpm2_onnx_synth\n"
        f"Then set {ENV_BIN} or pass --binary if you use another location."
    )


def prep(force: bool = False):
    """Download + cache all model weights/assets WITHOUT synthesizing. Idempotent."""
    from huggingface_hub import snapshot_download
    from rich.console import Console

    console = Console()
    model_path = Path(os.environ.get(ENV_MODEL_DIR, DEFAULT_MODEL_DIR)).expanduser()
    with console.status(f"[cyan]Downloading {MODEL}…"):
        snapshot = Path(snapshot_download(MODEL, force_download=force))
        bind_snapshot(snapshot, model_path)
    print(f"[prep] ready: {MODEL}")


def say(
    text: str,
    output: str = "output.wav",
    ref: str | None = None,
    ref_text: str | None = None,
    language: str = "english",
    instruct: str | None = None,
    max_steps: int = 160,
    binary: str | None = None,
    model_dir: str | None = None,
    play: bool = False,
    force: bool = False,
):
    """Generate VoxCPM2 ONNX speech through the external Soniqo host wrapper.

    Args:
        text: Text to synthesize.
        output: Output WAV path.
        ref: Optional reference WAV for cloning.
        ref_text: Optional transcript for ultimate cloning if supported by host.
        language: Language hint passed to host.
        instruct: Optional natural-language voice design prompt.
        max_steps: Autoregressive step cap. 160 is about 25.6s at 48kHz.
        binary: speech-core ONNX helper path. Defaults to VOXCPM2_ONNX_BIN.
        model_dir: Local `soniqo/VoxCPM2-ONNX` checkout. Defaults to VOXCPM2_ONNX_MODEL_DIR.
        play: Play with afplay after saving.
        force: Overwrite output if it already exists.
    """
    from rich.console import Console

    console = Console()
    if _skip_existing(output, force, play):
        console.print(f"[yellow]Skipped:[/yellow] {output} already exists (use --force)")
        return

    if ref and not Path(ref).exists():
        console.print(f"[red]Reference audio not found:[/red] {ref}")
        raise SystemExit(1)
    if model_dir and not Path(model_dir).exists():
        console.print(f"[red]VoxCPM2 ONNX model checkout not found:[/red] {model_dir}")
        raise SystemExit(1)

    binary = binary or os.environ.get(ENV_BIN) or DEFAULT_BIN
    model_dir = model_dir or os.environ.get(ENV_MODEL_DIR) or DEFAULT_MODEL_DIR

    resolved_binary = _existing_binary(binary)
    if not resolved_binary:
        console.print(
            "[red]speech-core VoxCPM2 ONNX helper is not available.[/red]\n"
            + _setup_message(f"Missing binary: {binary}")
        )
        raise SystemExit(1)

    model_path = Path(model_dir).expanduser()
    if not model_path.exists() or not model_path.is_dir():
        console.print(f"[red]VoxCPM2 ONNX model checkout not found:[/red] {model_dir}")
        raise SystemExit(1)

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = _build_host_command(
        binary=resolved_binary,
        model_dir=str(model_path),
        text=text,
        output=str(out_path),
        ref=ref,
        instruct=instruct or "clear, natural delivery",
        max_steps=max_steps,
        language=language,
    )

    if ref_text:
        console.print("[yellow]Note:[/yellow] ref_text is accepted for API compatibility but ignored by the current ONNX helper.")
    console.print(f"[dim]Model repo: {MODEL}[/dim]")
    console.print(f"[dim]Running speech-core ONNX helper: {resolved_binary}[/dim]")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        console.print(
            f"[red]speech-core VoxCPM2 ONNX helper failed with exit code {exc.returncode}.[/red]\n"
            f"Verify {model_dir} contains the exact {MODEL} bundle."
        )
        raise SystemExit(exc.returncode) from exc

    if not out_path.exists() or out_path.stat().st_size == 0:
        console.print(f"[red]External host did not create a non-empty output:[/red] {out_path}")
        raise SystemExit(1)

    if play:
        subprocess.run(["afplay", str(out_path)], check=True)

    console.print(f"[green]Saved:[/green] {out_path.resolve()}")


def info():
    """Print one-line JSON: name, repo, backend, languages, sample_rate_hz, modes, capabilities."""
    import json
    print(json.dumps({
        "name": "VoxCPM2 ONNX by Soniqo",
        "repo": MODEL,
        "backend": "onnx",
        "languages": ["en"],
        "sample_rate_hz": 48000,
        "modes": ["ivc"],
        "capabilities": ["voice_cloning"],
    }))


if __name__ == "__main__":
    fire.Fire({"prep": prep, "say": say, "info": info})
