#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = ["fire", "rich", "huggingface-hub"]
# ///
# this_file: src/vexy_audioflexy/tools/tts/tools/tts-breeze-mlx.py
"""Breeze-TTS-2 on MLX via the local breeze-tts-mlx checkout.

The engine is not importable as a library from this project: it pins its own
Python and MLX versions in ``local/breeze-tts-mlx/pyproject.toml`` and runs
through ``uv run`` inside that checkout. This wrapper therefore shells out to
its ``infer.py``, the same way the audio.cpp wrapper shells out to a binary.

Variants are whole checkpoint directories rather than flags: ``bf16`` is the
mlx-community conversion, ``8bit`` its quantized sibling. Both are downloaded
into the checkout, because ``infer.py`` takes a directory, not a repo id.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from _paths import runtime_path

VERSION = "1.0.0"

ROOT = Path(__file__).resolve().parent.parent
ENGINE_ROOT = Path(
    os.environ.get("BREEZE_MLX_ROOT", str(runtime_path('breeze-tts-mlx')))
)

# repo id -> checkout-relative directory infer.py is given
VARIANTS = {
    "bf16": ("mlx-community/Breeze-TTS-2-mlx", "chkpt-mlxcommunity-bf16"),
    "8bit": ("mlx-community/Breeze-TTS-2-mlx-8bit", "chkpt-mlxcommunity-8bit"),
}


def _variant(variant: str) -> tuple[str, Path]:
    if variant not in VARIANTS:
        raise SystemExit(
            f"unknown variant {variant!r}; choose one of {', '.join(VARIANTS)}"
        )
    repo, subdir = VARIANTS[variant]
    return repo, ENGINE_ROOT / subdir


def _require_engine() -> None:
    if not (ENGINE_ROOT / "infer.py").is_file():
        raise SystemExit(
            f"breeze-tts-mlx checkout not found at {ENGINE_ROOT}.\n"
            "Clone it there, then: cd local/breeze-tts-mlx && uv sync --locked"
        )


def prep(variant: str = "bf16") -> None:
    """Download into the provider cache; link the engine checkpoint location.

    Args:
        variant: ``bf16`` or ``8bit``.
    """
    from huggingface_hub import snapshot_download
    from rich.console import Console

    console = Console()
    _require_engine()
    repo, target = _variant(variant)
    if (target / "model.safetensors.index.json").is_file():
        console.print(f"[green][prep][/green] {variant} already present: {target}")
        return
    console.print(f"[cyan][prep][/cyan] {repo} -> {target}")
    snapshot = Path(snapshot_download(repo))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(snapshot, target_is_directory=True)
    print(f"[prep] ready: {repo} -> {target}")


def say(
    text: str,
    output: str = "output.wav",
    variant: str = "bf16",
    ref: str = "",
    ref_text: str = "",
    instruction: str = "",
    mode: str = "auto",
    seed: int = 42,
    max_new_tokens: int = 0,
    force: bool = False,
) -> None:
    """Synthesize one clip with Breeze-TTS-2.

    Args:
        text: Text to speak.
        output: Output WAV path.
        variant: ``bf16`` or ``8bit``.
        ref: Reference speaker WAV for cloning.
        ref_text: Transcript of *ref*. The engine rejects one without the other.
        instruction: Style instruction (selects the engine's guided/edit modes).
        mode: Prompt mode: auto, plain, guided, clone, edit.
        seed: Sampling seed.
        max_new_tokens: Override the engine's token budget; 0 keeps its default.
        force: Overwrite an existing output file.
    """
    from rich.console import Console

    console = Console()
    _require_engine()
    out_path = Path(output).resolve()
    if not force and out_path.exists() and out_path.stat().st_size > 0:
        console.print(f"[yellow]Skipped:[/yellow] {out_path} already exists")
        return

    _repo, model_dir = _variant(variant)
    if not (model_dir / "model.safetensors.index.json").is_file():
        raise SystemExit(f"{variant} weights missing at {model_dir}; run prep first")

    # The engine couples the two: passing one without the other is a hard error,
    # so drop a lone reference rather than letting it fail late.
    if bool(ref) != bool(ref_text.strip()):
        console.print(
            "[yellow]Ignoring reference:[/yellow] Breeze needs --ref and --ref_text together"
        )
        ref = ref_text = ""

    cmd = [
        "uv", "run", "python", "infer.py", str(model_dir),
        "--text", text,
        "--output", str(out_path),
        "--seed", str(seed),
        "--mode", mode,
    ]
    if ref:
        cmd += ["--ref-audio", str(Path(ref).resolve()), "--ref-text", ref_text]
    if instruction:
        cmd += ["--instruction", instruction]
    if max_new_tokens:
        cmd += ["--max-new-tokens", str(max_new_tokens)]

    console.print(f"[cyan]breeze-mlx[/cyan] v{VERSION} variant=[yellow]{variant}[/yellow]")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        cmd, cwd=str(ENGINE_ROOT), capture_output=True, text=True
    )
    if result.returncode != 0 or not out_path.exists():
        tail = "\n".join(
            ln for ln in (result.stdout + "\n" + result.stderr).splitlines()
            if ln.strip()
        )
        console.print("[red]infer.py failed[/red]")
        console.print(tail[-1500:])
        raise SystemExit(1)

    # infer.py's last line carries the measurements; keep it, it is the evidence
    # that the run produced audio rather than an empty header.
    summary = [ln for ln in result.stdout.splitlines() if ln.startswith("saved ")]
    console.print(f"[green]Saved:[/green] {out_path}")
    if summary:
        console.print(f"[dim]{summary[-1]}[/dim]")


def info(variant: str = "bf16") -> None:
    """Print one-line JSON with model metadata."""
    repo, model_dir = _variant(variant)
    print(json.dumps(
        {
            "name": f"Breeze-TTS-2 MLX {variant}",
            "repo": repo,
            "stack": "mlx",
            "runtime": "breeze-tts-mlx",
            "quant": variant,
            "model_dir": str(model_dir),
            "installed": (model_dir / "model.safetensors.index.json").is_file(),
            "modes": ["auto", "ivc"],
            "sample_rate_hz": 24000,
        },
        separators=(",", ":"),
    ))


if __name__ == "__main__":
    import fire

    fire.Fire({"prep": prep, "say": say, "info": info})
