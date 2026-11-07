#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = ["platformdirs>=4", 
#   "fire",
#   "rich",
#   "soundfile",
#   "DeepFilterLib",
#   "audiosronnx @ git+https://github.com/TigreGotico/audiosronnx@dev",
# ]
# ///
# this_file: src/vexy_audioflexy/tools/fx/tools/aud-audiosronnx.py
"""Driver for the audiosronnx ONNX engines: denoising and bandwidth extension.

audiosronnx is one pip package holding 17 speech-restoration engines, each backed
by a HuggingFace weight repo under `TigreGotico/audiosronnx-*`. Inference is pure
onnxruntime plus numpy/scipy — no torch, no Metal — so every engine runs on this
machine. That makes it the same shape as `aud-audiocpp.py`: one wrapper, many
models, selected by name.

Two jobs, two entry points, and they refuse each other's engines:

| job              | task    | engines                                              |
|------------------|---------|------------------------------------------------------|
| denoising        | denoise | dpdfnet, mossformer2, frcrn, mpsenet, gtcrn, cmgan,   |
|                  |         | metadenoiser, mossformergan, voicefixer, deepfilternet|
| bandwidth ext.   | sr      | lavasr, novasr, apbwe, hifiganbwe, flowhigh, sidon,   |
|                  |         | callenhancer                                          |

Denoisers emit their own native rate; extenders always emit 48 kHz. The two
compose in that order — denoise first, then extend, or the extender reconstructs
a high band out of the noise.

`deepfilternet` needs the small `DeepFilterLib` Rust wheel for its ERB front-end;
it is declared above so every engine works out of the box.

Weights download on first use and cache under `~/.local/share/audiosronnx`,
pinned by revision.
"""

from __future__ import annotations


import os
from pathlib import Path
from platformdirs import user_cache_path, user_data_path
RUNTIME_ROOT = Path(os.environ.get("VAFX_AUDIO_RUNTIME_ROOT", user_data_path("vexy-audioflexy") / "audio" )).expanduser()
CACHE_ROOT = Path(os.environ.get("VAFX_AUDIO_CACHE_ROOT", user_cache_path("vexy-audioflexy") / "audio")).expanduser()
import json
from pathlib import Path

VERSION = "1.0.0"

# Engines whose weights are not permissively licensed. Recorded, not enforced.
NONCOMMERCIAL = {"metadenoiser": "CC-BY-NC-4.0", "callenhancer": "CC-BY-NC-4.0"}


def _entry(package: str):
    """Registry entry for an engine alias, or exit listing the valid ones."""
    from audiosronnx import ENGINE_REGISTRY

    try:
        return ENGINE_REGISTRY[package]
    except KeyError:
        raise SystemExit(
            f"unknown audiosronnx engine '{package}'. "
            f"List them with: {Path(__file__).name} packages"
        ) from None


def _kind(package: str) -> str:
    """`denoise` or `sr` — which entry point this engine belongs to."""
    kind = getattr(_entry(package), "kind", "sr")
    return "denoise" if kind in ("denoise", "enhance") else "sr"


def _load(package: str):
    """Instantiate the engine through the entry point its kind requires."""
    from audiosronnx import load_denoise, load_sr

    return (load_denoise if _kind(package) == "denoise" else load_sr)(package)


def prep(package: str = "dpdfnet", force: bool = False) -> None:
    """Download one engine's weights and prove the graph runs.

    A quarter second of noise is pushed through the model: downloading an ONNX
    file says nothing about whether onnxruntime can execute it, and this is the
    cheapest thing that does.

    Args:
        package: engine alias (`dpdfnet`, `lavasr`, `voicefixer`, ...).
        force: accepted for symmetry with the other wrappers; downloads are
            cached by revision, so this changes nothing.
    """
    import numpy as np

    engine = _load(package)
    rate = int(getattr(_entry(package), "input_sample_rate", 0) or 16000)
    probe = (np.random.default_rng(0).standard_normal(rate // 4) * 0.05).astype("float32")
    if _kind(package) == "denoise":
        out, out_rate = engine.denoise(probe, rate)
    else:
        out, out_rate = engine.upscale(probe, rate)
    print(f"[prep] ready: {package} -> {len(out)} samples @ {out_rate} Hz")


def run(
    audio: str = "",
    package: str = "dpdfnet",
    output: str = "output.wav",
    task: str = "",
    force: bool = False,
    play: bool = False,
    **_ignored: object,
) -> None:
    """Clean up or band-extend one clip: audio in, audio out.

    Args:
        audio: Input WAV. Any rate; the engine resamples on the way in.
        package: engine alias. Its registry `kind` decides the operation, so a
            denoiser cannot be asked to extend, or the other way round.
        output: Output WAV path.
        task: `denoise` or `sr`. Optional — the engine's kind already implies it;
            a mismatch is an error rather than a silent reinterpretation.
        force: Overwrite an existing output file.
        play: Play the result with afplay instead of reporting the path.
    """
    import time

    from rich.console import Console
    from rich.panel import Panel

    if _ignored:
        # The batch runner sends every wrapper the same arguments. Dropping one
        # it does not understand would produce a plausible file from the wrong
        # request — a `--task_route` silently ignored is worse than a crash.
        raise SystemExit(
            f"{Path(__file__).name} does not accept: {', '.join(sorted(_ignored))}"
        )

    console = Console()
    out_path = Path(output).resolve()
    if not force and out_path.exists() and out_path.stat().st_size > 0:
        console.print(
            f"[yellow]Skipped:[/yellow] {output} already exists (use --force to overwrite)"
        )
        return
    if not audio:
        raise SystemExit("--audio is required")

    kind = _kind(package)
    if task and task != kind:
        raise SystemExit(
            f"engine '{package}' does {kind}, not '{task}' — pick another engine"
        )

    entry = _entry(package)
    console.print(
        Panel.fit(
            f"[bold cyan]audiosronnx[/bold cyan] v{VERSION}  "
            f"engine=[yellow]{package}[/yellow]  task=[yellow]{kind}[/yellow]\n"
            f"{getattr(entry, 'description', '')}",
            border_style="cyan",
        )
    )

    engine = _load(package)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    if kind == "denoise":
        engine.denoise_file(str(Path(audio).resolve()), str(out_path))
    else:
        engine.upscale_file(str(Path(audio).resolve()), str(out_path))
    elapsed = time.monotonic() - t0

    if not out_path.exists() or out_path.stat().st_size == 0:
        console.print(f"[red]{package} wrote nothing to {out_path}[/red]")
        raise SystemExit(1)

    import soundfile as sf

    details = sf.info(str(out_path))
    if play:
        import subprocess

        subprocess.run(["afplay", str(out_path)], check=True)
    else:
        rtf = elapsed / details.duration if details.duration else 0.0
        console.print(
            f"[green]Saved[/green] [bold]{out_path}[/bold] "
            f"({details.duration:.1f}s @ {details.samplerate} Hz, rtf={rtf:.2f})"
        )


def info(package: str = "dpdfnet") -> None:
    """Print one-line JSON with engine metadata."""
    entry = _entry(package)
    print(
        json.dumps(
            {
                "name": package,
                "family": "audiosronnx",
                "repo": f"TigreGotico/audiosronnx-{package}",
                "stack": "onnx",
                "quant": "f32",
                "op": _kind(package),
                "input_sample_rate": getattr(entry, "input_sample_rate", None),
                "output_sample_rate": getattr(entry, "output_sample_rate", None),
                "license": getattr(entry, "license", ""),
                "description": getattr(entry, "description", ""),
            },
            separators=(",", ":"),
        )
    )


def packages() -> None:
    """List every engine: alias, job, rates, licence."""
    from audiosronnx import ENGINE_REGISTRY

    for alias, entry in sorted(ENGINE_REGISTRY.items()):
        lic = getattr(entry, "license", "") or ""
        flag = " (noncommercial)" if alias in NONCOMMERCIAL else ""
        print(
            f"{alias:16} {_kind(alias):8} "
            f"{str(getattr(entry, 'input_sample_rate', '') or 'any'):>6} -> "
            f"{str(getattr(entry, 'output_sample_rate', '') or '?'):<6} {lic}{flag}"
        )


if __name__ == "__main__":
    import fire

    fire.Fire({"prep": prep, "run": run, "info": info, "packages": packages})
