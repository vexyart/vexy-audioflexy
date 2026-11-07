#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = ["platformdirs>=4", "fire", "rich", "mlx", "numpy", "scipy", "soundfile", "huggingface-hub"]
# ///
# this_file: src/vexy_audioflexy/tools/fx/tools/aud-resemble.py
"""Resemble Enhance on MLX — denoise plus generative restoration, 44.1 kHz out.

`ResembleAI/resemble-enhance` ships a DeepSpeed checkpoint and a package that
wants deepspeed, which has no working macOS wheel; issue 302 had to skip it.
`multimodalart/resemble-enhance-mlx` is the same model converted for MLX, and
unusually the *inference code is inside the weights repo* — a `resemble_enhance_mlx/`
package sitting beside the safetensors.

So this wrapper downloads the repo, puts it on `sys.path`, and calls it. That is
one more variation on "the code is not on PyPI" than `aud-mossformersr.py` (which
clones from GitHub), and it is handled the same way: fetch, then run in this
script's own environment.

`lambd` sets denoiser strength in [0, 1]; the model resamples to 44.1 kHz and
chunks long input itself.
"""

from __future__ import annotations


import os
from pathlib import Path
from platformdirs import user_cache_path, user_data_path
RUNTIME_ROOT = Path(os.environ.get("VAFX_AUDIO_RUNTIME_ROOT", user_data_path("vexy-audioflexy") / "audio" )).expanduser()
CACHE_ROOT = Path(os.environ.get("VAFX_AUDIO_CACHE_ROOT", user_cache_path("vexy-audioflexy") / "audio")).expanduser()
import json
import sys
from pathlib import Path

VERSION = "1.0.0"

WEIGHTS_REPO = "multimodalart/resemble-enhance-mlx"
CHECKPOINT = "resemble-enhance-mlx.safetensors"
PACKAGES = {"resemble-enhance": {"repo": WEIGHTS_REPO, "rate": 44100}}


def _check(package: str) -> None:
    if package not in PACKAGES:
        raise SystemExit(f"unknown package '{package}'. Known: {', '.join(PACKAGES)}")


def _snapshot() -> Path:
    """Download the repo — weights *and* the Python package inside it."""
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(WEIGHTS_REPO))


def _load(lambd: float):
    """Import `resemble_enhance_mlx` out of the downloaded repo and load weights."""
    directory = _snapshot()
    if str(directory) not in sys.path:
        # The package ships inside the model repo rather than on PyPI, so the
        # download directory is the import root.
        sys.path.insert(0, str(directory))
    import resemble_enhance_mlx as rem

    return rem, rem.load(str(directory / CHECKPOINT), lambd=lambd)


def prep(package: str = "resemble-enhance", force: bool = False) -> None:
    """Download the repo and prove the model loads and runs.

    Args:
        package: the only package here, `resemble-enhance`.
        force: accepted for symmetry; huggingface_hub caches by revision.
    """
    import numpy as np

    _check(package)
    rem, model = _load(0.5)
    probe = (np.random.default_rng(0).standard_normal(16000) * 0.05).astype("float32")
    out, rate = rem.enhance(model, probe, 16000)
    print(f"[prep] ready: {package} -> {np.asarray(out).size} samples @ {rate} Hz")


def run(
    audio: str = "",
    package: str = "resemble-enhance",
    output: str = "output.wav",
    task: str = "",
    lambd: float = 0.5,
    force: bool = False,
    play: bool = False,
    **_ignored: object,
) -> None:
    """Restore one clip: audio in, 44.1 kHz audio out.

    Args:
        audio: Input WAV, any rate.
        package: the only package here, `resemble-enhance`.
        output: Output WAV path.
        task: `denoise`. Accepted so the batch runner can pass its usual arguments.
        lambd: Denoiser strength in [0, 1]; 0 leaves the noise for the enhancer
            to deal with, 1 removes it before restoration.
        force: Overwrite an existing output file.
        play: Play the result with afplay instead of reporting the path.
    """
    import time

    import numpy as np
    import soundfile as sf
    from rich.console import Console
    from rich.panel import Panel

    if _ignored:
        raise SystemExit(
            f"{Path(__file__).name} does not accept: {', '.join(sorted(_ignored))}"
        )

    _check(package)
    console = Console()
    out_path = Path(output).resolve()
    if not force and out_path.exists() and out_path.stat().st_size > 0:
        console.print(
            f"[yellow]Skipped:[/yellow] {output} already exists (use --force to overwrite)"
        )
        return
    if not audio:
        raise SystemExit("--audio is required")
    if task and task != "denoise":
        raise SystemExit(f"'{package}' does denoise, not '{task}'")

    console.print(
        Panel.fit(
            f"[bold cyan]resemble-enhance[/bold cyan] v{VERSION}  "
            f"package=[yellow]{package}[/yellow]  task=[yellow]denoise[/yellow]\n"
            f"weights=[yellow]{WEIGHTS_REPO}[/yellow]  lambd={lambd}",
            border_style="cyan",
        )
    )

    data, in_rate = sf.read(str(Path(audio).resolve()), dtype="float32", always_2d=True)
    samples = np.mean(data, axis=1)
    rem, model = _load(lambd)
    t0 = time.monotonic()
    enhanced, out_rate = rem.enhance(model, samples, int(in_rate))
    elapsed = time.monotonic() - t0

    enhanced = np.asarray(enhanced).astype("float32").reshape(-1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), enhanced, int(out_rate))
    duration = enhanced.size / int(out_rate)
    if play:
        import subprocess

        subprocess.run(["afplay", str(out_path)], check=True)
    else:
        console.print(
            f"[green]Saved[/green] [bold]{out_path}[/bold] "
            f"({duration:.1f}s @ {out_rate} Hz, rtf={elapsed / duration if duration else 0:.2f})"
        )


def info(package: str = "resemble-enhance") -> None:
    """Print one-line JSON with package metadata."""
    _check(package)
    print(
        json.dumps(
            {
                "name": package,
                "family": "resemble-enhance",
                "repo": WEIGHTS_REPO,
                "stack": "mlx",
                "quant": "f32",
                "op": "dn",
                "sample_rate": 44100,
            },
            separators=(",", ":"),
        )
    )


def packages() -> None:
    """List the packages this wrapper drives."""
    for name, spec in PACKAGES.items():
        print(f"{name:18} denoise {spec['rate']:>7}  {spec['repo']}")


if __name__ == "__main__":
    import fire

    fire.Fire({"prep": prep, "run": run, "info": info, "packages": packages})
