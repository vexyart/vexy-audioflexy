#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = ["platformdirs>=4", "fire", "rich", "huggingface-hub"]
# ///
# this_file: src/vexy_audioflexy/tools/fx/tools/aud-mossformersr.py
"""MossFormer2 super-resolution (16 kHz -> 48 kHz) on MLX.

The odd one out among the wrappers. `starkdmi/MossFormer2_SR_48K_MLX` ships
weights, and its inference code lives in a GitHub repo of loose scripts rather
than a pip package, so there is nothing to `import`. The scripts are complete and
they run: this wrapper clones them under `local/` and drives `python/generate.py`
in its own uv environment, which is the same subprocess isolation `aud_all.py`
already applies one level up.

It earns the extra machinery by being the only *super-resolution* model on MLX
here — audiosronnx covers bandwidth extension on the CPU, `mlx-audio` covers
denoising on the GPU, and this fills the remaining corner.

`local/` is scratch space: delete it and the next `prep` clones it again.
"""

from __future__ import annotations


import os
from pathlib import Path
from platformdirs import user_cache_path, user_data_path
RUNTIME_ROOT = Path(os.environ.get("VAFX_AUDIO_RUNTIME_ROOT", user_data_path("vexy-audioflexy") / "audio" )).expanduser()
CACHE_ROOT = Path(os.environ.get("VAFX_AUDIO_CACHE_ROOT", user_cache_path("vexy-audioflexy") / "audio")).expanduser()
import json
import subprocess
from pathlib import Path

VERSION = "1.0.0"

ROOT = Path(__file__).resolve().parent.parent
REPO_URL = "https://github.com/starkdmi/mossformer_sr_mlx.git"
CHECKOUT = RUNTIME_ROOT / "engines/mossformer_sr_mlx"
GENERATOR = CHECKOUT / "python/generate.py"
WEIGHTS_REPO = "starkdmi/MossFormer2_SR_48K_MLX"

PACKAGES = {"mossformer2-sr-48k": {"repo": WEIGHTS_REPO, "rate": 48000}}

# generate.py's imports; requirements.txt pins nothing, and neither does this.
RUNTIME_DEPS = ["mlx", "numpy", "librosa", "soundfile", "huggingface-hub"]


def _check(package: str) -> None:
    if package not in PACKAGES:
        raise SystemExit(
            f"unknown package '{package}'. Known: {', '.join(PACKAGES)}"
        )


def _clone(force: bool = False) -> Path:
    """Clone or refresh the upstream scripts. Idempotent."""
    if CHECKOUT.exists() and not force:
        return CHECKOUT
    CHECKOUT.parent.mkdir(parents=True, exist_ok=True)
    if CHECKOUT.exists():
        subprocess.run(["git", "-C", str(CHECKOUT), "pull", "--ff-only"], check=True)
    else:
        subprocess.run(
            ["git", "clone", "--depth", "1", REPO_URL, str(CHECKOUT)], check=True
        )
    return CHECKOUT


def _generate_cmd(audio: Path, output: Path) -> list[str]:
    """The uv invocation that runs generate.py with its own dependencies."""
    cmd = ["uv", "run", "--no-project", "--quiet"]
    for dep in RUNTIME_DEPS:
        cmd += ["--with", dep]
    return cmd + [
        "python",
        str(GENERATOR),
        "--input",
        str(audio),
        "--output",
        str(output),
    ]


def prep(package: str = "mossformer2-sr-48k", force: bool = False) -> None:
    """Clone the inference scripts and cache the weights.

    Args:
        package: the only package here, `mossformer2-sr-48k`.
        force: re-pull the checkout even when it is already present.
    """
    _check(package)
    from huggingface_hub import hf_hub_download

    _clone(force)
    if not GENERATOR.exists():
        raise SystemExit(f"upstream layout changed: {GENERATOR} is missing")
    for filename in ("model_fp32.safetensors", "config.json"):
        hf_hub_download(repo_id=WEIGHTS_REPO, filename=filename)
    print(f"[prep] ready: {package} -> {CHECKOUT}")


def run(
    audio: str = "",
    package: str = "mossformer2-sr-48k",
    output: str = "output.wav",
    task: str = "",
    force: bool = False,
    play: bool = False,
    **_ignored: object,
) -> None:
    """Band-extend one clip to 48 kHz on the GPU: audio in, audio out.

    Args:
        audio: Input WAV; librosa resamples it to 48 kHz on the way in.
        package: the only package here, `mossformer2-sr-48k`.
        output: Output WAV path.
        task: `sr`. Accepted so the batch runner can pass its usual arguments.
        force: Overwrite an existing output file.
        play: Play the result with afplay instead of reporting the path.
    """
    import time

    from rich.console import Console
    from rich.panel import Panel

    _check(package)
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
    if task and task != "sr":
        raise SystemExit(f"'{package}' does sr, not '{task}'")
    if not GENERATOR.exists():
        prep(package)

    console.print(
        Panel.fit(
            f"[bold cyan]mossformer2-sr[/bold cyan] v{VERSION}  "
            f"package=[yellow]{package}[/yellow]  task=[yellow]sr[/yellow]\n"
            f"weights=[yellow]{WEIGHTS_REPO}[/yellow]  16 kHz -> 48 kHz",
            border_style="cyan",
        )
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    result = subprocess.run(
        _generate_cmd(Path(audio).resolve(), out_path),
        capture_output=True,
        text=True,
        # generate.py imports its siblings by bare name, so it only resolves
        # when run from its own directory.
        cwd=str(GENERATOR.parent),
    )
    elapsed = time.monotonic() - t0
    if result.returncode != 0 or not out_path.exists():
        console.print("[red]generate.py failed[/red]")
        console.print("\n".join((result.stdout + result.stderr).splitlines()[-12:]))
        raise SystemExit(1)

    if play:
        subprocess.run(["afplay", str(out_path)], check=True)
    else:
        console.print(
            f"[green]Saved[/green] [bold]{out_path}[/bold] ({elapsed:.1f}s wall)"
        )


def info(package: str = "mossformer2-sr-48k") -> None:
    """Print one-line JSON with package metadata."""
    _check(package)
    print(
        json.dumps(
            {
                "name": package,
                "family": "mossformer2_sr",
                "repo": WEIGHTS_REPO,
                "code": REPO_URL,
                "stack": "mlx",
                "quant": "f32",
                "op": "sr",
                "sample_rate": 48000,
            },
            separators=(",", ":"),
        )
    )


def packages() -> None:
    """List the packages this wrapper drives."""
    for name, spec in PACKAGES.items():
        print(f"{name:22} sr  {spec['rate']:>6}  {spec['repo']}")


if __name__ == "__main__":
    import fire

    fire.Fire({"prep": prep, "run": run, "info": info, "packages": packages})
