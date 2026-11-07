#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = ["platformdirs>=4", 
#   "fire",
#   "rich",
#   "soundfile",
#   "numpy",
#   "scipy",
#   "mlx-audio",
#   "sentencepiece",
# ]
# ///
# this_file: src/vexy_audioflexy/tools/fx/tools/aud-mlxaudio.py
"""Driver for the MLX speech-enhancement models — inference on the GPU, in MLX.

Where `aud-audiosronnx.py` runs ONNX graphs on the CPU, this wrapper runs the
same class of model as MLX arrays on Apple Silicon's GPU. `mlx-audio` supplies
the inference code; HuggingFace supplies weights converted from the PyTorch
originals, so a repo of bare `safetensors` becomes runnable without porting a
front-end by hand.

| package          | weights                          | rate    | what it does      |
|------------------|----------------------------------|---------|-------------------|
| `deepfilternet3` | mlx-community/DeepFilterNet-mlx  | 48 kHz  | noise suppression |
| `deepfilternet2` | mlx-community/DeepFilterNet-mlx  | 48 kHz  | noise suppression |
| `deepfilternet1` | mlx-community/DeepFilterNet-mlx  | 48 kHz  | noise suppression |
| `deepfilternet3-iky1e` | iky1e/DeepFilterNet3-MLX   | 48 kHz  | noise suppression |
| `mossformer2-se` | starkdmi/MossFormer2-SE-fp16     | 48 kHz  | noise suppression |

`deepfilternet3` and `deepfilternet3-iky1e` are the same upstream checkpoint
converted twice, by different people; both are registered because the issue names
both repos and "they should agree" is a claim worth measuring rather than assuming.

MLX allocates on the unified-memory GPU, so these compete for the same memory as
any audio.cpp Metal run — do not run both batches at once.
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

# package -> (module, class, repo, subfolder, sample rate)
PACKAGES: dict[str, dict[str, object]] = {
    "deepfilternet3": {
        "module": "mlx_audio.sts.models.deepfilternet",
        "cls": "DeepFilterNetModel",
        "repo": "mlx-community/DeepFilterNet-mlx",
        "subfolder": "v3",
        "rate": 48000,
    },
    "deepfilternet2": {
        "module": "mlx_audio.sts.models.deepfilternet",
        "cls": "DeepFilterNetModel",
        "repo": "mlx-community/DeepFilterNet-mlx",
        "subfolder": "v2",
        "rate": 48000,
    },
    "deepfilternet1": {
        "module": "mlx_audio.sts.models.deepfilternet",
        "cls": "DeepFilterNetModel",
        "repo": "mlx-community/DeepFilterNet-mlx",
        "subfolder": "v1",
        "rate": 48000,
    },
    "deepfilternet3-iky1e": {
        "module": "mlx_audio.sts.models.deepfilternet",
        "cls": "DeepFilterNetModel",
        "repo": "iky1e/DeepFilterNet3-MLX",
        "subfolder": "",
        "rate": 48000,
    },
    "mossformer2-se": {
        "module": "mlx_audio.sts.models.mossformer2_se",
        "cls": "MossFormer2SEModel",
        "repo": "starkdmi/MossFormer2-SE-fp16",
        "subfolder": "",
        "rate": 48000,
    },
}


def _spec(package: str) -> dict[str, object]:
    try:
        return PACKAGES[package]
    except KeyError:
        raise SystemExit(
            f"unknown mlx-audio package '{package}'. "
            f"Known: {', '.join(sorted(PACKAGES))}"
        ) from None


def _load(package: str):
    """Import the model class and pull its weights (cached by huggingface_hub)."""
    import importlib
    import inspect

    spec = _spec(package)
    module = importlib.import_module(str(spec["module"]))
    cls = getattr(module, str(spec["cls"]))
    kwargs: dict[str, object] = {}
    if "subfolder" in inspect.signature(cls.from_pretrained).parameters:
        # DeepFilterNetModel defaults to subfolder="v3" for the three-version
        # mlx-community repo; a single-version repo has no such directory, so the
        # default has to be cleared rather than merely left alone.
        kwargs["subfolder"] = spec["subfolder"] or None
    return cls.from_pretrained(str(spec["repo"]), **kwargs)


def _enhance(model, audio, rate: int):
    """Call whichever enhancement entry point this model class exposes."""
    if hasattr(model, "enhance_array"):
        return model.enhance_array(audio)
    return model.enhance(audio, rate)


def prep(package: str = "deepfilternet3", force: bool = False) -> None:
    """Download one package's weights and prove the graph runs.

    A quarter second of noise goes through the model: having the safetensors on
    disk says nothing about whether MLX can build and execute the network.

    Args:
        package: package name from the table above.
        force: accepted for symmetry with the other wrappers; huggingface_hub
            already caches by revision.
    """
    import numpy as np

    spec = _spec(package)
    rate = int(spec["rate"])  # type: ignore[arg-type]
    model = _load(package)
    probe = (np.random.default_rng(0).standard_normal(rate // 4) * 0.05).astype("float32")
    out = np.asarray(_enhance(model, probe, rate))
    print(f"[prep] ready: {package} -> {out.size} samples @ {rate} Hz")


def run(
    audio: str = "",
    package: str = "deepfilternet3",
    output: str = "output.wav",
    task: str = "",
    force: bool = False,
    play: bool = False,
    **_ignored: object,
) -> None:
    """Denoise one clip on the MLX GPU: audio in, audio out.

    Args:
        audio: Input WAV; resampled to the model's rate on the way in.
        package: package name from the table above.
        output: Output WAV path.
        task: `denoise`. Accepted (and checked) so the batch runner can pass the
            same arguments it passes every other wrapper.
        force: Overwrite an existing output file.
        play: Play the result with afplay instead of reporting the path.
    """
    import time

    import numpy as np
    import soundfile as sf
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
    if task and task != "denoise":
        raise SystemExit(f"mlx-audio package '{package}' does denoise, not '{task}'")

    spec = _spec(package)
    rate = int(spec["rate"])  # type: ignore[arg-type]
    console.print(
        Panel.fit(
            f"[bold cyan]mlx-audio[/bold cyan] v{VERSION}  "
            f"package=[yellow]{package}[/yellow]  task=[yellow]denoise[/yellow]\n"
            f"weights=[yellow]{spec['repo']}"
            f"{'/' + str(spec['subfolder']) if spec['subfolder'] else ''}[/yellow]",
            border_style="cyan",
        )
    )

    data, in_rate = sf.read(str(Path(audio).resolve()), dtype="float32", always_2d=True)
    samples = np.mean(data, axis=1)
    if in_rate != rate:
        # Polyphase resampling keeps the band edges clean; these models are
        # trained at 48 kHz and are not rate-adaptive.
        from math import gcd

        from scipy.signal import resample_poly  # noqa: PLC0415 - heavy, and only needed here

        g = gcd(int(in_rate), rate)
        samples = resample_poly(samples, rate // g, int(in_rate) // g).astype("float32")

    model = _load(package)
    t0 = time.monotonic()
    enhanced = np.asarray(_enhance(model, samples, rate)).astype("float32").reshape(-1)
    elapsed = time.monotonic() - t0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), enhanced, rate)
    duration = enhanced.size / rate
    if play:
        import subprocess

        subprocess.run(["afplay", str(out_path)], check=True)
    else:
        console.print(
            f"[green]Saved[/green] [bold]{out_path}[/bold] "
            f"({duration:.1f}s @ {rate} Hz, rtf={elapsed / duration if duration else 0:.2f})"
        )


def info(package: str = "deepfilternet3") -> None:
    """Print one-line JSON with package metadata."""
    spec = _spec(package)
    print(
        json.dumps(
            {
                "name": package,
                "family": "mlx-audio",
                "repo": spec["repo"],
                "subfolder": spec["subfolder"],
                "stack": "mlx",
                "quant": "fp16" if "fp16" in str(spec["repo"]) else "f32",
                "op": "denoise",
                "sample_rate": spec["rate"],
            },
            separators=(",", ":"),
        )
    )


def packages() -> None:
    """List every registered MLX package."""
    for name, spec in sorted(PACKAGES.items()):
        sub = f"/{spec['subfolder']}" if spec["subfolder"] else ""
        print(f"{name:22} denoise  {spec['rate']:>6}  {spec['repo']}{sub}")


if __name__ == "__main__":
    import fire

    fire.Fire({"prep": prep, "run": run, "info": info, "packages": packages})
