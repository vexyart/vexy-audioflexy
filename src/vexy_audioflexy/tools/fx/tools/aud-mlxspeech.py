#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.13"
# dependencies = ["platformdirs>=4", 
#   "fire",
#   "rich",
#   "soundfile",
#   "numpy",
#   "mlx-speech @ git+https://github.com/appautomaton/mlx-speech",
# ]
# ///
# this_file: src/vexy_audioflexy/tools/fx/tools/aud-mlxspeech.py
"""Driver for the audio-in/audio-out models in `mlx-speech`.

`mlx-speech` is mostly a TTS and ASR toolkit, and neither belongs in fx — the
sibling projects cover those. Two of its models are audio in, audio out, and
those are the ones registered here:

| package | op | what it does |
|---|---|---|
| `re-use-semamba` | `dn` | NVIDIA RE-USE / SEMamba universal speech enhancement, pure MLX |
| `step-audio-editx` | `edit` | Step-Audio-EditX: re-speaks a clip with an edited attribute |

`re-use-semamba` is the reason this wrapper exists. `nvidia/RE-USE` itself needs
`mamba-ssm` and its CUDA kernels, so issue 302 had to skip it; this conversion
runs the same weights as MLX arrays on the GPU. The weights stay **NSCLv1
non-commercial** — that restriction travels with them.

Step-Audio-EditX needs the source transcript as well as the audio: it re-speaks
the line with one attribute changed (emotion, style, speed), so it is the same
shape as VeVo2's editing route, not a denoiser.

PyPI's `mlx-speech` 0.1.0 does not carry the RE-USE module; the dependency above
pins the GitHub main branch, which does. Python 3.13 is upstream's floor.
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

PACKAGES: dict[str, dict[str, object]] = {
    "re-use-semamba": {
        "repo": "appautomaton/re-use-semamba-mlx",
        "op": "denoise",
        "rate": 0,  # rate-adaptive: output rate follows the input
        "license": "NSCLv1 (non-commercial)",
        "note": "RE-USE / SEMamba; the MLX conversion of a model whose original needs CUDA",
    },
    "step-audio-editx": {
        "repo": "appautomaton/step-audio-editx-8bit-mlx",
        "op": "edit",
        "rate": 24000,
        "license": "Apache-2.0",
        "note": "Re-speaks a clip with one attribute edited; needs the source transcript",
    },
}

DEFAULT_EDIT = ("emotion", "happy")


def _spec(package: str) -> dict[str, object]:
    try:
        return PACKAGES[package]
    except KeyError:
        raise SystemExit(
            f"unknown mlx-speech package '{package}'. Known: {', '.join(PACKAGES)}"
        ) from None


def _weights(package: str) -> Path:
    """Local directory holding the package's weights (downloads once)."""
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(str(_spec(package)["repo"])))


def _load_reuse(package: str):
    from mlx_speech.generation.reuse import REUSEEnhancer, load_mlx_semamba

    return REUSEEnhancer(load_mlx_semamba(_weights(package)))


def prep(package: str = "re-use-semamba", force: bool = False) -> None:
    """Download one package's weights, and for RE-USE prove the graph runs.

    Args:
        package: `re-use-semamba` or `step-audio-editx`.
        force: accepted for symmetry; huggingface_hub caches by revision.
    """
    import mlx.core as mx
    import numpy as np

    directory = _weights(package)
    if _spec(package)["op"] == "denoise":
        enhancer = _load_reuse(package)
        rate = int(_spec(package)["rate"]) or 16000  # type: ignore[arg-type]
        probe = (np.random.default_rng(0).standard_normal(rate // 2) * 0.05).astype("float32")
        out = np.asarray(enhancer.enhance(mx.array(probe), rate))
        print(f"[prep] ready: {package} -> {out.size} samples @ {rate} Hz")
        return
    # Step-Audio-EditX is ~4 GB across eight artifacts; downloading them is the
    # slow part, and loading the stack costs as much as a real run.
    print(f"[prep] ready: {package} -> {directory}")


def run(
    audio: str = "",
    package: str = "re-use-semamba",
    output: str = "output.wav",
    task: str = "",
    text: str = "",
    edit_type: str = "",
    edit_info: str = "",
    force: bool = False,
    play: bool = False,
    **_ignored: object,
) -> None:
    """Enhance or edit one clip: audio in, audio out.

    Args:
        audio: Input WAV.
        package: `re-use-semamba` or `step-audio-editx`.
        output: Output WAV path.
        task: `denoise` or `edit`; a mismatch is an error, not a reinterpretation.
        text: Transcript of *audio*. Required by `step-audio-editx`, unused by
            the enhancer.
        edit_type: What to change — `emotion`, `style`, `speed` (edit only).
        edit_info: The value to change it to, e.g. `happy` (edit only).
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

    console = Console()
    spec = _spec(package)
    op = str(spec["op"])
    out_path = Path(output).resolve()
    if not force and out_path.exists() and out_path.stat().st_size > 0:
        console.print(
            f"[yellow]Skipped:[/yellow] {output} already exists (use --force to overwrite)"
        )
        return
    if not audio:
        raise SystemExit("--audio is required")
    if task and task != op:
        raise SystemExit(f"mlx-speech package '{package}' does {op}, not '{task}'")

    console.print(
        Panel.fit(
            f"[bold cyan]mlx-speech[/bold cyan] v{VERSION}  "
            f"package=[yellow]{package}[/yellow]  task=[yellow]{op}[/yellow]\n"
            f"weights=[yellow]{spec['repo']}[/yellow]  {spec['license']}",
            border_style="cyan",
        )
    )

    data, in_rate = sf.read(str(Path(audio).resolve()), dtype="float32", always_2d=True)
    samples = np.mean(data, axis=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()

    if op == "denoise":
        import mlx.core as mx

        enhancer = _load_reuse(package)
        enhanced = np.asarray(enhancer.enhance(mx.array(samples), int(in_rate)))
        enhanced = enhanced.astype("float32").reshape(-1)
        # RE-USE is rate-adaptive: it returns audio at the rate it was given, so
        # writing the model's nominal 16 kHz would stretch the clip.
        sf.write(str(out_path), enhanced, int(in_rate))
        duration = enhanced.size / int(in_rate)
    else:
        import mlx_speech

        if not text:
            raise SystemExit("--text (the transcript of --audio) is required to edit")
        kind = edit_type or DEFAULT_EDIT[0]
        info = edit_info or DEFAULT_EDIT[1]
        model = mlx_speech.tts.load(str(spec["repo"]))
        result = model.generate(
            "",
            reference_audio=str(Path(audio).resolve()),
            reference_text=text,
            edit_type=kind,
            edit_info=info,
        )
        waveform = np.asarray(result.waveform).astype("float32").reshape(-1)
        sf.write(str(out_path), waveform, result.sample_rate)
        duration = waveform.size / result.sample_rate
        console.print(f"[dim]edit: {kind}={info}[/dim]")

    elapsed = time.monotonic() - t0
    if play:
        import subprocess

        subprocess.run(["afplay", str(out_path)], check=True)
    else:
        rtf = elapsed / duration if duration else 0.0
        console.print(
            f"[green]Saved[/green] [bold]{out_path}[/bold] ({duration:.1f}s, rtf={rtf:.2f})"
        )


def info(package: str = "re-use-semamba") -> None:
    """Print one-line JSON with package metadata."""
    spec = _spec(package)
    print(
        json.dumps(
            {
                "name": package,
                "family": "mlx-speech",
                "repo": spec["repo"],
                "stack": "mlx",
                "quant": "int8" if "8bit" in str(spec["repo"]) else "f32",
                "op": "dn" if spec["op"] == "denoise" else "edit",
                "sample_rate": spec["rate"],
                "license": spec["license"],
            },
            separators=(",", ":"),
        )
    )


def packages() -> None:
    """List the audio-in/audio-out packages this wrapper drives."""
    for name, spec in PACKAGES.items():
        rate = spec["rate"] or "input"
        print(f"{name:18} {spec['op']:8} {str(rate):>6}  {spec['repo']}  {spec['license']}")


if __name__ == "__main__":
    import fire

    fire.Fire({"prep": prep, "run": run, "info": info, "packages": packages})
