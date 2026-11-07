#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = ["platformdirs>=4", "fire", "rich", "sherpa-onnx", "soundfile", "numpy", "httpx"]
# ///
# this_file: src/vexy_audioflexy/tools/fx/tools/aud-sherpaonnx.py
"""sherpa-onnx speech enhancement and source separation: audio in, audio out.

Two jobs, two verbs, and they do not share engines:

| verb       | job                | engines                                    |
|------------|--------------------|--------------------------------------------|
| `run`      | denoising          | gtcrn, dpdfnet (baseline/2/4/8, 48 kHz hr) |
| `separate` | source separation  | spleeter 2-stems, UVR MDX-Net              |

`run` is the batch-runner verb — `aud_all.py` feeds it the noisy fixture and
scores the result. `separate` is deliberately *not* registered: it emits several
stems from music, so the harness's single-output, single-reference scoring has
nothing to compare it against. It is still a first-class CLI verb.

`gtcrn` and `dpdfnet` also exist in `aud-audiosronnx.py`. These are different
implementations of the same papers — a sherpa-onnx C++ front-end against the
k2-fsa exports, not the TigreGotico Python ones — so both are registered and the
scores are the comparison.

Weights are GitHub release assets: bare `.onnx` files for the denoisers and UVR,
a tarball for spleeter.
"""

from __future__ import annotations


import os
from pathlib import Path
import runpy
model_path = runpy.run_path(str(Path(__file__).with_name("_audioflexy_paths.py")))["model_path"]
from platformdirs import user_cache_path, user_data_path
RUNTIME_ROOT = Path(os.environ.get("VAFX_AUDIO_RUNTIME_ROOT", user_data_path("vexy-audioflexy") / "audio" )).expanduser()
CACHE_ROOT = Path(os.environ.get("VAFX_AUDIO_CACHE_ROOT", user_cache_path("vexy-audioflexy") / "audio")).expanduser()
import json
import os
import time
from pathlib import Path

VERSION = "1.0.0"

_REL = "https://github.com/k2-fsa/sherpa-onnx/releases/download"
_ENH = f"{_REL}/speech-enhancement-models"
_SEP = f"{_REL}/source-separation-models"

# name -> job, asset URL, family, notes. `job` decides which verb accepts it.
PACKAGES: dict[str, dict] = {
    "gtcrn": {
        "job": "denoise",
        "family": "gtcrn",
        "url": f"{_ENH}/gtcrn_simple.onnx",
        "quant": "f32",
        "repo": "Xiaobin-Rong/gtcrn (sherpa-onnx export)",
    },
    "dpdfnet-baseline": {
        "job": "denoise",
        "family": "dpdfnet",
        "url": f"{_ENH}/dpdfnet_baseline.onnx",
        "quant": "f32",
        "repo": "k2-fsa dpdfnet baseline",
    },
    "dpdfnet2": {
        "job": "denoise",
        "family": "dpdfnet",
        "url": f"{_ENH}/dpdfnet2.onnx",
        "quant": "f32",
        "repo": "k2-fsa dpdfnet2",
    },
    "dpdfnet4": {
        "job": "denoise",
        "family": "dpdfnet",
        "url": f"{_ENH}/dpdfnet4.onnx",
        "quant": "f32",
        "repo": "k2-fsa dpdfnet4",
    },
    "dpdfnet8": {
        "job": "denoise",
        "family": "dpdfnet",
        "url": f"{_ENH}/dpdfnet8.onnx",
        "quant": "f32",
        "repo": "k2-fsa dpdfnet8",
    },
    "dpdfnet2-48k-hr": {
        "job": "denoise",
        "family": "dpdfnet",
        "url": f"{_ENH}/dpdfnet2_48khz_hr.onnx",
        "quant": "f32",
        "repo": "k2-fsa dpdfnet2 48 kHz high-resolution",
    },
    "spleeter-2stems": {
        "job": "separate",
        "family": "spleeter",
        "url": f"{_SEP}/sherpa-onnx-spleeter-2stems-fp16.tar.bz2",
        "quant": "fp16",
        "stems": ["vocals", "accompaniment"],
        "repo": "deezer/spleeter 2stems (sherpa-onnx export)",
    },
    "uvr-mdxnet": {
        "job": "separate",
        "family": "uvr",
        "url": f"{_SEP}/UVR_MDXNET_9482.onnx",
        "quant": "f32",
        "stems": ["vocals", "accompaniment"],
        "repo": "Anjok07/ultimatevocalremover UVR_MDXNET_9482",
    },
}


def _root() -> Path:
    """Weight cache root. `SHERPA_ONNX_MODELS_ROOT` overrides it."""
    env = os.environ.get("SHERPA_ONNX_MODELS_ROOT")
    if env:
        return Path(env).expanduser()
    return model_path("sherpa-onnx")


def _spec(package: str) -> dict:
    try:
        return PACKAGES[package]
    except KeyError:
        raise SystemExit(
            f"unknown package '{package}'. Known: {', '.join(sorted(PACKAGES))}"
        ) from None


def _download(url: str, dest: Path) -> Path:
    import httpx
    from rich.console import Console

    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    with Console().status(f"[cyan]downloading {url.rsplit('/', 1)[-1]}…"):
        with httpx.stream("GET", url, follow_redirects=True, timeout=120.0) as r:
            r.raise_for_status()
            with part.open("wb") as fh:
                for chunk in r.iter_bytes(1 << 20):
                    fh.write(chunk)
    part.replace(dest)
    return dest


def _ensure(package: str) -> tuple[Path, dict]:
    """Fetch `package`; return the model file (or its directory) and the spec."""
    spec = _spec(package)
    root = _root()
    asset = spec["url"].rsplit("/", 1)[-1]
    if asset.endswith(".onnx"):
        return _download(spec["url"], root / "_enh" / asset), spec

    import tarfile

    tarball = _download(spec["url"], root / "_dl" / asset)
    root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tarball, "r:*") as tf:
        top = {Path(m.name).parts[0] for m in tf.getmembers() if m.name.strip("./")}
        if len(top) != 1:
            raise SystemExit(f"{asset} has {len(top)} top-level entries, expected 1")
        target = root / top.pop()
        if not target.is_dir():
            tf.extractall(root, filter="data")
    return target, spec


def _denoiser(package: str):
    import sherpa_onnx

    model, spec = _ensure(package)
    if spec["job"] != "denoise":
        raise SystemExit(f"'{package}' is a {spec['job']} engine — use `separate`")
    model_cfg = sherpa_onnx.OfflineSpeechDenoiserModelConfig(
        num_threads=2, provider="cpu", debug=False
    )
    if spec["family"] == "gtcrn":
        model_cfg.gtcrn = sherpa_onnx.OfflineSpeechDenoiserGtcrnModelConfig(model=str(model))
    else:
        model_cfg.dpdfnet = sherpa_onnx.OfflineSpeechDenoiserDpdfNetModelConfig(model=str(model))
    config = sherpa_onnx.OfflineSpeechDenoiserConfig(model=model_cfg)
    if not config.validate():
        raise SystemExit(f"invalid denoiser config for {package}:\n{config}")
    return sherpa_onnx.OfflineSpeechDenoiser(config)


def _separator(package: str):
    import sherpa_onnx

    path, spec = _ensure(package)
    if spec["job"] != "separate":
        raise SystemExit(f"'{package}' is a {spec['job']} engine — use `run`")
    model_cfg = sherpa_onnx.OfflineSourceSeparationModelConfig(
        num_threads=2, provider="cpu", debug=False
    )
    if spec["family"] == "spleeter":
        vocals = sorted(path.glob("vocals*.onnx"))
        accomp = sorted(path.glob("accompaniment*.onnx"))
        if not vocals or not accomp:
            raise SystemExit(f"{path} is missing a spleeter stem model")
        model_cfg.spleeter = sherpa_onnx.OfflineSourceSeparationSpleeterModelConfig(
            vocals=str(vocals[0]), accompaniment=str(accomp[0])
        )
    else:
        model_cfg.uvr = sherpa_onnx.OfflineSourceSeparationUvrModelConfig(model=str(path))
    config = sherpa_onnx.OfflineSourceSeparationConfig(model=model_cfg)
    if not config.validate():
        raise SystemExit(f"invalid separation config for {package}:\n{config}")
    return sherpa_onnx.OfflineSourceSeparation(config)


def prep(package: str = "gtcrn", force: bool = False) -> None:
    """Download one package's weights WITHOUT processing audio. Idempotent.

    Args:
        package: Registered package name; `packages` lists them.
        force: Unused; an existing download is never re-fetched.
    """
    path, spec = _ensure(package)
    print(f"[prep] ready: {package} ({spec['family']}, {spec['job']}) -> {path}")


def run(
    audio: str = "",
    package: str = "gtcrn",
    output: str = "output.wav",
    task: str = "",
    force: bool = False,
    play: bool = False,
    **_ignored: object,
) -> None:
    """Denoise one clip: audio in, audio out.

    Args:
        audio: Input WAV, any rate; the engine resamples on the way in.
        package: Denoising engine. A separation engine is refused rather than
            reinterpreted.
        output: Output WAV path.
        task: `denoise`, optional. A mismatch is an error, not a silent
            reinterpretation.
        force: Overwrite an existing output file.
        play: Play the result with afplay instead of reporting the path.
    """
    import numpy as np
    import soundfile as sf
    from rich.console import Console
    from rich.panel import Panel

    if _ignored:
        # The batch runner sends every wrapper the same arguments. Dropping one
        # would produce a plausible file from the wrong request.
        raise SystemExit(
            f"{Path(__file__).name} does not accept: {', '.join(sorted(_ignored))}"
        )

    console = Console()
    out_path = Path(output).resolve()
    if not force and out_path.exists() and out_path.stat().st_size > 0:
        console.print(f"[yellow]Skipped:[/yellow] {output} already exists (use --force)")
        return
    if not audio:
        raise SystemExit("--audio is required")
    spec = _spec(package)
    if task and task != "denoise":
        raise SystemExit(f"engine '{package}' does denoise, not '{task}'")
    if spec["job"] != "denoise":
        raise SystemExit(f"'{package}' does {spec['job']} — use `separate`")

    console.print(
        Panel.fit(
            f"[bold cyan]sherpa-onnx[/bold cyan] v{VERSION}  "
            f"engine=[yellow]{package}[/yellow]  family=[yellow]{spec['family']}[/yellow]  "
            f"task=[yellow]denoise[/yellow]",
            border_style="cyan",
        )
    )

    denoiser = _denoiser(package)
    data, rate = sf.read(str(Path(audio).resolve()), dtype="float32", always_2d=True)
    samples = np.ascontiguousarray(data[:, 0])

    t0 = time.monotonic()
    result = denoiser.run(samples, rate)
    elapsed = time.monotonic() - t0

    out_samples = np.asarray(result.samples, dtype="float32")
    if out_samples.size == 0 or not np.any(out_samples):
        # Exit 0 with silence reads as success; the harness would score it.
        console.print(f"[red]{package} produced silence[/red]")
        raise SystemExit(1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), out_samples, result.sample_rate, subtype="PCM_16")
    duration = out_samples.size / result.sample_rate
    console.print(
        f"[green]Saved[/green] [bold]{out_path}[/bold] "
        f"({duration:.1f}s @ {result.sample_rate} Hz, rtf={elapsed / duration:.2f})"
    )
    if play:
        import subprocess

        subprocess.run(["afplay", str(out_path)], check=True)


def separate(
    audio: str = "",
    package: str = "spleeter-2stems",
    output: str = "stem.wav",
    force: bool = False,
) -> None:
    """Split one clip into stems. Writes `output` with a `.<stem>` infix per stem.

    Args:
        audio: Input WAV. Stereo is used as given; mono is duplicated, because
            these models expect two channels.
        package: Separation engine (`spleeter-2stems`, `uvr-mdxnet`).
        output: Base output path; stems become `base.vocals.wav`, `base.1.wav`, …
        force: Overwrite existing stem files.
    """
    import numpy as np
    import soundfile as sf
    from rich.console import Console

    console = Console()
    if not audio:
        raise SystemExit("--audio is required")
    spec = _spec(package)
    if spec["job"] != "separate":
        raise SystemExit(f"'{package}' does {spec['job']} — use `run`")

    base = Path(output).resolve()
    stem_names = spec.get("stems", [])
    separator = _separator(package)

    data, rate = sf.read(str(Path(audio).resolve()), dtype="float32", always_2d=True)
    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    samples = np.ascontiguousarray(np.transpose(data))

    t0 = time.monotonic()
    result = separator.process(sample_rate=rate, samples=samples)
    elapsed = time.monotonic() - t0

    written = []
    for i, stem in enumerate(result.stems):
        name = stem_names[i] if i < len(stem_names) else str(i)
        path = base.with_suffix("").with_suffix(f".{name}.wav")
        if not force and path.exists() and path.stat().st_size > 0:
            console.print(f"[yellow]Skipped:[/yellow] {path} already exists")
            continue
        # A stem carries `.data` shaped (num_channels, num_samples); the rate
        # lives on the result, not on the stem.
        out = np.transpose(np.asarray(stem.data, dtype="float32"))
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(path), out, result.sample_rate, subtype="PCM_16")
        written.append(path)
    if not written:
        raise SystemExit(f"{package} wrote no stems")
    console.print(
        f"[green]Saved[/green] {len(written)} stems in {elapsed:.1f}s: "
        + ", ".join(p.name for p in written)
    )


def info(package: str = "gtcrn") -> None:
    """Print one-line JSON with engine metadata."""
    spec = _spec(package)
    print(json.dumps(
        {
            "name": package,
            "family": spec["family"],
            "repo": spec["repo"],
            "stack": "onnx",
            "runtime": "sherpa-onnx",
            "quant": spec["quant"],
            "op": "dn" if spec["job"] == "denoise" else "sep",
            "job": spec["job"],
            "url": spec["url"],
        },
        separators=(",", ":"),
    ))


def packages() -> None:
    """List every engine: name, job, family, quantization."""
    for name, spec in sorted(PACKAGES.items()):
        print(f"{name:20} {spec['job']:9} {spec['family']:9} {spec['quant']}")


if __name__ == "__main__":
    import fire

    fire.Fire({
        "prep": prep, "run": run, "separate": separate,
        "info": info, "packages": packages,
    })
