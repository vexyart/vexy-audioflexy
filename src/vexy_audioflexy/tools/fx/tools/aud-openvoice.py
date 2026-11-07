#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = ["platformdirs>=4", 
#   "fire", "rich", "torch", "torchaudio", "librosa", "soundfile", "numpy",
#   "httpx", "huggingface-hub", "setuptools<81",
#   "inflect", "unidecode", "eng-to-ipa", "pypinyin", "cn2an", "jieba",
# ]
# ///
# this_file: src/vexy_audioflexy/tools/fx/tools/aud-openvoice.py
"""OpenVoice v2 tone-colour conversion: audio in, audio out.

OpenVoice is two halves. The TTS half is MeloTTS, which lives in `../tts`. The
half that belongs here is the **tone-colour converter**: it takes speech from one
speaker and re-renders it in another speaker's timbre, which is voice conversion —
the same `vc` op as `aud-audiocpp.py`'s MeanVC2 and Seed-VC.

**Not pip-installable.** The package pins `faster-whisper==0.9.0`, which drags in
`av==10`, which does not build on macOS (`CompileError: av/logging.pyx`). It is
only needed by `se_extractor`, the VAD-chunked helper — and
`ToneColorConverter.extract_se()` embeds a speaker directly with no chunking and no
whisper. So this wrapper clones the repo into `local/OpenVoice` and imports it from
there, the same shape as `aud-mossformersr.py`, and declares only the dependencies
the converter itself uses.

`enable_watermark=False` for the same reason: watermarking pulls `wavmark`, another
dependency of a feature this harness does not use.

`openvoice.api` imports its text front-end at module load even though conversion
never touches text, so the text stack (`inflect`, `unidecode`, `eng-to-ipa`,
`pypinyin`, `cn2an`, `jieba`) is declared above. They are import-time weight, not
work this wrapper does.
"""

from __future__ import annotations


import os
from pathlib import Path
from platformdirs import user_cache_path, user_data_path
RUNTIME_ROOT = Path(os.environ.get("VAFX_AUDIO_RUNTIME_ROOT", user_data_path("vexy-audioflexy") / "audio" )).expanduser()
CACHE_ROOT = Path(os.environ.get("VAFX_AUDIO_CACHE_ROOT", user_cache_path("vexy-audioflexy") / "audio")).expanduser()
import json
import os
import subprocess
import sys
import time
from pathlib import Path

VERSION = "1.0.0"

REPO_URL = "https://github.com/myshell-ai/OpenVoice"
# The S3 archive the upstream docs point at (checkpoints_v2_0417.zip) now 404s.
# The same converter lives in the official HuggingFace repo, two files, ungated.
CKPT_REPO = "myshell-ai/OpenVoiceV2"
CKPT_FILES = ("converter/config.json", "converter/checkpoint.pth")

PACKAGES: dict[str, dict] = {
    "openvoice-v2": {
        "converter": "checkpoints_v2/converter",
        "quant": "fp32",
        "sample_rate": 22050,
        "repo": "myshell-ai/OpenVoice (checkpoints_v2)",
    },
}


def _root() -> Path:
    """Checkout root. `OPENVOICE_ROOT` overrides it."""
    env = os.environ.get("OPENVOICE_ROOT")
    if env:
        return Path(env).expanduser()
    return RUNTIME_ROOT / "engines" / "OpenVoice"


def _spec(package: str) -> dict:
    try:
        return PACKAGES[package]
    except KeyError:
        raise SystemExit(
            f"unknown package '{package}'. Known: {', '.join(PACKAGES)}"
        ) from None


def _ensure_checkout() -> Path:
    root = _root()
    if not (root / "openvoice" / "api.py").exists():
        root.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", REPO_URL, str(root)], check=True
        )
    return root


def _ensure_checkpoints(package: str) -> Path:
    """Fetch the converter checkpoint into the checkout, and return its directory."""
    from huggingface_hub import hf_hub_download

    root = _ensure_checkout()
    converter = root / _spec(package)["converter"]
    if (converter / "config.json").exists() and (converter / "checkpoint.pth").exists():
        return converter

    converter.mkdir(parents=True, exist_ok=True)
    for name in CKPT_FILES:
        src = Path(hf_hub_download(CKPT_REPO, name))
        dest = converter / Path(name).name
        if not dest.exists():
            dest.write_bytes(src.read_bytes())
    return converter


def _converter(package: str, device: str):
    root = _ensure_checkout()
    ckpt = _ensure_checkpoints(package)
    sys.path.insert(0, str(root))
    from openvoice.api import OpenVoiceBaseClass, ToneColorConverter

    # ToneColorConverter.__init__ reads `enable_watermark` with kwargs.get and then
    # forwards **kwargs to the base class, which rejects it — so the documented way
    # to skip watermarking raises TypeError, and the default path imports `wavmark`
    # for a feature this harness does not use. Build the object through the base
    # initialiser and set the two attributes the subclass would have set.
    converter = ToneColorConverter.__new__(ToneColorConverter)
    OpenVoiceBaseClass.__init__(converter, str(ckpt / "config.json"), device=device)
    converter.watermark_model = None
    converter.version = getattr(converter.hps, "_version_", "v1")
    converter.load_ckpt(str(ckpt / "checkpoint.pth"))
    return converter


def prep(package: str = "openvoice-v2", force: bool = False) -> None:
    """Clone the repo and download the v2 checkpoints. Idempotent.

    Args:
        package: Registered package name.
        force: Unused; an existing checkout and archive are reused.
    """
    ckpt = _ensure_checkpoints(package)
    print(f"[prep] ready: {package} -> {ckpt}")


def run(
    audio: str = "",
    package: str = "openvoice-v2",
    output: str = "output.wav",
    voice_ref: str = "",
    task: str = "",
    tau: float = 0.3,
    device: str = "cpu",
    force: bool = False,
    play: bool = False,
    **_ignored: object,
) -> None:
    """Re-render `audio` in the timbre of `voice_ref`: audio in, audio out.

    Args:
        audio: Source speech — the words and prosody that are kept.
        package: Registered package name.
        output: Output WAV path.
        voice_ref: Target speaker clip — the timbre that is applied. Required.
        task: `vc`, optional. A mismatch is an error, not a reinterpretation.
        tau: Conversion strength. Upstream's default is 0.3; higher moves further
            towards the target and further from the source's intelligibility.
        device: `cpu` (default), `mps` or `cuda`.
        force: Overwrite an existing output file.
        play: Play the result with afplay instead of reporting the path.
    """
    import numpy as np
    import soundfile as sf
    from rich.console import Console
    from rich.panel import Panel

    if _ignored:
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
    if not voice_ref:
        raise SystemExit("--voice_ref is required: this is voice conversion, not enhancement")
    if task and task != "vc":
        raise SystemExit(f"engine '{package}' does vc, not '{task}'")

    src = Path(audio).resolve()
    tgt = Path(voice_ref).resolve()
    for path in (src, tgt):
        if not path.exists():
            raise SystemExit(f"audio not found: {path}")

    spec = _spec(package)
    console.print(
        Panel.fit(
            f"[bold cyan]OpenVoice v2[/bold cyan] v{VERSION}  task=[yellow]vc[/yellow]  "
            f"tau=[yellow]{tau}[/yellow] device=[yellow]{device}[/yellow]\n"
            f"source={src.name}  target={tgt.name}",
            border_style="cyan",
        )
    )

    converter = _converter(package, device)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    source_se = converter.extract_se([str(src)])
    target_se = converter.extract_se([str(tgt)])
    converter.convert(
        audio_src_path=str(src), src_se=source_se, tgt_se=target_se,
        output_path=str(out_path), tau=tau,
    )
    elapsed = time.monotonic() - t0

    if not out_path.exists() or out_path.stat().st_size == 0:
        console.print(f"[red]{package} wrote nothing to {out_path}[/red]")
        raise SystemExit(1)
    data, rate = sf.read(str(out_path), dtype="float32", always_2d=True)
    if data.size == 0 or not np.any(data):
        console.print(f"[red]{package} produced silence[/red]")
        raise SystemExit(1)

    duration = data.shape[0] / rate
    console.print(
        f"[green]Saved[/green] [bold]{out_path}[/bold] "
        f"({duration:.1f}s @ {rate} Hz, rtf={elapsed / duration:.2f})"
    )
    if play:
        subprocess.run(["afplay", str(out_path)], check=True)


def info(package: str = "openvoice-v2") -> None:
    """Print one-line JSON with engine metadata."""
    spec = _spec(package)
    print(json.dumps(
        {
            "name": package,
            "family": "openvoice",
            "repo": spec["repo"],
            "stack": "torch",
            "runtime": "OpenVoice ToneColorConverter (PyTorch, local checkout)",
            "quant": spec["quant"],
            "op": "vc",
            "sample_rate": spec["sample_rate"],
        },
        separators=(",", ":"),
    ))


def packages() -> None:
    """List the registered engines."""
    for name, spec in PACKAGES.items():
        print(f"{name:14} vc  {spec['quant']:6} {spec['sample_rate']} Hz  {spec['repo']}")


if __name__ == "__main__":
    import fire

    fire.Fire({"prep": prep, "run": run, "info": info, "packages": packages})
