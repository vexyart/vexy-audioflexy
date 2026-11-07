#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "fire", "rich", "coqui-tts", "torch==2.5.1", "torchaudio==2.5.1",
#   "transformers==4.57.6", "soundfile",
# ]
# ///
# this_file: src/vexy_audioflexy/tools/tts/tools/tts-xtts.py
"""XTTS-v2 (Coqui). Backend: torch/coqui-tts, languages: 17 incl. en/pl, 24 kHz mono, modes: ivc, capabilities: voice_cloning.

The only engine in this project that clones a voice **into Polish**: XTTS-v2 takes
a reference clip and a language code, and Polish is one of its 17.

Three pins, each load-bearing:

- `torch==2.5.1` / `torchaudio==2.5.1`. From torch 2.9 torchaudio decodes through
  torchcodec, which links Homebrew's FFmpeg and needs `DYLD_FALLBACK_LIBRARY_PATH`
  set before the process starts. Coqui's own import fails outright without it.
  2.5.1 is already this project's pin for F5-TTS, so nothing new is introduced.
- `transformers==4.57.6`. Coqui imports `isin_mps_friendly` from
  `transformers.pytorch_utils`, which transformers 5 removed.

The weights are **CPML-licensed: non-commercial only**. Downloading them means
accepting that licence, which is what `COQUI_TOS_AGREED=1` says; the wrapper sets
it, so run this only where a non-commercial licence is acceptable.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

VERSION = "1.0.0"

MODELS: dict[str, dict] = {
    "xtts-v2": {
        "id": "tts_models/multilingual/multi-dataset/xtts_v2",
        "langs": ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl",
                  "cs", "ar", "zh-cn", "hu", "ko", "ja", "hi"],
        "clone": True,
        "sample_rate": 24000,
        "license": "CPML (non-commercial)",
    },
    "your-tts": {
        "id": "tts_models/multilingual/multi-dataset/your_tts",
        "langs": ["en", "fr-fr", "pt-br"],
        "clone": True,
        "sample_rate": 16000,
        "license": "CC BY-NC-ND 4.0 (non-commercial)",
    },
}


def _spec(model: str) -> dict:
    try:
        return MODELS[model]
    except KeyError:
        raise SystemExit(
            f"unknown model '{model}'. Known: {', '.join(MODELS)}"
        ) from None


def _api(model: str, device: str):
    # Accepting the Coqui Public Model Licence is a precondition of the download,
    # not a formality — see the module docstring.
    os.environ.setdefault("COQUI_TOS_AGREED", "1")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    import torch
    from TTS.api import TTS

    # Weights are frequently saved from CUDA machines; force them onto this one.
    _load = torch.load
    torch.load = lambda *a, **kw: _load(*a, **{**kw, "map_location": "cpu"})
    try:
        api = TTS(_spec(model)["id"])
    finally:
        torch.load = _load
    return api.to(device)


def prep(model: str = "xtts-v2", force: bool = False) -> None:
    """Download + cache the model WITHOUT synthesizing. Idempotent.

    Args:
        model: `xtts-v2` or `your-tts`.
        force: Unused; cached weights are never re-fetched.
    """
    spec = _spec(model)
    _api(model, "cpu")
    print(f"[prep] ready: {model} ({spec['id']}, {spec['license']})")


def say(
    text: str,
    output: str = "output.wav",
    model: str = "xtts-v2",
    ref: str = "",
    lang: str = "en",
    speed: float = 1.0,
    device: str = "cpu",
    play: bool = False,
    force: bool = False,
) -> None:
    """Clone the voice in `ref` and speak `text` in `lang`.

    Args:
        text: Text to synthesize.
        output: Output WAV path.
        model: `xtts-v2` or `your-tts`.
        ref: Reference clip to clone — required, this is a zero-shot cloner with
            no built-in voices.
        lang: Language code. Checked against the model's list; XTTS-v2 covers pl.
        speed: Speaking speed (XTTS only).
        device: `cpu` (default), `mps` or `cuda`. MPS falls back to the CPU for the
            operators XTTS uses that Metal lacks, which is slower than plain CPU
            more often than not — hence the default.
        play: Play the result with afplay instead of just saving it.
        force: Overwrite an existing output file.
    """
    import time

    from rich.console import Console
    from rich.panel import Panel

    console = Console()
    out_path = Path(output).resolve()
    if not force and out_path.exists() and out_path.stat().st_size > 0:
        console.print(f"[yellow]Skipped:[/yellow] {output} already exists (use --force)")
        return

    spec = _spec(model)
    if not ref:
        raise SystemExit(f"{model} is a zero-shot cloner: --ref is required")
    ref_path = Path(ref)
    if not ref_path.exists():
        raise SystemExit(f"reference audio not found: {ref_path}")
    if lang not in spec["langs"]:
        raise SystemExit(
            f"{model} speaks {', '.join(spec['langs'])} — not '{lang}'"
        )

    console.print(
        Panel.fit(
            f"[bold cyan]coqui {model}[/bold cyan] v{VERSION}  "
            f"lang=[yellow]{lang}[/yellow]  device=[yellow]{device}[/yellow]\n"
            f"ref={ref_path.name}  licence={spec['license']}",
            border_style="cyan",
        )
    )

    api = _api(model, device)
    kwargs: dict[str, object] = {
        "text": text,
        "speaker_wav": str(ref_path.resolve()),
        "language": lang,
        "file_path": str(out_path),
    }
    if model == "xtts-v2" and speed != 1.0:
        kwargs["speed"] = speed

    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    api.tts_to_file(**kwargs)
    wall = time.monotonic() - t0

    if not out_path.exists() or out_path.stat().st_size == 0:
        console.print(f"[red]{model} wrote nothing to {out_path}[/red]")
        raise SystemExit(1)

    import soundfile as sf

    details = sf.info(str(out_path))
    console.print(
        f"[green]Saved[/green] [bold]{out_path}[/bold] "
        f"({details.duration:.1f}s @ {details.samplerate} Hz, "
        f"rtf={wall / details.duration:.2f})"
    )
    if play:
        import subprocess

        subprocess.run(["afplay", str(out_path)], check=True)


def info(model: str = "xtts-v2") -> None:
    """Print one-line JSON with model metadata."""
    spec = _spec(model)
    print(json.dumps(
        {
            "name": model,
            "repo": f"coqui-ai/TTS ({spec['id']})",
            "stack": "torch",
            "runtime": "coqui-tts",
            "quant": "fp32",
            "mode": "ivc",
            "voice_clone": spec["clone"],
            "langs": spec["langs"],
            "sample_rate": spec["sample_rate"],
            "license": spec["license"],
        },
        separators=(",", ":"),
    ))


def models() -> None:
    """List the registered models: name, languages, licence."""
    for name, spec in MODELS.items():
        print(f"{name:10} {len(spec['langs']):>2} langs  {spec['sample_rate']} Hz  {spec['license']}")


if __name__ == "__main__":
    import fire

    fire.Fire({"prep": prep, "say": say, "info": info, "models": models})
