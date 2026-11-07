#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = ["fire", "rich", "torch", "torchaudio", "soundfile", "numpy", "omegaconf"]
# ///
# this_file: src/vexy_audioflexy/tools/tts/tools/tts-silero.py
"""Silero TTS. Backend: torch/torch.hub, languages: en/de/es/fr/ru/ua and others, 48 kHz mono, modes: auto, capabilities: none.

The smallest useful engine in this project: a ~50 MB JIT model, CPU-only, 119
English speakers, and no Python package at all — `torch.hub.load` fetches the repo
and the weights. That is also its one deployment quirk: the first call clones
`snakers4/silero-models` into `~/.cache/torch/hub`.

`omegaconf` is required and undeclared upstream; the hub entrypoint imports it to
read the model registry.

**No Polish.** Silero's v3 line covers en/de/es/fr/ru/ua/uz/tt/ba/xal and the Indic
set. Registered `en` only.
"""

from __future__ import annotations

import json
from pathlib import Path

VERSION = "1.0.0"
REPO = "snakers4/silero-models"

# language -> (hub speaker bundle, default voice). One bundle holds many voices.
BUNDLES: dict[str, tuple[str, str]] = {
    "en": ("v3_en", "en_0"),
    "de": ("v3_de", "eva_k"),
    "es": ("v3_es", "es_0"),
    "fr": ("v3_fr", "fr_0"),
    "ru": ("v4_ru", "aidar"),
    "ua": ("v4_ua", "mykyta"),
}


def _model(lang: str):
    import torch

    if lang not in BUNDLES:
        raise SystemExit(f"Silero speaks {', '.join(BUNDLES)} — not '{lang}'")
    bundle, _ = BUNDLES[lang]
    model, _example = torch.hub.load(
        REPO, "silero_tts", language=lang, speaker=bundle, trust_repo=True
    )
    return model


def prep(lang: str = "en", force: bool = False) -> None:
    """Fetch the hub repo and the JIT model. Idempotent.

    Args:
        lang: Language bundle to fetch (`en`, `de`, `es`, `fr`, `ru`, `ua`).
        force: Unused; the hub cache is never re-fetched.
    """
    model = _model(lang)
    print(f"[prep] ready: {REPO} {BUNDLES[lang][0]} — {len(model.speakers)} speakers")


def say(
    text: str,
    output: str = "output.wav",
    lang: str = "en",
    speaker: str = "",
    rate: int = 48000,
    ref: str = "",
    play: bool = False,
    force: bool = False,
) -> None:
    """Synthesize `text` with one Silero voice.

    Args:
        text: Text to synthesize.
        output: Output WAV path.
        lang: Language bundle.
        speaker: Voice id inside the bundle, e.g. `en_0`…`en_117`. Empty picks the
            bundle default.
        rate: Output sample rate; Silero offers 8000, 24000 and 48000.
        ref: Rejected. Silero has fixed voices and no cloning.
        play: Play the result with afplay instead of just saving it.
        force: Overwrite an existing output file.
    """
    import time

    import numpy as np
    import soundfile as sf
    from rich.console import Console
    from rich.panel import Panel

    console = Console()
    out_path = Path(output).resolve()
    if not force and out_path.exists() and out_path.stat().st_size > 0:
        console.print(f"[yellow]Skipped:[/yellow] {output} already exists (use --force)")
        return
    if ref:
        raise SystemExit(
            f"Silero has fixed voices and cannot clone {ref}. Use --speaker."
        )

    model = _model(lang)
    voice = speaker or BUNDLES[lang][1]
    if voice not in model.speakers:
        raise SystemExit(
            f"unknown speaker '{voice}'. {len(model.speakers)} available, "
            f"e.g. {', '.join(model.speakers[:6])}"
        )

    console.print(
        Panel.fit(
            f"[bold cyan]Silero TTS[/bold cyan] v{VERSION}  lang=[yellow]{lang}[/yellow]  "
            f"speaker=[yellow]{voice}[/yellow]  rate=[yellow]{rate}[/yellow]",
            border_style="cyan",
        )
    )

    t0 = time.monotonic()
    audio = model.apply_tts(text=text, speaker=voice, sample_rate=rate)
    wall = time.monotonic() - t0

    samples = np.asarray(audio.numpy(), dtype="float32").reshape(-1)
    if samples.size == 0 or not np.any(samples):
        console.print("[red]Silero produced silence[/red]")
        raise SystemExit(1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), samples, rate, subtype="PCM_16")
    duration = samples.size / rate
    console.print(
        f"[green]Saved[/green] [bold]{out_path}[/bold] "
        f"({duration:.1f}s @ {rate} Hz, rtf={wall / duration:.2f})"
    )
    if play:
        import subprocess

        subprocess.run(["afplay", str(out_path)], check=True)


def info(lang: str = "en") -> None:
    """Print one-line JSON with model metadata."""
    print(json.dumps(
        {
            "name": f"silero-{lang}",
            "repo": REPO,
            "stack": "torch",
            "runtime": "Silero TTS (torch.hub JIT, CPU)",
            "quant": "fp32",
            "mode": "auto",
            "voice_clone": False,
            "langs": list(BUNDLES),
            "sample_rate": 48000,
        },
        separators=(",", ":"),
    ))


def speakers(lang: str = "en") -> None:
    """List the voices in one language bundle."""
    for name in _model(lang).speakers:
        print(name)


if __name__ == "__main__":
    import fire

    fire.Fire({"prep": prep, "say": say, "info": info, "speakers": speakers})
