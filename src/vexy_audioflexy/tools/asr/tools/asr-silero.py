#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = ["fire", "rich", "torch", "torchaudio", "soundfile", "numpy", "omegaconf"]
# ///
# this_file: src/vexy_audioflexy/tools/asr/tools/asr-silero.py
"""Silero STT: a ~50 MB JIT CTC model from torch.hub, CPU-only.

The lightest ASR engine here by an order of magnitude, and the only one with no
Python package: `torch.hub.load` fetches `snakers4/silero-models` into
`~/.cache/torch/hub` and returns a scripted model plus its decoder.

Two upstream facts this wrapper works around:

- `omegaconf` is imported by the hub entrypoint and declared nowhere.
- The bundled `read_audio` helper calls `torchaudio.load`, which on torch 2.9
  routes through torchcodec and raises `TorchCodec is required for
  load_with_torchcodec`. Audio is therefore read with soundfile and resampled with
  `torchaudio.functional.resample`, neither of which needs a codec backend.

**No Polish.** Silero STT covers en/de/es/ua. Registered `en` only.

The model emits lower-case, unpunctuated text. That is a genuine model property and
shows up as WER against a punctuated reference, exactly as the scoring policy says
it should.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

VERSION = "1.0.0"
REPO = "snakers4/silero-models"
LANGS = ["en", "de", "es", "ua"]


def _model(lang: str):
    import torch

    if lang not in LANGS:
        raise SystemExit(f"Silero STT covers {', '.join(LANGS)} — not '{lang}'")
    model, decoder, _utils = torch.hub.load(
        REPO, "silero_stt", language=lang, device=torch.device("cpu"), trust_repo=True
    )
    return model, decoder


def _load_16k(path: Path):
    """Read `path` as mono float32 at 16 kHz without torchaudio's file loader."""
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio

    data, rate = sf.read(str(path), dtype="float32", always_2d=True)
    tensor = torch.from_numpy(np.ascontiguousarray(data[:, 0]))
    if rate != 16000:
        tensor = torchaudio.functional.resample(tensor, rate, 16000)
    return tensor, tensor.numel() / 16000


def prep(lang: str = "en", package: str = "", force: bool = False) -> None:
    """Fetch the hub repo and the JIT model. Idempotent.

    Args:
        lang: Language model to fetch (`en`, `de`, `es`, `ua`).
        package: Alias for `lang`. The batch runner passes `--package` to every
            wrapper's prep, and a prep that rejects it downloads nothing while
            reporting nothing — the weights then land inside the transcribe
            timeout instead.
        force: Unused; the hub cache is never re-fetched.
    """
    language = package or lang
    _model(language)
    print(f"[prep] ready: {REPO} silero_stt {language}")


def transcribe(
    audio: str,
    text_out: str = "",
    lang: str = "en",
    package: str = "",
    force: bool = False,
) -> None:
    """Transcribe one audio file.

    Args:
        audio: Input audio file, any rate; resampled to 16 kHz mono.
        text_out: Transcript path (default: *audio* with a `.txt` suffix).
        lang: Language model to use. Silero ships one model per language.
        package: Accepted as an alias for `lang`, so the batch runner can address
            this wrapper the same way as the others.
        force: Overwrite an existing transcript.
    """
    from rich.console import Console

    console = Console()
    src = Path(audio)
    if not src.exists():
        raise SystemExit(f"audio not found: {src}")
    language = package or lang
    out_path = Path(text_out) if text_out else src.with_suffix(".txt")
    if not force and out_path.exists() and out_path.stat().st_size > 0:
        console.print(f"[yellow]Skipped:[/yellow] {out_path} already exists")
        return

    console.print(f"[cyan]Silero STT[/cyan] v{VERSION} lang=[yellow]{language}[/yellow]")
    model, decoder = _model(language)
    samples, audio_s = _load_16k(src)

    t0 = time.monotonic()
    output = model(samples.unsqueeze(0))
    transcript = decoder(output[0].cpu()).strip()
    wall = time.monotonic() - t0

    if not transcript:
        console.print("[red]empty transcript[/red]")
        raise SystemExit(1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(transcript + "\n", encoding="utf-8")
    console.print(f"[green]Transcript[/green] {out_path.resolve()}\n  {transcript}")
    print("asr.result=" + json.dumps(
        {
            "text": transcript,
            "text_out": str(out_path),
            "wall_s": round(wall, 4),
            "audio_duration_s": round(audio_s, 4),
            "rtf": round(wall / audio_s, 6) if audio_s else 0.0,
            "package": language,
        },
        ensure_ascii=False, separators=(",", ":"),
    ))


def info(lang: str = "en") -> None:
    """Print one-line JSON with model metadata."""
    print(json.dumps(
        {
            "name": f"silero-stt-{lang}",
            "family": "silero",
            "repo": REPO,
            "stack": "torch",
            "runtime": "Silero STT (torch.hub JIT, CPU)",
            "quant": "fp32",
            "langs": LANGS,
            "tasks": ["asr"],
        },
        separators=(",", ":"),
    ))


def packages() -> None:
    """List the language models."""
    for lang in LANGS:
        print(lang)


if __name__ == "__main__":
    import fire

    fire.Fire({"prep": prep, "transcribe": transcribe, "info": info, "packages": packages})
