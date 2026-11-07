#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#   "fire", "rich", "melotts @ git+https://github.com/myshell-ai/MeloTTS",
#   "torch", "torchaudio", "soundfile", "numpy", "setuptools<81",
# ]
# ///
# this_file: src/vexy_audioflexy/tools/tts/tools/tts-melotts.py
"""MeloTTS (CPU). Backend: torch/melotts, languages: en/es/fr/zh/ja/ko, 44.1 kHz mono, modes: auto, capabilities: none.

The working MeloTTS path. `tools/tts-melotts-mlx.py` is the MLX attempt and stays in
`IGNORED_ENGINES`; this one runs the upstream PyTorch package on the CPU.

Three installation facts, all load-bearing:

- **Python 3.11, not 3.12.** MeloTTS pins `transformers==4.27.4`, which pins
  `tokenizers==0.13.3`, which has no arm64 wheel for 3.12 and fails to build from
  source (`failed with code 101` out of the Rust toolchain). 3.11 has the wheel.
- **git, not PyPI.** The `melotts` sdist on PyPI is broken — its build references a
  `src/requirements.txt` that is not in the archive.
- **`setuptools<81`.** MeloTTS's librosa pin imports `pkg_resources`, which is not
  part of Python and which setuptools 81 began removing — 84 no longer ships it at
  all. Unpinned, the first call dies on `ModuleNotFoundError: No module named
  'pkg_resources'`.
- **unidic downloads at runtime.** MeloTTS imports its Japanese front-end whatever
  language you ask for, and that front-end wants a MeCab dictionary that ships
  empty. `prep` runs `python -m unidic download`; without it every call dies on
  `no such file or directory: …/unidic/dicdir/mecabrc`.

Five English accents are separate speakers of one checkpoint, not separate models.
No Polish.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

VERSION = "1.0.0"
REPO = "myshell-ai/MeloTTS"

# CLI language code -> MeloTTS language id. Its own ids are upper-case and its
# English accents live inside the EN checkpoint as speakers.
LANGS: dict[str, str] = {
    "en": "EN",
    "es": "ES",
    "fr": "FR",
    "zh": "ZH",
    "ja": "JP",
    "ko": "KR",
}


def _ensure_unidic() -> None:
    """Fetch the MeCab dictionary MeloTTS imports unconditionally."""
    import unidic

    if not (Path(unidic.DICDIR) / "mecabrc").exists():
        subprocess.run([sys.executable, "-m", "unidic", "download"], check=True)


def _tts(lang: str, device: str):
    if lang not in LANGS:
        raise SystemExit(f"MeloTTS speaks {', '.join(LANGS)} — not '{lang}'")
    # The dictionary has to be on disk before melo.api is imported: its Japanese
    # front-end opens mecabrc at import time, whatever language was asked for.
    _ensure_unidic()
    from melo.api import TTS

    return TTS(language=LANGS[lang], device=device)


def prep(lang: str = "en", force: bool = False) -> None:
    """Download the checkpoint and the MeCab dictionary. Idempotent.

    Args:
        lang: Which language checkpoint to fetch (`en`, `es`, `fr`, `zh`, `ja`, `ko`).
        force: Unused; cached weights are never re-fetched.
    """
    model = _tts(lang, "cpu")
    print(f"[prep] ready: {REPO} {LANGS[lang]} speakers={list(model.hps.data.spk2id.keys())}")


def say(
    text: str,
    output: str = "output.wav",
    lang: str = "en",
    speaker: str = "",
    speed: float = 1.0,
    ref: str = "",
    device: str = "cpu",
    play: bool = False,
    force: bool = False,
) -> None:
    """Synthesize `text` with one MeloTTS speaker.

    Args:
        text: Text to synthesize.
        output: Output WAV path.
        lang: `en`, `es`, `fr`, `zh`, `ja` or `ko`.
        speaker: Speaker id within the checkpoint, e.g. `EN-US`, `EN-BR`,
            `EN_INDIA`, `EN-AU`, `EN-Default`. Empty picks the first.
        speed: Speaking speed multiplier.
        ref: Rejected. MeloTTS has fixed speakers and no cloning.
        device: `cpu` (default) or `mps`.
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
    if ref:
        raise SystemExit(
            f"MeloTTS has fixed speakers and cannot clone {ref}. Use --speaker."
        )

    model = _tts(lang, device)
    # spk2id is MeloTTS's HParams object, not a dict: iterating it goes through
    # __getitem__ with an int and raises. Always go via .keys()/.values().
    speakers = model.hps.data.spk2id
    if speaker and speaker not in speakers.keys():
        raise SystemExit(
            f"unknown speaker '{speaker}' for {LANGS[lang]}. Known: {', '.join(speakers.keys())}"
        )
    speaker_id = speakers[speaker] if speaker else next(iter(speakers.values()))
    chosen = speaker or next(iter(speakers.keys()))

    console.print(
        Panel.fit(
            f"[bold cyan]MeloTTS[/bold cyan] v{VERSION}  lang=[yellow]{LANGS[lang]}[/yellow]  "
            f"speaker=[yellow]{chosen}[/yellow]  speed=[yellow]{speed}[/yellow]",
            border_style="cyan",
        )
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    model.tts_to_file(text, speaker_id, str(out_path), speed=speed)
    wall = time.monotonic() - t0

    if not out_path.exists() or out_path.stat().st_size == 0:
        console.print(f"[red]MeloTTS wrote nothing to {out_path}[/red]")
        raise SystemExit(1)

    import soundfile as sf

    details = sf.info(str(out_path))
    console.print(
        f"[green]Saved[/green] [bold]{out_path}[/bold] "
        f"({details.duration:.1f}s @ {details.samplerate} Hz, "
        f"rtf={wall / details.duration:.2f})"
    )
    if play:
        subprocess.run(["afplay", str(out_path)], check=True)


def info(lang: str = "en") -> None:
    """Print one-line JSON with model metadata."""
    print(json.dumps(
        {
            "name": f"melotts-{lang}",
            "repo": REPO,
            "stack": "torch",
            "runtime": "MeloTTS (PyTorch, CPU, Python 3.11)",
            "quant": "fp32",
            "mode": "auto",
            "voice_clone": False,
            "langs": list(LANGS),
            "sample_rate": 44100,
        },
        separators=(",", ":"),
    ))


def speakers(lang: str = "en") -> None:
    """List the speakers inside one language checkpoint."""
    model = _tts(lang, "cpu")
    for name, sid in zip(model.hps.data.spk2id.keys(), model.hps.data.spk2id.values()):
        print(f"{name:14} {sid}")


if __name__ == "__main__":
    import fire

    fire.Fire({"prep": prep, "say": say, "info": info, "speakers": speakers})
