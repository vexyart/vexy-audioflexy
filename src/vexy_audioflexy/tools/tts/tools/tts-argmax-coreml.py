#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = ["fire", "rich", "soundfile"]
# ///
# this_file: src/vexy_audioflexy/tools/tts/tools/tts-argmax-coreml.py
"""Qwen3-TTS via argmax-oss-swift. Backend: coreml/Swift, languages: 10 incl. en, 24 kHz mono, modes: auto, capabilities: none.

`argmax-cli tts` is the TTS half of the same Swift binary whose ASR half `../asr`
runs as `whisperkit-*`. It executes Qwen3-TTS as CoreML on the Apple Neural Engine —
a different execution path for a model family this project already runs on MLX
(`tts-qwen3-mlx.py`) and on audio.cpp.

The binary is built, not installed. `prep` clones `argmaxinc/argmax-oss-swift` into
`../asr/local/argmax-oss-swift` — one checkout shared with the ASR side, since it
is one package — and runs `swift build -c release`. `ARGMAX_CLI_BIN` points at an
existing binary instead; `ARGMAX_ROOT` moves the checkout.

Nine fixed speakers, no cloning. `--output-format wav` is forced: the CLI defaults
to m4a, which the batch runner's downstream steps do not read.

No Polish: the CLI accepts english, chinese, japanese, korean, german, french,
russian, portuguese, spanish and italian.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from _paths import runtime_path

VERSION = "1.0.0"

REPO_URL = "https://github.com/argmaxinc/argmax-oss-swift"
SPEAKERS = ["aiden", "ryan", "ono-anna", "sohee", "eric", "dylan", "serena",
            "vivian", "uncle-fu"]
LANGS = {
    "en": "english", "zh": "chinese", "ja": "japanese", "ko": "korean",
    "de": "german", "fr": "french", "ru": "russian", "pt": "portuguese",
    "es": "spanish", "it": "italian",
}


def _root() -> Path:
    """Checkout root — shared with ../asr, which builds the same package."""
    env = os.environ.get("ARGMAX_ROOT")
    if env:
        return Path(env).expanduser()
    return runtime_path('argmax-oss-swift')


def _binary(build: bool = True) -> Path:
    env = os.environ.get("ARGMAX_CLI_BIN")
    if env:
        return Path(env).expanduser()
    root = _root()
    binary = root / ".build" / "release" / "whisperkit-cli"
    if binary.exists():
        return binary
    if not build:
        raise SystemExit(
            f"argmax CLI not built. Run `{Path(__file__).name} prep`, or set ARGMAX_CLI_BIN."
        )
    if not (root / "Package.swift").exists():
        root.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(root)], check=True)
    subprocess.run(
        ["swift", "build", "-c", "release", "--product", "whisperkit-cli"],
        cwd=str(root), check=True,
    )
    if not binary.exists():
        raise SystemExit(f"swift build finished but {binary} is missing")
    return binary


def prep(force: bool = False) -> None:
    """Clone + build the Swift CLI. Idempotent.

    Args:
        force: Unused; an existing build is reused.
    """
    print(f"[prep] ready: {_binary()}")


def say(
    text: str,
    output: str = "output.wav",
    speaker: str = "aiden",
    lang: str = "en",
    temperature: float = 0.9,
    ref: str = "",
    play: bool = False,
    force: bool = False,
) -> None:
    """Synthesize `text` with one Qwen3-TTS speaker on the Neural Engine.

    Args:
        text: Text to synthesize.
        output: Output WAV path (24 kHz mono).
        speaker: One of the nine fixed voices.
        lang: Language code; `en`, `zh`, `ja`, `ko`, `de`, `fr`, `ru`, `pt`, `es`, `it`.
        temperature: Sampling temperature; 0.0 is greedy.
        ref: Rejected. These are fixed voices with no cloning.
        play: Play the result with afplay instead of just saving it.
        force: Overwrite an existing output file.
    """
    from rich.console import Console
    from rich.panel import Panel

    console = Console()
    out_path = Path(output).resolve()
    if not force and out_path.exists() and out_path.stat().st_size > 0:
        console.print(f"[yellow]Skipped:[/yellow] {output} already exists (use --force)")
        return
    if ref:
        raise SystemExit(f"Qwen3-TTS here has fixed voices and cannot clone {ref}")
    if speaker not in SPEAKERS:
        raise SystemExit(f"unknown speaker '{speaker}'. Known: {', '.join(SPEAKERS)}")
    if lang not in LANGS:
        raise SystemExit(f"unsupported language '{lang}'. Known: {', '.join(LANGS)}")

    binary = _binary()
    console.print(
        Panel.fit(
            f"[bold cyan]Qwen3-TTS / argmax[/bold cyan] v{VERSION}  "
            f"speaker=[yellow]{speaker}[/yellow]  lang=[yellow]{LANGS[lang]}[/yellow]",
            border_style="cyan",
        )
    )

    # The CLI appends the format's extension to --output-path, so it is given the
    # stem and the result is moved into place.
    stem = out_path.with_suffix("")
    produced = stem.with_suffix(".wav")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(binary), "tts",
        "--text", text,
        "--speaker", speaker,
        "--language", LANGS[lang],
        "--output-path", str(stem),
        "--output-format", "wav",
        "--temperature", str(temperature),
    ]
    t0 = time.monotonic()
    result = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.monotonic() - t0
    if result.returncode != 0:
        console.print(result.stderr.strip()[-800:])
        raise SystemExit(result.returncode)
    if produced != out_path and produced.exists():
        produced.replace(out_path)
    if not out_path.exists() or out_path.stat().st_size == 0:
        console.print(f"[red]argmax CLI wrote nothing to {out_path}[/red]")
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


def info() -> None:
    """Print one-line JSON with model metadata."""
    print(json.dumps(
        {
            "name": "argmax-qwen3-tts",
            "repo": "argmaxinc/argmax-oss-swift (Qwen3-TTS CoreML)",
            "stack": "coreml",
            "runtime": "argmax-cli tts (Swift, CoreML, Neural Engine)",
            "quant": "f16",
            "mode": "auto",
            "voice_clone": False,
            "langs": list(LANGS),
            "speakers": SPEAKERS,
            "sample_rate": 24000,
        },
        separators=(",", ":"),
    ))


def speakers() -> None:
    """List the nine voices."""
    for name in SPEAKERS:
        print(name)


if __name__ == "__main__":
    import fire

    fire.Fire({"prep": prep, "say": say, "info": info, "speakers": speakers})
