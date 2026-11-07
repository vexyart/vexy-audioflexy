#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = ["fire", "rich", "pywhispercpp", "numpy", "soundfile"]
# ///
# this_file: src/vexy_audioflexy/tools/asr/tools/asr-whispercpp.py
"""whisper.cpp (ggml) via the pywhispercpp bindings — Metal on Apple Silicon.

The reference C++ whisper. `asr` already reaches whisper models through
audio.cpp and sherpa-onnx; this is the implementation those two are compared
against, with its own GGUF quantizations.

One wrapper, many models: `--package` is a whisper.cpp model name
(`tiny`, `small-q5_1`, `large-v3-turbo-q8_0`, …) and weights download on first use
to the pywhispercpp cache, or to `WHISPERCPP_MODELS_DIR` when it is set.

Two things the bindings do quietly and this wrapper does not:

- whisper.cpp guesses the language when none is given, and guesses badly on short
  clips. `--lang` is forwarded; without it the model runs in auto-detect and the
  guess is reported in the result line.
- `transcribe()` returns segments, and a run that produced no segments still exits
  cleanly. An empty transcript exits non-zero here instead, because a silent
  success scores 100% WER.
- The bindings refuse any WAV that is not already 16 kHz (`WAV file must be 16000
  Hz`) rather than resampling. Audio is resampled with ffmpeg on the way in and
  handed over as a float32 array, the same as `asr-sherpaonnx.py` does.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

VERSION = "1.0.0"

# whisper.cpp model names, grouped by what they are useful for here. The full
# list lives in pywhispercpp.constants.AVAILABLE_MODELS; `packages` prints it.
PACKAGES: dict[str, dict] = {
    "tiny": {"quant": "f16", "langs": ["en", "pl"]},
    "tiny-q5_1": {"quant": "q5_1", "langs": ["en", "pl"]},
    "base": {"quant": "f16", "langs": ["en", "pl"]},
    "base-q5_1": {"quant": "q5_1", "langs": ["en", "pl"]},
    "small": {"quant": "f16", "langs": ["en", "pl"]},
    "small-q5_1": {"quant": "q5_1", "langs": ["en", "pl"]},
    "medium-q5_0": {"quant": "q5_0", "langs": ["en", "pl"]},
    "large-v3-turbo": {"quant": "f16", "langs": ["en", "pl"]},
    "large-v3-turbo-q5_0": {"quant": "q5_0", "langs": ["en", "pl"]},
    "large-v3-q5_0": {"quant": "q5_0", "langs": ["en", "pl"]},
    "small.en": {"quant": "f16", "langs": ["en"]},
    "medium.en-q5_0": {"quant": "q5_0", "langs": ["en"]},
}

_REPO = "ggerganov/whisper.cpp (ggml weights)"


def _spec(package: str) -> dict:
    if package in PACKAGES:
        return PACKAGES[package]
    # A name this file does not list may still be a valid whisper.cpp model.
    from pywhispercpp.constants import AVAILABLE_MODELS

    if package in AVAILABLE_MODELS:
        return {"quant": "unknown", "langs": ["en", "pl"]}
    raise SystemExit(
        f"unknown package '{package}'. Registered: {', '.join(PACKAGES)}. "
        f"Any whisper.cpp model name also works: {', '.join(AVAILABLE_MODELS)}"
    )


def _models_dir() -> str | None:
    return os.environ.get("WHISPERCPP_MODELS_DIR") or None


def _model(package: str):
    from pywhispercpp.model import Model

    return Model(
        package,
        models_dir=_models_dir(),
        redirect_whispercpp_logs_to=False,
        print_progress=False,
        print_realtime=False,
        print_timestamps=False,
    )


def _duration(path: Path) -> float:
    import soundfile as sf

    try:
        info = sf.info(str(path))
        return info.frames / info.samplerate
    except Exception:
        return 0.0


def _load_16k(audio: Path):
    """Read `audio` as mono float32 at 16 kHz, resampling with ffmpeg if needed."""
    import subprocess
    import tempfile

    import numpy as np
    import soundfile as sf

    info = sf.info(str(audio))
    if info.samplerate == 16000 and info.channels == 1:
        data, _ = sf.read(str(audio), dtype="float32", always_2d=True)
        return np.ascontiguousarray(data[:, 0])
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "16k.wav"
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(audio),
             "-ac", "1", "-ar", "16000", str(wav)],
            check=True,
        )
        data, _ = sf.read(str(wav), dtype="float32", always_2d=True)
        return np.ascontiguousarray(data[:, 0])


def prep(package: str = "small", force: bool = False) -> None:
    """Download the ggml weights for one model WITHOUT transcribing. Idempotent.

    Args:
        package: whisper.cpp model name; `packages` lists the registered ones.
        force: Unused; an existing weight file is never re-fetched.
    """
    _spec(package)
    model = _model(package)
    print(f"[prep] ready: {package} -> {model.model_path}")


def transcribe(
    audio: str,
    package: str = "small",
    text_out: str = "",
    lang: str = "",
    translate: bool = False,
    n_threads: int = 0,
    force: bool = False,
) -> None:
    """Transcribe one audio file with whisper.cpp.

    Args:
        audio: Input audio file. whisper.cpp resamples to 16 kHz itself.
        package: whisper.cpp model name.
        text_out: Transcript path (default: *audio* with a `.txt` suffix).
        lang: Language code. Left empty, whisper.cpp auto-detects and the guess is
            reported in the result line.
        translate: Translate to English instead of transcribing in-language. Off by
            default: this harness scores against same-language references.
        n_threads: Decoder threads; 0 leaves whisper.cpp's own default.
        force: Overwrite an existing transcript.
    """
    from rich.console import Console

    console = Console()
    src = Path(audio)
    if not src.exists():
        raise SystemExit(f"audio not found: {src}")
    spec = _spec(package)
    if lang and lang not in spec["langs"]:
        raise SystemExit(f"package '{package}' covers {'/'.join(spec['langs'])}, not '{lang}'")
    out_path = Path(text_out) if text_out else src.with_suffix(".txt")
    if not force and out_path.exists() and out_path.stat().st_size > 0:
        console.print(f"[yellow]Skipped:[/yellow] {out_path} already exists")
        return

    console.print(f"[cyan]whisper.cpp[/cyan] v{VERSION} model=[yellow]{package}[/yellow]")
    model = _model(package)
    params: dict[str, object] = {"translate": translate}
    if lang:
        params["language"] = lang
    if n_threads:
        params["n_threads"] = n_threads

    t0 = time.monotonic()
    segments = model.transcribe(_load_16k(src), **params)
    wall = time.monotonic() - t0

    transcript = " ".join(s.text.strip() for s in segments if s.text.strip()).strip()
    if not transcript:
        console.print("[red]empty transcript[/red]")
        raise SystemExit(1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(transcript + "\n", encoding="utf-8")
    audio_s = _duration(src)
    detected = ""
    try:
        detected = model.get_params().get("language", "") or ""
    except Exception:
        pass
    console.print(f"[green]Transcript[/green] {out_path.resolve()}\n  {transcript}")
    print("asr.result=" + json.dumps(
        {
            "text": transcript,
            "text_out": str(out_path),
            "wall_s": round(wall, 4),
            "audio_duration_s": round(audio_s, 4),
            "rtf": round(wall / audio_s, 6) if audio_s else 0.0,
            "package": package,
            "language": lang or detected,
            "segments": len(segments),
        },
        ensure_ascii=False, separators=(",", ":"),
    ))


def info(package: str = "small") -> None:
    """Print one-line JSON with model metadata."""
    spec = _spec(package)
    print(json.dumps(
        {
            "name": package,
            "family": "whisper",
            "repo": _REPO,
            "stack": "whispercpp",
            "runtime": "whisper.cpp (ggml, Metal)",
            "quant": spec["quant"],
            "langs": spec["langs"],
            "tasks": ["asr", "translate"],
        },
        separators=(",", ":"),
    ))


def packages() -> None:
    """List the registered models, then every model whisper.cpp will accept."""
    from pywhispercpp.constants import AVAILABLE_MODELS
    from pywhispercpp.model import Model

    for name, spec in PACKAGES.items():
        print(f"{name:22} {spec['quant']:8} {'/'.join(spec['langs'])}")
    print(f"\nalso accepted: {', '.join(AVAILABLE_MODELS)}")
    print(f"\nbuild: {Model.system_info()}")


if __name__ == "__main__":
    import fire

    fire.Fire({"prep": prep, "transcribe": transcribe, "info": info, "packages": packages})
