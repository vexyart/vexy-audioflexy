#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = ["fire", "rich", "sherpa-onnx", "soundfile", "numpy", "httpx", "platformdirs"]
# ///
# this_file: src/vexy_audioflexy/tools/asr/tools/asr-sherpaonnx.py
"""sherpa-onnx offline ASR: one onnxruntime engine, many model families.

Like `asr-audiocpp.py`, this is one wrapper covering many models. sherpa-onnx
dispatches on which `OfflineRecognizer.from_*` factory is called, so a model is
addressed by `--package` and the family is a property of the package, not a flag.

Weights are GitHub release tarballs. `prep` downloads and extracts the whole
directory: a model is its `.onnx` files plus `tokens.txt`, and for the NeMo
transducers the `model_type="nemo_transducer"` that tells the engine how to drive
them — an icefall transducer decoded as a NeMo one produces fluent nonsense.

Audio reaches the engine as 16 kHz mono float32; anything else is resampled with
ffmpeg on the way in.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

from _audioflexy_paths import models_root

VERSION = "1.0.0"

_ASR = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models"

# name -> family, tarball, languages, quant. `family` selects the from_* factory.
PACKAGES: dict[str, dict] = {
    "whisper-tiny": {
        "family": "whisper",
        "url": f"{_ASR}/sherpa-onnx-whisper-tiny.tar.bz2",
        "langs": ["en", "pl"],
        "quant": "int8",
        "repo": "openai/whisper-tiny (sherpa-onnx export)",
    },
    "whisper-small": {
        "family": "whisper",
        "url": f"{_ASR}/sherpa-onnx-whisper-small.tar.bz2",
        "langs": ["en", "pl"],
        "quant": "int8",
        "repo": "openai/whisper-small (sherpa-onnx export)",
    },
    "sense-voice": {
        "family": "sense_voice",
        "url": f"{_ASR}/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2",
        # zh/en/ja/ko/yue only — Polish is not in the training set and the model
        # will still emit confident text for it, which is exactly why the batch
        # runner must not be told it speaks pl.
        "langs": ["en"],
        "quant": "int8",
        "repo": "FunAudioLLM/SenseVoiceSmall",
    },
    "zipformer-en": {
        "family": "transducer",
        "model_type": "transducer",
        "url": f"{_ASR}/sherpa-onnx-zipformer-en-2023-06-26.tar.bz2",
        "langs": ["en"],
        "quant": "int8",
        "repo": "k2-fsa icefall zipformer en 2023-06-26",
    },
    "nemo-fast-conformer-eu": {
        "family": "transducer",
        "model_type": "nemo_transducer",
        "url": f"{_ASR}/sherpa-onnx-nemo-fast-conformer-transducer-be-de-en-es-fr-hr-it-pl-ru-uk-20k.tar.bz2",
        "langs": ["en", "pl"],
        "quant": "f32",
        "repo": "nvidia stt_multilingual_fastconformer (be/de/en/es/fr/hr/it/pl/ru/uk)",
    },
    "parakeet-tdt-v3": {
        "family": "transducer",
        "model_type": "nemo_transducer",
        "url": f"{_ASR}/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2",
        "langs": ["en", "pl"],
        "quant": "int8",
        "repo": "nvidia/parakeet-tdt-0.6b-v3 (25 European languages)",
    },
    "moonshine-base-en": {
        "family": "moonshine",
        "url": f"{_ASR}/sherpa-onnx-moonshine-base-en-int8.tar.bz2",
        "langs": ["en"],
        "quant": "int8",
        "repo": "UsefulSensors/moonshine-base",
    },
}


def _root() -> Path:
    """Weight cache root. `SHERPA_ONNX_MODELS_ROOT` overrides it."""
    env = os.environ.get("SHERPA_ONNX_MODELS_ROOT")
    if env:
        return Path(env).expanduser()
    return models_root() / "sherpa-onnx"


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


def _extract(tarball: Path, into: Path) -> Path:
    import tarfile

    into.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tarball, "r:*") as tf:
        top = {Path(m.name).parts[0] for m in tf.getmembers() if m.name.strip("./")}
        if len(top) != 1:
            raise SystemExit(f"{tarball.name} has {len(top)} top-level entries, expected 1")
        target = into / top.pop()
        if not target.is_dir():
            tf.extractall(into, filter="data")
    return target


def _ensure(package: str) -> tuple[Path, dict]:
    spec = _spec(package)
    root = _root()
    tarball = _download(spec["url"], root / "_dl" / spec["url"].rsplit("/", 1)[-1])
    return _extract(tarball, root), spec


def _pick(d: Path, pattern: str, int8: bool) -> str:
    """One file matching `pattern`, preferring (or avoiding) the int8 export.

    Most packages ship both precisions side by side, so a bare glob would pick
    whichever sorts first and silently change the model between packages.
    """
    hits = sorted(d.glob(pattern))
    if not hits:
        return ""
    want = [p for p in hits if (".int8." in p.name) == int8]
    return str((want or hits)[0])


def _recognizer(package: str, lang: str = ""):
    """Build an OfflineRecognizer for `package`, resolving files from the tarball."""
    import sherpa_onnx

    d, spec = _ensure(package)
    family = spec["family"]
    int8 = spec["quant"] == "int8"
    tokens = _pick(d, "*tokens.txt", int8) or _pick(d, "tokens.txt", int8)
    if not tokens:
        raise SystemExit(f"{d} has no tokens.txt — layout changed")

    if family == "whisper":
        return sherpa_onnx.OfflineRecognizer.from_whisper(
            encoder=_pick(d, "*encoder*.onnx", int8),
            decoder=_pick(d, "*decoder*.onnx", int8),
            tokens=tokens,
            language=lang or "en",
            task="transcribe",
            num_threads=2,
        )
    if family == "sense_voice":
        return sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=_pick(d, "model*.onnx", int8),
            tokens=tokens,
            language=lang or "auto",
            use_itn=True,
            num_threads=2,
        )
    if family == "transducer":
        return sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=_pick(d, "encoder*.onnx", int8),
            decoder=_pick(d, "decoder*.onnx", int8),
            joiner=_pick(d, "joiner*.onnx", int8),
            tokens=tokens,
            model_type=spec.get("model_type", "transducer"),
            num_threads=2,
        )
    if family == "moonshine":
        return sherpa_onnx.OfflineRecognizer.from_moonshine(
            preprocessor=_pick(d, "preprocess*.onnx", int8),
            encoder=_pick(d, "encode*.onnx", int8),
            uncached_decoder=_pick(d, "uncached_decode*.onnx", int8),
            cached_decoder=_pick(d, "cached_decode*.onnx", int8),
            tokens=tokens,
            num_threads=2,
        )
    raise SystemExit(f"unsupported family '{family}'")


def _load_16k(audio: Path) -> tuple["object", int]:
    """Read `audio` as mono float32 at 16 kHz, resampling with ffmpeg if needed."""
    import numpy as np
    import soundfile as sf

    info = sf.info(str(audio))
    if info.samplerate == 16000 and info.channels == 1:
        data, rate = sf.read(str(audio), dtype="float32", always_2d=True)
        return np.ascontiguousarray(data[:, 0]), rate
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "16k.wav"
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(audio),
             "-ac", "1", "-ar", "16000", str(wav)],
            check=True,
        )
        data, rate = sf.read(str(wav), dtype="float32", always_2d=True)
        return np.ascontiguousarray(data[:, 0]), rate


def prep(package: str = "whisper-tiny", force: bool = False) -> None:
    """Download + extract one package's weights. Idempotent.

    Args:
        package: Registered package name; `packages` lists them.
        force: Unused; an existing download is never re-fetched.
    """
    d, spec = _ensure(package)
    print(f"[prep] ready: {package} ({spec['family']}) -> {d}")


def transcribe(
    audio: str,
    package: str = "whisper-tiny",
    text_out: str = "",
    lang: str = "",
    force: bool = False,
) -> None:
    """Transcribe one audio file.

    Args:
        audio: Input audio file, any rate; resampled to 16 kHz mono.
        package: Registered package name.
        lang: Language hint. Honoured by whisper and sense-voice; the transducers
            are multilingual without a language switch and ignore it.
        text_out: Transcript path (default: *audio* with a `.txt` suffix).
        force: Overwrite an existing transcript.
    """
    from rich.console import Console

    console = Console()
    src = Path(audio)
    if not src.exists():
        raise SystemExit(f"audio not found: {src}")
    spec = _spec(package)
    if lang and lang not in spec["langs"]:
        raise SystemExit(
            f"package '{package}' covers {'/'.join(spec['langs'])}, not '{lang}'"
        )
    out_path = Path(text_out) if text_out else src.with_suffix(".txt")
    if not force and out_path.exists() and out_path.stat().st_size > 0:
        console.print(f"[yellow]Skipped:[/yellow] {out_path} already exists")
        return

    console.print(
        f"[cyan]sherpa-onnx[/cyan] v{VERSION} package=[yellow]{package}[/yellow] "
        f"family=[yellow]{spec['family']}[/yellow]"
    )
    recognizer = _recognizer(package, lang)
    samples, rate = _load_16k(src)
    audio_s = len(samples) / rate

    t0 = time.monotonic()
    stream = recognizer.create_stream()
    stream.accept_waveform(rate, samples)
    recognizer.decode_stream(stream)
    wall = time.monotonic() - t0

    transcript = (stream.result.text or "").strip()
    if not transcript:
        # An empty transcript that exits 0 reads as success and scores 100% WER.
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
            "package": package,
        },
        ensure_ascii=False, separators=(",", ":"),
    ))


def info(package: str = "whisper-tiny") -> None:
    """Print one-line JSON with package metadata."""
    spec = _spec(package)
    print(json.dumps(
        {
            "name": package,
            "family": spec["family"],
            "repo": spec["repo"],
            "stack": "onnx",
            "runtime": "sherpa-onnx",
            "quant": spec["quant"],
            "langs": spec["langs"],
            "tasks": ["asr"],
            "url": spec["url"],
        },
        separators=(",", ":"),
    ))


def packages() -> None:
    """List every registered package: name, family, languages, quantization."""
    for name, spec in sorted(PACKAGES.items()):
        print(f"{name:26} {spec['family']:12} {'/'.join(spec['langs']):8} {spec['quant']}")


if __name__ == "__main__":
    import fire

    fire.Fire({"prep": prep, "transcribe": transcribe, "info": info, "packages": packages})
