#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = ["fire", "rich", "whisperx", "soundfile"]
# ///
# this_file: src/vexy_audioflexy/tools/asr/tools/asr-whisperx.py
"""WhisperX: faster-whisper transcription plus wav2vec2 forced alignment.

Two things this adds over the whisper models already here:

- **Word-level timings without a known transcript.** `asr-audiocpp.py align` needs
  the reference text; WhisperX transcribes first and aligns its own hypothesis, so
  `words` works on unlabelled audio. The alignment model is per language and
  downloads on demand; Polish is covered.
- **Speaker diarization**, optional and off by default: it pulls pyannote models
  that require accepting their terms on HuggingFace and an `HF_TOKEN`.

Runs on the **CPU**. WhisperX decodes through faster-whisper/CTranslate2, which
has no Metal backend, so `--device mps` is rejected rather than silently ignored.
`int8` is the default compute type; `float32` is slower and marginally better.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

VERSION = "1.0.0"

# faster-whisper model names. Anything CTranslate2 accepts works; these are the
# ones registered in asr_all.py.
PACKAGES: dict[str, dict] = {
    "tiny": {"quant": "int8", "langs": ["en", "pl"]},
    "base": {"quant": "int8", "langs": ["en", "pl"]},
    "small": {"quant": "int8", "langs": ["en", "pl"]},
    "medium": {"quant": "int8", "langs": ["en", "pl"]},
    "large-v3": {"quant": "int8", "langs": ["en", "pl"]},
    "large-v3-turbo": {"quant": "int8", "langs": ["en", "pl"]},
    "distil-large-v3": {"quant": "int8", "langs": ["en"]},
}

_REPO = "m-bain/whisperX (faster-whisper weights)"


def _spec(package: str) -> dict:
    if package in PACKAGES:
        return PACKAGES[package]
    return {"quant": "int8", "langs": ["en", "pl"]}


def _check_device(device: str) -> str:
    if device == "mps":
        raise SystemExit(
            "whisperX decodes through CTranslate2, which has no Metal backend. "
            "Use --device cpu."
        )
    return device


def _duration(path: Path) -> float:
    import soundfile as sf

    try:
        info = sf.info(str(path))
        return info.frames / info.samplerate
    except Exception:
        return 0.0


def prep(package: str = "small", lang: str = "", force: bool = False) -> None:
    """Download the ASR model, and the alignment model when `lang` is given.

    Args:
        package: faster-whisper model name.
        lang: Language whose alignment model to fetch as well (`en`, `pl`, …).
            The alignment model is per language and is otherwise fetched lazily
            on the first `--words` run.
        force: Unused; cached weights are never re-fetched.
    """
    import whisperx

    spec = _spec(package)
    whisperx.load_model(package, device="cpu", compute_type=spec["quant"])
    msg = f"[prep] ready: {package}"
    if lang:
        whisperx.load_align_model(language_code=lang, device="cpu")
        msg += f" + alignment model for {lang}"
    print(msg)


def transcribe(
    audio: str,
    package: str = "small",
    text_out: str = "",
    lang: str = "",
    words_out: str = "",
    words: bool = False,
    diarize: bool = False,
    batch_size: int = 8,
    device: str = "cpu",
    compute_type: str = "",
    force: bool = False,
) -> None:
    """Transcribe one file, optionally with word timings and speaker labels.

    Args:
        audio: Input audio file.
        package: faster-whisper model name.
        text_out: Transcript path (default: *audio* with a `.txt` suffix).
        lang: Language code. Left empty, whisper detects it and the detection is
            reported in the result line.
        words_out: Word-timing JSON path (default: *audio* with `.words.json`).
            Implies `--words`.
        words: Run forced alignment and write word timings.
        diarize: Also label speakers. Needs `HF_TOKEN` and accepted pyannote terms;
            it fails loudly rather than quietly returning unlabelled segments.
        batch_size: Segments decoded per batch.
        device: `cpu` only; see the module docstring.
        compute_type: CTranslate2 compute type; defaults to the package's `int8`.
        force: Overwrite existing outputs.
    """
    from rich.console import Console

    import whisperx

    console = Console()
    device = _check_device(device)
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
    want_words = words or bool(words_out)
    word_path = Path(words_out) if words_out else src.with_suffix(".words.json")

    console.print(
        f"[cyan]whisperX[/cyan] v{VERSION} model=[yellow]{package}[/yellow] "
        f"device=[yellow]{device}[/yellow]"
    )
    model = whisperx.load_model(
        package, device=device, compute_type=compute_type or spec["quant"]
    )
    samples = whisperx.load_audio(str(src))

    t0 = time.monotonic()
    result = model.transcribe(samples, batch_size=batch_size, language=lang or None)
    detected = result.get("language", "") or lang

    if want_words:
        align_model, meta = whisperx.load_align_model(language_code=detected, device=device)
        result = whisperx.align(
            result["segments"], align_model, meta, samples, device,
            return_char_alignments=False,
        )
    if diarize:
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        if not token:
            raise SystemExit("--diarize needs HF_TOKEN (and accepted pyannote terms)")
        from whisperx.diarize import DiarizationPipeline

        diarizer = DiarizationPipeline(use_auth_token=token, device=device)
        result = whisperx.assign_word_speakers(diarizer(str(src)), result)
    wall = time.monotonic() - t0

    segments = result.get("segments", [])
    transcript = " ".join(s.get("text", "").strip() for s in segments).strip()
    if not transcript:
        console.print("[red]empty transcript[/red]")
        raise SystemExit(1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(transcript + "\n", encoding="utf-8")

    word_count = 0
    if want_words:
        words_json = [
            {
                "word": w.get("word", ""),
                "start": float(w["start"]) if w.get("start") is not None else None,
                "end": float(w["end"]) if w.get("end") is not None else None,
                "score": float(w["score"]) if w.get("score") is not None else None,
                "speaker": w.get("speaker", ""),
            }
            for seg in segments
            for w in seg.get("words", [])
        ]
        word_count = len(words_json)
        word_path.parent.mkdir(parents=True, exist_ok=True)
        word_path.write_text(
            json.dumps({"audio": str(src), "model": package, "language": detected,
                        "words": words_json}, ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
        console.print(f"[green]Word timings[/green] {word_path.resolve()} ({word_count} words)")

    audio_s = _duration(src)
    console.print(f"[green]Transcript[/green] {out_path.resolve()}\n  {transcript}")
    print("asr.result=" + json.dumps(
        {
            "text": transcript,
            "text_out": str(out_path),
            "wall_s": round(wall, 4),
            "audio_duration_s": round(audio_s, 4),
            "rtf": round(wall / audio_s, 6) if audio_s else 0.0,
            "package": package,
            "language": detected,
            "segments": len(segments),
            "words": word_count,
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
            "stack": "whisperx",
            "runtime": "whisperX (faster-whisper / CTranslate2, CPU)",
            "quant": spec["quant"],
            "langs": spec["langs"],
            "tasks": ["asr", "align", "diarize"],
        },
        separators=(",", ":"),
    ))


def packages() -> None:
    """List the registered models."""
    for name, spec in PACKAGES.items():
        print(f"{name:18} {spec['quant']:6} {'/'.join(spec['langs'])}")


if __name__ == "__main__":
    import fire

    fire.Fire({"prep": prep, "transcribe": transcribe, "info": info, "packages": packages})
