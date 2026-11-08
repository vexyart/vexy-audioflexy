# this_file: src/vexy_audioflexy/speech/audiocpp.py
"""Direct audio.cpp Qwen and Nemotron routes without a private adapter workspace."""

import json
import os
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace

from . import qwen
from .hints import excerpt
from .io import sha256
from .types import Turn, Word


def model_path(value: str, default: str, variable: str) -> str:
    """Resolve user-installed native weights; never silently use an MLX repository."""
    value = (
        (
            os.getenv(variable.replace("VEXY_SUBS_", "VEXY_AUDIOFLEXY_"), "")
            or os.getenv(variable, "")
        )
        if not value or value == default
        else value
    )
    path = Path(value).expanduser() if value else None
    if path is None or not path.exists():
        raise FileNotFoundError(
            f"Set {variable} or pass the native model path; install weights with audiocpp_model_manager"
        )
    return str(path.resolve())


def binary_path(value: str = "") -> str:
    """Accept a configured executable or the normal PATH installation."""
    value = (
        value
        or os.getenv("VEXY_AUDIOFLEXY_AUDIOCPP_BIN", "")
        or os.getenv("VEXY_SUBS_AUDIOCPP_BIN", "")
        or "audiocpp_cli"
    )
    resolved = shutil.which(value)
    if not resolved:
        raise FileNotFoundError(
            "Install audiocpp_cli on PATH or set VEXY_SUBS_AUDIOCPP_BIN"
        )
    return resolved


def backend_name(value: str = "auto") -> str:
    """Prefer Metal on Apple Silicon; CPU works without an accelerator installation."""
    if value not in {"auto", "cpu", "metal", "cuda", "vulkan", "hip"}:
        raise ValueError("Unsupported audio.cpp compute backend")
    return (
        (
            "metal"
            if platform.system() == "Darwin" and platform.machine() == "arm64"
            else "cpu"
        )
        if value == "auto"
        else value
    )


def command(
    binary: str, family: str, task: str, model: str, wav: Path, backend: str
) -> list[str]:
    """Use the upstream common task contract with bounded process execution."""
    return [
        binary,
        "--task",
        task,
        "--family",
        family,
        "--model",
        model,
        "--backend",
        backend,
        "--audio",
        str(wav),
    ]


def execute(args: list[str]) -> None:
    """Keep native progress off JSON stdout; surface engine failures with context."""
    result = subprocess.run(
        args, check=False, capture_output=True, text=True, timeout=14400
    )
    if result.returncode:
        raise RuntimeError(
            f"audio.cpp failed ({result.returncode}): {result.stderr[-2000:]}"
        )


def transcribe(
    wav: Path,
    hints: str,
    language: str,
    model: str,
    aligner: str,
    binary: str,
    backend: str,
    reference: tuple = (),
) -> tuple[list[Word], dict]:
    """Run bounded requests in one loaded native session, avoiding word-filtering merges."""
    words, segments = [], []
    with tempfile.TemporaryDirectory() as temporary:
        folder = Path(temporary)
        prepared = list(qwen.chunks(wav, folder))
        requests = [
            {
                "id": f"{i:03}",
                "audio": str(audio),
                "language": qwen.LANGUAGES.get(language, language),
                "text": qwen.context(
                    hints, excerpt(list(reference), offset, offset + duration)
                ),
                "options": {
                    "max_tokens": qwen.MAX_TOKENS,
                    "audio_chunk_mode": "none",
                    "preserve_punctuation": True,
                    "clamp_timestamps_to_audio": True,
                },
            }
            for i, (audio, offset, duration) in enumerate(prepared)
        ]
        sequence = folder / "requests.json"
        sequence.write_text(json.dumps({"requests": requests}))
        args = [
            binary,
            "--task",
            "asr",
            "--family",
            "qwen3_asr",
            "--model",
            model,
            "--backend",
            backend,
            "--request-sequence",
            str(sequence),
            "--text-out",
            str(folder / "text.txt"),
            "--words-out",
            str(folder / "words.json"),
            "--session-option",
            f"qwen3_asr.forced_aligner_path={aligner}",
        ]
        if requests:
            execute(args)
        for i, (_, offset, duration) in enumerate(prepared):
            text = (folder / f"text_{i:03}.txt").read_text().strip()
            raw = (
                json.loads((folder / f"words_{i:03}.json").read_text()) if text else []
            )
            items = [
                SimpleNamespace(
                    text=w["word"],
                    start_time=w["start_sample"] / 16000,
                    end_time=w["end_sample"] / 16000,
                )
                for w in raw
            ]
            words.extend(qwen.retain_words(text, items, offset, duration))
            segments.append(
                {
                    "start": offset,
                    "end": offset + duration,
                    "text": text,
                    "native_words": raw,
                }
            )
    return words, {
        "text": " ".join(s["text"] for s in segments),
        "language": language,
        "backend": "audiocpp",
        "segments": segments,
        "alignment": {
            "model": aligner,
            "word_timestamps": "forced_alignment",
            "chunks": len(segments),
        },
    }


def diarize(
    wav: Path, model: str, binary: str, backend: str
) -> tuple[list[Turn], dict]:
    """Run Nemotron's continuous eight-channel session using the same native engine."""
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary) / "turns.json"
        args = command(binary, "nemotron_3_diar", "diar", model, wav, backend)
        execute(args + ["--mode", "offline", "--turns-out", str(output)])
        raw = json.loads(output.read_text())
    turns = [
        Turn(
            row["start_sample"] / 16000,
            row["end_sample"] / 16000,
            str(row["speaker_id"])
            if isinstance(row["speaker_id"], str)
            else f"speaker_{row['speaker_id']}",
        )
        for row in raw
    ]
    return turns, {
        "model": model,
        "capacity": 8,
        "backend": "audiocpp",
        "anonymous_ids": True,
        "turns": [
            {"start": t.start, "end": t.end, "speaker": t.speaker} for t in turns
        ],
    }


def identity(binary: str, model: str, aligner: str) -> dict:
    """Invalidate native caches when a binary or local weight package changes."""
    paths = [Path(binary), Path(model), Path(aligner)]
    files = [
        f
        for p in paths
        for f in (sorted(p.rglob("*")) if p.is_dir() else [p])
        if f.is_file()
    ]
    return {str(p): sha256(p) for p in files}
