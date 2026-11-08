# this_file: src/vexy_audioflexy/speech/qwen.py
"""Qwen recognition and forced alignment with auditable lexical retention."""

import platform
import tempfile
import wave
from pathlib import Path

from .hints import excerpt, metadata, vocabulary
from .types import Word

MODEL = "mlx-community/Qwen3-ASR-1.7B-8bit"
ALIGNER = "mlx-community/Qwen3-ForcedAligner-0.6B-8bit"
LANGUAGES = {
    "en": "English",
    "de": "German",
    "fr": "French",
    "it": "Italian",
    "es": "Spanish",
    "pt": "Portuguese",
    "ru": "Russian",
    "zh": "Chinese",
    "yue": "Cantonese",
    "ja": "Japanese",
    "ko": "Korean",
}
MAX_TOKENS = 1024
CHUNK_SECONDS = 30
# Aligner times sit on an 80 ms grid; a final word may start one step late, and its
# extrapolated end past the chunk is clamped rather than treated as invalid.
ALIGNER_STEP = 0.08
# Fast English speech peaks near 5 words/s; more means recited prompt text.
MAX_WORDS_PER_SECOND = 7


def mlx_available() -> bool:
    """Only select MLX on Apple Silicon with its actual inference runtime."""
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return False
    try:
        import mlx.core as mx
        import mlx_audio.stt  # noqa: F401

        return mx.metal.is_available()
    except (ImportError, OSError):
        return False


def select_backend(backend: str) -> str:
    """Resolve availability before creating cache keys, never after decode failure."""
    if backend not in {"auto", "mlx", "audiocpp", "workspace", "whisper"}:
        raise ValueError(
            "asr_backend must be auto, mlx, audiocpp, workspace or whisper"
        )
    return ("mlx" if mlx_available() else "audiocpp") if backend == "auto" else backend


def check_model(model: str) -> None:
    """Rejected benchmark families cannot enter through explicit paths or workspaces."""
    if any(name in model.casefold() for name in ("voxtral", "granite", "kroko")):
        raise ValueError("This ASR family was removed; use Qwen3-ASR")


def context(hints: str, reference: str = "") -> str:
    """Prefer a small glossary to free-form notes; tagged metadata is always kept.

    `reference` is the human-edited caption text overlapping one chunk only.
    """
    terms = vocabulary(hints)
    text = (
        "\n".join(
            filter(
                None,
                (
                    "Known terminology: "
                    + ", ".join(terms)
                    + ". Use these spellings only when spoken. Transcribe verbatim; do not add terms or summarize.",
                    metadata(hints),
                ),
            )
        )
        if terms
        else hints
    )
    if reference:
        text = "\n".join(
            filter(
                None,
                (
                    text,
                    "Human-edited transcript of this passage, for spelling only: "
                    + reference
                    + "\nTranscribe exactly what is spoken in the audio.",
                ),
            )
        )
    return text


def chunks(wav: Path, folder: Path):
    """Yield contiguous PCM chunks and true source offsets, including the last frame."""
    with wave.open(str(wav), "rb") as source:
        if source.getnchannels() != 1 or source.getframerate() != 16000:
            raise ValueError("Qwen requires decoded mono 16 kHz PCM")
        offset = 0
        while offset < source.getnframes():
            frames = min(CHUNK_SECONDS * 16000, source.getnframes() - offset)
            path = folder / f"{offset:012}.wav"
            with wave.open(str(path), "wb") as target:
                target.setparams(source.getparams())
                target.writeframes(source.readframes(frames))
            yield path, offset / 16000, frames / 16000
            offset += frames


def canonical(text: str) -> str:
    """Match aligner-cleaned words without losing punctuation in the saved transcript."""
    return "".join(
        c for c in text.replace("’", "'").casefold() if c.isalnum() or c == "'"
    )


def retain_words(text: str, items: list, offset: float, duration: float) -> list[Word]:
    """Group subword alignments; refuse lexical changes, invalid bounds or word loss."""
    words, position, previous = [], 0, 0.0
    for token in text.split():
        expected = canonical(token)
        if not expected:
            if not words:
                raise ValueError("Leading standalone punctuation has no alignment")
            words[-1].text += " " + token
            continue
        first, actual = position, ""
        while position < len(items) and len(actual) < len(expected):
            item = items[position]
            if (
                not (0 <= item.start_time <= item.end_time)
                or item.start_time > duration + ALIGNER_STEP
                or item.start_time < previous
            ):
                raise ValueError("Forced alignment returned invalid bounds or order")
            previous = item.start_time
            actual += canonical(item.text)
            position += 1
        if actual != expected or first == position:
            raise ValueError(
                f"Forced alignment changed recognized lexical words: {token!r} vs {actual!r} at item {first}"
            )
        words.append(
            Word(
                token,
                offset + min(items[first].start_time, duration),
                offset + min(items[position - 1].end_time, duration),
            )
        )
    if position != len(items):
        raise ValueError("Forced alignment added lexical words")
    return words


def transcribe(
    wav: Path,
    hints: str,
    language: str,
    model: str = MODEL,
    aligner_model: str = ALIGNER,
    reference: tuple = (),
) -> tuple[list[Word], dict]:
    """Load Qwen once; recognise and acoustically align each bounded audio chunk."""
    from mlx_audio.stt import load

    recognizer = load(model, strict=True)
    aligner = load(aligner_model, strict=True)
    words, segments, dropped = [], [], []
    with tempfile.TemporaryDirectory() as temporary:
        for audio, offset, duration in chunks(wav, Path(temporary)):
            prompt = context(hints, excerpt(list(reference), offset, offset + duration))
            # A long prompt can make Qwen recite it over short or quiet audio; retry bare.
            for system_prompt in dict.fromkeys((prompt or None, None)):
                try:
                    text, found = recognize(
                        recognizer,
                        aligner,
                        audio,
                        system_prompt,
                        language,
                        offset,
                        duration,
                    )
                    break
                except ValueError as error:
                    reason = str(error)
            else:
                text, found = "", []
                dropped.append(
                    {"start": offset, "end": offset + duration, "reason": reason}
                )
            segments.append({"start": offset, "end": offset + duration, "text": text})
            words.extend(found)
    return words, {
        "text": " ".join(s["text"] for s in segments),
        "language": language,
        "segments": segments,
        "backend": "mlx",
        "model": model,
        "alignment": {
            "model": aligner_model,
            "word_timestamps": "forced_alignment",
            "chunks": len(segments),
            "dropped_chunks": dropped,
        },
    }


def recognize(recognizer, aligner, audio, system_prompt, language, offset, duration):
    """One chunk's text and aligned words; ValueError when either looks invented."""
    result = recognizer.generate(
        str(audio),
        language=LANGUAGES.get(language, language) or None,
        system_prompt=system_prompt,
        max_tokens=MAX_TOKENS,
        temperature=0.0,
        verbose=False,
    )
    if result.generation_tokens >= MAX_TOKENS:
        raise ValueError("Qwen reached its token cap; transcript may be incomplete")
    text = result.text.strip()
    if len(text.split()) > MAX_WORDS_PER_SECOND * duration + 5:
        raise ValueError("Qwen returned more words than the audio can hold")
    if not text:
        return "", []
    aligned = aligner.generate(
        audio=str(audio),
        text=text,
        language=LANGUAGES.get(language, language) or "English",
    )
    return text, retain_words(text, aligned.items, offset, duration)
