# this_file: src/vexy_audioflexy/speech/alignment.py
"""Repair collapsed ASR timestamps against actual audio, preserving lexical words."""

from pathlib import Path

from .types import Word

MODEL = "mlx-community/Qwen3-ForcedAligner-0.6B-8bit"
LANGUAGES = {
    "en": "English",
    "de": "German",
    "fr": "French",
    "it": "Italian",
    "es": "Spanish",
    "pt": "Portuguese",
    "ru": "Russian",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
}


def windows(words: list[Word]) -> list[tuple[int, int]]:
    """Group defective alignments with surrounding acoustic/text context."""
    groups = []
    for index, word in enumerate(words):
        bad = word.end <= word.start or (index and word.start < words[index - 1].start)
        if not bad:
            continue
        start, end = max(0, index - 12), min(len(words), index + 13)
        if groups and start <= groups[-1][1]:
            groups[-1] = (groups[-1][0], end)
        else:
            groups.append((start, end))
    return groups


def repair(
    wav: Path, words: list[Word], language: str = "en", model: str = MODEL
) -> dict:
    """Only reload the forced aligner when native ASR word times are degenerate."""
    groups = windows(words)
    if not groups:
        return {"model": model, "windows": 0, "repaired_words": 0}
    import soundfile as sf
    from mlx_audio.stt import load

    aligner = load(model, strict=True)
    audio, rate = sf.read(wav, dtype="float32")
    changes = []
    for first, last in groups:
        selected = words[first:last]
        start = max(0, min(w.start for w in selected) - 0.5)
        end = min(len(audio) / rate, max(w.end for w in selected) + 0.5)
        if end - start > 60:
            raise ValueError("Alignment repair needs a shorter acoustic context")
        clean = aligner.aligner_processor.clean_token
        lexical = [w for w in selected if clean(w.text)]
        result = aligner.generate(
            audio=audio[round(start * rate) : round(end * rate)],
            text=" ".join(w.text for w in lexical),
            language=LANGUAGES.get(language, language),
        )
        if len(result.items) != len(lexical):
            raise ValueError("Forced alignment changed the token count")
        for word, item in zip(lexical, result.items, strict=True):
            if clean(word.text) != item.text:
                raise ValueError("Forced alignment changed lexical tokens")
            old = (word.start, word.end)
            if word.end > word.start:
                continue
            position = words.index(word)
            previous = max(
                (w.end for w in words[:position] if w.end > w.start), default=0
            )
            following = next(
                (w.start for w in words[position + 1 :] if w.end > w.start),
                len(audio) / rate,
            )
            candidate = (start + item.start_time, start + item.end_time)
            if not previous - 0.04 <= candidate[0] < candidate[1] <= following + 0.04:
                continue
            word.start, word.end = candidate
            changes.append(
                {"text": word.text, "original": old, "aligned": [word.start, word.end]}
            )
    return {
        "model": model,
        "windows": len(groups),
        "repaired_words": len(changes),
        "changes": changes,
    }
