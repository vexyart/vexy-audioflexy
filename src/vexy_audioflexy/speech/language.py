# this_file: src/vexy_audioflexy/speech/language.py
"""Optional langid language model and neutral timed-word sampling."""

from functools import lru_cache

from .types import Word


@lru_cache(maxsize=1)
def identifier():
    """Load langid's bundled 97-language Naive Bayes model once; no model download."""
    from langid.langid import LanguageIdentifier, model

    return LanguageIdentifier.from_modelstring(model, norm_probs=True)


def detect_language(words: list[Word], language: str = "") -> dict:
    """Sample across the recording, not only a short introductory greeting."""
    sample = " ".join(w.text for w in words[:: max(1, len(words) // 2000)])[:20000]
    detected, probability = (
        identifier().classify(sample) if sample.strip() else ("und", 0)
    )
    chosen = (language or detected).lower().replace("_", "-").split("-")[0]
    if chosen != "und" and chosen not in identifier().nb_classes:
        raise ValueError(f"Unsupported language code: {chosen}")
    return {
        "language": chosen,
        "detected": detected,
        "probability": float(probability),
        "overridden": bool(language),
        "model": "langid-1.1.6-naive-bayes",
    }
