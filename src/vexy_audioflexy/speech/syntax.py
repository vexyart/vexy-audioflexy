# this_file: src/vexy_audioflexy/speech/syntax.py
"""Small averaged-perceptron POS model for English phrase-safe line boundaries."""

from functools import lru_cache

from platformdirs import user_cache_path


@lru_cache(maxsize=1)
def tagger():
    """Use an existing NLTK model, or download its small JSON weights once."""
    import nltk
    from nltk.tag import PerceptronTagger

    cache = user_cache_path("vexy-audioflexy", appauthor=False) / "nltk"
    legacy = user_cache_path("vexy-subs", appauthor=False) / "nltk"
    nltk.data.path[:0] = [str(cache), str(legacy)]
    try:
        return PerceptronTagger()
    except LookupError:
        cache.mkdir(parents=True, exist_ok=True)
        if not nltk.download(
            "averaged_perceptron_tagger_eng", download_dir=str(cache), quiet=True
        ):
            raise RuntimeError(
                "Install NLTK averaged_perceptron_tagger_eng for English grammar"
            ) from None
        return PerceptronTagger()


def tags(tokens: list[str], language: str) -> tuple[str, ...]:
    """Tag once per speaker run; other languages use explicit phrase-boundary rules."""
    if language != "en":
        return tuple("" for _ in tokens)
    return tuple(tag for _, tag in tagger().tag(tokens))
