# this_file: src/vexy_audioflexy/speech/punctuation.py
"""Model-based punctuation restoration without rewriting the spoken words."""

from .hints import vocabulary
from .types import Word

MODEL = "oliverguhr/fullstop-punctuation-multilang-large"
FUNCTION_WORDS = {
    "a",
    "an",
    "the",
    "and",
    "but",
    "so",
    "or",
    "if",
    "then",
    "because",
    "as",
    "for",
    "to",
    "in",
    "on",
    "with",
    "by",
    "it",
    "this",
    "that",
    "these",
    "those",
    "there",
    "here",
    "we",
    "you",
    "he",
    "she",
    "they",
    "my",
    "our",
    "your",
    "his",
    "her",
    "their",
    "its",
    "of",
    "from",
    "at",
    "now",
    "okay",
    "yes",
    "no",
}


def plain_word(text: str) -> str:
    """Strip boundary punctuation while preserving apostrophes and decimal points."""
    return text.strip().strip('.,;:!?“”"') or text.strip()


def apply_labels(words: list[Word], predictions: list, hints: str = "") -> None:
    """Apply predicted punctuation one-to-one; retain word times and vocabulary."""
    if len(words) != len(predictions):
        raise ValueError("Punctuation model changed the word count")
    tokens = [prediction[0] for prediction in predictions]
    spelling = {}
    for term in vocabulary(hints):
        parts = term.split()
        for offset in range(len(tokens) - len(parts) + 1):
            if [t.casefold() for t in tokens[offset : offset + len(parts)]] == [
                p.casefold() for p in parts
            ]:
                spelling.update({offset + i: part for i, part in enumerate(parts)})
    sentence_start = True
    for index, (word, prediction) in enumerate(zip(words, predictions, strict=True)):
        token, label, _ = prediction
        if token != plain_word(word.text):
            raise ValueError("Punctuation model changed a lexical token")
        token = spelling.get(index, token)
        if index and word.speaker != words[index - 1].speaker:
            sentence_start = True
        if (
            not sentence_start
            and index not in spelling
            and token.casefold() in FUNCTION_WORDS
        ):
            token = token.lower()
        if sentence_start and token:
            token = token[0].upper() + token[1:]
        suffix = label if label in {".", ",", "?", "!", ":", ";", "-"} else ""
        word.text = token + suffix
        sentence_start = suffix in {".", "?", "!"}


def restore(words: list[Word], hints: str = "", model: str = MODEL) -> dict:
    """Run the maintained predictor with a Transformers-5-compatible pipeline."""
    if model == "asr":
        return {"model": "asr", "words": len(words), "preserved_asr_punctuation": True}
    import torch
    from deepmultilingualpunctuation import PunctuationModel
    from transformers import pipeline

    if not words:
        return {"model": model, "words": 0}
    predictor = PunctuationModel.__new__(PunctuationModel)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    predictor.pipe = pipeline(
        "token-classification", model=model, aggregation_strategy="none", device=device
    )
    tokens = [plain_word(w.text) for w in words]
    predictions = []
    for offset in range(0, len(tokens), 135):
        batch = tokens[offset : offset + 150]
        predicted = predictor.predict(batch)
        final = offset + 150 >= len(tokens)
        predictions.extend(predicted if final else predicted[:135])
        if final:
            break
    apply_labels(words, predictions, hints)
    return {"model": model, "device": device, "words": len(words)}
