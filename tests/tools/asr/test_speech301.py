# this_file: tests/tools/asr/test_speech301.py
"""A consumer must retain recognized wording and prove actual word alignment."""

import importlib.util
from pathlib import Path

import pytest


def wrapper():
    path = Path(__file__).resolve().parents[3] / "src/vexy_audioflexy/tools/asr" / "tools/speech-issue301.py"
    spec = importlib.util.spec_from_file_location("speech301", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_aligned_words_when_normalized_then_preserves_recognized_spelling():
    words = wrapper().retain_text(
        "QuasarAPI works!",
        [
            {"text": "quasarapi", "start": 0.1, "end": 0.5},
            {"text": "works", "start": 0.6, "end": 1.2},
        ],
    )
    assert [w["text"] for w in words] == ["QuasarAPI", "works!"], (
        "Aligners may normalize; recognized words stay intact"
    )
    assert words[1]["start"] == 0.6, "Retain actual alignment rather than interpolating"


@pytest.mark.parametrize("aligned", [[], [{"text": "different", "start": 0, "end": 1}]])
def test_alignment_when_word_omitted_or_rewritten_then_rejects(aligned):
    with pytest.raises(ValueError):
        wrapper().retain_text("Original 20", aligned)
