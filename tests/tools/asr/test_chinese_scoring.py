# this_file: tests/tools/asr/test_chinese_scoring.py
"""Chinese CER uses characters; WER uses actual word segmentation."""
import importlib.util
import sys
from pathlib import Path


def harness():
    spec = importlib.util.spec_from_file_location("asr_scoring", Path(__file__).resolve().parents[3] / "src/vexy_audioflexy/tools/asr" / "asr_all.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_chinese_words_are_segmented_and_spacing_does_not_create_character_errors():
    m = harness()
    r = m.score("这是 一个开始。", "这是一个开始", lang="zh")
    assert r["wer"] == 0 and r["cer"] == 0
    assert " " in r["ref_wer_norm"], "Chinese WER must not treat a whole sentence as one word"
    changed = m.score("这是一个开始。", "这是一个结束。", lang="zh")
    assert 0 < changed["wer"] < 1
    total = m.aggregate([dict(r, success=True), dict(changed, success=True)])
    assert 0 < total["wer"] < 1 and 0 < total["cer"] < 1


def test_english_scoring_keeps_existing_normalization_and_corpus_behavior():
    m = harness()
    assert m.score("A simple test!", "a simple test", lang="en") == m.score("A simple test!", "a simple test")
    r = m.score("a simple test", "a sample test", lang="en")
    assert abs(r["wer"] - 1 / 3) < 1e-12
    assert "ref_wer_norm" not in r, "English records keep the existing field shape"
    total = m.aggregate([dict(r, success=True), dict(m.score("three other words", "three other words"), success=True)])
    assert abs(total["wer"] - 1 / 6) < 1e-12
