# this_file: tests/tools/asr/test_mlxswift.py
"""Reject invalid native transcripts and timing before benchmark scoring."""
import importlib.util
from pathlib import Path

import pytest


def wrapper():
    spec = importlib.util.spec_from_file_location("swift_asr", Path(__file__).resolve().parents[3] / "src/vexy_audioflexy/tools/asr" / "tools/asr-mlxswift.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_native_output_requires_text_and_finite_positive_processing_time():
    module = wrapper()
    assert module.parse_result({"text": " spoken words ", "processing_time": 0.25}) == ("spoken words", 0.25)
    for value in [{}, {"text": 42, "processing_time": 1}, {"text": "", "processing_time": 1},
                  {"text": "words", "processing_time": -1}, {"text": "words", "processing_time": float("nan")},
                  {"text": "words", "processing_time": float("inf")}]:
        with pytest.raises(SystemExit):
            module.parse_result(value)


def test_audio8_uses_author_fork_and_exact_export_without_reusing_parakeet_binary():
    module = wrapper()
    spec = module.resolve("selcukkubur/Audio8-ASR-Infinite-mlx")
    assert spec["revision"] == "0e3f912dc09ac09dec0348e81c7eea96f19cac0e"
    assert spec["commit"] == "9473f480658d30834a9885cd078401584109e865"
    assert spec["langs"] == ["en", "zh"]
    assert spec["engine"] != module.resolve(module.REPO)["engine"]
    assert spec["build"] != module.resolve(module.REPO)["build"]
    with pytest.raises(SystemExit):
        module.resolve("other/a-similarly-named-model")


def test_audio8_requires_a_trained_language_prompt_before_native_execution():
    module = wrapper()
    spec = module.resolve("selcukkubur/Audio8-ASR-Infinite-mlx")
    for language in ["en", "zh"]:
        module.validate_language(spec, language)
    for language in ["", "auto", "pl"]:
        with pytest.raises(SystemExit):
            module.validate_language(spec, language)
    module.validate_language(module.resolve(module.REPO), "")
