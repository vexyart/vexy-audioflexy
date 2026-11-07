# this_file: tests/tools/asr/test_appautomaton.py
"""App Automaton Qwen ASR exact artifact and language boundaries."""
import importlib.util
from pathlib import Path

import pytest


def wrapper():
    spec = importlib.util.spec_from_file_location("appautomaton_asr", Path(__file__).resolve().parents[3] / "src/vexy_audioflexy/tools/asr" / "tools/asr-appautomaton.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_language_scope_is_not_widened_beyond_this_runtime():
    module = wrapper()
    assert module.language("en") == "English"
    assert module.language("zh") == "Chinese"
    assert module.language("") is None
    with pytest.raises(SystemExit):
        module.language("pl")


def test_empty_or_nontext_native_result_is_rejected():
    module = wrapper()
    assert module.text_result(type("Result", (), {"text": " actual words "})()) == "actual words"
    for value in [None, object(), type("Result", (), {"text": ""})()]:
        with pytest.raises(SystemExit):
            module.text_result(value)
