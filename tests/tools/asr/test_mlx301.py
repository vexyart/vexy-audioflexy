# this_file: tests/tools/asr/test_mlx301.py
"""Exact requested conversions use model-specific recognition contracts."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


def wrapper():
    path = Path(__file__).resolve().parents[3] / "src/vexy_audioflexy/tools/asr" / "tools/asr-mlx301.py"
    spec = importlib.util.spec_from_file_location("mlx301", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_granite_when_language_supplied_then_transcribes_instead_of_translating():
    kwargs = wrapper().generation_options(
        "divydeep/granite-speech-4.1-2b-mlx-8bit", "en", ""
    )
    assert "language" not in kwargs, "Granite language-only mode requests translation"
    assert "Transcribe" in kwargs["prompt"], "Use the documented recognition prompt"


def test_cohere_when_prompt_supplied_then_rejects_silently_ignored_context():
    with pytest.raises(ValueError):
        wrapper().generation_options(
            "littoralai/cohere-transcribe-mlx-fp16", "en", "Names: Example"
        )


def test_qwen_when_language_code_supplied_then_uses_native_language_name():
    kwargs = wrapper().generation_options(
        "majentik/Qwen3-ASR-1.7B-MLX-8bit", "pl", "context"
    )
    assert kwargs["language"] == "Polish", "Use native Qwen prompt language names"
    assert kwargs["system_prompt"] == "context", "Preserve the requested context"


@pytest.mark.parametrize(
    "result", [None, object(), SimpleNamespace(text=""), SimpleNamespace(text=None)]
)
def test_missing_model_text_when_result_object_returned_then_rejects_repr(result):
    with pytest.raises(ValueError):
        wrapper().text_result(result)
