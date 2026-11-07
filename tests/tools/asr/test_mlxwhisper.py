# this_file: tests/tools/asr/test_mlxwhisper.py
"""MLX Whisper adapter preserves native words and rejects false-success output."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def wrapper():
    path = Path(__file__).resolve().parents[3] / "src/vexy_audioflexy/tools/asr" / "tools/asr-mlxwhisper.py"
    spec = importlib.util.spec_from_file_location("mlxwhisper", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_native_words_when_prompt_file_used_then_preserves_context_and_times(
    tmp_path, monkeypatch
):
    module = wrapper()
    source = tmp_path / "input.wav"
    source.write_bytes(b"fixture")
    prompt = tmp_path / "hints.md"
    prompt.write_text("Terms: QuasarAPI, Python.\nProgramme context.")
    output = tmp_path / "result.txt"
    seen = {}

    def transcribe(audio, **kwargs):
        seen.update(kwargs)
        return {
            "text": "QuasarAPI works.",
            "segments": [
                {
                    "words": [
                        {"word": " QuasarAPI", "start": 0.1, "end": 0.7},
                        {"word": " works.", "start": 0.8, "end": 1.3},
                    ]
                }
            ],
        }

    monkeypatch.setitem(
        sys.modules, "mlx_whisper", SimpleNamespace(transcribe=transcribe)
    )
    monkeypatch.setattr(module, "prep", lambda *args: "pinned-model")
    monkeypatch.setattr(module, "duration", lambda _: 2.0)
    module.transcribe(str(source), text_out=str(output), prompt_file=str(prompt))
    data = json.loads(output.with_suffix(".native.json").read_text())
    assert seen["initial_prompt"] == "QuasarAPI, Python", (
        "Forward vocabulary rather than prose"
    )
    assert seen["word_timestamps"] is True, "Request native word timing"
    assert seen["path_or_hf_repo"] == "pinned-model", "Use the resolved pinned snapshot"
    assert data["words"][0] == {"text": "QuasarAPI", "start": 0.1, "end": 0.7}
    assert output.read_text().strip() == "QuasarAPI works."


@pytest.mark.parametrize(
    "result",
    [
        {"text": "", "segments": []},
        {"text": "Speech", "segments": []},
        {
            "text": "Speech",
            "segments": [{"words": [{"word": "Speech", "start": 2, "end": 1}]}],
        },
        {
            "text": "Speech",
            "segments": [
                {"words": [{"word": "Speech", "start": 0, "end": float("nan")}]}
            ],
        },
    ],
)
def test_invalid_native_output_when_forcing_then_preserves_previous_transcript(
    tmp_path, monkeypatch, result
):
    module = wrapper()
    source = tmp_path / "input.wav"
    source.write_bytes(b"fixture")
    output = tmp_path / "result.txt"
    output.write_text("previous transcript")
    monkeypatch.setitem(
        sys.modules, "mlx_whisper", SimpleNamespace(transcribe=lambda *a, **k: result)
    )
    monkeypatch.setattr(module, "prep", lambda *args: "pinned-model")
    monkeypatch.setattr(module, "duration", lambda _: 2.0)
    with pytest.raises(ValueError):
        module.transcribe(str(source), text_out=str(output), force=True)
    assert output.read_text() == "previous transcript", (
        "Failed inference must preserve existing output"
    )


def test_input_path_when_used_as_output_then_rejects_before_model_load(tmp_path):
    source = tmp_path / "input.wav"
    source.write_bytes(b"fixture")
    with pytest.raises(ValueError):
        wrapper().transcribe(str(source), text_out=str(source))
    assert source.read_bytes() == b"fixture", "Never overwrite source media"
