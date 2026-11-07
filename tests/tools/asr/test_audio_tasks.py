# this_file: tests/tools/asr/test_audio_tasks.py
"""Task outputs retain their meaning instead of being scored as ASR text."""

import importlib.util
from pathlib import Path

import pytest


def wrapper():
    path = Path(__file__).resolve().parents[3] / "src/vexy_audioflexy/tools/asr" / "tools/audio-task-audiocpp.py"
    spec = importlib.util.spec_from_file_location("audio_tasks", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_turn_schema_when_valid_then_preserves_probability_and_decision():
    data = wrapper().turn_result(
        'log\ncustom_schema_output={"schema":"smart_turn.v1","data":{"probability":0.75,"complete":true}}\n'
    )
    assert data == {"probability": 0.75, "complete": True}, (
        "Turn completion is not diarization"
    )


@pytest.mark.parametrize(
    "content",
    [
        "no schema",
        'custom_schema_output={"schema":"wrong","data":{}}',
        'custom_schema_output={"schema":"smart_turn.v1","data":{"probability":2,"complete":true}}',
        'custom_schema_output={"schema":"smart_turn.v1","data":{"probability":0.2,"complete":"false"}}',
    ],
)
def test_turn_schema_when_invalid_then_rejects_false_success(content):
    with pytest.raises(ValueError):
        wrapper().turn_result(content)


def test_alignment_when_sample_offsets_returned_then_converts_to_seconds():
    result = wrapper().alignment_words(
        [{"word": "hello", "start_sample": 1600, "end_sample": 8000}], 1
    )
    assert result == [{"text": "hello", "start": 0.1, "end": 0.5}], (
        "Sample offsets use 16 kHz input"
    )


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [{"word": "hello", "start_sample": 8000, "end_sample": 1600}],
        [{"word": "hello", "start_sample": True, "end_sample": 8000}],
        [{"word": "hello", "start_sample": 0, "end_sample": 32000}],
    ],
)
def test_alignment_when_invalid_then_rejects_bad_timeline(rows):
    with pytest.raises(ValueError):
        wrapper().alignment_words(rows, 1)
