# this_file: tests/tools/asr/test_indexecho.py
"""Native subtitle structure must not mix source words with translations."""
import importlib.util
from pathlib import Path

import pytest


def wrapper():
    spec = importlib.util.spec_from_file_location("index_echo", Path(__file__).resolve().parents[3] / "src/vexy_audioflexy/tools/asr" / "tools/asr-indexecho.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_subtitles_keep_source_translation_and_time_separate():
    cues = wrapper().parse_cues("[00:00.20-00:02.80]\n这是一个开始。\nThis is a beginning.\n[00:03.00-00:04.00]\n再见。\nGoodbye.", 5)
    assert len(cues) == 2
    assert cues[0] == {"start": .2, "end": 2.8, "source": "这是一个开始。", "translation": "This is a beginning."}


@pytest.mark.parametrize("text", ["garbage", "[00:00.00-00:01.00]\n中文", "[00:03.00-00:01.00]\n中文\nEnglish", "[00:00.00-00:08.00]\n中文\nEnglish"])
def test_incomplete_reversed_or_out_of_audio_cues_are_rejected(text):
    with pytest.raises(SystemExit):
        wrapper().parse_cues(text, 5)
