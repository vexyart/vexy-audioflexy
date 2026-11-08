# this_file: tests/test_speech_alignment.py
"""Alignment repair boundaries and lexical invariants use synthetic timestamps."""

from vexy_audioflexy.speech.alignment import repair, windows
from vexy_audioflexy.speech.types import Word


def test_collapsed_times_group_with_context():
    words = [Word(f"word{i}", i, i + 0.5) for i in range(40)]
    words[20].end = words[20].start
    assert windows(words) == [(8, 33)]
    words[22].end = words[22].start
    assert windows(words) == [(8, 35)]


def test_normal_times_do_not_load_alignment_model(tmp_path):
    result = repair(tmp_path / "absent.wav", [Word("hello", 0, 1)])
    assert result["windows"] == 0 and result["repaired_words"] == 0


def test_audio_alignment_preserves_words_and_records_actual_model_times(
    tmp_path, monkeypatch
):
    import sys
    from types import SimpleNamespace as N

    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        return N(
            items=[
                N(text="hello", start_time=0.1, end_time=0.4),
                N(text="world", start_time=0.5, end_time=0.9),
            ]
        )

    aligner = N(
        aligner_processor=N(clean_token=lambda text: text.strip(".")), generate=generate
    )
    monkeypatch.setitem(
        sys.modules, "soundfile", N(read=lambda *a, **k: ([0.0] * 32000, 16000))
    )
    monkeypatch.setitem(sys.modules, "mlx_audio.stt", N(load=lambda *a, **k: aligner))
    words = [Word("hello", 0, 0), Word("world.", 0.5, 1)]
    result = repair(tmp_path / "clip.wav", words)
    assert [w.text for w in words] == ["hello", "world."]
    assert [(w.start, w.end) for w in words] == [(0.1, 0.4), (0.5, 1)]
    assert result["repaired_words"] == 1 and calls[0]["language"] == "English"
