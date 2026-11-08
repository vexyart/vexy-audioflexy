# this_file: tests/test_speech_diarization.py
"""Backend API contracts and the eight-speaker capacity boundary."""

import sys
from types import SimpleNamespace as N

import pytest

from vexy_audioflexy.speech.inference import diarize


def detector(count):
    return N(
        set_streaming_config=lambda mode: None,
        generate=lambda *a, **k: N(
            segments=[N(start=i, end=i + 0.8, speaker=i) for i in range(count)]
        ),
    )


def test_default_preserves_continuous_nemotron_ids(tmp_path, monkeypatch):
    monkeypatch.setattr("vexy_audioflexy.speech.qwen.mlx_available", lambda: True)
    monkeypatch.setitem(
        sys.modules, "mlx_audio.vad", N(load=lambda *a, **k: detector(2))
    )
    turns, info = diarize(tmp_path / "clip.wav")
    assert [t.speaker for t in turns] == ["speaker_0", "speaker_1"]
    assert info["capacity"] == 8


def test_all_eight_slots_trigger_real_clustering_backend_contract(
    tmp_path, monkeypatch
):
    monkeypatch.setattr("vexy_audioflexy.speech.qwen.mlx_available", lambda: True)
    monkeypatch.setitem(
        sys.modules, "mlx_audio.vad", N(load=lambda *a, **k: detector(8))
    )
    seen = []

    def cluster(path, **kwargs):
        seen.append(kwargs)
        return N(
            num_speakers=10,
            segments=[
                N(start=i, end=i + 0.8, speaker=f"SPEAKER_{i:02}") for i in range(10)
            ],
        )

    monkeypatch.setitem(sys.modules, "diarize", N(diarize=cluster))
    turns, info = diarize(tmp_path / "clip.wav")
    assert len({t.speaker for t in turns}) == 10
    assert info["backend"] == "cluster" and seen == [{"max_speakers": 40}]


def test_unsupported_backend_fails_before_loading_weights(tmp_path):
    with pytest.raises(ValueError, match="diar_backend"):
        diarize(tmp_path / "clip.wav", backend="unknown")
