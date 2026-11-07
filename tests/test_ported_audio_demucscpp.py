"""Multi-stem validation for the native Demucs adapter.

this_file: tests/test_ported_audio_demucscpp.py
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

ROOT = (Path(__file__).resolve().parents[1] / "src/vexy_audioflexy/tools/fx")
spec = importlib.util.spec_from_file_location("demucscpp", ROOT / "tools/aud-demucscpp.py")
wrapper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wrapper)


def test_four_and_six_source_checkpoints_preserve_upstream_stem_order():
    assert wrapper.resolve("htdemucs-4s") == ("ggml-model-htdemucs-4s-f16.bin", ("drums", "bass", "other", "vocals"))
    assert wrapper.resolve("htdemucs-6s")[1] == ("drums", "bass", "other", "vocals", "guitar", "piano")
    with pytest.raises(SystemExit):
        wrapper.resolve("unpublished")


def test_inspection_retains_all_stems_and_reconstructs_sum(tmp_path):
    for index, name in enumerate(wrapper.STEMS[:4]):
        sf.write(tmp_path / f"target_{index}_{name}.wav", np.full((32, 2), 0.01 * (index + 1)), 44100, subtype="FLOAT")
    records, summed = wrapper.inspect_stems(tmp_path, wrapper.STEMS[:4], 32)
    assert [item["stem"] for item in records] == list(wrapper.STEMS[:4])
    assert np.allclose(summed, 0.1), "Reconstruction omitted or reordered a stem"
    assert all(len(item["sha256"]) == 64 for item in records)


@pytest.mark.parametrize("rate,signal", [(16000, np.ones((32, 2))), (44100, np.ones((32, 1))), (44100, np.full((32, 2), np.nan))])
def test_bad_stem_geometry_or_nonfinite_samples_fail(tmp_path, rate, signal):
    sf.write(tmp_path / "target_0_drums.wav", signal, rate, subtype="FLOAT")
    with pytest.raises(SystemExit):
        wrapper.inspect_stems(tmp_path, ("drums",), 32)


def test_all_silent_stems_fail(tmp_path):
    sf.write(tmp_path / "target_0_drums.wav", np.zeros((32, 2)), 44100)
    with pytest.raises(SystemExit):
        wrapper.inspect_stems(tmp_path, ("drums",), 32)
