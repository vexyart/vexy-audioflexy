# this_file: tests/test_ported_audio_localvqe.py
"""Native LocalVQE hop padding, latency compensation and input boundaries."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("localvqe", (Path(__file__).resolve().parents[1] / "src/vexy_audioflexy/tools/fx") / "tools/aud-localvqe.py")
wrapper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wrapper)


@pytest.mark.parametrize("length", [1, 255, 256, 257, 16000])
def test_padding_includes_one_flush_hop_and_preserves_input(length):
    mic = np.ones(length, dtype=np.float32)
    reference = np.zeros(length, dtype=np.float32)
    padded_mic, padded_reference = wrapper.prepare_streams(mic, reference)
    assert len(padded_mic) % 256 == 0
    assert len(padded_mic) >= length + 256, "Missing the one-hop flush for native latency"
    assert np.array_equal(padded_mic[:length], mic)
    assert np.array_equal(padded_reference[:length], reference)


def test_alignment_removes_latency_and_keeps_the_full_original_duration():
    raw = np.concatenate([np.zeros(256), np.arange(257), np.zeros(255)]).astype(np.float32)
    result = wrapper.align_output(raw, 257)
    assert np.array_equal(result, np.arange(257)), "Native hop latency was not removed"


@pytest.mark.parametrize("mic,far", [
    (np.array([]), np.array([])),
    (np.zeros(4), np.zeros(5)),
    (np.ones((4, 2)), np.ones((4, 2))),
    (np.array([np.nan]), np.array([0.])),
])
def test_invalid_streams_fail_before_native_inference(mic, far):
    with pytest.raises(SystemExit):
        wrapper.prepare_streams(mic, far)


def test_incomplete_or_nonfinite_output_is_rejected():
    with pytest.raises(SystemExit):
        wrapper.align_output(np.zeros(256), 10)
    with pytest.raises(SystemExit):
        wrapper.align_output(np.full(512, np.nan), 256)
