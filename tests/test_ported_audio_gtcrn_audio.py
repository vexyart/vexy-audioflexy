# this_file: tests/test_ported_audio_gtcrn_audio.py
"""Centered GTCRN framing must retain partial hops, silence and short inputs."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


def frontend():
    path = (Path(__file__).resolve().parents[1] / "src/vexy_audioflexy/tools/fx") / "tools/gtcrn_audio.py"
    spec = importlib.util.spec_from_file_location("gtcrn_audio", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("length", [1, 13, 256, 512, 97938])
def test_frontend_roundtrip_preserves_complete_timeline(length):
    module = frontend()
    samples = np.random.default_rng(201).normal(0, .1, length).astype(np.float32)
    np.testing.assert_allclose(module.synthesize(module.analyze(samples), length), samples, atol=1e-7)


def test_frontend_rejects_empty_and_nonfinite_and_preserves_silence():
    module = frontend()
    silence = np.zeros(16000, np.float32)
    assert not module.synthesize(module.analyze(silence), len(silence)).any()
    for value in [np.array([]), np.array([np.nan])]:
        with pytest.raises(ValueError):
            module.analyze(value)
