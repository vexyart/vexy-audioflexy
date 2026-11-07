# this_file: tests/test_ported_audio_speechlens.py
"""Reject invalid native completion and preserve the audio timeline."""
import importlib.util
from pathlib import Path

import pytest


def wrapper():
    spec = importlib.util.spec_from_file_location("speechlens", (Path(__file__).resolve().parents[1] / "src/vexy_audioflexy/tools/fx") / "tools/aud-speechlens.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_native_diagnostic_requires_completed_schema_and_finite_pipeline_time():
    module = wrapper()
    assert module.processing_seconds({"schemaVersion": 2, "outcome": "completed", "processingDurationSeconds": .25}) == .25
    for value in [None, -1, float("nan"), float("inf"), True]:
        with pytest.raises(SystemExit):
            module.processing_seconds({"schemaVersion": 2, "outcome": "completed", "processingDurationSeconds": value})
    with pytest.raises(SystemExit):
        module.processing_seconds({"schemaVersion": 2, "outcome": "failed", "processingDurationSeconds": 1})


def test_waveform_validation_rejects_truncation_or_nonfinite_output(tmp_path):
    import numpy as np
    import soundfile as sf

    source, output = tmp_path / "source.wav", tmp_path / "output.wav"
    samples = np.sin(np.arange(1600) / 10).astype("float32")
    sf.write(source, samples, 16000, subtype="FLOAT")
    sf.write(output, samples / 2, 16000, subtype="FLOAT")
    assert wrapper().verify_waveform(source, output)["frames"] == 1600
    sf.write(output, samples[:-1], 16000, subtype="FLOAT")
    with pytest.raises(SystemExit):
        wrapper().verify_waveform(source, output)

    samples[0] = np.nan
    sf.write(output, samples, 16000, subtype="FLOAT")
    with pytest.raises(SystemExit):
        wrapper().verify_waveform(source, output)


@pytest.mark.parametrize("subtype", ["PCM_16", "PCM_24", "FLOAT"])
def test_batch_audio_stats_recognizes_integer_and_float_wav(tmp_path, subtype):
    import numpy as np
    import soundfile as sf

    path = (Path(__file__).resolve().parents[1] / "src/vexy_audioflexy/tools/fx") / "aud_all.py"
    spec = importlib.util.spec_from_file_location("speechlens_batch", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    audio = tmp_path / "native.wav"
    sf.write(audio, np.full((1600, 2), .25), 16000, subtype=subtype)
    stats = module.audio_stats(audio)
    assert stats["rms"] == pytest.approx(.25, abs=1e-5), "Audible native PCM must not be reported as silence"
    assert stats["duration_s"] == .1
    assert stats["sample_rate"] == 16000
