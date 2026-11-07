# this_file: tests/test_ported_audio_restoration.py
"""48k restoration preserves Sidon duration and records UniverSR native cropping."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


def wrapper():
    path = (Path(__file__).resolve().parents[1] / "src/vexy_audioflexy/tools/fx") / "tools/aud-restoration.py"
    spec = importlib.util.spec_from_file_location("restoration", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_registered_weights_match_exact_published_artifact_hashes():
    module = wrapper()
    metadata = json.loads((module.ROOT / "catalogs/restoration-published-hashes.json").read_text())
    published = {f["rfilename"]: f.get("lfs", {}).get("sha256") for f in metadata["files"]}
    assert all(published[entry[1]] == entry[2] for entry in module.PACKAGES.values())


def test_output_contract_rejects_changed_rate_channels_nonfinite_and_missing_duration(tmp_path):
    module = wrapper()
    path = tmp_path / "out.wav"
    for rate, signal in [(24000, np.ones(48000)), (48000, np.ones((48000, 2))),
            (48000, np.full(48000, np.nan)), (48000, np.ones(47488))]:
        sf.write(path, signal, rate, subtype="FLOAT")
        with pytest.raises(ValueError):
            module.inspect_audio(path, 48000, "sidon")
    sf.write(path, np.ones(47500), 48000, subtype="FLOAT")
    assert module.inspect_audio(path, 48000, "universr")["missing_tail_samples"] == 500
    sf.write(path, np.ones(47487), 48000, subtype="FLOAT")
    with pytest.raises(ValueError):
        module.inspect_audio(path, 48000, "universr")


@pytest.mark.parametrize("package,rate,bandwidth", [("unknown", 16000, 0), ("universr_audio_orig", 44100, 0),
    ("universr_speech_orig", 16000, 11025), ("sidon_f32", 16000, 8000)])
def test_invalid_bandwidth_or_package_fails_before_inference(package, rate, bandwidth):
    with pytest.raises(SystemExit):
        wrapper().request_options(package, rate, bandwidth, 4, 42)


def test_native_duration_rounding_preserves_each_family_contract():
    module = wrapper()
    assert module.expected_output_frames(100, 44100, "universr") == 109
    assert module.expected_output_frames(100, 44100, "sidon") == 108
    assert module.expected_output_frames(100, 16000, "universr") == 300


@pytest.mark.parametrize("signal", [np.zeros(16000), np.zeros((16000, 2))])
def test_sidon_rejects_zero_input_before_loading_weights(tmp_path, monkeypatch, signal):
    module = wrapper()
    path = tmp_path / "silent.wav"
    sf.write(path, signal, 16000, subtype="FLOAT")
    monkeypatch.setattr(module, "prep", lambda *args: pytest.fail("Silent mix must fail before weights load"))
    with pytest.raises(SystemExit, match="non-silent input"):
        module.run(str(path), str(tmp_path / "out.wav"), package="sidon_f32")


@pytest.mark.parametrize("signal", [np.ones(16000) * .1, np.column_stack([np.ones(16000) * .1, -np.ones(16000) * .1])])
def test_failed_native_restoration_preserves_prior_output(tmp_path, monkeypatch, signal):
    import subprocess

    module = wrapper()
    source, target = tmp_path / "input.wav", tmp_path / "old.wav"
    sf.write(source, signal, 16000, subtype="FLOAT")
    target.write_bytes(b"keep previous artifact")
    binary = tmp_path / "build/macos-metal-release/bin/audiocpp_cli"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"test build")
    monkeypatch.setenv("AUDIOCPP_ISSUE201_ROOT", str(tmp_path))
    monkeypatch.setattr(module.subprocess, "check_output", lambda *args, **kwargs: module.COMMIT)
    monkeypatch.setattr(module, "prep", lambda *args: "pinned weights")
    monkeypatch.setattr(module.subprocess, "run", lambda argv, **kwargs: subprocess.CompletedProcess(argv, 1, "", "native error"))
    with pytest.raises(SystemExit, match="Native restoration failed"):
        module.run(str(source), str(target), force=True)
    assert target.read_bytes() == b"keep previous artifact", "Native failure must preserve the prior WAV"
