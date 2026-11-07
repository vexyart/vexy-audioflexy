# this_file: tests/test_ported_audio_musiccpp.py
"""Restoration/stem verification must preserve the native stereo timeline."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


def wrapper():
    path = (Path(__file__).resolve().parents[1] / "src/vexy_audioflexy/tools/fx") / "tools/aud-musiccpp.py"
    spec = importlib.util.spec_from_file_location("musiccpp", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_music_geometry_rejects_changed_channels_length_and_nonfinite_samples(tmp_path):
    path = tmp_path / "restored.wav"
    signal = np.ones((441, 2), np.float32) * .1
    sf.write(path, signal, 44100, subtype="FLOAT")
    record, _ = wrapper().inspect_audio(path, 441, 2)
    assert record["frames"] == 441 and record["rms"] == pytest.approx(.1)
    for bad in [signal[:-1], signal[:, :1], signal * np.nan]:
        sf.write(path, bad, 44100, subtype="FLOAT")
        with pytest.raises(ValueError):
            wrapper().inspect_audio(path, 441, 2)


def test_htdemucs_cpu_requires_exact_qualified_repair(tmp_path, monkeypatch):
    module = wrapper()
    monkeypatch.setenv("AUDIOCPP_HTDEMUCS_FIXED_ROOT", str(tmp_path))
    with pytest.raises(SystemExit, match="qualified repaired HTDemucs CPU build"):
        module.native_runtime("htdemucs_6stems_f16", "cpu")
    binary = tmp_path / "build/diagnostic/bin/audiocpp_cli"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"unqualified build")
    monkeypatch.setattr(module.subprocess, "check_output", lambda *args, **kwargs: module.COMMIT)
    with pytest.raises(SystemExit, match="qualified repaired HTDemucs CPU build"):
        module.native_runtime("htdemucs_6stems_q8_0", "cpu")


def test_htdemucs_cpu_admits_only_matching_binary_and_patch(tmp_path, monkeypatch):
    import hashlib

    module = wrapper()
    binary = tmp_path / "build/diagnostic/bin/audiocpp_cli"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"test qualified build")
    monkeypatch.setenv("AUDIOCPP_HTDEMUCS_FIXED_ROOT", str(tmp_path))
    monkeypatch.setattr(module, "CPU_BINARY_SHA256", hashlib.sha256(binary.read_bytes()).hexdigest())
    monkeypatch.setattr(module.subprocess, "check_output", lambda *args, **kwargs: module.COMMIT)
    actual, repair = module.native_runtime("htdemucs_6stems_f16", "cpu")
    assert actual == binary and repair == module.CPU_PATCH_SHA256
    monkeypatch.setattr(module, "CPU_PATCH_SHA256", "bad patch hash")
    with pytest.raises(SystemExit, match="qualified repaired HTDemucs CPU build"):
        module.native_runtime("htdemucs_6stems_f16", "cpu")


@pytest.mark.parametrize("repair_marker", [None, "current-patch"])
def test_cpu_resume_rejects_old_or_unqualified_binary_receipt(tmp_path, monkeypatch, repair_marker):
    import hashlib
    import json
    from types import SimpleNamespace

    path = (Path(__file__).resolve().parents[1] / "src/vexy_audioflexy/tools/fx") / "separate_all.py"
    spec = importlib.util.spec_from_file_location("separation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = tmp_path / "input.wav"
    source.write_bytes(b"fixture")
    directory = tmp_path / "out/htdemucs_6stems_f16"
    directory.mkdir(parents=True)
    binary = tmp_path / "engine/build/diagnostic/bin/audiocpp_cli"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"unqualified binary")
    monkeypatch.setenv("AUDIOCPP_HTDEMUCS_FIXED_ROOT", str(tmp_path / "engine"))
    package = wrapper()
    outputs = []
    for stem in package.STEMS:
        (directory / f"{stem}.wav").write_bytes(b"old stem")
        outputs.append({"file": f"{stem}.wav", "sha256": hashlib.sha256(b"old stem").hexdigest()})
    report = {"backend": "cpu", "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "outputs": outputs, "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "cpu_repair_patch_sha256": package.CPU_PATCH_SHA256 if repair_marker else None}
    (directory / "report.json").write_text(json.dumps(report))
    calls = []
    def failed_native(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=1, stdout="unqualified", stderr="")
    monkeypatch.setattr(module.subprocess, "run", failed_native)
    with pytest.raises(SystemExit):
        module.main(str(source), packages="htdemucs_6stems_f16", outdir=str(tmp_path / "out"), backend="cpu")
    assert len(calls) == 1, "A historical or unqualified CPU receipt must trigger native admission checks"
    assert not json.loads((tmp_path / "out/report.json").read_text())["results"][0]["resumed"]
