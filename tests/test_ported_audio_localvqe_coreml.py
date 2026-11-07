# this_file: tests/test_ported_audio_localvqe_coreml.py
"""Core ML LocalVQE boundaries retain delay-corrected native streaming audio."""
import importlib.util
import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


def wrapper():
    p = (Path(__file__).resolve().parents[1] / "src/vexy_audioflexy/tools/fx") / "tools/aud-localvqe-coreml.py"
    s = importlib.util.spec_from_file_location("localvqecoreml", p)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def test_four_exact_bundles_have_all_four_native_files():
    m = wrapper()
    assert len(m.CATALOG["packages"]) == 4
    assert all(set(p["files"]) == {"analytics/coremldata.bin", "coremldata.bin", "model.mil", "weights/weight.bin"}
        for p in m.CATALOG["packages"].values())


def test_output_geometry_keeps_only_native_resampling_rounding(tmp_path):
    m = wrapper()
    p = tmp_path / "out.wav"
    sf.write(p, np.ones(16001), 16000, subtype="FLOAT")
    assert m.inspect_audio(p, 24001, 24000)["frames"] == 16001
    sf.write(p, np.ones(16002), 16000, subtype="FLOAT")
    with pytest.raises(ValueError):
        m.inspect_audio(p, 24001, 24000)
    with pytest.raises(ValueError):
        m.inspect_audio(p, 16001, 16000)


def test_aec_requires_synchronized_reference_before_model_load(tmp_path, monkeypatch):
    m = wrapper()
    src, ref = tmp_path / "mic.wav", tmp_path / "ref.wav"
    sf.write(src, np.ones(16000), 16000)
    sf.write(ref, np.ones(15999), 16000)
    monkeypatch.setattr(m, "prep", lambda *a: pytest.fail("Invalid reference must fail before prep"))
    for path in ("", str(ref)):
        with pytest.raises(SystemExit, match="reference"):
            m.run(str(src), str(tmp_path / "out.wav"), task="aec", reference=path)


@pytest.mark.parametrize("signal", [np.zeros(16000), np.ones(16000) * .1])
def test_native_failure_preserves_previous_audio_and_admits_silence(tmp_path, monkeypatch, signal):
    m = wrapper()
    src, out = tmp_path / "mic.wav", tmp_path / "out.wav"
    sf.write(src, signal, 16000, subtype="FLOAT")
    out.write_bytes(b"previous audio")
    binary = tmp_path / ".build/release/fluidaudiocli"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"native binary")
    monkeypatch.setenv("FLUIDAUDIO_ISSUE201_ROOT", str(tmp_path))
    monkeypatch.setitem(m.CATALOG, "binary_sha256", m.sha(binary))
    monkeypatch.setattr(m.subprocess, "check_output", lambda *a, **k: m.CATALOG["runtime_commit"])
    monkeypatch.setattr(m, "prep", lambda *a: "pinned bundle")
    monkeypatch.setattr(m.subprocess, "run", lambda argv, **k: subprocess.CompletedProcess(argv, 1, "", "native error"))
    with pytest.raises(SystemExit, match="Native LocalVQE failed"):
        m.run(str(src), str(out), force=True)
    assert out.read_bytes() == b"previous audio", "Failed inference must preserve the existing WAV"


def test_eight_batch_routes_use_real_far_end_only_for_aec():
    import runpy

    m = runpy.run_path(str((Path(__file__).resolve().parents[1] / "src/vexy_audioflexy/tools/fx") / "aud_all.py"))
    m["TARGETS"]["aec-far"] = Path("synthetic-far.wav")
    rows = [r for r in m["MODELS"] if r["repo"] == "FluidInference/localvqe-coreml"]
    assert len(rows) == 8
    for r in rows:
        args = m["build_args"](r, "output.wav")
        assert ("--voice_ref" in args) == (r["task"] == "aec"), "Only AEC routes require real far-end input"
        assert r["stack"] == "coreml" and r["quant"] == "fp32"


def test_resume_rejects_changed_source_reference_output_binary_and_weights(tmp_path, monkeypatch):
    import json

    m = wrapper()
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / ".cache/issue201/huggingface"))
    import huggingface_hub.constants
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(tmp_path / ".cache/issue201/huggingface"))
    monkeypatch.setenv("FLUIDAUDIO_ISSUE201_ROOT", str(tmp_path))
    src, ref, out = (tmp_path / n for n in ("mic.wav", "far.wav", "out.wav"))
    binary = tmp_path / ".build/release/fluidaudiocli"
    binary.parent.mkdir(parents=True)
    for p in (src, ref, out, binary):
        p.write_bytes(p.name.encode())
    monkeypatch.setitem(m.CATALOG, "binary_sha256", m.sha(binary))
    info = m.CATALOG["packages"]["v13_256ms"]
    weight = tmp_path / f".cache/issue201/huggingface/models--FluidInference--localvqe-coreml/snapshots/{m.CATALOG['revision']}" / info["bundle"] / "weight.bin"
    weight.parent.mkdir(parents=True)
    weight.write_bytes(b"model weights")
    info["files"] = {"weight.bin": m.sha(weight)}
    receipt = {"repo": m.CATALOG["repo"], "revision": m.CATALOG["revision"], "package": "v13_256ms",
        "backend": "cpu", "runtime_commit": m.CATALOG["runtime_commit"], "model_files_sha256": info["files"],
        "binary_sha256": m.sha(binary), "input_sha256": m.sha(src), "reference_sha256": m.sha(ref),
        "audio": {"sha256": m.sha(out)}}
    out.with_suffix(".report.json").write_text(json.dumps(receipt))
    assert m.resume_ready(str(src), str(out), "v13_256ms", reference=str(ref))
    for p in (src, ref, out, binary, weight):
        prior = p.read_bytes()
        p.write_bytes(b"changed")
        assert not m.resume_ready(str(src), str(out), "v13_256ms", reference=str(ref)), f"Changed {p.name} must invalidate resume"
        p.write_bytes(prior)
