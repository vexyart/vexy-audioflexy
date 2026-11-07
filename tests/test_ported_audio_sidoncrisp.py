# this_file: tests/test_ported_audio_sidoncrisp.py
"""Exact cstr weights, native resampling geometry and atomic failure boundary."""
import importlib.util
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


def wrapper():
    path = (Path(__file__).resolve().parents[1] / "src/vexy_audioflexy/tools/fx") / "tools/aud-sidoncrisp.py"
    spec = importlib.util.spec_from_file_location("sidoncrisp", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_four_packages_match_published_lfs_hashes():
    m = wrapper()
    data = json.loads((m.ROOT / "catalogs/sidon-published-hashes.json").read_text())
    published = {f["rfilename"]: f.get("lfs", {}).get("sha256") for f in data["files"]}
    assert len(m.PACKAGES) == 4
    assert all(published[name] == digest for name, digest in m.PACKAGES.values())


def test_polyphase_duration_rounds_before_threefold_output():
    m = wrapper()
    assert m.output_frames(4411, 44100) == 4803
    assert m.output_frames(146907, 24000) == 293814
    assert m.output_frames(16000, 16000) == 48000


@pytest.mark.parametrize("signal,rate", [(np.ones(48000), 24000), (np.ones((48000, 2)), 48000),
    (np.full(48000, np.nan), 48000), (np.ones(47999), 48000)])
def test_invalid_native_output_is_rejected(tmp_path, signal, rate):
    p = tmp_path / "output.wav"
    sf.write(p, signal, rate, subtype="FLOAT")
    with pytest.raises(ValueError):
        wrapper().inspect_audio(p, 48000)


@pytest.mark.parametrize("signal", [np.zeros(16000), np.ones(16000) * .1])
def test_native_failure_preserves_prior_wave_and_accepts_silent_input(tmp_path, monkeypatch, signal):
    m = wrapper()
    src, target = tmp_path / "input.wav", tmp_path / "output.wav"
    sf.write(src, signal, 16000, subtype="FLOAT")
    target.write_bytes(b"keep prior wave")
    binary = tmp_path / "build/bin/crispasr"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"test native binary")
    monkeypatch.setenv("CRISPASR_ISSUE201_ROOT", str(tmp_path))
    monkeypatch.setattr(m, "BINARY_SHA256", m.sha(binary))
    monkeypatch.setattr(m.subprocess, "check_output", lambda *a, **k: m.COMMIT)
    monkeypatch.setattr(m, "prep", lambda *a: "verified.gguf")
    monkeypatch.setattr(m.subprocess, "run", lambda argv, **k: subprocess.CompletedProcess(argv, 15, "", "duration failure"))
    with pytest.raises(SystemExit, match="Native Sidon failed"):
        m.run(str(src), str(target), force=True)
    assert target.read_bytes() == b"keep prior wave", "Native failure must preserve previous output"


def test_unknown_package_fails_before_download():
    with pytest.raises(SystemExit, match="registered"):
        wrapper().prep("unknown")


def test_resume_rejects_source_output_package_or_binary_changes(tmp_path, monkeypatch):
    m = wrapper()
    src, target = tmp_path / "input.wav", tmp_path / "out.wav"
    src.write_bytes(b"source")
    target.write_bytes(b"output")
    binary = tmp_path / "build/bin/crispasr"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"binary")
    monkeypatch.setenv("CRISPASR_ISSUE201_ROOT", str(tmp_path))
    monkeypatch.setattr(m, "BINARY_SHA256", m.sha(binary))
    receipt = {"repo": m.REPO, "revision": m.REVISION, "package": "f16", "weights_sha256": m.PACKAGES["f16"][1],
        "backend": "metal", "runtime_commit": m.COMMIT, "binary_sha256": m.BINARY_SHA256,
        "input_sha256": m.sha(src), "audio": {"sha256": m.sha(target)}}
    target.with_suffix(".report.json").write_text(json.dumps(receipt))
    assert m.resume_ready(str(src), str(target), "f16"), "Exact receipt must resume"
    assert not m.resume_ready(str(src), str(target), "q4_k"), "Different package must not resume"
    for p in (src, target, binary):
        old = p.read_bytes()
        p.write_bytes(b"changed")
        assert not m.resume_ready(str(src), str(target), "f16"), f"Changed {p.name} must not resume"
        p.write_bytes(old)


def test_four_batch_routes_preserve_exact_repo_quant_and_default_native_backend():
    import runpy

    m = runpy.run_path(str((Path(__file__).resolve().parents[1] / "src/vexy_audioflexy/tools/fx") / "aud_all.py"))
    rows = [r for r in m["MODELS"] if r["repo"] == "cstr/Sidon-GGUF"]
    assert len(rows) == 4 and {r["quant"] for r in rows} == {"f16", "q8_0", "q6_k", "q4_k"}
    assert all(r["source"] == "noisy" and r["stack"] == "crispasr" for r in rows)
    assert all("--family" in m["build_args"](r, "out.wav") for r in rows)
