# this_file: tests/test_ported_audio_deepfilternet_coreml.py
"""Guard exact native DF3 assets, geometry, failed output and stale reuse."""
import importlib.util
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


def wrapper():
    p = (Path(__file__).resolve().parents[1] / "src/vexy_audioflexy/tools/fx") / "tools/aud-deepfilternet-coreml.py"
    s = importlib.util.spec_from_file_location("df3coreml", p)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def test_geometry_admits_only_original_resampler_rounding(tmp_path):
    m = wrapper()
    p = tmp_path / "out.wav"
    sf.write(p, np.zeros(4801), 48000, subtype="FLOAT")
    assert m.inspect_audio(p, 4411, 44100)["frames"] == 4801
    sf.write(p, np.zeros(4803), 48000, subtype="FLOAT")
    with pytest.raises(ValueError):
        m.inspect_audio(p, 4411, 44100)
    sf.write(p, np.zeros(4799), 48000, subtype="FLOAT")
    with pytest.raises(ValueError):
        m.inspect_audio(p, 4800, 48000)


@pytest.mark.parametrize("code", [0, 1])
def test_native_exit_without_wave_preserves_prior_output(tmp_path, monkeypatch, code):
    m = wrapper()
    src, out = tmp_path / "input.wav", tmp_path / "output.wav"
    sf.write(src, np.zeros(4800), 48000)
    out.write_bytes(b"previous audio")
    binary = tmp_path / m.CATALOG["binary_relative"]
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"native binary")
    monkeypatch.setenv("DF3_COREML_ISSUE201_ROOT", str(tmp_path))
    monkeypatch.setitem(m.CATALOG, "binary_sha256", m.sha(binary))
    monkeypatch.setattr(m.subprocess, "check_output", lambda *a, **k: m.CATALOG["runtime_commit"])
    monkeypatch.setattr(m, "prep", lambda *a: "pinned bundle")
    monkeypatch.setattr(m.subprocess, "run", lambda argv, **k: subprocess.CompletedProcess(argv, code, "engine failed", ""))
    with pytest.raises((ValueError, SystemExit, FileNotFoundError)):
        m.run(str(src), str(out), force=True)
    assert out.read_bytes() == b"previous audio", "Exit-zero engine failures must preserve earlier audio"


def test_bad_source_and_unknown_profile_fail_before_model_load(tmp_path, monkeypatch):
    m = wrapper()
    monkeypatch.setattr(m, "prep", lambda *a: pytest.fail("Invalid source must not fetch models"))
    p = tmp_path / "input.wav"
    sf.write(p, np.full(100, np.nan), 48000, subtype="FLOAT")
    with pytest.raises(SystemExit):
        m.run(str(p), str(tmp_path / "out.wav"))
    with pytest.raises(SystemExit):
        m.run(str(p), str(tmp_path / "out.wav"), package="unknown")


def test_prep_rejects_corrupt_or_symlinked_materialized_assets(tmp_path, monkeypatch):
    m = wrapper()
    monkeypatch.setattr(m, "CACHE_ROOT", tmp_path / ".cache/issue201")
    assets = tmp_path / ".cache/issue201/deepfilternet-coreml-materialized"
    assets.mkdir(parents=True)
    p = assets / "model.safetensors"
    p.write_bytes(b"pinned")
    monkeypatch.setitem(m.CATALOG, "files", {p.name: m.sha(p)})
    assert m.prep() == str(assets)
    p.write_bytes(b"corrupt")
    with pytest.raises(SystemExit, match="checksum"):
        m.prep()
    p.unlink()
    other = tmp_path / "original"
    other.write_bytes(b"pinned")
    p.symlink_to(other)
    with pytest.raises(SystemExit, match="checksum"):
        m.prep()


def test_registered_route_selects_exact_fp16_coreml_package():
    import runpy

    a = runpy.run_path(str((Path(__file__).resolve().parents[1] / "src/vexy_audioflexy/tools/fx") / "aud_all.py"))
    rows = [r for r in a["MODELS"] if r["repo"] == "iky1e/DeepFilterNet3-Streaming-CoreML"]
    assert len(rows) == 1, "Exact published DF3 graph requires its own consumer route"
    assert rows[0]["quant"] == "fp16" and rows[0]["stack"] == "coreml"
    assert rows[0]["package"] == "df3_fp16" and rows[0]["source"] == "noisy"


def test_resume_rejects_changed_input_wave_binary_or_model(tmp_path, monkeypatch):
    m = wrapper()
    monkeypatch.setattr(m, "CACHE_ROOT", tmp_path / ".cache/issue201")
    monkeypatch.setenv("DF3_COREML_ISSUE201_ROOT", str(tmp_path))
    src, out = tmp_path / "input.wav", tmp_path / "output.wav"
    binary = tmp_path / m.CATALOG["binary_relative"]
    model = tmp_path / ".cache/issue201/deepfilternet-coreml-materialized/model.safetensors"
    for p in (src, out, binary, model):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(p.name.encode())
    monkeypatch.setitem(m.CATALOG, "files", {"model.safetensors": m.sha(model)})
    monkeypatch.setitem(m.CATALOG, "binary_sha256", m.sha(binary))
    receipt = {"package": m.CATALOG["package"], "backend": "cpu", "repo": m.CATALOG["repo"],
               "revision": m.CATALOG["revision"], "runtime_commit": m.CATALOG["runtime_commit"],
               "model_files_sha256": m.CATALOG["files"], "binary_sha256": m.sha(binary),
               "input_sha256": m.sha(src), "audio": {"sha256": m.sha(out)}}
    out.with_suffix(".report.json").write_text(json.dumps(receipt))
    assert m.resume_ready(str(src), str(out), m.CATALOG["package"])
    for p in (src, out, binary, model):
        original = p.read_bytes(); p.write_bytes(b"changed")
        assert not m.resume_ready(str(src), str(out), m.CATALOG["package"]), f"Stale {p.name} must invalidate reuse"
        p.write_bytes(original)
