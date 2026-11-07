"""Command contracts for newer audio.cpp routes.

this_file: tests/test_ported_audio_audiocpp_new_models.py
"""
import importlib.util
import subprocess
from pathlib import Path

import pytest

ROOT = (Path(__file__).resolve().parents[1] / "src/vexy_audioflexy/tools/fx")
spec = importlib.util.spec_from_file_location("aud_audiocpp", ROOT / "tools/aud-audiocpp.py")
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)


@pytest.mark.parametrize("family,task,output_flag", [
    ("sam_audio", "s2s", "--out-dir"),
    ("sheetsage2", "midi", "--out"),
    ("minimax_music3", "gen", "--out"),
])
def test_new_family_output_contract(tmp_path, monkeypatch, family, task, output_flag):
    monkeypatch.setattr(w, "_require_engine", lambda: Path("audiocpp_cli"))
    monkeypatch.setattr(w, "_resolve", lambda _: {"family": family})
    monkeypatch.setattr(w, "_installed", lambda _: True)
    monkeypatch.setattr(w, "_model_path", lambda *args: tmp_path / "model")
    output = tmp_path / ("stems" if family == "sam_audio" else "result")
    calls = []
    def run(cmd, **kwargs):
        calls.append(cmd)
        if family == "sam_audio":
            output.mkdir()
            (output / "target.wav").write_bytes(b"target")
            (output / "residual.wav").write_bytes(b"residual")
        else:
            output.write_bytes(b"result")
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(w.subprocess, "run", run)
    w.run(audio="source.wav", package="test", output=str(output), text="prompt")
    assert calls[0][calls[0].index("--task") + 1] == task
    assert calls[0][calls[0].index(output_flag) + 1] == str(output)


def test_sam_requires_both_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(w, "_require_engine", lambda: Path("audiocpp_cli"))
    monkeypatch.setattr(w, "_resolve", lambda _: {"family": "sam_audio"})
    monkeypatch.setattr(w, "_installed", lambda _: True)
    monkeypatch.setattr(w, "_model_path", lambda *args: tmp_path / "model")
    output = tmp_path / "stems"
    def run(cmd, **kwargs):
        output.mkdir()
        (output / "target.wav").write_bytes(b"target")
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(w.subprocess, "run", run)
    with pytest.raises(SystemExit):
        w.run(audio="source.wav", package="test", output=str(output), text="speech")


def test_custom_binary_can_be_selected(tmp_path, monkeypatch):
    binary = tmp_path / "custom-audiocpp"
    binary.touch()
    monkeypatch.setenv("AUDIOCPP_BIN", str(binary))
    replacement = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(replacement)
    assert replacement._require_engine() == binary


def test_audio_package_listing_includes_score_and_music(monkeypatch, capsys):
    entries = {
        "score": {"family": "sheetsage2", "tasks": ["midi"], "precision": "f32"},
        "music": {"family": "minimax_music3", "tasks": ["music"], "precision": "q4_0"},
        "sam": {"family": "sam_audio", "tasks": ["s2s"], "precision": "q8_0"},
    }
    monkeypatch.setattr(w, "_specs", lambda: entries)
    w.packages()
    output = capsys.readouterr().out
    assert all(name in output for name in entries)


def test_multifile_install_requires_each_component(tmp_path, monkeypatch):
    entry = {"family": "minimax_music3", "target_directory": "music", "strip_prefix": "", "files": ["model.gguf", "vocoder.gguf"]}
    monkeypatch.setattr(w, "_resolve", lambda _: entry)
    monkeypatch.setattr(w, "MODELS_ROOT", tmp_path)
    root = tmp_path / "music"
    root.mkdir()
    (root / ".audiocpp-package-test.json").write_text("{}")
    (root / "model.gguf").write_bytes(b"weights")
    assert not w._installed("test")
    (root / "vocoder.gguf").write_bytes(b"weights")
    assert w._installed("test")


def test_prep_force_forwards_provider_force_and_checks_result(tmp_path, monkeypatch):
    monkeypatch.setattr(w, "_resolve", lambda _: {"target_directory": "music"})
    monkeypatch.setattr(w, "MODELS_ROOT", tmp_path)
    monkeypatch.setattr(w, "_require_engine", lambda: Path("audiocpp_cli"))
    monkeypatch.setattr(w, "_installed", lambda _: False)
    calls = []
    monkeypatch.setattr(w, "_install_package", lambda *args: calls.append(args))
    with pytest.raises(SystemExit, match="incomplete"):
        w.prep("test", force=True)
    assert calls[0] == ({"target_directory": "music"}, tmp_path, "test", True)


def test_relative_engine_and_models_roots_resolve_from_caller(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AUDIOCPP_ROOT", "engine")
    monkeypatch.setenv("AUDIOCPP_MODELS_ROOT", "weights")
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    assert adapter.ENGINE_ROOT == tmp_path / "engine"
    assert adapter.MODELS_ROOT == tmp_path / "weights"
