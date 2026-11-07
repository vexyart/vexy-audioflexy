# this_file: tests/test_ported_voice_speechcpp.py
"""Drama 3 CLI contracts without downloading weights."""

import importlib.util
import subprocess
import wave
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/vexy_audioflexy/tools/tts/tools"))

import pytest


def wrapper():
    spec = importlib.util.spec_from_file_location(
        "speechcpp", (Path(__file__).resolve().parents[1] / "src/vexy_audioflexy/tools/tts") / "tools/tts-speechcpp.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prep_selects_six_bit(monkeypatch):
    module = wrapper()
    monkeypatch.setattr(module, "binary", lambda: "speech")
    calls = []
    monkeypatch.setattr(module.subprocess, "run", lambda cmd, **kw: calls.append(cmd))
    module.prep()
    assert calls == [["speech", "pull", "drama-3:q6_k"]]


def test_clone_and_atomic_failure(tmp_path, monkeypatch):
    module = wrapper()
    monkeypatch.setattr(module, "binary", lambda: "speech")
    reference = tmp_path / "reference.wav"
    reference.write_bytes(b"reference fixture")
    calls = []

    def run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[1] == "tts":
            with wave.open(cmd[cmd.index("-o") + 1], "wb") as wav:
                wav.setparams((1, 2, 44100, 0, "NONE", "not compressed"))
                wav.writeframes(b"\x01\x00" * 4410)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(module.subprocess, "run", run)
    output = tmp_path / "out.wav"
    module.say(
        "Hello", str(output), ref=str(reference), ref_text="Reference words", lang="en"
    )
    assert calls[0][1:3] == ["voice", "drama-3:q6_k"]
    assert calls[0][calls[0].index("--transcript") + 1] == "Reference words"
    assert "--add-voice" in calls[1] and "--language" in calls[1]
    assert calls[1][2] == "drama-3:q6_k"
    before = output.read_bytes()

    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0])

    monkeypatch.setattr(module.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        module.say("Again", str(output), force=True)
    assert output.read_bytes() == before


@pytest.mark.parametrize(
    "ref,ref_text", [("missing.wav", "words"), ("", "words"), ("missing.wav", "")]
)
def test_bad_reference_fails_before_engine(ref, ref_text, monkeypatch):
    module = wrapper()
    monkeypatch.setattr(module, "binary", lambda: pytest.fail("must validate first"))
    with pytest.raises(SystemExit):
        module.say("Hello", ref=ref, ref_text=ref_text)


def test_empty_text_and_invalid_output_fail(tmp_path, monkeypatch):
    module = wrapper()
    with pytest.raises(SystemExit):
        module.say(" ")
    monkeypatch.setattr(module, "binary", lambda: "speech")
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: None)
    with pytest.raises((ValueError, FileNotFoundError)):
        module.say("Hello", str(tmp_path / "out.wav"))
    assert not (tmp_path / "out.wav").exists()


@pytest.mark.parametrize(
    "rate,frames", [(16000, b"\x01\x00" * 10), (44100, b"\x00\x00" * 10)]
)
def test_invalid_audio_preserves_existing_file(tmp_path, monkeypatch, rate, frames):
    module = wrapper()
    monkeypatch.setattr(module, "binary", lambda: "speech")

    def run(cmd, **kwargs):
        assert cmd[cmd.index("--voice") + 1] == "none"
        with wave.open(cmd[cmd.index("-o") + 1], "wb") as wav:
            wav.setparams((1, 2, rate, 0, "NONE", "not compressed"))
            wav.writeframes(frames)

    monkeypatch.setattr(module.subprocess, "run", run)
    output = tmp_path / "out.wav"
    output.write_bytes(b"previous output")
    with pytest.raises(ValueError):
        module.say("Hello", str(output), force=True)
    assert output.read_bytes() == b"previous output"


def test_binary_discovery_info_and_no_overwrite(tmp_path, monkeypatch):
    module = wrapper()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.delenv("SPEECHCPP_BIN", raising=False)
    monkeypatch.setattr(module.shutil, "which", lambda _: None)
    with pytest.raises(SystemExit, match="Install speech.cpp"):
        module.binary()
    monkeypatch.setenv("SPEECHCPP_BIN", "/chosen/speech")
    assert module.binary() == "/chosen/speech"
    calls = []
    monkeypatch.setattr(module.subprocess, "run", lambda cmd, **kw: calls.append(cmd))
    module.info()
    assert calls == [["/chosen/speech", "info", "drama-3:q6_k", "--json"]]
    existing = tmp_path / "out.wav"
    existing.write_bytes(b"keep")
    module.say("Hello", str(existing))
    assert len(calls) == 1 and existing.read_bytes() == b"keep"


def test_output_cannot_replace_reference(tmp_path):
    reference = tmp_path / "reference.wav"
    reference.write_bytes(b"keep recording")
    with pytest.raises(SystemExit, match="Output must differ"):
        wrapper().say(
            "Hello", str(reference), ref=str(reference), ref_text="Words", force=True
        )
    assert reference.read_bytes() == b"keep recording"
