# this_file: tests/tools/asr/test_nemospeechcpp.py
"""NVIDIA's pinned ASR files and native command contract."""
import hashlib
import importlib.util
import sys
import types
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("nemospeechcpp", Path(__file__).resolve().parents[3] / "src/vexy_audioflexy/tools/asr" / "tools/asr-nemospeechcpp.py")
wrapper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wrapper)


def test_native_command_selects_plain_text_path_and_language():
    args = wrapper.command("model.gguf", "input.wav", "a transcript.txt", "pl", "metal")
    assert args[1:3] == ["transcribe", "input.wav"]
    assert args[args.index("--output") + 1] == "a transcript.txt"
    assert args[args.index("--format") + 1] == "text"
    assert args[args.index("--language") + 1] == "pl"
    assert args[args.index("--device") + 1] == "metal"


def test_catalog_contains_four_pinned_asr_artifacts_only():
    assert len(wrapper.PACKAGES) == 4
    for spec in wrapper.PACKAGES.values():
        assert spec["role"] == "asr"
        assert len(spec["revision"]) == 40 and len(spec["sha256"]) == 64
    with pytest.raises(SystemExit, match="Unknown"):
        wrapper.resolve("diarization")


def test_corrupted_checkpoint_is_rejected(tmp_path, monkeypatch):
    model = tmp_path / "weights.gguf"
    model.write_bytes(b"corrupt")
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(hf_hub_download=lambda **_: str(model)))
    with pytest.raises(SystemExit, match="checksum"):
        wrapper.prep()


def test_matching_checkpoint_uses_the_pinned_revision(tmp_path, monkeypatch):
    model = tmp_path / "weights.gguf"
    model.write_bytes(b"weights")
    spec = {**wrapper.resolve("nemotron-3.5"), "sha256": hashlib.sha256(b"weights").hexdigest()}
    monkeypatch.setitem(wrapper.PACKAGES, "test", spec)
    calls = []
    def download(**kwargs):
        calls.append(kwargs)
        return str(model)
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(hf_hub_download=download))
    assert wrapper.prep("test") == str(model)
    assert calls[0]["revision"] == spec["revision"], "Unpinned model download"


def test_missing_audio_fails_before_model_download(tmp_path):
    with pytest.raises(SystemExit, match="Audio not found"):
        wrapper.transcribe(str(tmp_path / "missing.wav"))
