# this_file: tests/tools/asr/test_crispasr.py
"""Verify exact export selection and CrispASR's basename output contract."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("crispasr", Path(__file__).resolve().parents[3] / "src/vexy_audioflexy/tools/asr" / "tools/asr-crispasr.py")
wrapper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wrapper)


def test_resolve_selects_requested_exports():
    assert wrapper.resolve("phonon2")["repo"] == "cstr/phonon2-GGUF"
    assert wrapper.resolve("parakeet-redux")["filename"] == "parakeet-redux-q8_0.gguf"
    with pytest.raises(SystemExit, match="Unknown"):
        wrapper.resolve("parakeet-base")


def test_command_uses_basename_not_full_output_filename():
    cmd = wrapper.command("model.gguf", "input.wav", "/tmp/a transcript", "pl", "metal")
    assert cmd[cmd.index("--output-file") + 1] == "/tmp/a transcript"
    assert "--output-txt" in cmd, "CrispASR needs both the output format and basename"
    assert cmd[cmd.index("--backend") + 1] == "parakeet"
    assert cmd[cmd.index("--gpu-backend") + 1] == "metal"


def test_transcribe_rejects_missing_input_before_download(tmp_path):
    with pytest.raises(SystemExit, match="Audio not found"):
        wrapper.transcribe(str(tmp_path / "missing.wav"))
