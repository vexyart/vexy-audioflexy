"""Native Qwen model identities and mode-specific command contracts.

this_file: tests/test_ported_voice_qwenttscpp.py
"""
import importlib.util
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/vexy_audioflexy/tools/tts/tools"))

import pytest

ROOT = (Path(__file__).resolve().parents[1] / "src/vexy_audioflexy/tools/tts")
spec = importlib.util.spec_from_file_location("qwenttscpp", ROOT / "tools/tts-qwenttscpp.py")
wrapper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wrapper)




@pytest.mark.parametrize("mode,ref,text,instruct,speaker", [
    ("base", "", "", "warm", ""),
    ("base", "", "", "", "ryan"),
    ("base", "", "transcript", "", ""),
    ("customvoice", "ref.wav", "", "", ""),
    ("voicedesign", "", "", "", ""),
    ("voicedesign", "", "", "warm", "ryan"),
])
def test_incompatible_mode_inputs_fail_before_download(mode, ref, text, instruct, speaker):
    with pytest.raises(SystemExit):
        wrapper.validate_mode(mode, ref, text, instruct, speaker)


def test_reference_text_is_a_file_argument_and_text_never_enters_argv():
    args = wrapper.command("talker.gguf", "codec.gguf", "out.wav", "en", 42,
                           "base", "", "reference.wav", "reference.txt", "", 128)
    assert args[args.index("--ref-text") + 1] == "reference.txt"
    assert "--speaker" not in args
    assert args[args.index("--lang") + 1] == "English"


def test_failed_inference_preserves_existing_output(tmp_path, monkeypatch):
    output = tmp_path / "existing.wav"
    output.write_bytes(b"preserve")
    (tmp_path / "build").mkdir()
    (tmp_path / "build/qwen-tts").touch()
    monkeypatch.setattr(wrapper, "ENGINE_ROOT", tmp_path)
    monkeypatch.setattr(wrapper, "prep", lambda *_: ("talker", "codec"))
    import subprocess
    monkeypatch.setattr(wrapper.subprocess, "run", lambda *a, **kw: subprocess.CompletedProcess(a, 1, "", "failure"))
    with pytest.raises(SystemExit):
        wrapper.say("Hello", str(output), force=True)
    assert output.read_bytes() == b"preserve", "Inference failure overwrote prior output"
