# this_file: tests/tools/asr/test_openasr.py
"""Original pinned packs and OpenASR's family-specific language contract."""
import importlib.util
import subprocess
import wave
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("openasr", Path(__file__).resolve().parents[3] / "src/vexy_audioflexy/tools/asr" / "tools/asr-openasr.py")
wrapper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wrapper)


def test_command_selects_native_pinned_pack_and_disables_network():
    args = wrapper.command("weights.oasr", "audio.wav", "output.txt", "pl", "auto")
    assert args[args.index("--model-pack") + 1] == "weights.oasr"
    assert "--offline" in args
    assert args[args.index("--language") + 1] == "pl"


def test_nonwhisper_family_does_not_receive_rejected_language_flag():
    args = wrapper.command("weights.oasr", "audio.wav", "output.txt", "pl", "auto", "parakeet-tdt")
    assert "--language" not in args
    with pytest.raises(SystemExit, match="auto"):
        wrapper.command("weights.oasr", "audio.wav", "output.txt", "en", "metal")


def test_original_catalog_has_pinned_pack_artifacts():
    assert len(wrapper.PACKAGES) == 27
    for entry in wrapper.PACKAGES.values():
        assert entry["filename"].endswith(".oasr")
        assert len(entry["revision"]) == 40 and len(entry["sha256"]) == 64
    assert wrapper.resolve("moonshine-tiny")["repo"] == "OpenASR/moonshine-tiny"


def test_failed_inference_preserves_previous_output(tmp_path, monkeypatch):
    src, out, binary = [tmp_path / name for name in ("input.wav", "output.txt", "openasr")]
    src.touch(); binary.touch(); out.write_text("preserve")
    monkeypatch.setattr(wrapper, "BINARY", binary)
    monkeypatch.setattr(wrapper, "prep", lambda *_: "weights.oasr")
    def run(args, **kwargs):
        if args[0] == "ffmpeg":
            with wave.open(args[-1], "wb") as audio:
                audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                audio.writeframes(b"\x01\x00" * 160)
            return subprocess.CompletedProcess(args, 0, "", "")
        return subprocess.CompletedProcess(args, 1, "", "native inference failed")
    monkeypatch.setattr(wrapper.subprocess, "run", run)
    with pytest.raises(SystemExit):
        wrapper.transcribe(str(src), text_out=str(out), force=True)
    assert out.read_text() == "preserve", "Inference failure overwrote the prior transcript"
