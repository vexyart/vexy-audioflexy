# this_file: tests/test_ported_voice_smallgguf.py
"""Exact small GGUF TTS packages and checked atomic native output."""
import importlib.util
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/vexy_audioflexy/tools/tts/tools"))

import numpy as np
import pytest
import soundfile as sf


def wrapper():
    path = (Path(__file__).resolve().parents[1] / "src/vexy_audioflexy/tools/tts") / "tools/tts-smallgguf.py"
    spec = importlib.util.spec_from_file_location("smallgguf", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module




@pytest.mark.parametrize("package,lang,voice", [("unknown", "en", ""), ("kitten_tts_mini_0_8_orig", "pl", ""),
    ("kitten_tts_mini_0_8_orig", "en", "af_heart"), ("kokoro_82m_q8_0", "en", "zf_xiaobei"),
    ("kokoro_82m_q8_0", "pl", "")])
def test_unsupported_package_language_voice_fails_before_download(package, lang, voice):
    with pytest.raises(SystemExit):
        wrapper().resolve(package, lang, voice)


def test_valid_voice_language_resolution_and_output_geometry(tmp_path):
    module = wrapper()
    assert module.resolve("kokoro_82m_bf16", "zh", "") == ("zh", "zf_xiaobei")
    assert module.resolve("kitten_tts_mini_0_8_orig", "en", "Bella") == ("en", "Bella")
    path = tmp_path / "test.wav"
    sf.write(path, np.ones(2400) * .1, 24000, subtype="FLOAT")
    assert module.inspect_audio(path)["duration_s"] == .1
    for rate, samples in [(16000, np.ones(32)), (24000, np.zeros(32)), (24000, np.full(32, np.nan)),
            (24000, np.ones((32, 2)))]:
        sf.write(path, samples, rate, subtype="FLOAT")
        with pytest.raises(ValueError):
            module.inspect_audio(path)


def test_kitten_native_contract_omits_unsupported_language_and_failure_preserves_wave(tmp_path, monkeypatch):
    import subprocess

    module = wrapper()
    binary = tmp_path / "build/macos-metal-release/bin/audiocpp_cli"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"native test build")
    monkeypatch.setenv("AUDIOCPP_ISSUE201_ROOT", str(tmp_path))
    monkeypatch.setattr(module.subprocess, "check_output", lambda *args, **kwargs: module.COMMIT)
    monkeypatch.setattr(module, "prep", lambda *args: "pinned weights")
    def synthesize(argv, **kwargs):
        assert "--language" not in argv, "Kitten rejects the native language request option"
        sf.write(argv[argv.index("--out") + 1], np.ones(2400) * .1, 24000, subtype="FLOAT")
        return subprocess.CompletedProcess(argv, 0, "session.wall_ms 1", "")
    monkeypatch.setattr(module.subprocess, "run", synthesize)
    output = tmp_path / "speech.wav"
    module.say("A sentence", str(output))
    before = output.read_bytes()
    monkeypatch.setattr(module.subprocess, "run", lambda argv, **kwargs: subprocess.CompletedProcess(argv, 1, "", "failed"))
    with pytest.raises(SystemExit):
        module.say("A new sentence", str(output), force=True)
    assert output.read_bytes() == before, "Failed synthesis must preserve the previous waveform"
