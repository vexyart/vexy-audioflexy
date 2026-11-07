# this_file: tests/test_ported_audio_s2scpp.py
"""Local voice server boundaries and actual received-PCM artifact handling."""
import array
import importlib.util
import types
import wave
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("s2scpp", (Path(__file__).resolve().parents[1] / "src/vexy_audioflexy/tools/fx") / "tools/aud-s2scpp.py")
wrapper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wrapper)


def test_server_command_binds_loopback_and_rejects_invalid_ports():
    args = wrapper.server_command("models", 18089)
    assert args[args.index("--host") + 1] == "127.0.0.1"
    assert "--llm-url" not in args
    for port in (0, -1, 65536):
        with pytest.raises(SystemExit):
            wrapper.server_command("models", port)


def test_failed_server_startup_is_reported_without_polling_unrelated_service():
    with pytest.raises(SystemExit, match="exited"):
        wrapper.wait_ready(types.SimpleNamespace(poll=lambda: 1), 18089, 10)


def test_received_pcm_is_saved_with_full_duration_and_hash(tmp_path):
    raw, wav = tmp_path / "received.f32", tmp_path / "received.wav"
    raw.write_bytes(array.array("f", [0.25] * 24000).tobytes())
    duration, digest = wrapper.capture_wav(raw, wav)
    assert duration == 1 and len(digest) == 64
    with wave.open(str(wav), "rb") as audio:
        assert (audio.getframerate(), audio.getnchannels(), audio.getsampwidth(), audio.getnframes()) == (24000, 1, 2, 24000)


@pytest.mark.parametrize("samples", [[], [0.] * 10, [float("nan")]])
def test_invalid_received_pcm_cannot_be_claimed_as_speech(tmp_path, samples):
    raw = tmp_path / "received.f32"
    raw.write_bytes(array.array("f", samples).tobytes())
    with pytest.raises(SystemExit):
        wrapper.capture_wav(raw, tmp_path / "output.wav")
