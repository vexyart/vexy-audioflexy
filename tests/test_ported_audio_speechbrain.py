# this_file: tests/test_ported_audio_speechbrain.py
"""Input failures must be diagnosed before loading enhancement dependencies."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("enhancement", (Path(__file__).resolve().parents[1] / "src/vexy_audioflexy/tools/fx") / "tools/aud-speechbrain.py")
wrapper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wrapper)


def test_package_rejects_other_speechbrain_recipes():
    wrapper.check_package("metricgan-plus-voicebank")
    with pytest.raises(SystemExit, match="Unknown SpeechBrain"):
        wrapper.check_package("unknown")


def test_run_rejects_task_before_loading_model(tmp_path):
    with pytest.raises(SystemExit, match="task=denoise"):
        wrapper.run(str(tmp_path / "missing.wav"), task="vc")


def test_run_rejects_missing_audio_before_loading_model(tmp_path):
    with pytest.raises(SystemExit, match="Audio not found"):
        wrapper.run(str(tmp_path / "missing.wav"))


def test_run_rejects_overwriting_input(tmp_path):
    audio = tmp_path / "input.wav"
    audio.touch()
    with pytest.raises(SystemExit, match="differ from input"):
        wrapper.run(str(audio), output=str(audio), force=True)
