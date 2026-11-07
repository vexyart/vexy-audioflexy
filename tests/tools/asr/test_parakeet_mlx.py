# this_file: tests/tools/asr/test_parakeet_mlx.py
"""Local Parakeet model selection and failed-output protection use synthetic assets."""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

PATH = Path(__file__).resolve().parents[3] / "src/vexy_audioflexy/tools/asr/tools/asr-parakeet-mlx.py"
spec = importlib.util.spec_from_file_location("parakeet_mlx", PATH)
wrapper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wrapper)


def model(tmp_path):
    directory = tmp_path / "misleading-name"
    directory.mkdir()
    (directory / "config.json").write_text(json.dumps({"sample_rate": 16000,
        "encoder": {"_target_": "nemo.ConformerEncoder"},
        "decoder": {"_target_": "nemo.RNNTDecoder"},
        "model_defaults": {"tdt_durations": [0, 1, 2, 3, 4]}, "quantization": {"bits": 8}}))
    (directory / "model.safetensors").write_bytes(b"synthetic weights")
    (directory / "tokenizer.model").write_bytes(b"synthetic tokenizer")
    return directory


def test_model_when_wrong_family_or_quantization_then_rejects(tmp_path):
    directory = model(tmp_path)
    assert wrapper.model_config(str(directory))[0] == directory
    config = json.loads((directory / "config.json").read_text())
    config["quantization"]["bits"] = 4
    (directory / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="Parakeet TDT"):
        wrapper.model_config(str(directory))


@pytest.mark.parametrize("text", ["recognized synthetic text", "", None])
def test_transcribe_when_local_model_then_explicit_type_and_preserves_output_on_failure(tmp_path, monkeypatch, capsys, text):
    directory = model(tmp_path)
    source, output = tmp_path / "audio.wav", tmp_path / "result.txt"
    source.touch()
    output.write_text("previous transcript")
    import soundfile
    monkeypatch.setattr(soundfile, "info", lambda *a: SimpleNamespace(duration=1.0))
    requests = []
    utils = ModuleType("mlx_audio.stt.utils")
    utils.load = lambda *a, **k: requests.append((a, k)) or SimpleNamespace(generate=lambda *a: SimpleNamespace(text=text))
    monkeypatch.setitem(sys.modules, "mlx_audio", ModuleType("mlx_audio"))
    monkeypatch.setitem(sys.modules, "mlx_audio.stt", ModuleType("mlx_audio.stt"))
    monkeypatch.setitem(sys.modules, "mlx_audio.stt.utils", utils)
    if text:
        wrapper.transcribe(str(source), str(directory), str(output), force=True)
        assert output.read_text() == text + "\n"
        assert json.loads(capsys.readouterr().out.split("asr.result=", 1)[1])["model_type"] == "parakeet"
    else:
        with pytest.raises(ValueError, match="recognized text"):
            wrapper.transcribe(str(source), str(directory), str(output), force=True)
        assert output.read_text() == "previous transcript"
    assert requests == [((str(directory),), {"model_type": "parakeet"})]
