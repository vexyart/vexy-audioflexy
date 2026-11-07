# this_file: tests/tools/asr/test_nemotron_salm.py
"""SALM identity, explicit encoder routing and failure protection use synthetic data."""

import hashlib
import importlib.util
import json
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

PATH = Path(__file__).resolve().parents[3] / "src/vexy_audioflexy/tools/asr/tools/asr-nemotron-salm.py"
spec = importlib.util.spec_from_file_location("nemotron_salm", PATH)
wrapper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wrapper)


def test_runtime_when_speechlm_disabled_or_commit_wrong_then_rejects(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="SPEECHLM=ON"):
        wrapper.runtime(str(tmp_path), 10)
    binary = tmp_path / "build/metal-asr/bin/nemo-speech"
    binary.parent.mkdir(parents=True)
    binary.touch()
    (tmp_path / "build/metal-asr/CMakeCache.txt").write_text("NEMO_SPEECH_BUILD_SPEECHLM:BOOL=ON\n")
    monkeypatch.setattr(wrapper.subprocess, "check_output", lambda *a, **k: "other-commit")
    with pytest.raises(ValueError, match="commit"):
        wrapper.runtime(str(tmp_path), 10)
    monkeypatch.setattr(wrapper.subprocess, "check_output", lambda *a, **k: wrapper.COMMIT)
    assert wrapper.runtime(str(tmp_path), 10) == binary


def test_exact_model_guard_when_metadata_or_bytes_differ_then_rejects(tmp_path, monkeypatch):
    from gguf import GGUFWriter

    paths, expected = [], []
    for index, profile in enumerate(wrapper.MODELS):
        path = tmp_path / f"synthetic-{index}.gguf"
        writer = GGUFWriter(path, profile["architecture"])
        writer.add_string("general.name", profile["name"])
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        paths.append(path)
        expected.append({**profile, "size": path.stat().st_size,
                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    monkeypatch.setattr(wrapper, "MODELS", tuple(expected))
    assert wrapper.validate_models(*(str(p) for p in paths)) == tuple(paths)
    expected[1]["architecture"] = "whisper"
    with pytest.raises(ValueError, match="metadata"):
        wrapper.validate_models(*(str(p) for p in paths))
    expected[1]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA256"):
        wrapper.validate_models(*(str(p) for p in paths))


def test_command_when_decoder_and_encoder_are_separate_then_routes_both_without_copy():
    argv = wrapper.command(Path("native binary"), Path("input.wav"), Path("decoder path.gguf"),
                           Path("encoder path.gguf"), Path("result.txt"), "en", 4)
    assert argv[argv.index("--model") + 1] == "decoder path.gguf"
    assert argv[argv.index("--asr.salm.encoder_path") + 1] == "encoder path.gguf"
    assert argv[argv.index("--asr.salm.max_sequences") + 1] == "1"
    assert "--asr.salm.batched=false" in argv


@pytest.mark.parametrize("returncode,text", [(1, "partial"), (0, ""), (0, "synthetic transcript")])
def test_transcribe_when_native_fails_then_keeps_previous_output(tmp_path, monkeypatch, capsys, returncode, text):
    source, output = tmp_path / "input.wav", tmp_path / "result.txt"
    source.touch()
    output.write_text("previous valid transcript")
    monkeypatch.setattr(wrapper, "runtime", lambda *a: Path("native binary"))
    monkeypatch.setattr(wrapper, "validate_models", lambda *a: (Path("decoder"), Path("encoder")))

    def run(argv, **kwargs):
        if argv[0] == "ffmpeg":
            with wave.open(argv[-1], "wb") as wav:
                wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                wav.writeframes(b"\0\0" * 16000)
            return SimpleNamespace(returncode=0)
        Path(argv[argv.index("--output") + 1]).write_text(text)
        return SimpleNamespace(returncode=returncode, stdout="", stderr="")

    monkeypatch.setattr(wrapper.subprocess, "run", run)
    if returncode or not text:
        with pytest.raises((SystemExit, ValueError)):
            wrapper.transcribe(str(source), "decoder", "encoder", "engine", str(output), force=True)
        assert output.read_text() == "previous valid transcript"
    else:
        wrapper.transcribe(str(source), "decoder", "encoder", "engine", str(output), force=True)
        assert output.read_text() == text + "\n"
        result = json.loads(capsys.readouterr().out.split("asr.result=", 1)[1])
        assert result["package"] == "nemotron-3.5-ad-q5-0-xl"
