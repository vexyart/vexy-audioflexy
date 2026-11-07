# this_file: tests/tools/asr/test_diarization.py
"""Native timestamp units, boundary failures and optimal speaker assignment."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3] / "src/vexy_audioflexy/tools/asr"


def load(filename):
    spec = importlib.util.spec_from_file_location(filename, ROOT / "tools" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_native_sample_offsets_convert_to_seconds():
    result = load("diarize-native.py").normalize([
        {"start_sample": 1600, "end_sample": 32000, "speaker_id": "speaker_0"}], "audiocpp", 3)
    assert result == [{"start": .1, "end": 2., "speaker": "speaker_0"}]


def test_nemo_ids_are_preserved_and_overlaps_are_valid():
    segments = [{"start": 0., "end": 2., "speaker": 1},
                {"start": 1., "end": 3., "speaker": 2}]
    assert len(load("diarize-native.py").normalize({"segments": segments}, "nemo", 3)) == 2


@pytest.mark.parametrize("start,end,speaker", [(2, 1, "a"), (-1, 2, "a"),
    (0, 4, "a"), (0, float("nan"), "a"), (0, 1, ""), (True, 2, "a")])
def test_invalid_turns_fail_before_report_is_written(start, end, speaker):
    with pytest.raises(SystemExit):
        load("diarize-native.py").normalize({"segments": [
            {"start": start, "end": end, "speaker": speaker}]}, "nemo", 3)


def test_empty_output_is_valid_for_silent_input():
    assert load("diarize-native.py").normalize([], "audiocpp", 3) == []


def test_scoring_maps_arbitrary_speaker_ids_and_counts_misses():
    reference = [{"start": 0., "end": 2., "speaker": "A"},
                 {"start": 2., "end": 4., "speaker": "B"}]
    hypothesis = [{"start": 0., "end": 2., "speaker": "42"},
                  {"start": 2., "end": 4., "speaker": "99"}]
    module = load("diar-metrics.py")
    assert module.measure(reference, hypothesis, 4., collar=0.)["diarization error rate"] == 0.
    result = module.measure(reference, hypothesis[:1], 4., collar=0.)
    assert result["missed detection"] == 2.
    assert result["diarization error rate"] == .5


def test_wrapper_converts_input_and_preserves_native_receipt(tmp_path, monkeypatch):
    import soundfile as sf

    wrapper = load("diarize-native.py")
    source = tmp_path / "source.wav"
    sf.write(source, [0.] * 8000, 8000)
    original = source.read_bytes()
    weight = tmp_path / "extensionless-cache-blob"
    weight.write_bytes(b"test-model")
    monkeypatch.setattr(wrapper, "prep", lambda _: str(weight))
    real_run = wrapper.subprocess.run

    def run(args, **kwargs):
        if args[0] == "ffmpeg":
            return real_run(args, **kwargs)
        assert args[args.index("--model") + 1].endswith(".gguf")
        assert sf.info(args[args.index("--audio") + 1]).samplerate == 16000
        Path(args[args.index("--turns-out") + 1]).write_text("[]")
        return wrapper.subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(wrapper.subprocess, "run", run)
    output = tmp_path / "result"
    wrapper.run(str(source), str(output))
    assert json.loads((output / "report.json").read_text())["turns"] == []
    assert (output / "native.json").read_text().strip() == "[]"
    assert source.read_bytes() == original, "The wrapper modified source audio"
    with pytest.raises(SystemExit, match="Output exists"):
        wrapper.run(str(source), str(output))


def test_mode_dispatch_and_invalid_package():
    wrapper = load("diarize-native.py")
    item = wrapper.resolve("nemo-nemotron-3-diarization-q8")
    _, offline = wrapper.command(item, "model", Path("audio"), Path("raw"), "metal", "offline")
    _, streaming = wrapper.command(item, "model", Path("audio"), Path("raw"), "metal", "streaming")
    assert "--offline" in offline and "--offline" not in streaming
    with pytest.raises(SystemExit, match="Unknown"):
        wrapper.resolve("unknown")
    with pytest.raises(SystemExit, match="diarize-mlx"):
        wrapper.resolve("mlx-nemotron3-8bit")


def test_mlx_prep_selects_the_exact_eight_bit_revision(monkeypatch):
    import huggingface_hub

    module = load("diarize-mlx.py")
    requests = []
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *args, **kwargs: requests.append((args, kwargs)) or "local")
    assert module.prep() == "local"
    assert requests[0][0] == ("mlx-community/Nemotron-3-Diarization-8bit",)
    assert requests[0][1]["revision"] == "dd8b8ce3d69a80540a7e50150e2beed1c34fee37"
    with pytest.raises(SystemExit, match="Expected"):
        module.prep("unquantized")
