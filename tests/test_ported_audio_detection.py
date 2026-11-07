# this_file: tests/test_ported_audio_detection.py
"""Native detection JSON must prove execution, task identity and valid spans."""
import importlib.util
from pathlib import Path

import pytest


def wrapper():
    path = (Path(__file__).resolve().parents[1] / "src/vexy_audioflexy/tools/fx") / "tools/aud-detection.py"
    spec = importlib.util.spec_from_file_location("detection", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def native(task, payload=""):
    family = "pulsevad" if task == "vad" else "smart_turn"
    return f"family={family}\ntask={task}\nmode=offline\n{payload}\n[TIMING ts=20261003] session.wall_ms 12.5\n"


def test_empty_vad_requires_completed_native_task_and_timing():
    assert wrapper().parse_result(native("vad"), "pulsevad", "vad", 16000, .5)[0] == {"segments": []}
    for text in ["", native("turn"), native("vad").replace("12.5", "nan")]:
        with pytest.raises(ValueError):
            wrapper().parse_result(text, "pulsevad", "vad", 16000, .5)


@pytest.mark.parametrize("span", [{"start_sample": -1, "end_sample": 100}, {"start_sample": 0, "end_sample": 16001}, {"start_sample": 12, "end_sample": 12}, {"start_sample": True, "end_sample": 50}])
def test_vad_rejects_invalid_or_out_of_timeline_spans(span):
    import json
    with pytest.raises(ValueError):
        wrapper().parse_result(native("vad", "speech_segments=" + json.dumps([span])), "pulsevad", "vad", 16000, .5)


def test_turn_strict_threshold_and_schema_are_preserved():
    import json
    payload = {"schema": "smart_turn.v1", "data": {"probability": .5, "complete": False}}
    result, seconds = wrapper().parse_result(native("turn", "custom_schema_output=" + json.dumps(payload)), "smart_turn", "turn", 16000, .5)
    assert result["complete"] is False and seconds == .0125
    for changes in [{"probability": .5, "complete": True}, {"probability": 2, "complete": True}, {"probability": True, "complete": True}]:
        payload["data"] = changes
        with pytest.raises(ValueError):
            wrapper().parse_result(native("turn", "custom_schema_output=" + json.dumps(payload)), "smart_turn", "turn", 16000, .5)
    payload["data"] = None
    with pytest.raises(ValueError):
        wrapper().parse_result(native("turn", "custom_schema_output=" + json.dumps(payload)), "smart_turn", "turn", 16000, .5)


def test_adapter_rejects_wrong_input_rate_before_native_execution(tmp_path):
    import numpy as np
    import soundfile as sf
    path = tmp_path / "wrong-rate.wav"
    sf.write(path, np.zeros(24000), 24000)
    with pytest.raises(SystemExit, match="mono16k"):
        wrapper().run(str(path), str(tmp_path / "result.json"))
    with pytest.raises(SystemExit, match="registered package"):
        wrapper().run(str(path), str(tmp_path / "result.json"), package="substitute")


def test_detection_resume_rejects_changed_source(tmp_path):
    import json

    import numpy as np
    import soundfile as sf
    path = (Path(__file__).resolve().parents[1] / "src/vexy_audioflexy/tools/fx") / "detect_all.py"
    spec = importlib.util.spec_from_file_location("detection_batch", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    inputs, outputs = tmp_path / "inputs", tmp_path / "outputs"
    inputs.mkdir(); outputs.mkdir()
    sf.write(inputs / "test.wav", np.zeros(1600), 16000)
    (outputs / "pulsevad_2_1k_f32__cpu__test.json").write_text(json.dumps({"source_sha256": "different"}))
    with pytest.raises(SystemExit, match="differs from source/config"):
        module.main(str(inputs), str(outputs), packages="pulsevad_2_1k_f32", backends="cpu")
