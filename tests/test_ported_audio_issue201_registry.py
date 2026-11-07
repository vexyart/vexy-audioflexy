# this_file: tests/test_ported_audio_issue201_registry.py
"""The same native checkout must serve package preparation and inference."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

spec = importlib.util.spec_from_file_location("aud_registry", (Path(__file__).resolve().parents[1] / "src/vexy_audioflexy/tools/fx") / "aud_all.py")
registry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(registry)


def test_issue201_models_use_distinct_names_and_correct_operations():
    names = [m["name"] for m in registry.MODELS]
    assert len(set(names)) == len(names), "Every audio model must retain its own result row"
    models = {m["name"]: m for m in registry.MODELS}
    assert models["issue201-reuse-f32"]["source"] == "noisy"
    assert models["issue201-universr-speech-orig"]["source"] == "lowres"
    assert models["speechbrain-metricgan-plus"]["task"] == "denoise"


def test_prep_and_run_forward_identical_native_checkout(tmp_path, monkeypatch):
    calls = []
    model = next(m for m in registry.MODELS if m["name"] == "issue201-reuse-f32")

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(registry.subprocess, "run", run)
    assert registry.prep_model(model, 10, False)[0], "Preparation should pass"
    assert registry.run_model(model, str(tmp_path / "out.wav"), 10, False)[0], "Inference should pass"
    expected = model["env"]["AUDIOCPP_ROOT"]
    assert len(calls) == 2, "Both prep and inference must be invoked"
    assert all(kwargs["env"]["AUDIOCPP_ROOT"] == expected for _, kwargs in calls), "Prep and inference must use the same checkout"
