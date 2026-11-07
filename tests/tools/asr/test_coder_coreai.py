# this_file: tests/tools/asr/test_coder_coreai.py
"""Coder543 conversion identity and distinct host profiles."""
import importlib.util
from pathlib import Path

import pytest


def load():
    spec = importlib.util.spec_from_file_location("coder_coreai", Path(__file__).resolve().parents[3] / "src/vexy_audioflexy/tools/asr" / "tools/diarize-coder-coreai.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_two_graph_profiles_are_preserved():
    module = load()
    assert module.profile_for("coreai-coder-nemotron3-fast32") == "fast32"
    assert module.profile_for("coreai-coder-nemotron3-fast128") == "fast128"
    with pytest.raises(SystemExit, match="Unknown"):
        module.profile_for("n3d_streaming_float16")


def test_prep_pins_exact_repo_and_fetches_only_requested_bundle(monkeypatch):
    import huggingface_hub

    module = load()
    requests = []
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *args, **kwargs: requests.append((args, kwargs)) or "/snapshot")
    assert module.prep("coreai-coder-nemotron3-fast128") == "/snapshot/fast128"
    assert requests[0][0] == ("coder543/nemotron-3-diarization-coreai",)
    assert requests[0][1]["revision"] == "99f6a8a5b614b28d61bfd16190edb92eab745198"
    assert requests[0][1]["allow_patterns"] == ["fast128/*", "LICENSE", "NOTICE"]
