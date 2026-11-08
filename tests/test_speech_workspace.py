# this_file: tests/test_speech_workspace.py
"""Workspace recognition must reach captions with real timing and cache identity."""

import json
from pathlib import Path


def test_workspace_recognition_when_worker_returns_words_then_keeps_native_times(
    tmp_path, monkeypatch
):
    from vexy_audioflexy.speech import workspace

    root = tmp_path / "models"
    worker = root / "myasr/tools/speech-issue301.py"
    worker.parent.mkdir(parents=True)
    worker.write_text("# fixture")
    wav = tmp_path / "input.wav"
    wav.write_bytes(b"fixture")
    observed = []

    def run(args, **kwargs):
        observed.append(args)
        output = Path(
            next(x.split("=", 1)[1] for x in args if x.startswith("--output="))
        )
        output.write_text(
            json.dumps(
                {
                    "words": [{"text": "API", "start": 0.1, "end": 0.6}],
                    "raw": {
                        "language": "en",
                        "alignment": {"native_word_estimates": True},
                    },
                }
            )
        )

    monkeypatch.setattr(workspace.subprocess, "run", run)
    words, raw = workspace.transcribe(wav, "Terms: API", "en", "chosen", str(root))
    assert words[0].start == 0.1 and words[0].end == 0.6, (
        "Do not interpolate actual word times"
    )
    assert raw["alignment"]["native_word_estimates"] is True
    assert "--model=chosen" in observed[0], "Use the requested workspace model"
    assert "--prompt=Terms: API" in observed[0], (
        "Forward recognition context as one argv value"
    )


def test_workspace_signature_when_worker_changes_then_invalidates_stage(tmp_path):
    from vexy_audioflexy.speech import workspace

    worker = tmp_path / "myasr/tools/speech-issue301.py"
    worker.parent.mkdir(parents=True)
    worker.write_text("first")
    first = workspace.signature(str(tmp_path))
    worker.write_text("second")
    assert workspace.signature(str(tmp_path)) != first, (
        "Worker changes must not reuse old inference"
    )


def test_packaged_signature_tracks_runtime_bytes_and_configured_location(
    tmp_path, monkeypatch
):
    from vexy_audioflexy.speech import workspace
    from vexy_audioflexy.tools.asr.tools import _audioflexy_paths

    monkeypatch.delenv("VEXY_AUDIOFLEXY_MODEL_WORKSPACE", raising=False)
    monkeypatch.delenv("VEXY_SUBS_MODEL_WORKSPACE", raising=False)
    monkeypatch.delenv("AUDIOCPP_ISSUE301_ROOT", raising=False)
    monkeypatch.delenv("VAFX_RUNTIME_DIR", raising=False)
    monkeypatch.setattr(
        _audioflexy_paths, "user_cache_path", lambda *a, **kw: tmp_path / "cache"
    )
    default = (
        tmp_path
        / "cache/runtimes/audio.cpp-issue301/build/macos-metal-release/bin/audiocpp_cli"
    )
    default.parent.mkdir(parents=True)
    default.write_bytes(b"first")
    first = workspace.signature("")
    default.write_bytes(b"second")
    second = workspace.signature("")
    assert first != second, (
        "Replacing the platform-cache executable must invalidate stages"
    )

    configured = (
        tmp_path
        / "configured/audio.cpp-issue301/build/macos-metal-release/bin/audiocpp_cli"
    )
    configured.parent.mkdir(parents=True)
    configured.write_bytes(b"second")
    monkeypatch.setenv("VAFX_RUNTIME_DIR", str(tmp_path / "configured"))
    third = workspace.signature("")
    assert third != second, (
        "Moving to identical bytes in a configured runtime must invalidate stages"
    )
    configured.write_bytes(b"third")
    assert workspace.signature("") != third

    override = tmp_path / "override/build/macos-metal-release/bin/audiocpp_cli"
    override.parent.mkdir(parents=True)
    override.write_bytes(b"third")
    monkeypatch.setenv("AUDIOCPP_ISSUE301_ROOT", str(tmp_path / "override"))
    fourth = workspace.signature("")
    assert fourth != third, "The adapter-specific engine override must take precedence"
    override.write_bytes(b"fourth")
    assert workspace.signature("") != fourth
    override.unlink()
    assert workspace.signature("") != fourth, (
        "A missing runtime must not reuse previous inference"
    )


def test_legacy_workspace_signature_retains_local_native_binary(tmp_path):
    from vexy_audioflexy.speech import workspace

    worker = tmp_path / "myasr/tools/speech-issue301.py"
    worker.parent.mkdir(parents=True)
    worker.write_text("# synthetic legacy worker")
    binary = (
        tmp_path
        / ".external/audio.cpp-issue301/build/macos-metal-release/bin/audiocpp_cli"
    )
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"first")
    first = workspace.signature(str(tmp_path))
    binary.write_bytes(b"second")
    assert workspace.signature(str(tmp_path)) != first
