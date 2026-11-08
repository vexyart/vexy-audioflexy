# this_file: tests/test_speech_qwen.py
"""Qwen selection, lexical retention and actual alignment contracts."""

import json
import wave
from types import SimpleNamespace as N

import pytest


def test_auto_when_mlx_missing_then_selects_qwen_native(monkeypatch):
    from vexy_audioflexy.speech import qwen

    monkeypatch.setattr(qwen, "mlx_available", lambda: False)
    assert qwen.select_backend("auto") == "audiocpp", "Fallback must still use Qwen"
    monkeypatch.setattr(qwen, "mlx_available", lambda: True)
    assert qwen.select_backend("auto") == "mlx", "Prefer MLX when available"


@pytest.mark.parametrize("model", ["voxtral-small", "granite-speech", "kroko-en"])
def test_rejected_model_when_selected_then_fails_before_loading(model):
    from vexy_audioflexy.speech.qwen import check_model

    with pytest.raises(ValueError, match="removed"):
        check_model(model)


def test_aligner_when_punctuation_changes_then_preserves_original_text():
    from vexy_audioflexy.speech.qwen import retain_words

    words = retain_words(
        "FontLab 8.4, isn't bad!",
        [
            N(text="FontLab", start_time=0.1, end_time=0.5),
            N(text="84", start_time=0.5, end_time=1),
            N(text="isn't", start_time=1, end_time=1.3),
            N(text="bad", start_time=1.3, end_time=1.7),
        ],
        30,
        2,
    )
    assert [w.text for w in words] == ["FontLab", "8.4,", "isn't", "bad!"]
    assert words[0].start == 30.1 and words[-1].end == 31.7


def test_alignment_when_last_word_end_extrapolated_past_chunk_then_clamps():
    from vexy_audioflexy.speech.qwen import retain_words

    words = retain_words(
        "it happen",
        [
            N(text="it", start_time=29.6, end_time=29.76),
            N(text="happen", start_time=29.84, end_time=30.4),
        ],
        240,
        30,
    )
    assert words[-1].end == 270, "Aligner overshoot is clamped to the chunk end"


@pytest.mark.parametrize(
    "text,items",
    [
        ("20 units", [N(text="2", start_time=0, end_time=1)]),
        ("hello", [N(text="hello", start_time=1, end_time=0)]),
        ("hello", [N(text="hello", start_time=2.5, end_time=3)]),
    ],
)
def test_alignment_when_content_or_bounds_invalid_then_rejects(text, items):
    from vexy_audioflexy.speech.qwen import retain_words

    with pytest.raises(ValueError):
        retain_words(text, items, 0, 2)


def test_mlx_when_audio_spans_chunks_then_aligns_each_and_offsets_times(
    tmp_path, monkeypatch
):
    import sys

    from vexy_audioflexy.speech import qwen

    wav = tmp_path / "input.wav"
    with wave.open(str(wav), "wb") as out:
        out.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        out.writeframes(b"\0\0" * 496000)
    observed = []

    def generate(audio, **kwargs):
        observed.append(kwargs)
        return N(text="API works.", generation_tokens=3)

    aligner = N(
        generate=lambda **kwargs: N(
            items=[
                N(text="API", start_time=0, end_time=0.4),
                N(text="works", start_time=0.4, end_time=0.8),
            ]
        )
    )
    monkeypatch.setitem(
        sys.modules,
        "mlx_audio.stt",
        N(
            load=lambda model, **kwargs: (
                aligner if "Aligner" in model else N(generate=generate)
            )
        ),
    )
    words, raw = qwen.transcribe(wav, "Terms: API.", "en", qwen.MODEL, qwen.ALIGNER)
    assert len(raw["segments"]) == 2 and words[2].start == 30
    assert observed[0]["system_prompt"] == observed[1]["system_prompt"]
    assert "API" in observed[0]["system_prompt"]
    assert raw["alignment"]["word_timestamps"] == "forced_alignment"


def test_mlx_when_prompt_recited_then_retries_bare_and_drops_hopeless_chunk(
    tmp_path, monkeypatch
):
    import sys

    from vexy_audioflexy.speech import qwen

    wav = tmp_path / "input.wav"
    with wave.open(str(wav), "wb") as out:
        out.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        out.writeframes(b"\0\0" * 496000)  # 30 s chunk + 1 s tail
    prompts = []

    def generate(audio, **kwargs):
        prompts.append(kwargs["system_prompt"])
        recited = kwargs["system_prompt"] or "31" in str(audio)
        return N(text="word " * 300 if recited else "API works.", generation_tokens=3)

    aligner = N(
        generate=lambda **kwargs: N(
            items=[
                N(text="API", start_time=0, end_time=0.4),
                N(text="works", start_time=0.4, end_time=0.8),
            ]
        )
    )
    monkeypatch.setitem(
        sys.modules,
        "mlx_audio.stt",
        N(
            load=lambda model, **kwargs: (
                aligner if "Aligner" in model else N(generate=generate)
            )
        ),
    )
    monkeypatch.setattr(
        qwen, "chunks", lambda wav, folder: iter([("a00", 0, 30), ("a31", 30, 1)])
    )
    words, raw = qwen.transcribe(wav, "Terms: API.", "en")
    assert prompts[0] and prompts[1] is None, "Recited text triggers one bare retry"
    assert [w.text for w in words] == ["API", "works."]
    assert raw["segments"][1]["text"] == ""
    assert raw["alignment"]["dropped_chunks"][0]["start"] == 30


def test_native_when_outputs_present_then_preserves_text_and_sample_times(
    tmp_path, monkeypatch
):
    from vexy_audioflexy.speech import audiocpp

    wav = tmp_path / "clip.wav"
    with wave.open(str(wav), "wb") as out:
        out.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        out.writeframes(b"\0\0" * 32000)
    model, aligner = tmp_path / "qwen.gguf", tmp_path / "aligner.gguf"
    model.touch()
    aligner.touch()
    observed = []

    def run(args, **kwargs):
        observed.append(args)
        from pathlib import Path

        Path(args[args.index("--text-out") + 1]).with_name("text_000.txt").write_text(
            "API works."
        )
        Path(args[args.index("--words-out") + 1]).with_name(
            "words_000.json"
        ).write_text(
            json.dumps(
                [
                    {"word": "API", "start_sample": 1600, "end_sample": 8000},
                    {"word": "works", "start_sample": 8000, "end_sample": 16000},
                ]
            )
        )
        return N(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(audiocpp.subprocess, "run", run)
    words, raw = audiocpp.transcribe(
        wav, "Terms: API.", "en", str(model), str(aligner), "fixture", "cpu"
    )
    assert words[-1].text == "works." and words[0].start == 0.1
    assert observed[0][observed[0].index("--family") + 1] == "qwen3_asr"
    assert raw["alignment"]["word_timestamps"] == "forced_alignment"


def test_native_configuration_when_missing_then_gives_setup_guidance(
    tmp_path, monkeypatch
):
    from vexy_audioflexy.speech import audiocpp

    monkeypatch.delenv("VEXY_SUBS_QWEN_MODEL", raising=False)
    with pytest.raises(FileNotFoundError, match="VEXY_SUBS_QWEN_MODEL"):
        audiocpp.model_path("", "default", "VEXY_SUBS_QWEN_MODEL")
    model = tmp_path / "qwen.gguf"
    model.write_bytes(b"weights")
    monkeypatch.setenv("VEXY_SUBS_QWEN_MODEL", str(model))
    assert audiocpp.model_path("default", "default", "VEXY_SUBS_QWEN_MODEL") == str(
        model
    )
    with pytest.raises(FileNotFoundError, match="audiocpp_cli"):
        audiocpp.binary_path("nonexistent-vexy-executable")
    with pytest.raises(ValueError):
        audiocpp.backend_name("unknown")


def test_native_failure_when_engine_errors_then_reports_without_fallback(monkeypatch):
    from vexy_audioflexy.speech import audiocpp

    monkeypatch.setattr(
        audiocpp.subprocess, "run", lambda *a, **k: N(returncode=1, stderr="bad model")
    )
    with pytest.raises(RuntimeError, match="bad model"):
        audiocpp.execute(["fixture"])
