# this_file: tests/test_speech_models.py
"""Optional model contracts with isolated engine fakes and synthetic text."""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace as N

import pytest

from vexy_audioflexy.speech import (
    Turn,
    Word,
    hints,
    language,
    punctuation,
    syntax,
    workspace,
)


def test_types_reject_invalid_boundaries():
    for args in [
        ("", 0, 1),
        ("word", -1, 0),
        ("word", 2, 1),
        ("word", 0, float("inf")),
    ]:
        with pytest.raises(ValueError):
            Word(*args)
    with pytest.raises(ValueError):
        Turn(0, 0, "speaker_0")


def test_language_preserves_probability_override_and_empty_contract(monkeypatch):
    monkeypatch.setattr(
        language,
        "identifier",
        lambda: N(classify=lambda text: ("de", 0.9), nb_classes=["en", "de"]),
    )
    result = language.detect_language([Word("hello", 0, 1)], "EN_us")
    assert result == {
        "language": "en",
        "detected": "de",
        "probability": 0.9,
        "overridden": True,
        "model": "langid-1.1.6-naive-bayes",
    }
    assert language.detect_language([])["language"] == "und"
    with pytest.raises(ValueError, match="Unsupported"):
        language.detect_language([], "invalid")


def test_syntax_non_english_needs_no_model_and_english_uses_tagger(monkeypatch):
    monkeypatch.setattr(
        syntax, "tagger", lambda: N(tag=lambda tokens: [(t, "NN") for t in tokens])
    )
    assert syntax.tags(["hello"], "en") == ("NN",)
    assert syntax.tags(["hello"], "de") == ("",)


def test_syntax_legacy_cache_when_offline_preserves_weights(monkeypatch, tmp_path):
    legacy = tmp_path / "vexy-subs" / "nltk"
    legacy.mkdir(parents=True)
    weights = legacy / "synthetic-weights.json"
    weights.write_bytes(b'{"synthetic": true}')
    paths = []

    def load_tagger():
        if str(legacy) not in paths:
            raise LookupError("Legacy weights are not on the search path")
        assert weights.read_bytes() == b'{"synthetic": true}'
        return N(tag=lambda tokens: [(token, "NN") for token in tokens])

    def download(*args, **kwargs):
        pytest.fail("Existing legacy weights must work without downloading")

    monkeypatch.setitem(sys.modules, "nltk", N(data=N(path=paths), download=download))
    monkeypatch.setitem(sys.modules, "nltk.tag", N(PerceptronTagger=load_tagger))
    monkeypatch.setattr(syntax, "user_cache_path", lambda app, **kw: tmp_path / app)
    syntax.tagger.cache_clear()
    try:
        assert syntax.tags(["hello"], "en") == ("NN",)
        assert paths[0] == str(tmp_path / "vexy-audioflexy" / "nltk")
        assert weights.read_bytes() == b'{"synthetic": true}'
        assert not (tmp_path / "vexy-audioflexy").exists()
    finally:
        syntax.tagger.cache_clear()


def test_hints_keep_metadata_and_only_overlapping_reference():
    from vexy_audioflexy.speech.qwen import context

    prompt = context(
        "Terms: QuasarAPI.\nTitle: A sample",
        hints.excerpt([(0, 1, "early"), (30, 31, "current")], 30, 60),
    )
    assert "Title: A sample" in prompt and "QuasarAPI" in prompt
    assert "current" in prompt and "early" not in prompt


def test_punctuation_preserves_timing_and_rejects_lexical_rewrites():
    words = [Word("hello", 0, 1), Word("quasarapi", 1, 2)]
    punctuation.apply_labels(
        words, [["hello", "?", 1], ["quasarapi", ".", 1]], "Terms: QuasarAPI."
    )
    assert [w.text for w in words] == ["Hello?", "QuasarAPI."]
    assert [(w.start, w.end) for w in words] == [(0, 1), (1, 2)]
    with pytest.raises(ValueError, match="lexical"):
        punctuation.apply_labels(words, [["changed", "0", 1], ["quasarapi", "0", 1]])
    assert punctuation.restore(words, model="asr")["preserved_asr_punctuation"]


def test_workspace_defaults_to_complete_packaged_resources(monkeypatch):
    monkeypatch.delenv("VEXY_SUBS_MODEL_WORKSPACE", raising=False)
    monkeypatch.delenv("VEXY_AUDIOFLEXY_MODEL_WORKSPACE", raising=False)
    root = workspace.adapter_root()
    assert (root / "tools/speech-issue301.py").is_file()
    assert (root / "catalogs/issue301-audiocpp.json").is_file()
    signature = workspace.signature("")
    assert "asr/tools/speech-issue301.py" in signature
    assert "asr/catalogs/issue301-audiocpp.json" in signature


def test_packaged_workspace_result_api_and_failures(tmp_path, monkeypatch):
    seen = []

    def run(args, **kwargs):
        seen.append(args)
        output = Path(
            next(a.split("=", 1)[1] for a in args if a.startswith("--output="))
        )
        output.mkdir()
        (output / "report.json").write_text(
            json.dumps(
                {"result": {"turns": [{"start": 0, "end": 1, "speaker": "speaker_0"}]}}
            )
        )

    monkeypatch.setattr(workspace.subprocess, "run", run)
    report = workspace.diarize(tmp_path / "input.wav", "sortformer_diar_v2")
    assert report["capacity"] == 4 and report["turns"][0]["speaker"] == "speaker_0"
    assert "--mode=streaming" in seen[0] and "--script" in seen[0]

    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(7, args[0])

    monkeypatch.setattr(workspace.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError) as error:
        workspace.transcribe(tmp_path / "input.wav", "", "en", "chosen")
    assert error.value.returncode == 7


def test_core_imports_do_not_import_optional_engines():
    code = "import sys; from vexy_audioflexy.speech import qwen,inference,alignment,audiocpp,punctuation,syntax,language; assert not any(m in sys.modules for m in ('torch','mlx_audio','mlx_whisper','langid','nltk','diarize'))"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_whisper_forwards_glossary_and_requires_word_times(tmp_path, monkeypatch):
    from vexy_audioflexy.speech import inference

    seen = []
    result = {
        "text": "API",
        "segments": [{"words": [{"word": " API ", "start": 0.1, "end": 0.5}]}],
    }

    def transcribe(*args, **kwargs):
        seen.append(kwargs)
        return result

    monkeypatch.setitem(sys.modules, "mlx_whisper", N(transcribe=transcribe))
    words, raw = inference.whisper(tmp_path / "clip.wav", "Terms: API.", "en")
    assert words[0].text == "API" and raw is result
    assert seen[0]["initial_prompt"] == "API" and seen[0]["word_timestamps"]
    result["segments"] = []
    with pytest.raises(ValueError, match="word timestamps"):
        inference.whisper(tmp_path / "clip.wav", "", "")


def test_native_identity_and_environment_aliases(tmp_path, monkeypatch):
    from vexy_audioflexy.speech import audiocpp

    model = tmp_path / "model.gguf"
    model.write_bytes(b"synthetic")
    monkeypatch.setenv("VEXY_AUDIOFLEXY_QWEN_MODEL", str(model))
    assert audiocpp.model_path("", "default", "VEXY_SUBS_QWEN_MODEL") == str(model)
    first = audiocpp.identity(str(model), str(model), str(model))
    model.write_bytes(b"changed")
    assert first != audiocpp.identity(str(model), str(model), str(model))
