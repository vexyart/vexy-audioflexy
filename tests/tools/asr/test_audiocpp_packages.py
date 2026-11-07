# this_file: tests/tools/asr/test_audiocpp_packages.py
"""A sibling quantization must not count as the requested package's weights."""
import importlib.util
import json

import pytest
import wave
from pathlib import Path

spec = importlib.util.spec_from_file_location("audiocpp", Path(__file__).resolve().parents[3] / "src/vexy_audioflexy/tools/asr" / "tools/asr-audiocpp.py")
wrapper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wrapper)


def test_installed_requires_exact_declared_gguf(tmp_path, monkeypatch):
    specs = tmp_path / "specs"
    specs.mkdir()
    (specs / "test.json").write_text(json.dumps({
        "family": "test", "packages": [{"id": "test-q8", "target_directory": "Shared",
        "files": ["Shared/test-q8.gguf"], "strip_prefix": "Shared"}]}))
    monkeypatch.setattr(wrapper, "SPECS_DIR", specs)
    monkeypatch.setattr(wrapper, "MODELS_ROOT", tmp_path / "models")
    wrapper._specs.cache_clear()
    root = wrapper.MODELS_ROOT / "Shared"
    root.mkdir(parents=True)
    (root / "test-f16.gguf").write_bytes(b"other quantization")
    assert wrapper._model_path("test-q8") == root / "test-q8.gguf"
    assert not wrapper._installed("test-q8"), "A different quantization is not installed support"
    (root / "test-q8.gguf").write_bytes(b"requested weights")
    assert wrapper._installed("test-q8"), "The exact declared file must resolve"
    wrapper._specs.cache_clear()


def test_prep_when_specs_are_frozen_then_uses_provider_snapshot_and_native_links(tmp_path, monkeypatch):
    import huggingface_hub

    calls = []
    snapshot = tmp_path / "hf-cache/snapshot"
    (snapshot / "remote").mkdir(parents=True)
    (snapshot / "remote/model.gguf").write_bytes(b"exact cached quantization")
    monkeypatch.setattr(wrapper, "MODELS_ROOT", tmp_path / "models")
    monkeypatch.setattr(wrapper, "_require_engine", lambda: tmp_path / "engine")
    entry = {"download_kind": "huggingface_snapshot", "family": "test",
             "target_directory": "Test", "repo": "public/test", "revision": "pinned",
             "files": ["remote/model.gguf"], "strip_prefix": "remote"}
    monkeypatch.setattr(wrapper, "_resolve", lambda _: entry)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **k: calls.append((a, k)) or str(snapshot))
    monkeypatch.setattr(wrapper.subprocess, "run", lambda *a, **k: pytest.fail("Native downloader must not run"))
    wrapper.prep("test-f16")
    assert calls == [(("public/test",), {"revision": "pinned", "allow_patterns": ["remote/model.gguf"]})]
    link = tmp_path / "models/Test/model.gguf"
    assert link.is_symlink() and link.samefile(snapshot / "remote/model.gguf")


def test_native_command_when_paths_are_relative_then_resolves_before_engine_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "input.wav"
    with wave.open(str(source), "wb") as audio:
        audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        audio.writeframes(b"\x00\x00" * 160)
    monkeypatch.setattr(wrapper, "SPECS_DIR", tmp_path / "frozen-specs")
    monkeypatch.setattr(wrapper, "_require_engine", lambda: tmp_path / "engine")
    monkeypatch.setattr(wrapper, "_resolve", lambda _: {"family": "test"})
    monkeypatch.setattr(wrapper, "_installed", lambda _: True)
    monkeypatch.setattr(wrapper, "_model_path", lambda _: tmp_path / "model.gguf")

    def run(args, console):
        assert args[args.index("--audio") + 1] == str(source), "Native cwd must not alter input identity"
        assert args[args.index("--text-out") + 1] == str(tmp_path / "result.txt")
        assert args[args.index("--model-spec-override") + 1] == str(wrapper.SPECS_DIR)
        (tmp_path / "result.txt").write_text("recognized speech")
        return "", "", 0

    monkeypatch.setattr(wrapper, "_run_engine", run)
    wrapper.transcribe("input.wav", package="test-f16", text_out="result.txt")


def test_multifile_package_requires_all_declared_files(tmp_path, monkeypatch):
    monkeypatch.setattr(wrapper, "MODELS_ROOT", tmp_path)
    entry = {"target_directory": "shared", "files": ["remote/model.gguf", "remote/mmproj.gguf"], "strip_prefix": "remote"}
    monkeypatch.setattr(wrapper, "_resolve", lambda _: entry)
    root = tmp_path / "shared"
    root.mkdir()
    (root / "model.gguf").write_bytes(b"model")
    assert not wrapper._installed("wanted")
    (root / "mmproj.gguf").write_bytes(b"projector")
    assert wrapper._installed("wanted")


def test_binary_override_can_select_full_build(tmp_path, monkeypatch):
    binary = tmp_path / "audiocpp_cli"
    binary.touch()
    monkeypatch.setenv("AUDIOCPP_BIN", str(binary))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module._require_engine() == binary


def test_package_download_overrides_inherit_defaults(tmp_path, monkeypatch):
    specs = tmp_path / "specs"
    specs.mkdir()
    (specs / "test.json").write_text(json.dumps({"family": "test", "package_defaults": {"download": {"kind": "huggingface_snapshot", "repo": "default/repo"}}, "packages": [{"id": "test", "download": {"repo": "override/repo"}}]}))
    monkeypatch.setattr(wrapper, "SPECS_DIR", specs)
    monkeypatch.setattr(wrapper, "EXTRA_SPECS_DIR", tmp_path / "extra")
    wrapper._specs.cache_clear()
    assert wrapper._resolve("test")["download_kind"] == "huggingface_snapshot"
    assert wrapper._resolve("test")["repo"] == "override/repo"
    wrapper._specs.cache_clear()


def test_parakeet_omits_unsupported_language_option(tmp_path, monkeypatch):
    source = tmp_path / "input.wav"
    with wave.open(str(source), "wb") as audio:
        audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        audio.writeframes(b"\x00\x00" * 160)
    monkeypatch.setattr(wrapper, "_require_engine", lambda: tmp_path / "engine")
    monkeypatch.setattr(wrapper, "_resolve", lambda _: {"family": "parakeet_tdt"})
    monkeypatch.setattr(wrapper, "_installed", lambda _: True)
    monkeypatch.setattr(wrapper, "_model_path", lambda _: tmp_path / "model.gguf")
    def run(args, console):
        assert "--language" not in args
        (tmp_path / "result.txt").write_text("recognized speech")
        return "", "", 0
    monkeypatch.setattr(wrapper, "_run_engine", run)
    wrapper.transcribe(str(source), package="test", text_out=str(tmp_path / "result.txt"), lang="en")


def test_central_audio_cpp_packages_use_exact_repository_paths(monkeypatch):
    monkeypatch.setattr(wrapper, "EXTRA_SPECS_DIR", Path(__file__).resolve().parents[3] / "src/vexy_audioflexy/tools/asr" / "model_specs")
    wrapper._specs.cache_clear()
    for package, family, filename in [
        ("phonon2_central_q8", "parakeet_tdt", "phonon-2-mixed-q8.gguf"),
        ("fun_asr_nano_central_q8", "fun_asr_nano", "fun-asr-nano-2512-q8_0.gguf"),
        ("hviske_central_q8", "hviske_asr", "hviske-v5.3-q8_0.gguf"),
        ("mms_aligner_central_f16", "mms_forced_aligner", "mms-forced-aligner-f16.gguf"),
        ("vibevoice_streaming_central_q4", "vibevoice_asr_streaming", "vibevoice-asr-streaming-7b-q4_k.gguf"),
    ]:
        entry = wrapper._resolve(package)
        assert entry["repo"] == "audio-cpp/audio.cpp-gguf"
        assert entry["family"] == family
        assert Path(entry["files"][0]).name == filename
    wrapper._specs.cache_clear()


def test_align_resolves_relative_paths_before_engine_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "input.wav"
    with wave.open(str(source), "wb") as audio:
        audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        audio.writeframes(b"\x00\x00" * 160)
    monkeypatch.setattr(wrapper, "_require_engine", lambda: tmp_path / "engine")
    monkeypatch.setattr(wrapper, "_resolve", lambda _: {"family": "mms_forced_aligner"})
    monkeypatch.setattr(wrapper, "_installed", lambda _: True)
    monkeypatch.setattr(wrapper, "_model_path", lambda _: tmp_path / "model.gguf")
    def run(args, console):
        assert args[args.index("--audio") + 1] == str(source)
        assert args[args.index("--words-out") + 1] == str(tmp_path / "words.json")
        (tmp_path / "words.json").write_text('[{"word":"hello"}]')
        return "", "", 0
    monkeypatch.setattr(wrapper, "_run_engine", run)
    wrapper.align("input.wav", "hello", package="test", words_out="words.json", lang="eng")


def test_prep_force_replaces_layout_link_and_checks_result(tmp_path, monkeypatch):
    import huggingface_hub

    snapshot = tmp_path / "hf-cache"
    snapshot.mkdir()
    source = snapshot / "exact.gguf"
    source.write_bytes(b"pinned snapshot model")
    target = tmp_path / "models/Test"
    target.mkdir(parents=True)
    (target / "exact.gguf").write_bytes(b"older native copy")
    entry = {"download_kind": "huggingface_snapshot", "family": "test",
             "target_directory": "Test", "repo": "public/test", "files": ["exact.gguf"]}
    monkeypatch.setattr(wrapper, "_resolve", lambda _: entry)
    monkeypatch.setattr(wrapper, "MODELS_ROOT", tmp_path / "models")
    monkeypatch.setattr(wrapper, "_require_engine", lambda: tmp_path / "engine")
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **k: str(snapshot))
    monkeypatch.setattr(wrapper, "_installed", lambda _: False)
    with pytest.raises(SystemExit, match="incomplete"):
        wrapper.prep("test", force=True)
    assert (target / "exact.gguf").is_symlink()
    assert (target / "exact.gguf").samefile(source)


@pytest.mark.parametrize("name", ["../escape.gguf", "/absolute.gguf", "", "folder\\escape.gguf"])
def test_provider_layout_when_unsafe_paths_then_rejects(name):
    with pytest.raises(ValueError, match="Unsafe"):
        wrapper._safe_relative(name)


def test_provider_layout_when_existing_other_file_then_retains_it(tmp_path):
    snapshot, target = tmp_path / "snapshot", tmp_path / "models"
    snapshot.mkdir()
    target.mkdir()
    (snapshot / "model.gguf").write_bytes(b"pinned")
    (target / "model.gguf").write_bytes(b"owner copy")
    with pytest.raises(FileExistsError, match="provider file"):
        wrapper._link_snapshot({"files": ["model.gguf"]}, snapshot, target, False)
    assert (target / "model.gguf").read_bytes() == b"owner copy"


def test_relative_engine_and_models_roots_resolve_from_caller(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AUDIOCPP_ROOT", "engine")
    monkeypatch.setenv("AUDIOCPP_MODELS_ROOT", "weights")
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    assert adapter.ENGINE_ROOT == tmp_path / "engine"
    assert adapter.MODELS_ROOT == tmp_path / "weights"
