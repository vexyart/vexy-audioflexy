# this_file: tests/test_ported_asr.py
"""Test catalog coverage and subprocess failure handling without model downloads."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("transcribecpp", ROOT / "src/vexy_audioflexy/tools/asr/tools/asr-transcribecpp.py")
wrapper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wrapper)


def test_catalog_has_only_transcription_variants_and_exact_downloads():
    rows = wrapper.packages()
    assert len(rows) == 74, "Pinned catalog coverage changed"
    for package, row in rows.items():
        assert row["capabilities"]["transcribe"]["supported"], package
        resolved, download = wrapper.resolve(package)
        assert resolved == row, package
        assert download["filename"].endswith(".gguf"), package
        assert download["size_bytes"] > 0, package


@pytest.mark.parametrize("package,quant", [("missing", "Q8_0"), ("moonshine-streaming-tiny", "invalid")])
def test_resolve_rejects_unknown_package_or_quant(package, quant):
    with pytest.raises(SystemExit):
        wrapper.resolve(package, quant)


def test_command_preserves_paths_and_uses_file_output():
    cmd = wrapper.command("model with spaces.gguf", "input with spaces.wav", "out.txt", "pl", "cpu")
    assert cmd[-1] == "input with spaces.wav", "Input path must be a single argument"
    assert cmd[cmd.index("-m") + 1] == "model with spaces.gguf"
    assert cmd[cmd.index("-o") + 1] == "out.txt"
    assert cmd[cmd.index("--backend") + 1] == "cpu"
    assert cmd[cmd.index("--language") + 1] == "pl"


def test_transcribe_rejects_missing_audio(tmp_path):
    with pytest.raises(SystemExit, match="Audio not found"):
        wrapper.transcribe(str(tmp_path / "missing.wav"))


def test_language_hint_accepts_only_declared_locales_or_their_base_languages():
    assert wrapper.language_hint(["en-US", "en-GB", "pl-PL"], "en") == "en-US"
    assert wrapper.language_hint(["en-US", "pl-PL"], "pl") == "pl-PL"
    assert wrapper.language_hint(["en-US", "en-GB"], "en-GB") == "en-GB"
    with pytest.raises(SystemExit, match="does not declare"):
        wrapper.language_hint(["en-US"], "pl")


@pytest.mark.parametrize("returncode,text", [(1, "partial"), (0, "")])
def test_transcribe_failure_does_not_replace_existing_output(tmp_path, monkeypatch, returncode, text):
    import wave

    engine = tmp_path / "engine"
    binary = engine / "build/bin/transcribe-cli"
    binary.parent.mkdir(parents=True)
    binary.touch()
    audio = tmp_path / "input.wav"
    audio.touch()
    output = tmp_path / "existing.txt"
    output.write_text("previous valid transcript")
    monkeypatch.setattr(wrapper, "ENGINE_ROOT", engine)
    monkeypatch.setattr(wrapper, "prep", lambda *args: "model.gguf")

    def run(argv, **kwargs):
        if argv[0] == "ffmpeg":
            with wave.open(argv[-1], "wb") as wav:
                wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                wav.writeframes(b"\x00\x00" * 16000)
            return SimpleNamespace(returncode=0)
        Path(argv[argv.index("-o") + 1]).write_text(text)
        return SimpleNamespace(returncode=returncode, stdout="", stderr="")

    monkeypatch.setattr(wrapper.subprocess, "run", run)
    with pytest.raises(SystemExit):
        wrapper.transcribe(str(audio), text_out=str(output), force=True)
    assert output.read_text() == "previous valid transcript", "Failed inference must retain previous evidence"


def test_installed_resource_resolves_catalog_and_runtime_override(tmp_path, monkeypatch):
    from importlib.resources import files
    import runpy

    monkeypatch.setenv("TRANSCRIBECPP_ROOT", str(tmp_path / "native engine"))
    resource = files("vexy_audioflexy").joinpath("tools/asr/tools/asr-transcribecpp.py")
    module = runpy.run_path(str(resource))
    assert module["ENGINE_ROOT"] == tmp_path / "native engine"
    assert len(module["packages"]()) == 74


def test_missing_engine_fails_before_download(tmp_path, monkeypatch):
    audio = tmp_path / "input.wav"
    audio.touch()
    monkeypatch.setattr(wrapper, "ENGINE_ROOT", tmp_path / "missing engine")
    monkeypatch.setattr(wrapper, "prep", lambda *args: pytest.fail("must not download"))
    with pytest.raises(SystemExit, match="Build transcribe.cpp first"):
        wrapper.transcribe(str(audio))


def test_local_ultra_when_wrong_identity_then_never_uses_stock_parakeet(tmp_path):
    with pytest.raises(SystemExit, match="package parakeet-ultra"):
        wrapper.local_model(str(tmp_path / "model.gguf"), "parakeet-tdt-0.6b-v3", "Q8_0")
    with pytest.raises(SystemExit, match="byte size"):
        wrapper.local_model(str(tmp_path / "missing.gguf"), "parakeet-ultra", "Q8_0")


@pytest.mark.parametrize("wrong_metadata", [False, True])
def test_local_ultra_when_gguf_metadata_checked_then_rejects_other_variant(tmp_path, monkeypatch, wrong_metadata):
    import hashlib
    from gguf import GGUFWriter

    model = tmp_path / "synthetic.gguf"
    writer = GGUFWriter(model, "parakeet")
    for key, value in wrapper.LOCAL_ULTRA["metadata"].items():
        if key == "general.architecture":
            continue
        if key == "stt.variant" and wrong_metadata:
            value = "stock-parakeet"
        if isinstance(value, str):
            writer.add_string(key, value)
        else:
            writer.add_uint32(key, value)
    writer.add_array("general.languages", ["en", "pl"])
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    expected = {**wrapper.LOCAL_ULTRA, "size_bytes": model.stat().st_size,
                "sha256": hashlib.sha256(model.read_bytes()).hexdigest()}
    monkeypatch.setattr(wrapper, "LOCAL_ULTRA", expected)
    if wrong_metadata:
        with pytest.raises(SystemExit, match="stt.variant"):
            wrapper.local_model(str(model), "parakeet-ultra", "Q8_0")
    else:
        path, spec = wrapper.local_model(str(model), "parakeet-ultra", "Q8_0")
        assert path == str(model)
        assert spec["languages"] == ["en", "pl"]
        expected["sha256"] = "0" * 64
        with pytest.raises(SystemExit, match="SHA256"):
            wrapper.local_model(str(model), "parakeet-ultra", "Q8_0")


def test_local_ultra_when_engine_commit_differs_then_rejects_before_download(tmp_path, monkeypatch):
    audio = tmp_path / "input.wav"
    audio.touch()
    engine = tmp_path / "engine"
    binary = engine / "build/bin/transcribe-cli"
    binary.parent.mkdir(parents=True)
    binary.touch()
    monkeypatch.setattr(wrapper, "ENGINE_ROOT", engine)
    monkeypatch.setattr(wrapper, "local_model", lambda *a: ("exact.gguf", {"languages": ["en"]}))
    monkeypatch.setattr(wrapper.subprocess, "check_output", lambda *a, **k: "wrong-commit")
    monkeypatch.setattr(wrapper, "prep", lambda *a: pytest.fail("Local models must not download"))
    with pytest.raises(SystemExit, match="pinned transcribe.cpp"):
        wrapper.transcribe(str(audio), package="parakeet-ultra", local_model_path="exact.gguf")


def test_command_when_threads_given_then_keeps_cpu_limit_explicit():
    argv = wrapper.command("model.gguf", "input.wav", "out.txt", "en", "cpu", 4)
    assert argv[argv.index("--threads") + 1] == "4"


def test_current_catalog_when_expanded_then_keeps_non_asr_roles_separate():
    assert len(wrapper.packages()) == 74
    assert {"orukeet", "parakeet-ultra", "phonon-2"} <= wrapper.packages().keys()
    assert {row["role"] for row in wrapper.roles().values()} == {"diarize", "langid", "vad"}
    for package, row in wrapper.roles().items():
        assert not row["capabilities"]["transcribe"]["supported"]
        with pytest.raises(SystemExit, match="Unknown"):
            wrapper.resolve(package)
        assert len(row["published_revision"]) == 40
    assert wrapper._role_spec("silero-vad-v6.2", "")[1]["quant"] == "F32"


def test_phonon_when_official_repository_unavailable_then_no_substitute_download(monkeypatch):
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **k: pytest.fail("Unavailable provider must not download substitutes"))
    with pytest.raises(SystemExit, match="unavailable"):
        wrapper.prep("phonon-2")


def test_prep_when_published_revision_pinned_then_uses_exact_provider_identity(tmp_path, monkeypatch):
    import hashlib
    import huggingface_hub

    model = tmp_path / "model.gguf"
    model.write_bytes(b"synthetic model")
    requests = []
    monkeypatch.setattr(wrapper, "resolve", lambda *a: ({"published_repo": "public/test", "published_revision": "f" * 40},
                                                        {"filename": "exact.gguf", "size_bytes": model.stat().st_size,
                                                         "published_sha256": hashlib.sha256(model.read_bytes()).hexdigest()}))
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda **k: requests.append(k) or str(model))
    assert wrapper.prep("synthetic") == str(model)
    assert requests == [{"repo_id": "public/test", "filename": "exact.gguf", "revision": "f" * 40}]


@pytest.mark.parametrize("role,text,key", [
    ("langid", "candidate: 1 en index=20 p=0.9 logit=3.2\n", "candidates"),
    ("vad", "segment: 0 start=1600 end=16000 t0=0.100 t1=1.000\n", "segments"),
    ("diarize", "[   0.10 ->    1.00] S2\n", "turns"),
])
def test_native_role_records_when_valid_then_typed_json_without_transcript(role, text, key):
    result = wrapper.role_result(role, text, 2.0)
    assert len(result[key]) == 1
    assert "text" not in result


@pytest.mark.parametrize("role,text", [
    ("langid", ""), ("langid", "candidate: 1 en index=20 p=nan logit=3.2"),
    ("vad", "segment: 0 start=1600 end=999999 t0=0.100 t1=62.5"),
    ("diarize", "[   1.00 ->    0.10] S2"), ("langid", "a transcript"),
])
def test_native_role_records_when_malformed_then_rejects(role, text):
    with pytest.raises(ValueError):
        wrapper.role_result(role, text, 2.0)


def test_silence_when_vad_or_diarization_then_empty_segments_are_valid():
    assert wrapper.role_result("vad", "", 2.0) == {"segments": []}
    assert wrapper.role_result("diarize", "", 2.0) == {"turns": []}


@pytest.mark.parametrize("returncode,native", [(1, "partial"), (0, "malformed"), (0, "")])
def test_role_when_fails_or_silence_then_previous_output_is_protected(tmp_path, monkeypatch, returncode, native):
    import wave

    source, output = tmp_path / "input.wav", tmp_path / "role.json"
    source.touch()
    output.write_text("previous valid role output")
    binary = tmp_path / "engine/build/bin/transcribe-cli"
    binary.parent.mkdir(parents=True)
    binary.touch()
    monkeypatch.setattr(wrapper, "ENGINE_ROOT", tmp_path / "engine")
    monkeypatch.setattr(wrapper, "_require_revision", lambda *a: None)
    monkeypatch.setattr(wrapper, "prep_role", lambda *a: "exact-role-model.gguf")

    def run(argv, **kwargs):
        if argv[0] == "ffmpeg":
            with wave.open(argv[-1], "wb") as wav:
                wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                wav.writeframes(b"\0\0" * 16000)
            return SimpleNamespace(returncode=0)
        Path(argv[argv.index("-o") + 1]).write_text(native)
        return SimpleNamespace(returncode=returncode, stdout="", stderr="")

    monkeypatch.setattr(wrapper.subprocess, "run", run)
    if returncode or native:
        with pytest.raises((SystemExit, ValueError)):
            wrapper.role(str(source), "silero-vad-v6.2", str(output), force=True)
        assert output.read_text() == "previous valid role output"
    else:
        import json
        wrapper.role(str(source), "silero-vad-v6.2", str(output), force=True)
        assert json.loads(output.read_text())["segments"] == []


def test_sortformer_when_final_frame_is_padded_then_retains_native_end_and_clips_to_audio():
    result = wrapper.role_result("diarize", "[   0.00 ->    2.72] S1\n", 2.67075)
    assert result["turns"] == [{"start": 0.0, "end": 2.67075, "speaker": "1", "native_end": 2.72}]
    with pytest.raises(ValueError):
        wrapper.role_result("diarize", "[   0.00 ->    3.00] S1\n", 2.67075)



def test_catalog_when_compared_with_5700_and_provider_metadata_then_exact_variant_and_artifact_sets():
    import hashlib
    import json

    rows = list(wrapper.packages().values()) + list(wrapper.roles().values())
    variants = "\n".join(sorted(row["variant"] for row in rows))
    # Independent oracle: upstream5700 catalog plus immutable provider-size corrections.
    assert hashlib.sha256(variants.encode()).hexdigest() == "1670fefbf8eb730f1488fa199f168cca75551efa3a12d60fdf7f923296a17dee"
    artifacts = sorted((row["variant"], a["quant"], a["filename"], a["size_bytes"]) for row in rows for a in row["downloads"])
    assert hashlib.sha256(json.dumps(artifacts, separators=(",", ":")).encode()).hexdigest() == "a523df09161e6c7df76cf51714a6d59ed7c28fd9440326a911f00ca4f8e3b06e"
    assert len(artifacts) == 398
    assert sum(row["availability"] == "repository-unavailable" for row in rows) == 1


def test_new_published_variant_when_runtime_old_then_rejects_before_download(tmp_path, monkeypatch):
    source = tmp_path / "input.wav"
    source.touch()
    engine = tmp_path / "engine"
    binary = engine / "build/bin/transcribe-cli"
    binary.parent.mkdir(parents=True)
    binary.touch()
    monkeypatch.setattr(wrapper, "ENGINE_ROOT", engine)
    monkeypatch.setattr(wrapper.subprocess, "check_output", lambda *a, **k: "old-revision")
    monkeypatch.setattr(wrapper, "prep", lambda *a: pytest.fail("Runtime rejection must precede downloads"))
    with pytest.raises(SystemExit, match="requires transcribe.cpp commit"):
        wrapper.transcribe(str(source), package="orukeet")


def test_role_download_when_provider_hash_differs_then_rejects(tmp_path, monkeypatch):
    import huggingface_hub

    model = tmp_path / "synthetic.gguf"
    model.write_bytes(b"synthetic role weight")
    monkeypatch.setattr(wrapper, "_role_spec", lambda *a: ({"published_repo": "public/role", "published_revision": "f" * 40},
        {"filename": "exact.gguf", "size_bytes": model.stat().st_size, "published_sha256": "0" * 64}))
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **k: str(model))
    with pytest.raises(SystemExit, match="SHA256"):
        wrapper.prep_role("synthetic")


def test_phonon_local_conversion_when_exact_profile_then_no_gguf_schema_substitution(tmp_path, monkeypatch):
    import hashlib
    from gguf import GGUFWriter

    model = tmp_path / "phonon-2-F32.gguf"
    writer = GGUFWriter(model, "parakeet")
    for key, value in wrapper.LOCAL_PHONON["metadata"].items():
        if key == "general.architecture":
            continue
        if isinstance(value, str):
            writer.add_string(key, value)
        else:
            writer.add_uint32(key, value)
    writer.add_array("general.languages", ["en"])
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    expected = {**wrapper.LOCAL_PHONON, "size_bytes": model.stat().st_size,
                "sha256": hashlib.sha256(model.read_bytes()).hexdigest()}
    monkeypatch.setattr(wrapper, "LOCAL_PHONON", expected)
    path, spec = wrapper.local_model(str(model), "phonon-2", "F32")
    assert path == str(model)
    assert spec["languages"] == ["en"]
    assert spec["local_profile"]["runtime_commit"] == "5700f127e38dd53c4735ea23ef743b4b538eacd4"
    with pytest.raises(SystemExit, match="quant F32"):
        wrapper.local_model(str(model), "phonon-2", "Q8_0")


def test_phonon_local_conversion_when_legacy_runtime_then_rejects(tmp_path, monkeypatch):
    audio = tmp_path / "input.wav"
    audio.touch()
    engine = tmp_path / "engine"
    binary = engine / "build/bin/transcribe-cli"
    binary.parent.mkdir(parents=True)
    binary.touch()
    monkeypatch.setattr(wrapper, "ENGINE_ROOT", engine)
    monkeypatch.setattr(wrapper, "local_model", lambda *a: ("exact-phonon.gguf", {"languages": ["en"], "local_profile": wrapper.LOCAL_PHONON}))
    monkeypatch.setattr(wrapper.subprocess, "check_output", lambda *a, **k: "077110eb00ca6880e13bef4ddebf766b3b76896c")
    monkeypatch.setattr(wrapper, "prep", lambda *a: pytest.fail("Local conversion must not download a substitute"))
    with pytest.raises(SystemExit, match="pinned transcribe.cpp"):
        wrapper.transcribe(str(audio), package="phonon-2", quant="F32", local_model_path="exact-phonon.gguf")


def test_prep_when_converted_local_model_supplied_then_validates_without_network(monkeypatch):
    import huggingface_hub

    monkeypatch.setattr(wrapper, "local_model", lambda path, package, quant: (path, {"languages": ["en"]}))
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **k: pytest.fail("Explicit local prep must not download"))
    assert wrapper.prep("phonon-2", "F32", local_model_path="converted-phonon.gguf") == "converted-phonon.gguf"


@pytest.mark.parametrize("kind", ["asr", "role"])
@pytest.mark.parametrize("correct_hash", [False, True])
def test_published_prep_when_same_size_bytes_then_requires_exact_hash_without_cache_mutation(tmp_path, monkeypatch, capsys, kind, correct_hash):
    import hashlib
    import huggingface_hub

    model = tmp_path / "provider-owned.gguf"
    original = b"same-size cached model bytes"
    model.write_bytes(original)
    artifact = {"filename": "exact.gguf", "size_bytes": len(original),
                "published_sha256": hashlib.sha256(original).hexdigest() if correct_hash else "0" * 64}
    spec = {"published_repo": "public/test", "published_revision": "f" * 40}
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **k: str(model))
    if kind == "asr":
        monkeypatch.setattr(wrapper, "resolve", lambda *a: (spec, artifact))
        prepare = wrapper.prep
    else:
        monkeypatch.setattr(wrapper, "_role_spec", lambda *a: (spec, artifact))
        prepare = wrapper.prep_role
    if correct_hash:
        assert prepare("synthetic") == str(model)
    else:
        with pytest.raises(SystemExit, match="SHA256"):
            prepare("synthetic")
        assert "ready" not in capsys.readouterr().out
    assert model.read_bytes() == original


def test_transcribe_when_same_size_provider_cache_has_wrong_hash_then_preserves_output_before_native_call(tmp_path, monkeypatch):
    import huggingface_hub

    audio, output, cached = tmp_path / "audio.wav", tmp_path / "transcript.txt", tmp_path / "cached.gguf"
    audio.touch()
    output.write_text("previous valid transcript")
    original = b"wrong but expected-size cached bytes"
    cached.write_bytes(original)
    binary = tmp_path / "engine/build/bin/transcribe-cli"
    binary.parent.mkdir(parents=True)
    binary.touch()
    monkeypatch.setattr(wrapper, "ENGINE_ROOT", tmp_path / "engine")
    monkeypatch.setattr(wrapper, "resolve", lambda *a: ({"published_repo": "public/test", "published_revision": "f" * 40, "languages": ["en"]},
        {"filename": "exact.gguf", "size_bytes": len(original), "published_sha256": "0" * 64}))
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda **k: str(cached))
    monkeypatch.setattr(wrapper.subprocess, "run", lambda *a, **k: pytest.fail("Wrong cached identity must fail before native processes"))
    with pytest.raises(SystemExit, match="SHA256"):
        wrapper.transcribe(str(audio), package="synthetic", text_out=str(output), force=True)
    assert output.read_text() == "previous valid transcript"
    assert cached.read_bytes() == original


def test_published_artifact_when_hash_missing_then_readiness_is_rejected(tmp_path):
    model = tmp_path / "owner.gguf"
    model.write_bytes(b"owner bytes")
    with pytest.raises(SystemExit, match="declare its SHA256"):
        wrapper._verify_published(model, {"size_bytes": model.stat().st_size})
    assert model.read_bytes() == b"owner bytes"
