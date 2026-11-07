# this_file: tests/test_final_boundaries.py
"""Synthetic batch resume and provider preparation boundaries; no model jobs."""
import importlib.util
from importlib.resources import files
import json
from pathlib import Path
import subprocess
import sys
import types

import pytest

TOOLS = Path(str(files("vexy_audioflexy").joinpath("tools")))


def load(relative):
    spec = importlib.util.spec_from_file_location("boundary_" + Path(relative).stem.replace("-", "_"), TOOLS / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(TOOLS / "tts"))
    monkeypatch.syspath_prepend(str(TOOLS / "tts/tools"))
    monkeypatch.syspath_prepend(str(TOOLS / "asr/tools"))
    monkeypatch.setenv("VAFX_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("VAFX_VOICE_ROOT", str(tmp_path / "voice"))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))


def voice_fixture(tmp_path, monkeypatch, name="inflect-micro-v2-onnx"):
    batch = load("tts/tts_all.py")
    text = tmp_path / "text.txt"; text.write_text("Synthetic speech.")
    refs = tmp_path / "refs"; refs.mkdir()
    (refs / "en_test.wav").write_bytes(b"synthetic reference")
    (refs / "en_test.lab").write_text("Synthetic reference.")
    prompts = tmp_path / "prompts.json"; prompts.write_text('["A calm voice."]')
    monkeypatch.setattr(batch, "get_audio_duration", lambda _: 1.0)
    monkeypatch.setattr(batch, "save_dashboard", lambda out: out / "index.html")
    calls = []
    def run_model(**kwargs):
        calls.append(kwargs)
        Path(kwargs["out"]).write_bytes(b"generated wave")
        return True, 0.5, ""
    def convert(cmd, **kwargs):
        assert cmd[0] == "ffmpeg"
        Path(cmd[-1]).write_bytes(b"generated mp3")
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(batch, "run_model", run_model)
    monkeypatch.setattr(batch.subprocess, "run", convert)
    kwargs = dict(outdir=str(tmp_path / "out"), text_en=str(text), references=str(refs),
                  models=name, langs="en", prep=False, design_prompts_file=str(prompts))
    return batch, kwargs, calls


def voice_result(tmp_path):
    records = json.loads((tmp_path / "out/report.json").read_text())["results"]
    return next(row for row in records.values() if not row.get("ignored"))


def test_tts_relative_reference_reaches_actual_child_argv(tmp_path, monkeypatch):
    batch, kwargs, _ = voice_fixture(tmp_path, monkeypatch, "audio8-06b-onnx-int4")
    actual_run = load("tts/tts_all.py").run_model
    monkeypatch.setattr(batch, "run_model", actual_run)
    monkeypatch.chdir(tmp_path)
    kwargs.update(outdir="out", text_en="text.txt", references="refs")
    calls = []
    def child(cmd, **options):
        calls.append((cmd, options))
        return subprocess.CompletedProcess(cmd, 1, "", "synthetic stop before inference")
    monkeypatch.setattr(batch.subprocess, "run", child)
    batch.main(**kwargs)
    cmd, options = calls[0]
    ref = Path(cmd[cmd.index("--ref") + 1])
    assert ref.is_absolute() and ref == tmp_path / "refs/en_test.wav"
    assert ref.is_file() and options["cwd"] == str(tmp_path / "out")
    assert batch.REF_TEXTS["en"] == "Synthetic reference."


@pytest.mark.parametrize("change", ["text", "reference", "transcript", "prompt", "options", "tampered", "legacy", "raw"])
def test_tts_full_batch_refuses_stale_or_unbound_outputs(tmp_path, monkeypatch, change):
    name = "qwen3-voicedesign-mlx" if change == "prompt" else "audio8-06b-onnx-int4"
    batch, kwargs, calls = voice_fixture(tmp_path, monkeypatch, name)
    batch.main(**kwargs)
    assert voice_result(tmp_path)["success"]
    batch.main(**kwargs)
    assert len(calls) == 1 and voice_result(tmp_path)["success"]
    mp3 = next((tmp_path / "out").glob("*.mp3"))
    if change == "text": Path(kwargs["text_en"]).write_text("Changed speech.")
    elif change == "reference": (tmp_path / "refs/en_test.wav").write_bytes(b"changed reference")
    elif change == "transcript": (tmp_path / "refs/en_test.lab").write_text("Changed transcript.")
    elif change == "prompt": Path(kwargs["design_prompts_file"]).write_text('["A loud voice."]')
    elif change == "options": next(m for m in batch.MODELS if m["name"] == name)["env"] = {"SYNTHETIC_OPTION": "changed"}
    elif change == "tampered": mp3.write_bytes(b"owner edited mp3")
    elif change == "legacy":
        for receipt in (tmp_path / "out").glob("*request.json"): receipt.unlink()
        (tmp_path / "out/report.json").unlink()
    else:
        mp3.unlink(); mp3.with_suffix(".wav").write_bytes(b"owner raw wav")
    preserved = {p.name: p.read_bytes() for p in (tmp_path / "out").glob("*.mp3")}
    preserved.update({p.name: p.read_bytes() for p in (tmp_path / "out").glob("*.wav")})
    batch.main(**kwargs)
    assert len(calls) == 1, "stale output requires explicit force"
    assert not voice_result(tmp_path)["success"]
    assert "force" in voice_result(tmp_path)["error"]
    assert all((tmp_path / "out" / name).read_bytes() == content for name, content in preserved.items())
    batch.main(**kwargs, force=True)
    assert len(calls) == 2 and calls[-1]["force"] and voice_result(tmp_path)["success"]


@pytest.mark.parametrize("change", ["audio", "transcript", "options", "tampered", "legacy"])
def test_asr_full_batch_refuses_same_basename_and_other_stale_outputs(tmp_path, monkeypatch, change):
    batch = load("asr/asr_all.py")
    corpus = tmp_path / "a"; corpus.mkdir()
    (corpus / "en_same.wav").write_bytes(b"first input")
    (corpus / "en_same.lab").write_text("Synthetic speech.")
    calls = []
    def transcribe(model, audio, out, lang, timeout, verbose, force):
        calls.append(force)
        # Exercise the existence-only behavior retained by standalone adapters.
        if not out.exists() or force: out.write_text("Synthetic speech.")
        return True, 0.5, "", {}
    monkeypatch.setattr(batch, "run_model", transcribe)
    monkeypatch.setattr(batch, "save_dashboard", lambda out: out / "index.html")
    kwargs = dict(eval_dir=str(corpus), outdir=str(tmp_path / "out"), models="qwen3-asr-17b-mlx-8bit", langs="en", prep=False)
    def result():
        data = json.loads((tmp_path / "out/report.json").read_text())
        return data["results"][batch._result_key("qwen3-asr-17b-mlx-8bit", "en")]["clips"][0]
    batch.main(**kwargs)
    assert result()["success"]
    batch.main(**kwargs)
    assert result()["success"]
    resumed_calls = len(calls)
    output = Path(result()["transcript_path"])
    if change == "audio":
        corpus2 = tmp_path / "b"; corpus2.mkdir()
        (corpus2 / "en_same.wav").write_bytes(b"replacement input")
        (corpus2 / "en_same.lab").write_text("New reference.")
        kwargs["eval_dir"] = str(corpus2)
    elif change == "transcript": (corpus / "en_same.lab").write_text("New reference.")
    elif change == "options": next(m for m in batch.MODELS if m["name"] == "qwen3-asr-17b-mlx-8bit")["env"] = {"SYNTHETIC_OPTION": "changed"}
    elif change == "tampered": output.write_text("Owner transcript.")
    else:
        for receipt in (tmp_path / "out").glob("*request.json"): receipt.unlink()
        (tmp_path / "out/report.json").unlink()
    old = output.read_bytes()
    batch.main(**kwargs)
    assert len(calls) == resumed_calls and not result()["success"]
    assert "force" in result()["error"] and output.read_bytes() == old
    assert "hypothesis" not in result()
    batch.main(**kwargs, force=True)
    assert calls == [False, True] and result()["success"]


@pytest.mark.parametrize("namespace,filename", [("fx", "aud-audiocpp.py"), ("tts", "tts-audiocpp.py")])
def test_audiocpp_prep_uses_provider_and_native_links(tmp_path, monkeypatch, namespace, filename):
    adapter = load(f"{namespace}/tools/{filename}")
    specs = tmp_path / "specs"; specs.mkdir()
    spec = {"family": "synthetic", "package_defaults": {"download": {"kind": "huggingface_snapshot", "repo": "synthetic/repo", "revision": "pinned"}},
            "packages": [{"id": "synthetic", "target_directory": "native", "strip_prefix": "remote", "files": ["remote/model.gguf", "remote/config.json"]}]}
    (specs / "model.json").write_text(json.dumps(spec))
    adapter._specs.cache_clear()
    monkeypatch.setattr(adapter, "SPECS_DIR", specs)
    if hasattr(adapter, "EXTRA_SPECS_DIR"): monkeypatch.setattr(adapter, "EXTRA_SPECS_DIR", tmp_path / "absent")
    monkeypatch.setattr(adapter, "MODELS_ROOT", tmp_path / "managed")
    monkeypatch.setattr(adapter, "_require_engine", lambda: Path("fake-engine"))
    snapshot = tmp_path / "hub/models--synthetic--repo/snapshots/pinned"
    (snapshot / "remote").mkdir(parents=True)
    (snapshot / "remote/model.gguf").write_bytes(b"provider weights")
    (snapshot / "remote/config.json").write_text("{}")
    calls = []
    def download(repo, **kwargs):
        calls.append((repo, kwargs)); return str(snapshot)
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(snapshot_download=download))
    def forbidden(*args, **kwargs): pytest.fail("native installer/HTTP must not execute")
    monkeypatch.setattr(adapter.subprocess, "run", forbidden)
    adapter.prep("synthetic")
    assert calls == [("synthetic/repo", {"revision": "pinned", "allow_patterns": spec["packages"][0]["files"]})]
    target = tmp_path / "managed/native"
    assert (target / "model.gguf").is_symlink()
    assert (target / "model.gguf").resolve() == snapshot / "remote/model.gguf"
    assert (target / ".audiocpp-package-synthetic.json").is_file()
    assert adapter._installed("synthetic")
    adapter.prep("synthetic")
    assert len(calls) == 1
    marker = target / ".audiocpp-package-synthetic.json"
    marker.unlink()
    (target / "model.gguf").unlink()
    (target / "config.json").unlink()
    (target / "config.json").write_bytes(b"owner config")
    with pytest.raises(FileExistsError): adapter.prep("synthetic")
    assert not marker.exists() and (target / "config.json").read_bytes() == b"owner config"
    adapter.prep("synthetic", force=True)
    assert adapter._installed("synthetic") and (target / "config.json").is_symlink()
    assert all(p.is_symlink() for p in target.rglob("*") if p.suffix in {".gguf", ".safetensors"})


@pytest.mark.parametrize("target_kind", ["direct", "alias", "old", "companion_conflict", "external"])
def test_indextts_preflights_complete_view_without_mutating_provider(tmp_path, monkeypatch, target_kind):
    adapter = load("tts/tools/tts-indextts2-mlx.py")
    engine = tmp_path / "engine"; engine.mkdir(); (engine / "pyproject.toml").touch()
    hub = tmp_path / "hub"
    snapshot = hub / "models--synthetic--model/snapshots/new"
    snapshot.mkdir(parents=True); (snapshot / "weights.safetensors").write_bytes(b"new weights")
    old = snapshot.parent / "old"; old.mkdir(); (old / "weights.safetensors").write_bytes(b"old weights")
    camp = hub / "models--synthetic--camp/snapshots/main/camp.bin"
    camp.parent.mkdir(parents=True); camp.write_bytes(b"companion")
    target = tmp_path / "view"
    if target_kind == "direct": target = snapshot
    elif target_kind == "alias": target.symlink_to(snapshot, target_is_directory=True)
    elif target_kind == "old": target = old
    elif target_kind == "companion_conflict":
        (target / "hf_cache").mkdir(parents=True)
        (target / "hf_cache" / adapter.CAMPPLUS_FILE).write_bytes(b"owner companion")
    def provider_state():
        return {str(p.relative_to(hub)): ("link", str(p.readlink()), p.read_bytes()) if p.is_symlink()
                else ("file", p.read_bytes()) if p.is_file() else ("directory",)
                for p in hub.rglob("*")}
    before = provider_state()
    monkeypatch.setattr(adapter, "ENGINE_ROOT", engine)
    monkeypatch.setattr(adapter, "MODEL_DIR", target)
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(snapshot_download=lambda *a, **k: str(snapshot), hf_hub_download=lambda *a, **k: str(camp)))
    def run(cmd, **kwargs):
        exec(cmd[-1], {})
        return subprocess.CompletedProcess(cmd, 0)
    monkeypatch.setattr(adapter.subprocess, "run", run)
    if target_kind == "external":
        adapter.prep()
        assert (target / "weights.safetensors").is_symlink()
        assert (target / "hf_cache" / adapter.CAMPPLUS_FILE).resolve() == camp
    else:
        with pytest.raises((ValueError, FileExistsError)):
            adapter.prep()
        if target_kind == "companion_conflict": assert not (target / "weights.safetensors").exists()
    assert provider_state() == before


def test_voice_audiocpp_companions_use_same_provider_route_and_retry_budget(tmp_path, monkeypatch):
    adapter = load("tts/tools/tts-audiocpp.py")
    monkeypatch.setattr(adapter, "_require_engine", lambda: Path("fake-engine"))
    monkeypatch.setattr(adapter, "_installed", lambda _: False)
    monkeypatch.setattr(adapter, "_model_path", lambda package: tmp_path / package)
    monkeypatch.setattr(adapter, "_resolve", lambda package: {"id": package})
    calls = []
    def install(entry, root, package, force):
        calls.append((package, force))
        if len(calls) < 3: raise OSError("synthetic transport failure")
    monkeypatch.setattr(adapter, "_install_package", install)
    with pytest.raises(SystemExit, match="incomplete"):
        adapter.prep("main", deps="companion", force=True)
    assert calls == [("companion", True)] * 3
    calls.clear()
    installed = set()
    monkeypatch.setattr(adapter, "_installed", lambda package: package in installed)
    def success(entry, root, package, force):
        calls.append((package, force)); installed.add(package)
    monkeypatch.setattr(adapter, "_install_package", success)
    adapter.prep("main", deps="companion", force=True)
    assert calls == [("companion", True), ("main", True)]


@pytest.mark.parametrize("value", ["../escape", "/absolute", "", ".", "back\\slash"])
def test_provider_rejects_unsafe_spec_paths(value):
    helper = load("_audiocpp_provider.py")
    with pytest.raises(ValueError): helper.safe_relative(value)


def test_asr_batch_blocks_actual_qwen_existence_skip_but_standalone_keeps_it(tmp_path, monkeypatch):
    batch = load("asr/asr_all.py")
    adapter = load("asr/tools/asr-qwen3-mlx.py")
    corpus = tmp_path / "corpus"; corpus.mkdir()
    wav = corpus / "en_same.wav"; wav.write_bytes(b"replacement input")
    (corpus / "en_same.lab").write_text("New reference.")
    model = next(m for m in batch.MODELS if m["name"] == "qwen3-asr-17b-mlx-8bit")
    out = tmp_path / "out"; out.mkdir()
    transcript = out / batch.output_filename(model, "en", wav.stem)
    transcript.write_text("Old hypothesis.")
    # The actual standalone adapter exits before importing mlx_audio.
    adapter.transcribe(str(wav), text_out=str(transcript))
    calls = []
    def transport(cmd, **kwargs):
        calls.append(cmd)
        adapter.transcribe(str(wav), text_out=str(transcript))
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(batch.subprocess, "run", transport)
    monkeypatch.setattr(batch, "save_dashboard", lambda out: out / "index.html")
    batch.main(str(corpus), str(out), models=model["name"], langs="en", prep=False)
    row = json.loads((out / "report.json").read_text())["results"][batch._result_key(model["name"], "en")]["clips"][0]
    assert not calls and not row["success"] and "hypothesis" not in row
    assert transcript.read_text() == "Old hypothesis."
