# this_file: tests/test_ported_audio_migration.py
"""Portable resources retain pins and require caller-provided private inputs."""
import ast
import runpy
import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1] / "src/vexy_audioflexy/tools/fx"


def test_all_resources_compile_and_have_no_machine_paths():
    paths = list(ROOT.rglob("*.py"))
    assert len(paths) == 32
    for path in paths:
        source = path.read_text()
        ast.parse(source)
        assert "/Users/" not in source and "/Volumes/" not in source
        assert "this_file:" in source


def test_runtime_roots_follow_environment_not_installed_resources(tmp_path, monkeypatch):
    monkeypatch.setenv("VAFX_AUDIO_RUNTIME_ROOT", str(tmp_path / "runtime"))
    monkeypatch.setenv("VAFX_AUDIO_CACHE_ROOT", str(tmp_path / "cache"))
    sys.path.insert(0, str(ROOT / "tools"))
    try:
        for path in ROOT.glob("tools/aud-*.py"):
            module = runpy.run_path(str(path))
            assert module["RUNTIME_ROOT"] == tmp_path / "runtime"
            assert module["CACHE_ROOT"] == tmp_path / "cache"
    finally:
        sys.path.remove(str(ROOT / "tools"))


def test_batch_requires_manifest_and_rejects_missing_source_before_prep(tmp_path):
    module = runpy.run_path(str(ROOT / "aud_all.py"))
    assert not module["SOURCES"] and not module["TARGETS"]
    manifest = tmp_path / "inputs.json"
    manifest.write_text('{"sources": {}}')
    with pytest.raises(SystemExit, match="sources and targets"):
        module["main"](str(manifest), models="audiosr-basic-f32")


def test_noisy_fixture_requires_input_and_output():
    import inspect
    module = runpy.run_path(str(ROOT / "tools/make_noisy_fixture.py"))
    for name in ("source", "output"):
        assert inspect.signature(module["main"]).parameters[name].default is inspect.Parameter.empty


def test_batch_reads_text_from_manifest_before_calling_models(tmp_path, monkeypatch):
    import json
    module = runpy.run_path(str(ROOT / "aud_all.py"))
    run = module["main"]
    source, target = tmp_path / "source.wav", tmp_path / "target.wav"
    source.touch()
    target.touch()
    manifest = tmp_path / "inputs.json"
    manifest.write_text(json.dumps({"sources": {"speech": str(source)}, "targets": {"spkb": str(target)},
                                    "source_text": "Synthetic source", "target_text": "Synthetic target"}))
    def stop_before_download(*args):
        args = run.__globals__["build_args"](args[0], "out.wav")
        assert "Synthetic source" in args and "Synthetic target" in args
        raise RuntimeError("verified before download")
    monkeypatch.setitem(run.__globals__, "prep_model", stop_before_download)
    model = next(m for m in module["MODELS"] if m.get("route") == "editing")
    with pytest.raises(RuntimeError, match="verified before download"):
        run(str(manifest), models=model["name"], outdir=str(tmp_path / "outputs"))


def test_changed_manifest_inputs_and_text_cannot_reuse_prior_output(tmp_path, monkeypatch):
    import json
    module = runpy.run_path(str(ROOT / "aud_all.py"))
    main = module["main"]
    source, target, reference = (tmp_path / name for name in ("source.wav", "target.wav", "reference.wav"))
    for path in (source, target, reference):
        path.write_bytes(b"synthetic original")
    manifest = tmp_path / "inputs.json"
    data = {"sources": {"speech": str(source)}, "targets": {"spkb": str(target)},
            "references": {"speech": str(reference)}, "source_text": "Synthetic source", "target_text": "Synthetic target"}
    calls = []
    def execute(model, output, timeout, verbose, force):
        calls.append(force)
        Path(output).write_bytes(f"synthetic run {len(calls)}".encode())
        return True, 1.0, ""
    monkeypatch.setitem(main.__globals__, "run_model", execute)
    monkeypatch.setitem(main.__globals__, "audio_stats", lambda path: {"duration_s": 1.0, "sample_rate": 16000, "rms": .1})
    monkeypatch.setitem(main.__globals__, "print_summary", lambda *args: None)
    monkeypatch.setitem(main.__globals__, "save_report_html", lambda *args: tmp_path / "report.html")
    monkeypatch.setitem(main.__globals__, "build_dashboard", lambda *args: None)
    model = next(m for m in module["MODELS"] if m.get("route") == "editing")
    def run():
        manifest.write_text(json.dumps(data))
        main(str(manifest), models=model["name"], outdir=str(tmp_path / "outputs"), prep=False, metrics=False)
    run()
    run()
    assert calls == [False], "Identical request must reuse the original output"
    source.write_bytes(b"different source with same key")
    run()
    assert calls == [False, True], "Changed source bytes must rerun even with same logical key"
    data["target_text"] = "Different synthetic request"
    run()
    assert calls == [False, True, True], "Changed conditioned text must rerun"
    target.write_bytes(b"different target")
    reference.write_bytes(b"different reference")
    run()
    assert calls == [False, True, True, True], "Changed target/reference bytes must rerun"


def test_saved_model_root_and_specific_override_reach_isolated_script(tmp_path, monkeypatch):
    import subprocess
    script = ROOT / "tools/aud-sherpaonnx.py"
    config = tmp_path / "config/vexy-audioflexy"
    config.mkdir(parents=True)
    models = tmp_path / "weights"
    (config / "config.toml").write_text(f'models_dir = "{models}"\n')
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    # platformdirs uses XDG on Linux and Library/Application Support on macOS.
    mac_config = tmp_path / "home/Library/Application Support/vexy-audioflexy"
    mac_config.mkdir(parents=True)
    (mac_config / "config.toml").write_text(f'models_dir = "{models}"\n')
    monkeypatch.delenv("VAFX_MODELS_DIR", raising=False)
    monkeypatch.delenv("SHERPA_ONNX_MODELS_ROOT", raising=False)
    code = 'import runpy; print(runpy.run_path(__import__("sys").argv[1])["_root"]())'
    result = subprocess.run([sys.executable, "-c", code, str(script)], check=True, text=True, capture_output=True)
    assert Path(result.stdout.strip()) == models / "sherpa-onnx"
    monkeypatch.setenv("SHERPA_ONNX_MODELS_ROOT", str(tmp_path / "specific"))
    result = subprocess.run([sys.executable, "-c", code, str(script)], check=True, text=True, capture_output=True)
    assert Path(result.stdout.strip()) == tmp_path / "specific"


def test_standalone_dashboard_has_metadata_and_rebuilds_synthetic_report(tmp_path):
    import subprocess
    output = tmp_path / "out"
    output.mkdir()
    (output / "report.json").write_text('{"results": {}, "summary": {}}')
    script = ROOT / "tools/build_dashboard.py"
    assert "# /// script" in script.read_text()
    result = subprocess.run([sys.executable, str(script)], cwd=tmp_path, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert "audio enhancement" in (output / "index.html").read_text()


def test_registered_model_identity_is_resolved_and_ambiguity_rejected(tmp_path, monkeypatch):
    helper = runpy.run_path(str(ROOT / "tools/_audioflexy_paths.py"))["model_path"]
    monkeypatch.setitem(helper.__globals__, "user_data_path", lambda *args, **kwargs: tmp_path)
    monkeypatch.setitem(helper.__globals__, "user_config_path", lambda *args, **kwargs: tmp_path)
    registry = tmp_path / "models.toml"
    path = tmp_path / "registered-sherpa"
    registry.write_text(f'[[models]]\nidentity = "sherpa-onnx"\nprovider = "other"\npath = "{path}"\n')
    assert helper("sherpa-onnx") == path
    with registry.open("a") as stream:
        stream.write(f'[[models]]\nidentity = "sherpa-onnx"\nprovider = "other"\npath = "{tmp_path / "other"}"\n')
    with pytest.raises(ValueError, match="Ambiguous"):
        helper("sherpa-onnx")
