# this_file: tests/test_ported_voice.py
"""Voice resource closure and isolated runtime routing, without inference."""
import importlib.util
from pathlib import Path
import sys
import subprocess
import pytest

ROOT = Path(__file__).resolve().parents[1] / 'src/vexy_audioflexy/tools/tts'
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def isolated_voice_storage(tmp_path, monkeypatch):
    import _paths
    monkeypatch.setattr(_paths, '_app_path', lambda kind: tmp_path / kind)
    for variable in ('HOME', 'USERPROFILE', 'XDG_CONFIG_HOME', 'XDG_DATA_HOME', 'APPDATA', 'LOCALAPPDATA'):
        monkeypatch.setenv(variable, str(tmp_path / variable))
    monkeypatch.setenv('VAFX_MODELS_DIR', str(tmp_path / 'models'))
    monkeypatch.setenv('VAFX_VOICE_ROOT', str(tmp_path / 'voice'))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runtime_paths_respect_explicit_user_root(tmp_path, monkeypatch):
    helper = load('voice_paths', ROOT / 'tools/_paths.py')
    monkeypatch.setenv('VAFX_VOICE_ROOT', str(tmp_path))
    assert helper.runtime_path('scratch') == tmp_path / 'scratch'


def test_batch_has_no_private_reference_defaults_and_keeps_script_closure():
    batch = load('voice_batch', ROOT / 'tts_all.py')
    assert batch.REFS == {'en': None, 'pl': None}
    for model in batch.MODELS:
        assert (ROOT / model['script']).is_file(), model['script']


def test_batch_forwards_nonzero_and_uses_uv_script(tmp_path, monkeypatch):
    batch = load('voice_batch_failure', ROOT / 'tts_all.py')
    calls = []
    def fail(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 23, '', 'synthetic failure')
    monkeypatch.setattr(batch.subprocess, 'run', fail)
    ok, _, message = batch.run_model({'script': 'tools/tts-qwenttscpp.py', 'args_fn': lambda *args: ['hello world', '--output', str(tmp_path / 'clip.wav')]}, 'en', 'hello world', '', str(tmp_path / 'clip.wav'), 3, False)
    assert not ok and 'synthetic failure' in message
    assert calls[0][:3] == ['uv', 'run', '--script']
    assert 'hello world' in calls[0]


def test_batch_rejects_invalid_caller_design_prompts_before_dispatch(tmp_path):
    import pytest
    batch = load('voice_batch_prompts', ROOT / 'tts_all.py')
    prompts = tmp_path / 'prompts.json'
    prompts.write_text('["valid", 3]')
    with pytest.raises(SystemExit, match='Design prompts'):
        batch.main(design_prompts_file=str(prompts))


def test_smallgguf_prepares_distinct_files_in_configured_store(tmp_path, monkeypatch):
    import hashlib
    import types
    adapter = load('voice_small_prep', ROOT / 'tools/tts-smallgguf.py')
    monkeypatch.setenv('VAFX_MODELS_DIR', str(tmp_path / 'models'))
    calls = []
    def download(repo, filename, **kwargs):
        calls.append(kwargs)
        result = tmp_path / Path(filename).name
        result.write_bytes(filename.encode())
        return str(result)
    monkeypatch.setitem(sys.modules, 'huggingface_hub', types.SimpleNamespace(hf_hub_download=download))
    packages = {'first': ('kitten', 'one/model-one.gguf', hashlib.sha256(b'one/model-one.gguf').hexdigest()), 'second': ('kokoro', 'two/model-two.gguf', hashlib.sha256(b'two/model-two.gguf').hexdigest())}
    monkeypatch.setattr(adapter, 'PACKAGES', packages)
    one = Path(adapter.prep('first'))
    two = Path(adapter.prep('second'))
    assert one != two and one.read_bytes() == b'one/model-one.gguf'
    assert two.read_bytes() == b'two/model-two.gguf'
    assert all('cache_dir' not in call and 'local_dir' not in call for call in calls)
    assert all(path.is_relative_to(tmp_path / 'models') for path in (one, two))


def test_prune_report_imports_batch_in_fresh_process(tmp_path):
    import json
    (tmp_path / 'report.json').write_text(json.dumps({'results': {}}))
    result = subprocess.run([sys.executable, str(ROOT / 'tools/prune-report.py'), '--outdir', str(tmp_path)], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_batch_relative_output_uses_callers_directory(tmp_path, monkeypatch):
    batch = load('voice_batch_relative', ROOT / 'tts_all.py')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(batch, 'resolve_models', lambda _: [])
    batch.main(outdir='clips', prep_only=True)
    assert (tmp_path / 'clips').is_dir()
    assert not (ROOT / 'clips').exists()


def test_adapter_model_lookup_honors_config_and_registry(tmp_path, monkeypatch):
    import _paths
    config = tmp_path / 'config'
    data = tmp_path / 'data'
    config.mkdir(); data.mkdir()
    models = tmp_path / 'consolidated'
    (config / 'config.toml').write_text(f'models_dir = "{models}"\n')
    monkeypatch.delenv('VAFX_MODELS_DIR', raising=False)
    monkeypatch.setattr(_paths, '_app_path', lambda kind: config if kind == 'config' else data)
    sherpa = load('voice_sherpa_paths', ROOT / 'tools/tts-sherpaonnx.py')
    assert sherpa._root() == models / 'sherpa-onnx'
    registered = tmp_path / 'provider-snapshot'
    (data / 'models.toml').write_text(f'[[models]]\nidentity = "sherpa-onnx"\nprovider = "local"\npath = "{registered}"\n')
    assert sherpa._root() == registered
    monkeypatch.setenv('SHERPA_ONNX_MODELS_ROOT', str(tmp_path / 'explicit'))
    assert sherpa._root() == tmp_path / 'explicit'


def test_breeze_downloads_provider_snapshot_and_links_expected_layout(tmp_path, monkeypatch):
    import types
    adapter = load('voice_breeze_prep', ROOT / 'tools/tts-breeze-mlx.py')
    engine = tmp_path / 'engine'; engine.mkdir(); (engine / 'infer.py').touch()
    snapshot = tmp_path / 'hf-snapshot'; snapshot.mkdir()
    (snapshot / 'model.safetensors.index.json').write_text('{}')
    calls = []
    def download(repo, **kwargs):
        calls.append(kwargs)
        return str(snapshot)
    monkeypatch.setitem(sys.modules, 'huggingface_hub', types.SimpleNamespace(snapshot_download=download))
    monkeypatch.setattr(adapter, 'ENGINE_ROOT', engine)
    adapter.prep()
    assert calls == [{}]
    _, target = adapter._variant('bf16')
    assert target.is_symlink() and target.resolve() == snapshot


def test_indextts_prep_executes_provider_downloads_and_compatibility_links(tmp_path, monkeypatch):
    import types
    adapter = load('voice_index_prep', ROOT / 'tools/tts-indextts2-mlx.py')
    engine = tmp_path / 'engine'; engine.mkdir(); (engine / 'pyproject.toml').touch()
    snapshot = tmp_path / 'hf-snapshot'; snapshot.mkdir(); (snapshot / 'weights.safetensors').write_bytes(b'weights')
    camp = tmp_path / 'campplus.bin'; camp.write_bytes(b'encoder')
    calls = []
    def snapshot_download(repo, **kwargs):
        calls.append(kwargs)
        return str(snapshot)
    def hf_hub_download(*args, **kwargs):
        calls.append(kwargs)
        return str(camp)
    monkeypatch.setitem(sys.modules, 'huggingface_hub', types.SimpleNamespace(snapshot_download=snapshot_download, hf_hub_download=hf_hub_download))
    monkeypatch.setattr(adapter, 'ENGINE_ROOT', engine)
    model_dir = tmp_path / 'compatibility'
    monkeypatch.setattr(adapter, 'MODEL_DIR', model_dir)
    def run(cmd, **kwargs):
        exec(cmd[-1], {})
        return subprocess.CompletedProcess(cmd, 0)
    monkeypatch.setattr(adapter.subprocess, 'run', run)
    adapter.prep()
    assert all('local_dir' not in call and 'cache_dir' not in call for call in calls)
    assert (model_dir / 'weights.safetensors').is_symlink()
    assert (model_dir / 'hf_cache' / adapter.CAMPPLUS_FILE).resolve() == camp
    assert not (snapshot / 'hf_cache').exists()


def test_audiocpp_default_model_root_uses_saved_configuration(tmp_path, monkeypatch):
    import _paths
    config = tmp_path / 'config'
    config.mkdir()
    store = tmp_path / 'consolidated'
    (config / 'config.toml').write_text(f'models_dir = "{store}"\n')
    monkeypatch.delenv('VAFX_MODELS_DIR', raising=False)
    monkeypatch.delenv('AUDIOCPP_MODELS_ROOT', raising=False)
    adapter = load('voice_audiocpp_configured', ROOT / 'tools/tts-audiocpp.py')
    assert adapter.MODELS_ROOT == store
    monkeypatch.setattr(adapter, '_resolve', lambda package: {'target_directory': 'synthetic-package', 'family': 'synthetic'})
    assert adapter._model_path('synthetic') == store / 'synthetic-package'


def test_windows_paths_match_platformdirs_local_defaults(tmp_path, monkeypatch):
    import types
    import platformdirs.windows
    helper = load('voice_windows_paths', ROOT / 'tools/_paths.py')
    local = tmp_path / 'Local'
    roaming = tmp_path / 'Roaming'
    monkeypatch.setenv('LOCALAPPDATA', str(local))
    monkeypatch.setenv('APPDATA', str(roaming))
    monkeypatch.setattr(helper, 'sys', types.SimpleNamespace(platform='win32'))
    monkeypatch.setattr(platformdirs.windows, 'get_win_folder', lambda name: str(local if name == 'CSIDL_LOCAL_APPDATA' else roaming))
    expected = platformdirs.windows.Windows('vexy-audioflexy', appauthor=False)
    assert helper._app_path('config') == Path(expected.user_config_dir)
    assert helper._app_path('data') == Path(expected.user_data_dir)


def test_all_voice_main_entries_are_classified_runnable():
    import ast
    dispatch = load('voice_dispatch_classification', ROOT.parents[1] / 'dispatch.py')
    for path in ROOT.rglob('*.py'):
        tree = ast.parse(path.read_text())
        has_main = any(isinstance(node, ast.If) and isinstance(node.test, ast.Compare) and isinstance(node.test.left, ast.Name) and node.test.left.id == '__name__' and any(isinstance(value, ast.Constant) and value.value == '__main__' for value in node.test.comparators) for node in tree.body)
        if has_main:
            assert dispatch._kind(path) == 'runnable', path.name


@pytest.mark.parametrize('name', ['build_dashboard.py', 'build_report.py', 'prune-report.py'])
def test_report_entrypoints_help_in_fresh_dependency_qualified_process(name, tmp_path):
    import os
    result = subprocess.run([sys.executable, str(ROOT / 'tools' / name), '--help'], cwd=tmp_path, env={**os.environ, 'UV_OFFLINE': '1'}, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert 'usage:' in result.stdout.lower()
    assert not list(tmp_path.rglob('*.html'))


def test_supertonic_coreml_provider_assets_and_compile_stay_separate(tmp_path, monkeypatch):
    import types
    adapter = load('voice_supertonic_coreml_provider', ROOT / 'tools/tts-supertonic-3-coreml.py')
    snapshot = tmp_path / 'provider' / 'snapshot-commit'; snapshot.mkdir(parents=True)
    package = snapshot / 'DurationPredictor.mlpackage'; package.mkdir()
    (package / 'weights.bin').write_bytes(b'provider weights')
    (snapshot / 'tts.json').write_text('{}')
    styles = snapshot / 'voice_styles'; styles.mkdir(); (styles / 'M1.json').write_text('{}')
    fallback = tmp_path / 'provider' / 'fallback.json'; fallback.write_text('{}')
    calls = []
    def snapshot_download(**kwargs):
        calls.append(kwargs)
        return str(snapshot)
    def hf_hub_download(**kwargs):
        calls.append(kwargs)
        if kwargs['repo_id'] != 'Supertone/supertonic-3':
            raise FileNotFoundError('conversion omits F1')
        return str(fallback)
    monkeypatch.setitem(sys.modules, 'huggingface_hub', types.SimpleNamespace(snapshot_download=snapshot_download, hf_hub_download=hf_hub_download))
    original = {str(p.relative_to(snapshot)): p.read_bytes() for p in snapshot.rglob('*') if p.is_file()}
    bundle = adapter._download_bundle('synthetic/conversion')
    assert not bundle.is_relative_to(snapshot)
    assert (bundle / 'DurationPredictor.mlpackage').is_symlink()
    assert (bundle / 'voice_styles').is_dir() and not (bundle / 'voice_styles').is_symlink()
    assert adapter._download_voice_style('M1', bundle).resolve() == styles / 'M1.json'
    assert adapter._download_voice_style('F1', bundle, 'synthetic/conversion').resolve() == fallback
    compile_calls = []
    def compile_model(source):
        destination = tmp_path / "native-compiled-temporary.mlmodelc"
        assert Path(source).resolve() == package
        destination.mkdir(); (destination / "compiled.bin").write_bytes(b"compiled")
        compile_calls.append(source)
        return str(destination)
    def compiled_model(path, compute_units):
        assert not Path(path).is_relative_to(snapshot)
        assert Path(path).is_relative_to(tmp_path / "voice")
        assert Path(path).is_dir()
        return (path, compute_units)
    monkeypatch.setitem(sys.modules, 'coremltools', types.SimpleNamespace(models=types.SimpleNamespace(utils=types.SimpleNamespace(compile_model=compile_model), CompiledMLModel=compiled_model)))
    first = adapter._load_coreml_model(bundle / package.name, 'CPU')
    assert adapter._load_coreml_model(bundle / package.name, 'CPU') == first
    assert len(compile_calls) == 1
    assert all('local_dir' not in call and 'cache_dir' not in call for call in calls)
    assert {str(p.relative_to(snapshot)): p.read_bytes() for p in snapshot.rglob('*') if p.is_file()} == original
    assert not (snapshot / 'voice_styles/F1.json').exists()


def test_supertonic_compatibility_link_preserves_conflicting_file(tmp_path):
    adapter = load('voice_supertonic_conflict', ROOT / 'tools/tts-supertonic-3-coreml.py')
    source = tmp_path / 'provider.json'; source.write_text('provider')
    target = tmp_path / 'owner.json'; target.write_text('owner')
    with pytest.raises(FileExistsError, match='conflicts'):
        adapter._link_cached(source, target)
    assert target.read_text() == 'owner'


@pytest.mark.parametrize('name', ['tts-magpie-gguf.py', 'tts-soniqo-voxcpm2-onnx.py'])
def test_live_updaters_preserve_provider_and_recovery_on_cached_revision_conflict(name, tmp_path, monkeypatch):
    import types
    adapter = load('voice_updater_' + name.replace('-', '_'), ROOT / 'tools' / name)
    provider = tmp_path / 'provider'; provider.mkdir()
    old_blob = provider / 'old-blob'; old_blob.write_bytes(b'old canonical')
    new_blob = provider / 'new-blob'; new_blob.write_bytes(b'new cached revision')
    recovery = tmp_path / 'recovery.bin'; recovery.write_bytes(old_blob.read_bytes())
    old = provider / 'old-snapshot'; old.mkdir(); (old / 'model.bin').symlink_to(old_blob)
    new = provider / 'new-snapshot'; new.mkdir(); (new / 'model.bin').symlink_to(new_blob)
    (new / 'additional.json').write_text('{}')
    engine = tmp_path / 'engine'; engine.mkdir()
    if name == 'tts-magpie-gguf.py':
        binary = engine / 'magpie-tts'; binary.touch()
        monkeypatch.setattr(adapter, '_find_binary', lambda _: binary)
        target = engine / 'weights'
    else:
        target = engine / 'model'
        monkeypatch.setenv(adapter.ENV_MODEL_DIR, str(target))
    current = [old]
    calls = []
    def snapshot_download(*args, **kwargs):
        assert 'local_dir' not in kwargs and 'cache_dir' not in kwargs
        calls.append(kwargs)
        return str(current[0])
    monkeypatch.setitem(sys.modules, 'huggingface_hub', types.SimpleNamespace(snapshot_download=snapshot_download))
    adapter.prep()
    assert (target / 'model.bin').is_symlink()
    assert (target / 'model.bin').resolve() == old_blob
    current[0] = new
    with pytest.raises(FileExistsError, match='conflicts'):
        adapter.prep(force=True)
    assert old_blob.read_bytes() == recovery.read_bytes() == b'old canonical'
    assert new_blob.read_bytes() == b'new cached revision'
    assert (target / 'model.bin').resolve() == old_blob
    assert not (target / 'additional.json').exists(), 'Conflict must be checked before any new binding'
    assert calls[-1]['force_download'] is True


@pytest.mark.parametrize('name', ['tts-magpie-gguf.py', 'tts-soniqo-voxcpm2-onnx.py'])
def test_updater_standalone_help_does_not_prepare_weights(name, tmp_path):
    import os
    result = subprocess.run([sys.executable, str(ROOT / 'tools' / name), '--help'], cwd=tmp_path, env={**os.environ, 'HF_HUB_OFFLINE': '1', 'UV_OFFLINE': '1'}, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert 'prep' in result.stdout + result.stderr
    assert not list(tmp_path.rglob('*.bin'))


def test_snapshot_binding_rejects_overlap_and_provider_store_aliases(tmp_path, monkeypatch):
    import _paths
    hub = tmp_path / 'hub'
    snapshot = hub / 'models--synthetic--repo/snapshots/current'
    snapshot.mkdir(parents=True)
    (snapshot / 'weights.bin').write_bytes(b'provider')
    peer = snapshot.parent / 'old'; peer.mkdir(); (peer / 'old.bin').write_bytes(b'recovery')
    monkeypatch.setenv('HF_HUB_CACHE', str(hub))
    alias = tmp_path / 'alias'; alias.symlink_to(peer, target_is_directory=True)
    before = sorted((str(p.relative_to(hub)), p.read_bytes() if p.is_file() else None) for p in hub.rglob('*'))
    for target in [snapshot / 'compat', snapshot.parent, peer / 'compat', alias / 'compat']:
        with pytest.raises(ValueError):
            _paths.bind_snapshot(snapshot, target)
        assert sorted((str(p.relative_to(hub)), p.read_bytes() if p.is_file() else None) for p in hub.rglob('*')) == before
    _paths.bind_snapshot(snapshot, snapshot)
    exact = tmp_path / 'exact'; exact.symlink_to(snapshot, target_is_directory=True)
    _paths.bind_snapshot(snapshot, exact)
    external = tmp_path / 'managed-view'
    _paths.bind_snapshot(snapshot, external)
    assert (external / 'weights.bin').resolve() == snapshot / 'weights.bin'
    assert sorted((str(p.relative_to(hub)), p.read_bytes() if p.is_file() else None) for p in hub.rglob('*')) == before
