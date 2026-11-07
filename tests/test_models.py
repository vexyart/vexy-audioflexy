# this_file: tests/test_models.py
"""Synthetic model stores; never inspect the user's actual provider caches."""

import subprocess
import sys
import tomllib

import pytest

from vexy_audioflexy.models import Models


def test_module_import_in_fresh_interpreter():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from vexy_audioflexy.models import Models; assert callable(Models.register)",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.fixture
def models(tmp_path, monkeypatch):
    for name in ("HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "XDG_CACHE_HOME"):
        monkeypatch.delenv(name, raising=False)
    return Models(
        config_dir=tmp_path / "config", data_dir=tmp_path / "data", home=tmp_path
    )


def test_config_precedence_and_preserves_other_settings(models, tmp_path):
    assert models.config()["models_dir"] == str(tmp_path / "data/models")
    assert not (tmp_path / "data").exists()
    config = tmp_path / "config/config.toml"
    config.parent.mkdir()
    config.write_text('other = "retained"\n')
    models.configure(str(tmp_path / "saved"))
    assert models.config()["models_dir"] == str(tmp_path / "saved")
    override = Models(
        models_dir=tmp_path / "override",
        config_dir=config.parent,
        data_dir=tmp_path / "data",
    )
    assert override.config()["models_dir"] == str(tmp_path / "override")
    assert tomllib.loads(config.read_text())["other"] == "retained"


def test_saved_tilde_path_is_absolute_for_config_routing_and_discovery(
    models, tmp_path, monkeypatch
):
    monkeypatch.setenv("HOME", str(tmp_path))
    config = tmp_path / "config/config.toml"
    config.parent.mkdir()
    config.write_text('models_dir = "~/synthetic-models"\n')
    root = tmp_path / "synthetic-models"
    root.mkdir()
    model = root / "example.bin"
    model.write_bytes(b"synthetic")
    assert models.config()["models_dir"] == str(root)
    assert models.download_destination("other") == str(root)
    assert models.discover() == [
        {"identity": model.name, "provider": "local", "path": str(model)}
    ]
    assert config.read_text() == 'models_dir = "~/synthetic-models"\n'


@pytest.mark.parametrize(
    "content",
    ["models_dir = [", "models_dir = 42", 'models_dir = ""', 'models_dir = "relative"'],
)
def test_invalid_config_is_not_overwritten(models, tmp_path, content):
    config = tmp_path / "config/config.toml"
    config.parent.mkdir()
    config.write_text(content)
    with pytest.raises(ValueError):
        models.config()
    with pytest.raises(ValueError):
        models.configure(str(tmp_path / "replacement"))
    assert config.read_text() == content


def test_empty_config_and_missing_pointer_discover_nothing(models, tmp_path):
    assert models.discover() == []
    assert models.list() == []
    assert models.download_destination("lmstudio") == str(tmp_path / ".lmstudio/models")
    assert models.download_destination("other") == str(tmp_path / "data/models")
    assert list(tmp_path.iterdir()) == []


def test_discovery_finds_snapshots_and_symlinks_without_writes(
    models, tmp_path, monkeypatch
):
    hub = tmp_path / "hub"
    snapshot = hub / "models--example--synthetic/snapshots/revision"
    snapshot.mkdir(parents=True)
    weight = tmp_path / "weight"
    weight.write_bytes(b"synthetic")
    (snapshot / "model.bin").symlink_to(weight)
    monkeypatch.setenv("HF_HUB_CACHE", str(hub))
    studio = tmp_path / "studio"
    model = studio / "models/example/synthetic"
    model.mkdir(parents=True)
    (tmp_path / ".lmstudio-home-pointer").write_text(str(studio) + "\n")
    local = tmp_path / "data/models"
    local.mkdir(parents=True)
    (local / "linked.bin").symlink_to(weight)
    records = models.discover()
    assert {r["provider"] for r in records} == {"huggingface", "lmstudio", "local"}
    assert records[0]["path"] == str(snapshot)
    assert not (tmp_path / "data/models.toml").exists()
    assert (snapshot / "model.bin").is_symlink()
    models.discover(save=True)
    models.discover(save=True)
    assert models.list() == records


def test_provider_cache_environment_precedence(models, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert models.download_destination("huggingface") == str(
        tmp_path / "xdg/huggingface/hub"
    )
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    assert models.download_destination("huggingface") == str(tmp_path / "hf/hub")
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(tmp_path / "legacy"))
    assert models.download_destination("huggingface") == str(tmp_path / "legacy")
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    assert models.download_destination("huggingface") == str(tmp_path / "hub")


def test_register_retains_conflicts_and_registry_symlink(models, tmp_path):
    target = tmp_path / "registry.toml"
    target.write_text('other = "retained"\n')
    registry = tmp_path / "data/models.toml"
    registry.parent.mkdir()
    registry.symlink_to(target)
    models.register("synthetic", "local", str(tmp_path / "one"))
    models.register("synthetic", "local", str(tmp_path / "two"))
    models.register("synthetic", "local", str(tmp_path / "one"))
    assert len(models.list()) == 2
    assert registry.is_symlink()
    assert tomllib.loads(target.read_text())["other"] == "retained"


@pytest.mark.parametrize(
    "content",
    [
        'models = "bad"',
        "models = [{}]",
        'models = [{identity="a", provider="b", path="relative"}]',
        "invalid = [",
    ],
)
def test_malformed_registry_prevents_registration(models, tmp_path, content):
    registry = tmp_path / "data/models.toml"
    registry.parent.mkdir()
    registry.write_text(content)
    with pytest.raises(ValueError):
        models.register("synthetic", "local", str(tmp_path / "one"))
    assert registry.read_text() == content


def test_empty_pointer_is_an_error(models, tmp_path):
    (tmp_path / ".lmstudio-home-pointer").write_text("\n")
    with pytest.raises(ValueError, match="non-empty"):
        models.discover()


def test_platformdirs_defaults(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "platformdirs.user_data_path", lambda *a, **kw: tmp_path / "data"
    )
    monkeypatch.setattr(
        "platformdirs.user_config_path", lambda *a, **kw: tmp_path / "config"
    )
    assert Models().config() == {
        "models_dir": str(tmp_path / "data/models"),
        "config_file": str(tmp_path / "config/config.toml"),
        "registry_file": str(tmp_path / "data/models.toml"),
    }
