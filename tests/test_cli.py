# this_file: tests/test_cli.py
"""Exercise the real CLI process and installed metadata."""

import json
import subprocess
import sys
from importlib.metadata import version

import pytest
import yaml

from vexy_audioflexy.cli import version as cli_version


def test_version_matches_installed_metadata() -> None:
    assert cli_version() == version("vexy-audioflexy")


def test_cli_version_exits_successfully() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "vexy_audioflexy.cli", "version"],
        capture_output=True, text=True, check=True,
    )
    assert result.stdout.strip() == cli_version()


def test_cli_help_lists_version() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "vexy_audioflexy.cli", "--", "--help"],
        capture_output=True, text=True, check=True,
    )
    assert "version" in result.stdout + result.stderr


def test_cli_unknown_command_fails() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "vexy_audioflexy.cli", "unknown-command"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode != 0


def test_installed_models_configure_and_override(tmp_path) -> None:
    """Use installed entry point with isolated model configuration and home."""
    from pathlib import Path
    import os
    import tomllib

    executable = Path(sys.executable).parent / "vafx"
    command = [
        str(executable), "models",
        f"--config_dir={tmp_path / 'config'}",
        f"--data_dir={tmp_path / 'data'}",
        f"--home={tmp_path / 'home'}",
    ]
    env = {**os.environ, "HF_HUB_CACHE": str(tmp_path / "hf")}
    saved = tmp_path / "saved-models"
    subprocess.run(command + ["configure", str(saved)], env=env, check=True,
                   capture_output=True, text=True)
    with (tmp_path / "config/config.toml").open("rb") as stream:
        assert tomllib.load(stream)["models_dir"] == str(saved)
    override = tmp_path / "override-models"
    result = subprocess.run(command + [f"--models_dir={override}", "config"],
                            env=env, check=True, capture_output=True, text=True)
    assert str(override) in result.stdout
    assert str(saved) not in result.stdout
    assert not (tmp_path / "data").exists()


@pytest.mark.parametrize("json_flag", [[], ["--json"]])
@pytest.mark.parametrize("command", ["list", "config", "discover"])
def test_models_when_output_selected_then_complete_parseable_document(tmp_path, monkeypatch, command, json_flag):
    from vexy_audioflexy.models import Models

    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hf"))
    model = tmp_path / "synthetic żółw.bin"
    model.write_bytes(b"synthetic fixture, not model weights")
    settings = {name: str(tmp_path / name) for name in ("config_dir", "data_dir", "home")}
    models = Models(**settings)
    models.register("synthetic/żółw", "local", str(model))
    import tomli_w

    records = models.list()
    records[0]["metadata"] = {"enabled": True, "labels": ["yes", "null", "001", "żółw"]}
    with (tmp_path / "data_dir" / "models.toml").open("wb") as stream:
        tomli_w.dump({"models": records}, stream)
    expected = getattr(models, command)()
    result = subprocess.run(
        [sys.executable, "-m", "vexy_audioflexy.cli", "models",
         *[f"--{key}={value}" for key, value in settings.items()], command, *json_flag],
        capture_output=True, text=True, check=True,
    )
    parse = json.loads if json_flag else yaml.safe_load
    assert parse(result.stdout) == expected
    if command == "list":
        assert "żółw" in result.stdout
        if not json_flag:
            assert result.stdout.startswith("- identity: synthetic/żółw\n")


@pytest.mark.parametrize("json_flag", [[], ["--json"]])
def test_tools_when_output_selected_then_complete_parseable_document(json_flag):
    from vexy_audioflexy.dispatch import Tools

    result = subprocess.run(
        [sys.executable, "-m", "vexy_audioflexy.cli", "tools", "list", "asr", *json_flag],
        capture_output=True, text=True, check=True,
    )
    parse = json.loads if json_flag else yaml.safe_load
    assert parse(result.stdout) == Tools().list("asr")


@pytest.mark.parametrize("prefix", [[], ["--json"]])
def test_models_when_registry_empty_then_explicit_empty_list(tmp_path, prefix):
    result = subprocess.run(
        [sys.executable, "-m", "vexy_audioflexy.cli", *prefix, "models",
         f"--data_dir={tmp_path}", "list"], capture_output=True, text=True, check=True,
    )
    assert result.stdout == "[]\n"


def test_version_when_json_requested_then_json_string():
    result = subprocess.run(
        [sys.executable, "-m", "vexy_audioflexy.cli", "version", "--json"],
        capture_output=True, text=True, check=True,
    )
    assert json.loads(result.stdout) == cli_version()


@pytest.mark.parametrize("args", [[], ["--json"], ["models", "--json"],
                                  ["--json", "--", "--help"]])
def test_cli_when_output_switch_used_for_help_then_help_still_works(args):
    result = subprocess.run(
        [sys.executable, "-m", "vexy_audioflexy.cli", *args],
        capture_output=True, text=True, check=True,
    )
    assert "SYNOPSIS" in result.stdout + result.stderr


def test_installed_models_discover_is_read_only(tmp_path) -> None:
    from pathlib import Path
    import os

    models = tmp_path / "models"
    models.mkdir()
    (models / "synthetic.bin").write_bytes(b"synthetic fixture, not model weights")
    result = subprocess.run(
        [str(Path(sys.executable).parent / "vafx"), "models",
         f"--config_dir={tmp_path / 'config'}", f"--data_dir={tmp_path / 'data'}",
         f"--home={tmp_path / 'home'}", f"--models_dir={models}", "discover"],
        env={**os.environ, "HF_HUB_CACHE": str(tmp_path / "hf")},
        check=True, capture_output=True, text=True,
    )
    assert "synthetic.bin" in result.stdout
    assert not (tmp_path / "config").exists()
    assert not (tmp_path / "data").exists()
