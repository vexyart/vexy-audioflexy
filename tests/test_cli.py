# this_file: tests/test_cli.py
"""Exercise the real CLI process and installed metadata."""

from importlib.metadata import version
import subprocess
import sys

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
