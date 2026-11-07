# this_file: src/vexy_audioflexy/tools/asr/tools/_audioflexy_paths.py
"""Portable storage locations for independently executed ASR scripts."""

import os
import shutil
import tomllib
from pathlib import Path

from platformdirs import user_cache_path, user_config_path, user_data_path


def data_root() -> Path:
    """Match the public CLI's platformdirs data location."""
    return user_data_path("vexy-audioflexy", appauthor=False)


def runtime_root(name: str) -> Path:
    """Native source/build checkouts are separate from model weights."""
    root = os.environ.get("VAFX_RUNTIME_DIR")
    base = Path(root).expanduser() if root else user_cache_path("vexy-audioflexy", appauthor=False) / "runtimes"
    if not base.is_absolute():
        raise ValueError("VAFX_RUNTIME_DIR must be an absolute path")
    return base / name


def models_root() -> Path:
    """Use an explicit environment override, CLI TOML configuration, or app data."""
    override = os.environ.get("VAFX_MODELS_DIR")
    config = user_config_path("vexy-audioflexy", appauthor=False) / "config.toml"
    saved = {}
    if config.is_file():
        with config.open("rb") as stream:
            saved = tomllib.load(stream)
    value = override if override is not None else saved.get("models_dir", str(data_root() / "models"))
    if not isinstance(value, str) or not value.strip():
        raise ValueError("models_dir must be a non-empty absolute path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError("models_dir must be an absolute path")
    return path


def stage_native_host(source: Path, destination: Path) -> None:
    """Copy only distributed host sources before Swift writes its build directory."""
    paths = [source / name for name in ("Package.swift", "THIRD_PARTY_LICENSE.txt")]
    paths += [p for name in ("Sources", "Tests") for p in (source / name).rglob("*") if p.is_file()]
    for path in paths:
        target = destination / path.relative_to(source)
        if target.exists() and target.read_bytes() != path.read_bytes():
            raise FileExistsError(f"Native host differs: {target}; choose a fresh runtime directory")
    for path in paths:
        target = destination / path.relative_to(source)
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
