# this_file: src/vexy_audioflexy/tools/fx/tools/_audioflexy_paths.py
"""Resolve configured model storage without importing the CLI environment."""
import os
import tomllib
from pathlib import Path

from platformdirs import user_config_path, user_data_path


def model_path(identity: str) -> Path:
    """Use an exact registered identity, or its folder under configured models_dir."""
    data = user_data_path("vexy-audioflexy", appauthor=False)
    config = user_config_path("vexy-audioflexy", appauthor=False) / "config.toml"
    saved = tomllib.loads(config.read_text()) if config.exists() else {}
    value = os.environ.get("VAFX_MODELS_DIR", saved.get("models_dir", str(data / "models")))
    registry = data / "models.toml"
    records = tomllib.loads(registry.read_text()).get("models", []) if registry.exists() else []
    matches = [row["path"] for row in records if row.get("identity") == identity]
    if len(set(matches)) > 1:
        raise ValueError(f"Ambiguous registered model {identity}; set its explicit adapter override")
    value = matches[0] if matches else value
    if not isinstance(value, str) or not value.strip() or not Path(value).expanduser().is_absolute():
        raise ValueError("Model storage must be a non-empty absolute path")
    path = Path(value).expanduser()
    return path if matches else path / identity
