# this_file: src/vexy_audioflexy/models.py
"""Local model configuration and non-downloading provider discovery."""

from __future__ import annotations

import os
import tempfile
import tomllib
from pathlib import Path

import platformdirs
import tomli_w


def _read(path: Path) -> dict:
    if not path.exists():
        if path.is_symlink():
            raise ValueError(f"Broken TOML symlink: {path}")
        return {}
    with path.open("rb") as stream:
        return tomllib.load(stream)


def _write(path: Path, value: dict) -> None:
    """Replace atomically, retaining an existing user-managed symlink."""
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            tomli_w.dump(value, stream)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _path(value: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Model paths must be non-empty strings")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError("Model paths must be absolute")
    return path


class Models:
    """Fire-compatible model commands; discovery never loads or downloads weights."""

    def __init__(self, models_dir=None, config_dir=None, data_dir=None, home=None):
        self._data = (
            Path(data_dir)
            if data_dir
            else platformdirs.user_data_path("vexy-audioflexy", appauthor=False)
        )
        config = (
            Path(config_dir)
            if config_dir
            else platformdirs.user_config_path("vexy-audioflexy", appauthor=False)
        )
        self._config = config / "config.toml"
        self._registry = self._data / "models.toml"
        self._home = Path(home) if home else Path.home()
        self._override = str(_path(str(models_dir))) if models_dir is not None else None

    def config(self) -> dict:
        """Show effective locations: CLI override, saved setting, then app data."""
        saved = _read(self._config)
        if "models_dir" in saved:
            saved["models_dir"] = str(_path(saved["models_dir"]))
        directory = self._override or saved.get(
            "models_dir", str(self._data / "models")
        )
        return {
            "models_dir": directory,
            "config_file": str(self._config),
            "registry_file": str(self._registry),
        }

    def configure(self, models_dir: str) -> dict:
        """Save a models folder while retaining unrelated TOML settings."""
        self.config()
        saved = _read(self._config)
        saved["models_dir"] = str(_path(models_dir))
        _write(self._config, saved)
        return self.config()

    def list(self) -> list[dict]:
        """Read registered locations, including records whose files are missing."""
        records = _read(self._registry).get("models", [])
        if not isinstance(records, list):
            raise ValueError("Registry models must be an array of tables")  # noqa: TRY004
        for record in records:
            if not isinstance(record, dict) or any(
                not isinstance(record.get(key), str) or not record[key].strip()
                for key in ("identity", "provider", "path")
            ):
                raise ValueError("Each model needs identity, provider and path strings")
            _path(record["path"])
        return records

    def register(self, identity: str, provider: str, path: str) -> list[dict]:
        """Add a location; retain same-identity records at other locations."""
        if not all(isinstance(v, str) and v.strip() for v in (identity, provider)):
            raise ValueError("Model identity and provider must be non-empty strings")
        record = {"identity": identity, "provider": provider, "path": str(_path(path))}
        records = self.list()
        if record not in records:
            records.append(record)
            saved = _read(self._registry)
            saved["models"] = records
            _write(self._registry, saved)
        return records

    def download_destination(self, provider: str) -> str:
        """Return the preferred store root; perform no download or directory creation."""
        if provider == "huggingface":
            hf_home = os.environ.get("HF_HOME") or str(
                Path(os.environ.get("XDG_CACHE_HOME", str(self._home / ".cache")))
                / "huggingface"
            )
            return str(
                _path(
                    os.environ.get("HF_HUB_CACHE")
                    or os.environ.get("HUGGINGFACE_HUB_CACHE")
                    or str(Path(hf_home) / "hub")
                )
            )
        if provider == "lmstudio":
            pointer = self._home / ".lmstudio-home-pointer"
            base = (
                _path(pointer.read_text().strip())
                if pointer.exists()
                else self._home / ".lmstudio"
            )
            return str(base / "models")
        return self.config()["models_dir"]

    def discover(self, save: bool = False) -> list[dict]:
        """Find provider snapshots/model folders and direct app model entries."""
        records = []
        hf = Path(self.download_destination("huggingface"))
        for snapshot in sorted(hf.glob("models--*/snapshots/*")):
            if snapshot.is_dir():
                identity = snapshot.parent.parent.name.removeprefix("models--").replace(
                    "--", "/"
                )
                records.append(
                    {
                        "identity": identity,
                        "provider": "huggingface",
                        "path": str(snapshot),
                    }
                )
        lm = Path(self.download_destination("lmstudio"))
        for model in sorted(lm.glob("*/*")):
            if model.is_dir():
                records.append(
                    {
                        "identity": str(model.relative_to(lm)),
                        "provider": "lmstudio",
                        "path": str(model),
                    }
                )
        root = Path(self.config()["models_dir"])
        if root.is_dir():
            for model in sorted(root.iterdir()):
                if model.exists():
                    records.append(
                        {
                            "identity": model.name,
                            "provider": "local",
                            "path": str(model),
                        }
                    )
        if save:
            for record in records:
                self.register(**record)
        return records
