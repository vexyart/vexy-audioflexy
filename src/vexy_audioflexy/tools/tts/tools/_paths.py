# this_file: src/vexy_audioflexy/tools/tts/tools/_paths.py
"""Writable user storage for isolated voice scripts; never mutate package resources."""
import os
import sys
from pathlib import Path


def runtime_path(relative: str) -> Path:
    if relative == "models":
        return model_path("")
    if relative.startswith("models/"):
        return model_path(relative.removeprefix("models/"))
    root = os.environ.get("VAFX_VOICE_ROOT")
    if root:
        return Path(root).expanduser() / relative
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    elif sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / "vexy-audioflexy" / "voice" / relative


def _app_path(kind: str) -> Path:
    if sys.platform == 'darwin':
        return Path.home() / 'Library' / 'Application Support' / 'vexy-audioflexy'
    if sys.platform == 'win32':
        return Path(os.environ.get('LOCALAPPDATA', Path.home() / 'AppData' / 'Local')) / 'vexy-audioflexy'
    variable = 'XDG_CONFIG_HOME' if kind == 'config' else 'XDG_DATA_HOME'
    fallback = Path.home() / ('.config' if kind == 'config' else '.local/share')
    return Path(os.environ.get(variable, fallback)) / 'vexy-audioflexy'


def model_path(identity: str) -> Path:
    """Honor consolidated model configuration and exact registered identities."""
    import tomllib
    config = _app_path('config') / 'config.toml'
    registry = _app_path('data') / 'models.toml'
    if registry.exists():
        with registry.open('rb') as stream:
            records = tomllib.load(stream).get('models', [])
        for record in records:
            if record.get('identity') == identity:
                path = Path(record['path']).expanduser()
                if not path.is_absolute():
                    raise ValueError('Registered model path must be absolute')
                return path
    saved = {}
    if config.exists():
        with config.open('rb') as stream:
            saved = tomllib.load(stream)
    value = os.environ.get('VAFX_MODELS_DIR', saved.get('models_dir', str(_app_path('data') / 'models')))
    if not isinstance(value, str) or not value.strip() or not Path(value).expanduser().is_absolute():
        raise ValueError('models_dir must be a nonempty absolute path')
    return Path(value).expanduser() / identity


def bind_snapshot(snapshot: Path, target: Path, companions: dict[str, Path] | None = None) -> None:
    """Link provider files after checking every binding; never overwrite owner data."""
    snapshot = snapshot.expanduser().resolve()
    target = target.expanduser()
    resolved_target = target.resolve()
    companions = companions or {}
    if resolved_target == snapshot and not companions:
        return  # Existing exact provider view needs no writes.
    if resolved_target.is_relative_to(snapshot) or snapshot.is_relative_to(resolved_target):
        raise ValueError('Provider snapshot and model view must not overlap')
    hf_home = Path(os.environ.get('HF_HOME', Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache')) / 'huggingface'))
    hub = Path(os.environ.get('HF_HUB_CACHE', os.environ.get('HUGGINGFACE_HUB_CACHE', hf_home / 'hub'))).expanduser().resolve()
    stores = [hub]
    for source in [snapshot, *companions.values()]:
        for ancestor in source.resolve().parents:
            if ancestor.name == 'snapshots' and ancestor.parent.name.startswith('models--'):
                stores.append(ancestor.parent.parent)
    if any(resolved_target.is_relative_to(store) for store in stores):
        raise ValueError('Model view must be outside the immutable provider store')
    if target.is_symlink():
        raise FileExistsError(f'Model directory conflicts: {target}; choose a fresh directory')
    entries = [(source, target / source.relative_to(snapshot)) for source in snapshot.rglob('*')]
    for relative, source in companions.items():
        path = Path(relative)
        if path.is_absolute() or '..' in path.parts or not path.parts or '\\' in relative:
            raise ValueError(f'Unsafe companion path: {relative}')
        if not source.is_file() or source.stat().st_size == 0:
            raise FileNotFoundError(f'Companion file is missing: {source}')
        destination = target / path
        for parent in destination.parents:
            if parent == target:
                break
            if parent.is_symlink() or (parent.exists() and not parent.is_dir()):
                raise FileExistsError(f'Model binding conflicts: {parent}; choose a fresh directory')
        entries.append((source.resolve(), destination))
    bindings = {}
    for source, destination in entries:
        if destination in bindings and bindings[destination].resolve() != source.resolve():
            raise FileExistsError(f'Model binding conflicts: {destination}; choose a fresh directory')
        bindings[destination] = source
        if source.is_dir():
            conflict = destination.is_symlink() or (destination.exists() and not destination.is_dir())
        else:
            conflict = (destination.exists() or destination.is_symlink()) and destination.resolve() != source.resolve()
        if conflict:
            raise FileExistsError(f'Model binding conflicts: {destination}; choose a fresh directory')
    target.mkdir(parents=True, exist_ok=True)
    for source, destination in entries:
        if source.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
        elif not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.symlink_to(source)
