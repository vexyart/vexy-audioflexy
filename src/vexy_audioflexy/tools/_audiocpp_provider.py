# this_file: src/vexy_audioflexy/tools/_audiocpp_provider.py
"""Prepare audio.cpp compatibility links from exact Hugging Face package specs."""
import json
from pathlib import Path
import time


def safe_relative(value: str) -> Path:
    """A native package may only bind paths inside its selected layout."""
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value or value == ".":
        raise ValueError(f"Unsafe relative model path: {value}")
    return path


def install_package(entry: dict, models_root: Path, package: str, force: bool) -> None:
    """Download into the provider cache, then preflight native links and marker."""
    from huggingface_hub import snapshot_download

    if entry.get("download_kind") != "huggingface_snapshot":
        raise SystemExit(f"package '{package}': prep only installs huggingface_snapshot packages")
    if not entry.get("repo") or not entry.get("files"):
        raise SystemExit("HF packages must declare a repository and exact files")
    target = models_root / safe_relative(entry["target_directory"])
    files = entry["files"]
    for name in files:
        safe_relative(name)
    snapshot = Path(snapshot_download(entry["repo"], revision=entry.get("revision", "main"), allow_patterns=files)).resolve()
    if target.resolve().is_relative_to(snapshot) or snapshot.is_relative_to(target.resolve()):
        raise ValueError("Provider snapshot and native model view must not overlap")
    for ancestor in snapshot.parents:
        if ancestor.name == "snapshots" and ancestor.parent.name.startswith("models--"):
            if target.resolve().is_relative_to(ancestor.parent.parent):
                raise ValueError("Native model view must be outside the provider store")
    marker = target / f".audiocpp-package-{package}.json"
    if marker.is_symlink() or (marker.exists() and not marker.is_file()):
        raise FileExistsError(f"Package marker conflicts: {marker}")
    plan = []
    prefix = entry.get("strip_prefix", "").rstrip("/")
    for remote in files:
        if prefix and not remote.startswith(prefix + "/"):
            raise ValueError(f"File does not match strip_prefix: {remote}")
        source = snapshot / remote
        output = target / safe_relative(remote[len(prefix) + 1:] if prefix else remote)
        if not source.is_file() or source.stat().st_size == 0:
            raise SystemExit(f"Provider snapshot is missing declared file: {remote}")
        if not output.parent.resolve().is_relative_to(target.resolve()):
            raise ValueError(f"Native layout escapes its target: {output}")
        exists = output.exists() or output.is_symlink()
        same = output.is_file() and output.samefile(source)
        if exists and not same and (not force or output.is_dir()):
            raise FileExistsError(f"Model path differs from provider file: {output}; use --force")
        plan.append((source, output, exists, same))
    for source, output, exists, same in plan:
        if same:
            continue
        output.parent.mkdir(parents=True, exist_ok=True)
        if exists:
            output.unlink()
        output.symlink_to(source.resolve())
    payload = {"schema_version": 1, "package_id": package, "repo": entry["repo"],
               "requested_revision": entry.get("revision", "main"), "resolved_revision": snapshot.name,
               "installed_at_unix": int(time.time()),
               "files": {name: {"size": (snapshot / name).stat().st_size, "etag": ""} for name in files}}
    marker.write_text(json.dumps(payload, indent=2) + "\n")
