#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = ["fire", "rich", "platformdirs"]
# ///
# this_file: src/vexy_audioflexy/tools/tts/tools/scan-models.py
"""
Scan local cache folders for downloaded TTS models and produce a local catalog override JSON.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import fire
import platformdirs
from rich.console import Console
from rich.table import Table
from _paths import runtime_path

console = Console()

CATALOG_PATH = runtime_path('models/catalog.json')


def _load_catalog() -> dict:
    """Load the bundled model catalog, returning empty dict on failure."""
    if not CATALOG_PATH.exists():
        return {}
    try:
        return json.loads(CATALOG_PATH.read_text())
    except Exception:
        return {}


def _dir_size(path: Path) -> int:
    """Return total byte size of a directory tree."""
    total = 0
    try:
        for entry in path.rglob("*"):
            if entry.is_file():
                try:
                    total += entry.stat().st_size
                except OSError:
                    pass
    except PermissionError:
        pass
    return total


def _fmt_size(n: int) -> str:
    """Human-readable file size."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.0f}{unit}"
        n = int(n // 1024)
    return f"{n:.0f}PB"


def _build_repo_index(catalog: dict) -> dict[str, tuple[str, str]]:
    """
    Build a mapping of lowercase repo_id -> (model_key, variant_key)
    from the catalog so we can match found HF dirs back to catalog entries.
    """
    index: dict[str, tuple[str, str]] = {}
    for key, entry in catalog.get("models", {}).items():
        for variant_key, variant in entry.get("variants", {}).items():
            repo = variant.get("repo", "")
            if repo:
                index[repo.lower()] = (key, variant_key)
    return index


# ---------------------------------------------------------------------------
# Scanner functions — each returns a list of dicts with keys:
#   model_key, variant, display_name, local_path, size_bytes, source
# ---------------------------------------------------------------------------

def _scan_huggingface(cache_dir: Path, repo_index: dict, verbose: bool) -> list[dict]:
    """Scan ~/.cache/huggingface/hub for models--org--name directories."""
    results = []
    if not cache_dir.exists():
        if verbose:
            console.print(f"  [dim]skip[/dim] {cache_dir}  (not found)")
        return results

    for entry in sorted(cache_dir.iterdir()):
        if not entry.is_dir():
            continue
        name = entry.name
        if not name.startswith("models--"):
            continue
        # models--org--name → org/name
        parts = name[len("models--"):].split("--", 1)
        if len(parts) != 2:
            continue
        repo_id = f"{parts[0]}/{parts[1]}"
        size = _dir_size(entry)
        model_key, variant_key = repo_index.get(repo_id.lower(), (None, None))
        results.append(
            {
                "model_key": model_key,
                "variant": variant_key,
                "display_name": repo_id,
                "local_path": str(entry),
                "size_bytes": size,
                "source": "huggingface",
            }
        )
        if verbose:
            tag = f"[green]{model_key}:{variant_key}[/green]" if model_key else "[dim]unknown[/dim]"
            console.print(f"  HF {tag}  {repo_id}  {_fmt_size(size)}")
    return results


def _scan_flat_dir(cache_dir: Path, source: str, verbose: bool) -> list[dict]:
    """Scan a flat cache dir where each child dir/file is a model."""
    results = []
    if not cache_dir.exists():
        if verbose:
            console.print(f"  [dim]skip[/dim] {cache_dir}  (not found)")
        return results

    for entry in sorted(cache_dir.iterdir()):
        if entry.name.startswith("."):
            continue
        size = _dir_size(entry) if entry.is_dir() else entry.stat().st_size
        results.append(
            {
                "model_key": None,
                "variant": None,
                "display_name": entry.name,
                "local_path": str(entry),
                "size_bytes": size,
                "source": source,
            }
        )
        if verbose:
            console.print(f"  {source}  [dim]unknown[/dim]  {entry.name}  {_fmt_size(size)}")
    return results


def _scan_torch_hub(cache_dir: Path, verbose: bool) -> list[dict]:
    """Scan PyTorch hub checkpoints (flat files, not dirs)."""
    results = []
    if not cache_dir.exists():
        if verbose:
            console.print(f"  [dim]skip[/dim] {cache_dir}  (not found)")
        return results

    for entry in sorted(cache_dir.iterdir()):
        if entry.is_file() and not entry.name.startswith("."):
            size = entry.stat().st_size
            results.append(
                {
                    "model_key": None,
                    "variant": None,
                    "display_name": entry.name,
                    "local_path": str(entry),
                    "size_bytes": size,
                    "source": "torch_hub",
                }
            )
            if verbose:
                console.print(f"  torch_hub  [dim]unknown[/dim]  {entry.name}  {_fmt_size(size)}")
    return results


def _scan_ollama(models_dir: Path, verbose: bool) -> list[dict]:
    """Scan ~/.ollama/models — manifests + blobs."""
    results = []
    if not models_dir.exists():
        if verbose:
            console.print(f"  [dim]skip[/dim] {models_dir}  (not found)")
        return results

    manifests_dir = models_dir / "manifests"
    if manifests_dir.exists():
        # manifests/registry.ollama.ai/library/<name>/<tag>
        for mfile in sorted(manifests_dir.rglob("*")):
            if mfile.is_file() and not mfile.name.startswith("."):
                # Build a human-readable name from path components after manifests/
                rel = mfile.relative_to(manifests_dir)
                parts = rel.parts
                # Drop registry host if present
                display = "/".join(parts[-3:]) if len(parts) >= 3 else str(rel)
                size = mfile.stat().st_size
                results.append(
                    {
                        "model_key": None,
                        "variant": None,
                        "display_name": f"ollama:{display}",
                        "local_path": str(mfile),
                        "size_bytes": size,
                        "source": "ollama",
                    }
                )
                if verbose:
                    console.print(f"  ollama  [dim]manifest[/dim]  {display}  {_fmt_size(size)}")
    return results


def _scan_lmstudio(pointer_file: Path, verbose: bool) -> list[dict]:
    """
    Read $HOME/.lmstudio-home-pointer to get the LM Studio home dir,
    then scan for models inside it.
    """
    results = []
    if not pointer_file.exists():
        if verbose:
            console.print(f"  [dim]skip[/dim] {pointer_file}  (not found)")
        return results

    try:
        lm_home = Path(pointer_file.read_text().strip())
    except Exception as exc:
        if verbose:
            console.print(f"  [dim]skip[/dim] {pointer_file}  (read error: {exc})")
        return results

    # Common LM Studio model locations
    candidates = [
        lm_home / "models",
        lm_home,
    ]
    for models_dir in candidates:
        if not models_dir.exists() or not models_dir.is_dir():
            continue
        for entry in sorted(models_dir.iterdir()):
            if entry.name.startswith(".") or not entry.is_dir():
                continue
            # LM Studio stores models as org/model directories
            for sub in sorted(entry.iterdir()):
                if sub.name.startswith("."):
                    continue
                size = _dir_size(sub) if sub.is_dir() else sub.stat().st_size
                display = f"{entry.name}/{sub.name}"
                results.append(
                    {
                        "model_key": None,
                        "variant": None,
                        "display_name": display,
                        "local_path": str(sub),
                        "size_bytes": size,
                        "source": "lmstudio",
                    }
                )
                if verbose:
                    console.print(f"  lmstudio  [dim]unknown[/dim]  {display}  {_fmt_size(size)}")
        break  # only scan first found
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def prep(force: bool = False):
    """No model weights to download for this scanner utility. Idempotent no-op."""
    print("[prep] ready: (scan-models — no weights to download)")


def say(output: str | None = None, verbose: bool = False, json_out: bool = False):
    """
    Scan local cache folders for downloaded TTS models.

    Args:
        output: If provided, write local catalog JSON to this path.
        verbose: Show all scanned paths and findings.
        json_out: Output as JSON (default: human-readable table).
    """
    home = Path.home()
    cache_base = Path(platformdirs.user_cache_dir())

    scan_paths = [
        ("lmstudio", home / ".lmstudio-home-pointer"),
        ("huggingface", cache_base / "huggingface" / "hub"),
        ("whisper_cpp", cache_base / "whisper.cpp"),
        ("whisper", cache_base / "whisper"),
        ("mlx_audio", cache_base / "mlx-audio"),
        ("torch_hub", cache_base / "torch" / "hub" / "checkpoints"),
        ("ollama", home / ".ollama" / "models"),
    ]

    catalog = _load_catalog()
    repo_index = _build_repo_index(catalog)

    all_models: list[dict] = []

    if verbose:
        console.print("\n[bold]Scanning locations...[/bold]")

    for source, path in scan_paths:
        if verbose:
            console.print(f"\n[cyan]{source}[/cyan]  →  {path}")
        if source == "lmstudio":
            all_models.extend(_scan_lmstudio(path, verbose))
        elif source == "huggingface":
            all_models.extend(_scan_huggingface(path, repo_index, verbose))
        elif source == "torch_hub":
            all_models.extend(_scan_torch_hub(path, verbose))
        elif source == "ollama":
            all_models.extend(_scan_ollama(path, verbose))
        else:
            all_models.extend(_scan_flat_dir(path, source, verbose))

    # Build output catalog structure
    scanned_at = datetime.now(timezone.utc).isoformat()
    scan_path_strs = [str(p) for _, p in scan_paths]

    # Build models dict keyed by model_key where known, else by display_name
    models_out: dict[str, dict] = {}
    for m in all_models:
        key = m["model_key"] or m["display_name"]
        entry = {
            "found_variant": m["variant"],
            "local_path": m["local_path"],
            "size_bytes": m["size_bytes"],
            "source": m["source"],
            "display_name": m["display_name"],
        }
        # If same key appears multiple times (e.g. different variants), keep largest
        if key in models_out:
            if m["size_bytes"] > models_out[key]["size_bytes"]:
                models_out[key] = entry
        else:
            models_out[key] = entry

    result_json = {
        "$schema": "models-catalog-local-v1",
        "scanned_at": scanned_at,
        "scan_paths": scan_path_strs,
        "models": models_out,
    }

    # Decide output mode
    emit_json = json_out or (output is not None)

    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result_json, indent=2))
        console.print(f"[green]Wrote[/green] {len(models_out)} models to {out_path}")

    if emit_json and not output:
        print(json.dumps(result_json, indent=2))
        return

    if not emit_json:
        # Human-readable table
        console.print(f"\nScanned [bold]{len(scan_paths)}[/bold] locations, found [bold]{len(all_models)}[/bold] models:\n")

        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Model", style="white", no_wrap=False, min_width=35)
        table.add_column("Source", style="dim", min_width=12)
        table.add_column("Catalog Key", style="green", min_width=18)
        table.add_column("Size", justify="right", min_width=8)

        for m in sorted(all_models, key=lambda x: x["source"]):
            catalog_tag = m["model_key"] or "[dim]—[/dim]"
            if m["variant"]:
                catalog_tag += f"[dim]:{m['variant']}[/dim]"
            table.add_row(
                m["display_name"],
                m["source"],
                catalog_tag,
                _fmt_size(m["size_bytes"]),
            )

        if all_models:
            console.print(table)
        else:
            console.print("[dim]No models found in scanned locations.[/dim]")

        console.print("\nUse [bold]--output local-catalog.json[/bold] to save as catalog override.")
        console.print("Use [bold]--json[/bold] to print JSON to stdout.")


def info():
    """Print one-line JSON metadata for this utility."""
    print(json.dumps({
        "name": "scan-models",
        "repo": None,
        "backend": "utility",
        "languages": [],
        "sample_rate_hz": None,
        "modes": [],
        "capabilities": ["local_cache_scan"],
    }))


if __name__ == "__main__":
    fire.Fire({"prep": prep, "say": say, "info": info})
