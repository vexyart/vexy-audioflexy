# this_file: src/vexy_audioflexy/tools/_audiocpp_catalog.py
"""Registry rows for audio.cpp families from the generated shared catalog.

``gen_audiocpp_catalog.py`` writes ``catalogs/audiocpp-<sha7>.json``; the
TTS, ASR and audio-to-audio registries call :func:`entries` to append one row
per family routed to them that no hand-written entry already covers. Rows carry
the catalog status (``pending runtime``) until a run verifies them. Each
registry builds its own ``args_fn`` from ``params``.
"""

from __future__ import annotations

import json
import os
import runpy
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
CATALOG = TOOLS / "catalogs" / "audiocpp-83b1354.json"
NAME_PREFIX = "ac83-"


def engine_root() -> str:
    """``AUDIOCPP_AC83_ROOT`` override, else the shared ``runtime_path('audio.cpp')``."""
    override = os.environ.get("AUDIOCPP_AC83_ROOT")
    if override:
        return override
    return str(runpy.run_path(str(TOOLS / "tts/tools/_paths.py"))["runtime_path"]("audio.cpp"))


def load(path: Path = CATALOG) -> dict:
    return json.loads(path.read_text())


def entries(registry: str, path: Path = CATALOG) -> list[dict]:
    """One generic row per catalog route that targets ``registry``."""
    catalog = load(path)
    root = engine_root()
    rows = []
    for family in catalog["families"]:
        packages = {p["id"]: p for p in family["packages"]}
        for route in family["routes"]:
            if route["route"] != registry:
                continue
            package = packages[route["package"]]
            rows.append({
                "name": NAME_PREFIX + package["id"].replace("_", "-"),
                "package": package["id"],
                "family": family["family"],
                "repo": package["repo"],
                "quant": package["precision"],
                "langs": route["langs"],
                "params": route["params"],
                "status": route["status"],
                "source_revision": catalog["source_revision"],
                "stack": "audiocpp",
                "env": {"AUDIOCPP_ROOT": root},
            })
    return rows
