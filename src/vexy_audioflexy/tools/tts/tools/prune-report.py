#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["fire", "rich"]
# ///
# this_file: src/vexy_audioflexy/tools/tts/tools/prune-report.py
"""Drop result rows that the model registry no longer claims.

`out/report.json` accumulates across runs, so it outlives the registry: when a
model's language list is corrected, its old rows stay behind and keep asserting a
success that is no longer claimed. That is how a `chatterbox-turbo/pl` row survived
the discovery that Turbo does not actually deliver Polish.

A row is pruned when its model is still registered but no longer declares that
language. Rows for models that have left the registry entirely are kept — those are
history, not drift.

A successful row whose clip is still on disk is never dropped, and is reported
instead: a playable sample in a language the registry does not claim is evidence
that the registry under-declares, and deleting the record would throw that evidence
away. Decide those by listening.

    ./tools/prune-report.py            # show what would go
    ./tools/prune-report.py --apply    # write it back
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from _paths import runtime_path

ROOT = Path(__file__).resolve().parent.parent


def _registry() -> dict[str, set[str]]:
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location("tts_all", ROOT / "tts_all.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["tts_all"] = module
    spec.loader.exec_module(module)
    return {m["name"]: set(m["langs"]) for m in module.MODELS}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="write the pruned report back")
    parser.add_argument("--outdir", default=str(runtime_path('out')))
    args = parser.parse_args()

    outdir = Path(args.outdir)
    path = outdir / "report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    langs_by_model = _registry()

    results = report.get("results", {})
    as_dict = isinstance(results, dict)
    items = list(results.items()) if as_dict else [(None, r) for r in results]

    keep, drop, review = [], [], []
    for key, entry in items:
        # Voice-design runs are recorded as "model:desiN"; the registry knows "model".
        base = (entry.get("model") or "").split(":", 1)[0]
        declared = langs_by_model.get(base)
        stale = declared is not None and entry.get("lang") not in declared
        if not stale:
            keep.append((key, entry))
            continue
        clip = Path(entry.get("output_path") or "").name
        if entry.get("success") and clip and (outdir / clip).exists():
            review.append((key, entry))
            keep.append((key, entry))
        else:
            drop.append((key, entry))

    for _, entry in drop:
        print(f"  drop  {entry.get('model')}/{entry.get('lang')}"
              f"  success={entry.get('success')}")
    for _, entry in review:
        print(f"  KEPT — playable clip in an undeclared language, listen and decide: "
              f"{Path(entry.get('output_path') or '').name}")
    print(f"{len(drop)} stale row(s) dropped, {len(review)} held for review, "
          f"{len(keep)} kept")

    if not args.apply or not drop:
        if drop:
            print("re-run with --apply to write it back")
        return

    report["results"] = dict(keep) if as_dict else [e for _, e in keep]
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
