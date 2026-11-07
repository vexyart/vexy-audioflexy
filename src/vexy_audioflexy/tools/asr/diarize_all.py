#!/usr/bin/env -S uv run
# /// script
# dependencies = ["fire"]
# ///
# this_file: src/vexy_audioflexy/tools/asr/diarize_all.py
"""Batch native speaker diarization; structured turns use their own task metrics."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main(audio: str, outdir: str, packages: str = "audiocpp-nemotron3-q8_0", reference: str = "",
         backend: str = "metal", mode: str = "offline",
         timeout: int = 900, force: bool = False, latency_profile: str = "low") -> None:
    """Run selected pinned packages and retain failures without losing successful turns."""
    source = Path(audio).resolve()
    selected = packages.split(",") if isinstance(packages, str) else list(packages)
    known = {row["package"]: row for row in json.loads((ROOT / "catalogs/diarization.json").read_text())["models"]}
    if not source.is_file() or not selected or any(p not in known for p in selected) or timeout <= 0:
        raise SystemExit("Require existing audio, known packages and positive timeout")
    if reference and not Path(reference).is_file():
        raise SystemExit("Reference JSON must exist")
    destination = Path(outdir).resolve()
    records = []
    for package in selected:
        output = destination / package
        script = {"mlx": "diarize-mlx.py", "coreai-zoo": "diarize-coreai.py",
                  "coreai-coder": "diarize-coder-coreai.py"}.get(known[package]["engine"], "diarize-native.py")
        args = ["uv", "run", "--script", str(ROOT / "tools" / script), "run", str(source),
                str(output), "--package", package, "--backend", backend, "--mode", mode,
                "--timeout", str(timeout)]
        if force:
            args.append("--force")
        if known[package]["engine"] == "coreai-zoo":
            args.extend(["--latency-profile", latency_profile])
        try:
            result = subprocess.run(args, capture_output=True, text=True, timeout=timeout + 60, check=False)
            record = {"package": package, "success": result.returncode == 0}
            if record["success"]:
                record["report"] = json.loads((output / "report.json").read_text())
                if reference:
                    result = subprocess.run(["uv", "run", "--script", str(ROOT / "tools/diar-metrics.py"),
                        str(Path(reference).resolve()), str(output / "report.json"),
                        "--output", str(output / "metrics.json")], capture_output=True, text=True, timeout=timeout, check=False)
                    if result.returncode:
                        record.update(success=False, error=result.stdout + result.stderr)
                    else:
                        record["metrics"] = json.loads((output / "metrics.json").read_text())
            else:
                record["error"] = (result.stdout + result.stderr)[-3000:]
        except subprocess.TimeoutExpired:
            record = {"package": package, "success": False, "error": "Deadline exceeded"}
        records.append(record)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "report.json").write_text(json.dumps({"results": records}, indent=2) + "\n")
    print(json.dumps({"results": records}, indent=2))
    if not all(row["success"] for row in records):
        raise SystemExit(1)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
