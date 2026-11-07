#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = ["platformdirs>=4", "fire"]
# ///
# this_file: src/vexy_audioflexy/tools/fx/separate_all.py
"""Batch native source separation, preserving every named stem and its report.

uv run separate_all.py --audio path/to/music.wav --packages htdemucs-4s,htdemucs-6s
"""
from __future__ import annotations


import os
from pathlib import Path
from platformdirs import user_cache_path, user_data_path
RUNTIME_ROOT = Path(os.environ.get("VAFX_AUDIO_RUNTIME_ROOT", user_data_path("vexy-audioflexy") / "audio" )).expanduser()
CACHE_ROOT = Path(os.environ.get("VAFX_AUDIO_CACHE_ROOT", user_cache_path("vexy-audioflexy") / "audio")).expanduser()
import json
import os
import runpy
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PACKAGES = ("htdemucs-4s", "htdemucs-6s", "htdemucs_6stems_f16", "htdemucs_6stems_q8_0")


def main(audio: str, packages: str = "htdemucs-4s,htdemucs-6s",
         outdir: str = "out/separation", timeout: int = 1800, force: bool = False, backend: str = "metal") -> None:
    """Run selected variants, resume validated reports and record individual failures."""
    source = Path(audio).resolve()
    selected = packages.split(",") if isinstance(packages, str) else list(packages)
    if not source.is_file() or timeout <= 0 or not selected or any(p not in PACKAGES for p in selected):
        raise SystemExit("Require input audio, positive timeout and known packages: " + ",".join(PACKAGES))
    destination = Path(outdir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    for package in selected:
        output = destination / package
        musiccpp = package.startswith("htdemucs_6stems_")
        report_path = output / "report.json"
        if report_path.exists() and not force:
            report = json.loads(report_path.read_text())
            import hashlib
            source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            stems = report.get("outputs" if musiccpp else "stems", [])
            repaired = True
            if musiccpp and backend == "cpu":
                engine = Path(os.environ.get("AUDIOCPP_HTDEMUCS_FIXED_ROOT", str(RUNTIME_ROOT / "engines/audio.cpp-demucs-diagnostic")))
                binary = engine / "build/diagnostic/bin/audiocpp_cli"
                patch = ROOT / "patches/ggml-conv-transpose2d-batch.patch"
                qualified = runpy.run_path(str(ROOT / "tools/aud-musiccpp.py"))
                repaired = binary.is_file() and report.get("cpu_repair_patch_sha256") == qualified["CPU_PATCH_SHA256"] == hashlib.sha256(patch.read_bytes()).hexdigest() and report.get("binary_sha256") == qualified["CPU_BINARY_SHA256"] == hashlib.sha256(binary.read_bytes()).hexdigest()
            if repaired and report.get("input_sha256") == source_hash and (not musiccpp or report.get("backend") == backend) and len(stems) == (4 if package.endswith("4s") else 6) and all(
                (output / stem["file"]).is_file() and hashlib.sha256((output / stem["file"]).read_bytes()).hexdigest() == stem["sha256"]
                for stem in stems
            ):
                records.append({"package": package, "success": True, "resumed": True, "report": report})
                continue
        argv = ["uv", "run", "--script", str(ROOT / ("tools/aud-musiccpp.py" if musiccpp else "tools/aud-demucscpp.py")), "run", str(source),
                str(output), "--package", package, "--timeout", str(timeout), "--force"]
        if musiccpp:
            argv += ["--backend", backend]
        try:
            result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout + 60, check=False)
            success = result.returncode == 0 and report_path.is_file()
            record = {"package": package, "success": success, "resumed": False}
            if success:
                record["report"] = json.loads(report_path.read_text())
            else:
                record["error"] = (result.stdout + result.stderr)[-3000:]
        except subprocess.TimeoutExpired:
            record = {"package": package, "success": False, "error": "Batch deadline exceeded"}
        records.append(record)
        (destination / "report.json").write_text(json.dumps({"results": records}, indent=2) + "\n")
    (destination / "report.json").write_text(json.dumps({"results": records}, indent=2) + "\n")
    print(json.dumps({"results": records}, indent=2))
    if not all(record["success"] for record in records):
        raise SystemExit(1)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
