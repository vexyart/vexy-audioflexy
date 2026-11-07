#!/usr/bin/env -S uv run -s
# /// script
# dependencies = ["platformdirs>=4", "fire"]
# ///
# this_file: src/vexy_audioflexy/tools/fx/detect_all.py
"""Batch native speech-activity and turn decisions, retaining every JSON/log pair."""
from __future__ import annotations


import os
from pathlib import Path
from platformdirs import user_cache_path, user_data_path
RUNTIME_ROOT = Path(os.environ.get("VAFX_AUDIO_RUNTIME_ROOT", user_data_path("vexy-audioflexy") / "audio" )).expanduser()
CACHE_ROOT = Path(os.environ.get("VAFX_AUDIO_CACHE_ROOT", user_cache_path("vexy-audioflexy") / "audio")).expanduser()
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS = ["pulsevad_2_1k_f32", "pulsevad_81k_f32", "smart_turn_v3_2_f32"]


def main(inputdir: str, outdir: str = "out/detection", packages: str = "all",
         backends: str = "cpu,metal", force: bool = False) -> None:
    """Run every requested clip/package/backend; resume only matching source/config."""
    inputs, destination = Path(inputdir).resolve(), (Path(outdir).expanduser().resolve()).resolve()
    selected = MODELS if packages == "all" else list(packages) if isinstance(packages, tuple) else packages.split(",")
    engines = list(backends) if isinstance(backends, tuple) else backends.split(",")
    clips = sorted(inputs.glob("*.wav"))
    if not clips or not selected or not engines or len(set(selected)) != len(selected) or len(set(engines)) != len(engines) or any(name not in MODELS for name in selected) or any(name not in {"cpu", "metal"} for name in engines):
        raise SystemExit("Require WAV inputs, registered packages and CPU/Metal backends")
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    for package in selected:
        for backend in engines:
            for source in clips:
                output = destination / f"{package}__{backend}__{source.stem}.json"
                source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
                resumed = False
                if output.is_file() and not force:
                    prior = json.loads(output.read_text())
                    resumed = all(prior.get(key) == value for key, value in {
                        "source_sha256": source_hash, "package": package, "backend": backend, "threshold": .5,
                        "runtime_commit": "2892ed3e94b6ccbdacc111fa2896aaae0a5dc8fc",
                        "revision": "351dbab8d8534675ee29440bb402e348b09e55e2"}.items())
                    if not resumed:
                        raise SystemExit(f"Existing detection result differs from source/config: {output}; use --force")
                if not resumed:
                    command = ["uv", "run", "--script", str(ROOT / "tools/aud-detection.py"), "run",
                        f"--audio={source}", f"--output={output}", f"--package={package}", f"--backend={backend}"]
                    if force:
                        command.append("--force")
                    native = subprocess.run(command, capture_output=True, text=True, timeout=180, check=False)
                    output.with_suffix(".harness.log").write_text(native.stdout + native.stderr)
                    native.check_returncode()
                records.append({"receipt": str(output), "resumed": resumed, **json.loads(output.read_text())})
                print(f"{'resume' if resumed else 'run'} {package} {backend} {source.stem}", flush=True)
    report = {"models": selected, "backends": engines, "clips": [str(path) for path in clips], "results": records}
    (destination / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"{len(records)} decisions retained: {destination / 'report.json'}")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
