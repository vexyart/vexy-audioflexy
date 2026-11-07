#!/usr/bin/env -S uv run -s
# /// script
# dependencies = ["platformdirs>=4", "fire", "numpy", "soundfile"]
# ///
# this_file: src/vexy_audioflexy/tools/fx/tools/make_music_fixture.py
"""Reproduce a64k MP3 degradation and channel-distinct music restoration smoke."""

import os
from pathlib import Path
from platformdirs import user_cache_path, user_data_path
RUNTIME_ROOT = Path(os.environ.get("VAFX_AUDIO_RUNTIME_ROOT", user_data_path("vexy-audioflexy") / "audio" )).expanduser()
CACHE_ROOT = Path(os.environ.get("VAFX_AUDIO_CACHE_ROOT", user_cache_path("vexy-audioflexy") / "audio")).expanduser()
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf


def main(source: str, output: str) -> None:
    """Preserve existing containers; require exact PCM when F32 WAV timestamps differ."""
    reference, destination = Path(source).resolve(), Path(output).resolve()
    samples, rate = sf.read(reference, dtype="float32", always_2d=True)
    if rate != 44100 or samples.shape[1] != 2 or len(samples) < 44100 or not np.isfinite(samples).all():
        raise SystemExit("Require finite stereo44.1kHz reference with at least one second")
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        stage = Path(temporary)
        distinct = samples[:44100].copy()
        distinct[:, 1] = .65 * np.concatenate((np.zeros(137, np.float32), distinct[:-137, 0]))
        sf.write(stage / "distinct-stereo.wav", distinct, 44100, subtype="FLOAT")
        for input_path, prefix in [(reference, "compressed"), (stage / "distinct-stereo.wav", "distinct-stereo")]:
            encoded = stage / f"{prefix}.mp3"
            decoded = stage / ("compressed.wav" if prefix == "compressed" else "distinct-stereo-compressed.wav")
            subprocess.run(["ffmpeg", "-v", "error", "-i", str(input_path), "-codec:a", "libmp3lame", "-b:a", "64k", str(encoded)], check=True, timeout=30)
            subprocess.run(["ffmpeg", "-v", "error", "-i", str(encoded), "-c:a", "pcm_f32le", str(decoded)], check=True, timeout=30)
        artifacts = []
        for path in sorted(stage.iterdir()):
            target = destination / path.name
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                if path.suffix != ".wav":
                    raise SystemExit(f"Compressed fixture bytes differ: {target}")
                before, before_rate = sf.read(target, dtype="float32", always_2d=True)
                after, after_rate = sf.read(path, dtype="float32", always_2d=True)
                if before_rate != after_rate or not np.array_equal(before, after):
                    raise SystemExit(f"Fixture differs from existing byte/PCM provenance: {target}")
                digest = hashlib.sha256(target.read_bytes()).hexdigest()
            artifacts.append({"file": path.name, "sha256": digest})
        for path in stage.iterdir():
            if not (destination / path.name).exists():
                path.replace(destination / path.name)
    manifest = {"source": str(reference), "source_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
        "numpy": np.__version__, "soundfile": sf.__version__,
        "ffmpeg": subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0],
        "codec": "libmp3lame64kbit/s; gapless decode to F32 WAV", "distinct_stereo": "first1s; right=left delayed137samples and scaled0.65",
        "reproduction": "Preserve existing WAV container bytes; require exact PCM if libsndfile PEAK timestamps differ. MP3 bytes must match.",
        "artifacts": artifacts}
    (destination / "fixture-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Verified fixture PCM and MP3 bytes; preserved existing WAV containers")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
