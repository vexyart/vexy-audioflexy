#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = ["fire", "rich", "soundfile", "platformdirs"]
# ///
# this_file: src/vexy_audioflexy/tools/asr/tools/asr-whisperkit.py
"""WhisperKit (argmax-oss-swift): CoreML whisper on the Apple Neural Engine.

The only Swift engine in this workspace, and the only one that runs on the ANE
rather than the GPU or the CPU. `argmax-cli transcribe` is a built binary, so this
wrapper shells out to it exactly as `asr-audiocpp.py` shells out to `audiocpp_cli`.

The checkout is not vendored: `prep` clones `argmaxinc/argmax-oss-swift` into
`local/argmax-oss-swift` and builds `whisperkit-cli` with `swift build -c release`
(about 30 seconds with Xcode's toolchain already installed). `WHISPERKIT_BIN`
points at an existing binary instead.

Model weights are CoreML packages from `argmaxinc/whisperkit-coreml`, downloaded by
the CLI on first use.

Compute units default to `cpuAndNeuralEngine`, which is the point of this engine;
`--compute-units` overrides both encoder and decoder.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

from _audioflexy_paths import runtime_root

VERSION = "1.0.0"

REPO_URL = "https://github.com/argmaxinc/argmax-oss-swift"

# WhisperKit model names, as `--model` takes them.
PACKAGES: dict[str, dict] = {
    "tiny": {"quant": "f16", "langs": ["en", "pl"]},
    "base": {"quant": "f16", "langs": ["en", "pl"]},
    "small": {"quant": "f16", "langs": ["en", "pl"]},
    "large-v3_turbo": {"quant": "f16", "langs": ["en", "pl"]},
    "large-v3_turbo_632MB": {"quant": "mixed", "langs": ["en", "pl"]},
}


def _spec(package: str) -> dict:
    if package in PACKAGES:
        return PACKAGES[package]
    # WhisperKit accepts any model name published in its CoreML repo.
    return {"quant": "unknown", "langs": ["en", "pl"]}


def _root() -> Path:
    return runtime_root("argmax-oss-swift")


def _binary(build: bool = True) -> Path:
    """The built CLI. Clones and builds it when missing unless `build` is False."""
    env = os.environ.get("WHISPERKIT_BIN")
    if env:
        return Path(env).expanduser()
    root = _root()
    binary = root / ".build" / "release" / "whisperkit-cli"
    if binary.exists():
        return binary
    if not build:
        raise SystemExit(
            f"whisperkit-cli not built. Run `{Path(__file__).name} prep`, or set "
            f"WHISPERKIT_BIN."
        )
    if not (root / "Package.swift").exists():
        root.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(root)], check=True)
    subprocess.run(
        ["swift", "build", "-c", "release", "--product", "whisperkit-cli"],
        cwd=str(root), check=True,
    )
    if not binary.exists():
        raise SystemExit(f"swift build finished but {binary} is missing")
    return binary


def _duration(path: Path) -> float:
    import soundfile as sf

    try:
        info = sf.info(str(path))
        return info.frames / info.samplerate
    except Exception:
        return 0.0


def prep(package: str = "small", force: bool = False) -> None:
    """Clone + build the CLI and download one model's CoreML package. Idempotent.

    Args:
        package: WhisperKit model name; `packages` lists the registered ones.
        force: Unused; an existing build and cached weights are reused.
    """
    import tempfile
    import wave

    binary = _binary()
    # The CLI fetches weights lazily on a real transcription, so prep transcribes
    # one second of silence. A prep that quietly downloads nothing is worse than
    # no prep: the model then lands inside the transcribe timeout.
    with tempfile.TemporaryDirectory() as tmp:
        silence = Path(tmp) / "silence.wav"
        with wave.open(str(silence), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(b"\x00\x00" * 16000)
        result = subprocess.run(
            [str(binary), "transcribe", "--audio-path", str(silence),
             "--model", package],
            capture_output=True, text=True,
        )
    if result.returncode != 0:
        print(result.stderr.strip()[-800:])
        raise SystemExit(f"whisperkit-cli could not fetch model '{package}'")
    print(f"[prep] ready: {binary} model={package}")


def transcribe(
    audio: str,
    package: str = "small",
    text_out: str = "",
    lang: str = "",
    compute_units: str = "cpuAndNeuralEngine",
    force: bool = False,
) -> None:
    """Transcribe one audio file with WhisperKit.

    Args:
        audio: Input audio file.
        package: WhisperKit model name.
        text_out: Transcript path (default: *audio* with a `.txt` suffix).
        lang: Language code. Without it WhisperKit detects the language.
        compute_units: `cpuAndNeuralEngine` (default), `cpuAndGPU`, `cpuOnly` or `all`.
        force: Overwrite an existing transcript.
    """
    from rich.console import Console

    console = Console()
    src = Path(audio)
    if not src.exists():
        raise SystemExit(f"audio not found: {src}")
    spec = _spec(package)
    if lang and lang not in spec["langs"]:
        raise SystemExit(f"package '{package}' covers {'/'.join(spec['langs'])}, not '{lang}'")
    out_path = Path(text_out) if text_out else src.with_suffix(".txt")
    if not force and out_path.exists() and out_path.stat().st_size > 0:
        console.print(f"[yellow]Skipped:[/yellow] {out_path} already exists")
        return

    binary = _binary()
    cmd = [
        str(binary), "transcribe",
        "--audio-path", str(src.resolve()),
        "--model", package,
        "--audio-encoder-compute-units", compute_units,
        "--text-decoder-compute-units", compute_units,
    ]
    if lang:
        cmd += ["--language", lang]

    console.print(f"[cyan]WhisperKit[/cyan] v{VERSION} model=[yellow]{package}[/yellow] "
                  f"units=[yellow]{compute_units}[/yellow]")
    t0 = time.monotonic()
    result = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.monotonic() - t0
    if result.returncode != 0:
        console.print(result.stderr.strip()[-800:])
        raise SystemExit(result.returncode)

    # The CLI prints progress lines and then the transcript; the transcript is
    # whatever non-empty text survives, joined.
    transcript = " ".join(
        line.strip() for line in result.stdout.splitlines()
        if line.strip() and not line.strip().startswith(("[", "Downloading", "Loading"))
    ).strip()
    if not transcript:
        console.print("[red]empty transcript[/red]")
        raise SystemExit(1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(transcript + "\n", encoding="utf-8")
    audio_s = _duration(src)
    console.print(f"[green]Transcript[/green] {out_path.resolve()}\n  {transcript}")
    print("asr.result=" + json.dumps(
        {
            "text": transcript,
            "text_out": str(out_path),
            "wall_s": round(wall, 4),
            "audio_duration_s": round(audio_s, 4),
            "rtf": round(wall / audio_s, 6) if audio_s else 0.0,
            "package": package,
            "compute_units": compute_units,
        },
        ensure_ascii=False, separators=(",", ":"),
    ))


def info(package: str = "small") -> None:
    """Print one-line JSON with model metadata."""
    spec = _spec(package)
    print(json.dumps(
        {
            "name": package,
            "family": "whisper",
            "repo": "argmaxinc/argmax-oss-swift (argmaxinc/whisperkit-coreml weights)",
            "stack": "coreml",
            "runtime": "WhisperKit (Swift, CoreML, Neural Engine)",
            "quant": spec["quant"],
            "langs": spec["langs"],
            "tasks": ["asr", "translate"],
        },
        separators=(",", ":"),
    ))


def packages() -> None:
    """List the registered models."""
    for name, spec in PACKAGES.items():
        print(f"{name:24} {spec['quant']:8} {'/'.join(spec['langs'])}")


if __name__ == "__main__":
    import fire

    fire.Fire({"prep": prep, "transcribe": transcribe, "info": info, "packages": packages})
