#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = ["fire", "rich"]
# ///
# this_file: src/vexy_audioflexy/tools/tts/tools/tts-indextts2-mlx.py
"""IndexTTS-2 natively on Apple Silicon (mlx-community/IndexTTS-2-MLX).

Backend: mlx via github.com/Jup33Q/mlx-indextts. Languages: zh, en. 22.05 kHz.
Mode: ivc (zero-shot cloning from a 5-15 s reference clip), with optional explicit
emotion control.

mlx-audio also ships an `indextts` module, but it targets the 1.5 layout; the v2
checkpoint (separate gpt / s2mel / bigvgan / vq2emb safetensors plus the emotion
matrices) is driven by mlx-indextts, which runs in its own uv project. This wrapper
shells out to that project's CLI rather than importing it, so its pinned
torch/torchaudio stack stays isolated from the rest of this repo.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from _paths import runtime_path

VERSION = "1.0.0"
MODEL_REPO = "mlx-community/IndexTTS-2-MLX"
# Speaker encoder the s2mel stage conditions on; not shipped in the MLX repo.
CAMPPLUS_REPO = "funasr/campplus"
CAMPPLUS_FILE = "campplus_cn_common.bin"

ROOT = Path(__file__).resolve().parent.parent
ENGINE_ROOT = runtime_path('mlx-indextts')
MODEL_DIR = runtime_path('models/IndexTTS-2-MLX')

EMOTIONS = ["happy", "angry", "sad", "afraid", "disgusted", "melancholic", "surprised", "calm"]

BUILD = f"""\
  git clone --depth 1 https://github.com/Jup33Q/mlx-indextts {ENGINE_ROOT}
  cd {ENGINE_ROOT} && uv sync --extra v2"""


def _require_engine() -> None:
    if not (ENGINE_ROOT / "pyproject.toml").exists():
        raise SystemExit(f"mlx-indextts not checked out at {ENGINE_ROOT}\nSet it up with:\n{BUILD}")


def _env() -> dict[str, str]:
    """torchaudio decodes through torchcodec, which needs Homebrew's FFmpeg libs."""
    env = os.environ.copy()
    hint = "/opt/homebrew/lib"
    if Path(hint).is_dir():
        current = env.get("DYLD_FALLBACK_LIBRARY_PATH", "")
        if hint not in current.split(":"):
            env["DYLD_FALLBACK_LIBRARY_PATH"] = ":".join([p for p in (current, hint) if p])
    return env


def prep(force: bool = False) -> None:
    """Download the MLX checkpoint and the CAMPPlus speaker encoder. Idempotent."""
    _require_engine()
    subprocess.run(
        [
            "uv", "run", "--with", "einops", "python", "-c",
            "from huggingface_hub import snapshot_download, hf_hub_download; "
            "import pathlib, runpy\n"
            f"d = pathlib.Path(snapshot_download({MODEL_REPO!r}, force_download={force!r}))\n"
            f"c = pathlib.Path(hf_hub_download({CAMPPLUS_REPO!r}, {CAMPPLUS_FILE!r}, force_download={force!r}))\n"
            f"target = pathlib.Path({str(MODEL_DIR)!r})\n"
            f"bind = runpy.run_path({str(Path(__file__).with_name('_paths.py'))!r})['bind_snapshot']\n"
            f"bind(d, target, companions={{{('hf_cache/' + CAMPPLUS_FILE)!r}: c}})\n",
        ],
        check=True,
        cwd=str(ENGINE_ROOT),
        env=_env(),
    )
    print(f"[prep] ready: {MODEL_REPO} -> {MODEL_DIR}")


def say(
    text: str,
    output: str = "output.wav",
    ref: str = "",
    emotion: str = "",
    emo_alpha: float = 0.6,
    speed: float = 1.0,
    seed: int = 0,
    play: bool = False,
    force: bool = False,
) -> None:
    """Generate speech with IndexTTS-2 on MLX.

    Args:
        text: Text to synthesize (zh or en).
        output: Output WAV path.
        ref: Reference speaker WAV — required; IndexTTS-2 has no preset voices.
        emotion: Optional explicit emotion (happy, angry, sad, afraid, disgusted,
            melancholic, surprised, calm).
        emo_alpha: Strength of the explicit emotion, 0-1.
        speed: Speaking-rate multiplier.
        seed: Fixed sampling seed; 0 leaves the engine default.
        play: Play the result with afplay instead of reporting the path.
        force: Overwrite an existing output file.
    """
    from rich.console import Console
    from rich.panel import Panel

    console = Console()
    _require_engine()
    out_path = Path(output).resolve()

    if not force and out_path.exists() and out_path.stat().st_size > 0:
        console.print(
            f"[yellow]Skipped:[/yellow] {output} already exists (use --force to overwrite)"
        )
        return
    if not ref:
        raise SystemExit("IndexTTS-2 needs a reference clip: pass --ref path/to/voice.wav")
    ref_path = Path(ref).expanduser().resolve()
    if not ref_path.is_file():
        raise SystemExit(f"Reference audio not found: {ref}")
    if emotion and emotion not in EMOTIONS:
        raise SystemExit(f"unknown emotion '{emotion}'. Choose from: {', '.join(EMOTIONS)}")
    if not MODEL_DIR.exists():
        prep()

    # `einops` is imported by the MaskGCT semantic codec but missing from the
    # project's own dependency set, so it is supplied per-run.
    cmd = [
        "uv", "run", "--with", "einops", "mlx-indextts", "generate",
        "-m", str(MODEL_DIR),
        "-r", str(ref_path),
        "-t", text,
        "-o", str(out_path),
        "--speed", str(speed),
    ]
    if emotion:
        cmd += ["--emotion", emotion, "--emo-alpha", str(emo_alpha)]
    if seed:
        cmd += ["--seed", str(seed)]

    console.print(
        Panel.fit(
            f"[bold cyan]IndexTTS-2 (MLX)[/bold cyan] v{VERSION}\n"
            f"repo=[yellow]{MODEL_REPO}[/yellow]",
            border_style="cyan",
        )
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=str(ENGINE_ROOT), env=_env()
    )
    if result.returncode != 0 or not out_path.exists():
        lines = [
            ln
            for ln in (result.stdout + "\n" + result.stderr).splitlines()
            if ln.strip() and "UserWarning" not in ln
        ]
        console.print("[red]mlx-indextts failed[/red]")
        console.print("\n".join(lines[-12:]))
        raise SystemExit(1)

    if play:
        subprocess.run(["afplay", str(out_path)], check=True)
    else:
        console.print(f"[green]Saved[/green] [bold]{out_path}[/bold]")


def info() -> None:
    """Print one-line JSON with model metadata."""
    print(json.dumps({
        "name": "indextts2-mlx",
        "repo": MODEL_REPO,
        "backend": "mlx",
        "quant": "fp32",
        "languages": ["zh", "en"],
        "sample_rate_hz": 22050,
        "modes": ["ivc"],
        "capabilities": ["voice_cloning", "emotion_control"],
    }, separators=(",", ":")))


if __name__ == "__main__":
    import fire

    fire.Fire({"prep": prep, "say": say, "info": info})
