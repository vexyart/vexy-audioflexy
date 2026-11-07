#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = ["fire", "rich", "huggingface-hub"]
# ///
# this_file: src/vexy_audioflexy/tools/tts/tools/tts-magpie-gguf.py
"""Magpie TTS (NVIDIA NeMo port) via the external magpie-tts.cpp ggml binary.

Backend: ggml (CPU/Metal). Repo: m1el/magpie-tts-multilingual-357m-gguf.
Languages: multilingual (en used here). Sample rate: 22050 Hz mono. Mode: auto
(preset speaker IDs, no voice cloning). The binary is built separately from
github.com/m1el/magpie-tts.cpp; say() shells out to it. Because the binary is
linked with a relative rpath to its bundled ggml libs, it is invoked with its
own directory as the working directory.
"""

from pathlib import Path
import os
import shutil
import subprocess
from _paths import runtime_path, bind_snapshot

VERSION = "1.0.0"
REPO = "m1el/magpie-tts-multilingual-357m-gguf"

# Repo root is two levels up from tools/; the magpie-tts.cpp checkout lives under
# .external/ with its compiled binary and downloaded GGUF weights.
ROOT = Path(__file__).resolve().parent.parent
_MAGPIE_DIR = runtime_path('magpie-tts.cpp')

DEFAULT_MODEL = "weights/magpie-357m-f32.gguf"
DEFAULT_CODEC = "weights/nano-codec-f32.gguf"


def _find_binary(binary: str | None) -> Path:
    candidates = [
        binary,
        os.environ.get("MAGPIE_TTS_BIN"),
        str(_MAGPIE_DIR / "magpie-tts"),
        shutil.which("magpie-tts"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if path.exists() and os.access(path, os.X_OK):
            return path.resolve()
    raise SystemExit(
        "magpie-tts binary not found.\n"
        "Expected: a built magpie-tts.cpp executable on PATH or via --binary.\n"
        "Upstream project: https://github.com/m1el/magpie-tts.cpp\n"
        "Build route:\n"
        "  git clone https://github.com/m1el/magpie-tts.cpp\n"
        "  cd magpie-tts.cpp && git clone https://github.com/ggml-org/ggml.git\n"
        "  cmake -S ggml -B ggml/build -DGGML_METAL=ON -DBUILD_SHARED_LIBS=ON\n"
        "  make -C ggml/build -j8 && make magpie-tts\n"
        "Then download weights from https://huggingface.co/" + REPO
    )


def _resolve_weight(value: str, binary_dir: Path, label: str) -> Path:
    """Resolve a weight path: absolute as-is, else relative to the binary dir."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = binary_dir / path
    if not path.exists():
        raise SystemExit(
            f"{label} not found: {path}\n"
            f"Download weights with:\n"
            f"  uv run --script {Path(__file__).resolve()} prep"
        )
    return path.resolve()


def prep(force: bool = False) -> None:
    """Download + cache the GGUF weights into the binary's weights/ dir. Idempotent."""
    from huggingface_hub import snapshot_download

    binary = _find_binary(None)
    weights_dir = binary.parent / "weights"
    print(f"[prep] downloading {REPO} -> {weights_dir}")
    snapshot = Path(snapshot_download(repo_id=REPO, force_download=force))
    bind_snapshot(snapshot, weights_dir)
    print(f"[prep] ready: {REPO}")


def say(
    text: str,
    output: str = "output.wav",
    speaker: int = 0,
    model: str = DEFAULT_MODEL,
    codec: str = DEFAULT_CODEC,
    temperature: float = 0.7,
    top_k: int = 80,
    binary: str | None = None,
    play: bool = False,
    force: bool = False,
) -> None:
    """Generate speech with Magpie TTS via the magpie-tts.cpp binary (22050 Hz).

    Args:
        text: Text to synthesize.
        output: Output WAV path (22050 Hz mono).
        speaker: Preset speaker ID (default 0).
        model: TTS GGUF path (absolute, or relative to the binary's dir).
        codec: Nano-codec GGUF path (absolute, or relative to the binary's dir).
        temperature: Sampling temperature (0 = deterministic).
        top_k: Top-k sampling.
        binary: Path to the magpie-tts binary. Also read from MAGPIE_TTS_BIN.
        play: Play the result via afplay (macOS) instead of keeping only the file.
        force: Overwrite output when it exists.
    """
    from rich.console import Console

    console = Console()
    out_path = Path(output).expanduser().resolve()
    if not force and out_path.exists() and out_path.stat().st_size > 0:
        console.print(
            f"[yellow]Skipped:[/yellow] {output} already exists (use --force to overwrite)"
        )
        return

    binary_path = _find_binary(binary)
    binary_dir = binary_path.parent
    model_path = _resolve_weight(model, binary_dir, "TTS model GGUF")
    codec_path = _resolve_weight(codec, binary_dir, "Codec GGUF")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(binary_path),
        "-m",
        str(model_path),
        "-c",
        str(codec_path),
        "-t",
        text,
        "-o",
        str(out_path),
        "-s",
        str(speaker),
        "--temp",
        str(temperature),
        "--top-k",
        str(top_k),
    ]
    console.print(f"[dim]Repo: {REPO}[/dim]")
    console.print(f"[dim]Command: {' '.join(cmd)}[/dim]")
    try:
        # cwd = binary dir so the relative @rpath to bundled ggml libs resolves.
        subprocess.run(cmd, cwd=str(binary_dir), check=True)
    except FileNotFoundError as exc:
        raise SystemExit(
            f"magpie-tts binary vanished before execution: {binary_path}"
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"magpie-tts failed with exit code {exc.returncode}.") from exc

    if not out_path.exists() or out_path.stat().st_size == 0:
        raise SystemExit(
            f"magpie-tts completed but did not create a non-empty output: {out_path}"
        )
    console.print(f"[green]Saved:[/green] {out_path} (22050 Hz)")

    if play:
        subprocess.run(["afplay", str(out_path)], check=False)


def info() -> None:
    """Print one-line JSON: name, repo, backend, languages, sample_rate_hz, modes, capabilities."""
    import json

    print(
        json.dumps(
            {
                "name": "magpie-gguf",
                "repo": REPO,
                "backend": "gguf",
                "languages": ["en"],
                "sample_rate_hz": 22050,
                "modes": ["auto"],
                "capabilities": ["preset_voices"],
            }
        )
    )


if __name__ == "__main__":
    import fire

    fire.Fire({"prep": prep, "say": say, "info": info})
