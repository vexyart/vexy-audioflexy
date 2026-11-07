#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = ["fire", "huggingface-hub", "mlx-audio @ git+https://github.com/Blaizzy/mlx-audio.git", "numpy", "rich", "soundfile"]
# ///
# this_file: src/vexy_audioflexy/tools/tts/tools/tts-miso-mlx.py
"""MisoTTS MLX — MisoLabs voice-clone TTS via mlx-audio Sesame-compatible loader. Backend: MLX (Apple Silicon). Languages: en. Sample rate: 24000 Hz. Modes: ivc (identity voice cloning via ref audio). Variants: 8bit (mlx-community/MisoLabs-MisoTTS-8bit), bf16 (mlx-community/MisoLabs-MisoTTS-bf16). Capabilities: voice_cloning."""

from pathlib import Path
from typing import Any
import importlib
import json

VERSION = "1.0.0"
MODEL_VARIANTS = {
    "8bit": "mlx-community/MisoLabs-MisoTTS-8bit",
    "bf16": "mlx-community/MisoLabs-MisoTTS-bf16",
}
DEFAULT_VARIANT = "8bit"


def _validate_ref(ref: str | None) -> None:
    if not ref:
        return
    ref_path = Path(ref).expanduser()
    if not ref_path.exists() or not ref_path.is_file():
        raise SystemExit(f"Reference audio not found: {ref}")


def _verify_sesame_config(model_repo: str) -> None:
    from huggingface_hub import hf_hub_download

    config_path = hf_hub_download(model_repo, "config.json")
    config = json.loads(Path(config_path).read_text())
    if config.get("model_type") != "sesame":
        raise SystemExit(
            f"{model_repo} config.json has model_type={config.get('model_type')!r}; "
            "this wrapper only supports mlx-audio Sesame-compatible models."
        )


def _audio_from_results(results: list[Any]) -> tuple[Any, int]:
    import numpy as np

    if not results:
        raise SystemExit("MisoTTS returned no audio.")
    chunks = [np.array(result.audio, copy=False) for result in results]
    audio = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
    sample_rate = int(getattr(results[0], "sample_rate", 24000))
    return audio, sample_rate


def prep(force: bool = False) -> None:
    """Download and cache all MisoTTS MLX model weights without synthesizing audio. Idempotent."""
    import importlib as _importlib

    from huggingface_hub import snapshot_download

    model_repo = MODEL_VARIANTS[DEFAULT_VARIANT]
    _verify_sesame_config(model_repo)
    load_model = getattr(_importlib.import_module("mlx_audio.tts.utils"), "load_model")
    load_model(model_repo)
    print(f"[prep] ready: {model_repo}")


def say(
    text: str,
    output: str = "output.wav",
    variant: str = DEFAULT_VARIANT,
    model: str | None = None,
    ref: str | None = None,
    ref_text: str | None = None,
    play: bool = False,
    force: bool = False,
) -> None:
    """Generate speech with MisoTTS MLX through mlx-audio load_model.

    Args:
        text: Text to synthesize.
        output: Output WAV path.
        variant: Explicit model variant: "8bit", "bf16", or a full HF repo ID.
        model: Alias for variant, used by tts_all.py.
        ref: Optional reference WAV if the installed mlx-audio Sesame generator supports it.
        ref_text: Optional reference transcript for compatible generators.
        play: Play with afplay instead of keeping output.
        force: Overwrite output when it exists.
    """
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn

    console = Console()
    out_path = Path(output)
    if not play and not force and out_path.exists() and out_path.stat().st_size > 0:
        console.print(
            f"[yellow]Skipped:[/yellow] {output} already exists (use --force to overwrite)"
        )
        return
    _validate_ref(ref)

    selected_variant = model or variant
    model_repo = MODEL_VARIANTS.get(selected_variant, selected_variant)
    _verify_sesame_config(model_repo)

    import soundfile as sf

    load_model = getattr(importlib.import_module("mlx_audio.tts.utils"), "load_model")
    console.print(f"[dim]Model: {model_repo}[/dim]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading MisoTTS MLX...", total=None)
        model = load_model(model_repo)
        progress.update(task, description="Generating speech...")
        kwargs: dict[str, Any] = {"text": text}
        if ref:
            kwargs["ref_audio"] = ref
        if ref_text:
            kwargs["ref_text"] = ref_text
        try:
            results = list(model.generate(**kwargs))
        except TypeError as exc:
            raise SystemExit(
                "Installed mlx-audio does not accept the MisoTTS generation arguments. "
                "Upgrade mlx-audio from Blaizzy/mlx-audio and verify Sesame model support."
            ) from exc
        progress.update(task, description="Done.")

    audio, sample_rate = _audio_from_results(results)

    if play:
        import subprocess
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, audio, sample_rate)
            tmp_path = Path(tmp.name)
        subprocess.run(["afplay", str(tmp_path)], check=True)
        tmp_path.unlink(missing_ok=True)
        console.print("[green]Done.[/green]")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), audio, sample_rate)
    console.print(f"[green]Saved:[/green] {out_path.resolve()} ({sample_rate} Hz)")


def info() -> None:
    """Print one-line JSON with model metadata."""
    import json as _json

    print(_json.dumps({
        "name": "MisoTTS MLX",
        "repo": list(MODEL_VARIANTS.values()),
        "backend": "mlx",
        "languages": ["en"],
        "sample_rate_hz": 24000,
        "modes": ["ivc"],
        "capabilities": ["voice_cloning"],
    }))


if __name__ == "__main__":
    import fire
    fire.Fire({"prep": prep, "say": say, "info": info})
