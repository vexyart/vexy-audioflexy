#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = ["fire", "rich", "omnivoice", "soundfile", "torch", "torchaudio", "numpy", "huggingface-hub"]
# ///
# this_file: src/vexy_audioflexy/tools/tts/tools/tts-omnivoice-torch.py
"""OmniVoice zero-shot TTS via PyTorch/MPS on Apple Silicon. Backend: omnivoice pip package
(Qwen3-0.6B backbone + iterative masked diffusion), repo k2-fsa/OmniVoice, 24 kHz mono,
600+ languages, mode: ivc (instant voice cloning via ref audio + optional ref_text)."""

from pathlib import Path

REPO = "k2-fsa/OmniVoice"
VERSION = "1.0.0"
MODEL = REPO


def prep(force: bool = False):
    """Download + cache all OmniVoice model weights without synthesizing audio. Idempotent."""
    import os

    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn

    console = Console()

    import torch

    _orig_torch_load = torch.load

    def _cpu_load(*args, **kwargs):
        kwargs.setdefault("map_location", "cpu")
        return _orig_torch_load(*args, **kwargs)

    torch.load = _cpu_load

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float32

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"Downloading/caching {REPO}...", total=None)
        from omnivoice import OmniVoice

        OmniVoice.from_pretrained(REPO, device_map=device, dtype=dtype)
        progress.update(task, description="Done.")

    print(f"[prep] ready: {REPO}")


def say(
    text: str,
    ref: str | None = None,
    ref_text: str | None = None,
    output: str = "output.wav",
    play: bool = False,
    force: bool = False,
):
    """Generate speech using OmniVoice (PyTorch) with optional voice cloning.

    Zero-shot voice cloning TTS supporting 600+ languages. 24kHz mono output.
    Uses MPS acceleration on Apple Silicon.

    Args:
        text: Text to synthesize.
        ref: Reference audio WAV for voice cloning (optional).
        ref_text: Transcript of reference audio (optional, improves cloning).
        output: Output file path (default: output.wav, 24kHz mono).
        play: Play audio via afplay instead of saving (macOS).
        force: Overwrite output file if it exists.
    """
    import os

    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn

    console = Console()

    if (
        not play
        and not force
        and Path(output).exists()
        and Path(output).stat().st_size > 0
    ):
        console.print(
            f"[yellow]Skipped:[/yellow] {output} already exists (use --force to overwrite)"
        )
        return

    if ref:
        ref_path = Path(ref)
        if not ref_path.exists():
            console.print(f"[red]Reference audio not found: {ref}[/red]")
            raise SystemExit(1)

    import torch
    import soundfile as sf

    # Monkey-patch torch.load to force CPU loading (weights may be saved from CUDA)
    _orig_torch_load = torch.load

    def _cpu_load(*args, **kwargs):
        kwargs.setdefault("map_location", "cpu")
        return _orig_torch_load(*args, **kwargs)

    torch.load = _cpu_load

    from omnivoice import OmniVoice

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float32  # MPS doesn't support all float16 ops

    console.print(f"[dim]Model: {MODEL} (device={device})[/dim]")
    if ref:
        console.print(f"[dim]Reference: {ref}[/dim]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading OmniVoice...", total=None)
        model = OmniVoice.from_pretrained(
            MODEL,
            device_map=device,
            dtype=dtype,
        )
        progress.update(task, description="Generating speech...")

        gen_kwargs = {"text": text}
        if ref:
            gen_kwargs["ref_audio"] = ref
        if ref_text:
            gen_kwargs["ref_text"] = ref_text

        audio_list = model.generate(**gen_kwargs)
        progress.update(task, description="Done.")

    if not audio_list:
        console.print("[red]Error: model returned no audio.[/red]")
        raise SystemExit(1)

    audio = audio_list[0]
    sample_rate = 24000

    if play:
        import subprocess
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, audio, sample_rate)
            tmp_path = tmp.name

        console.print(
            f"[cyan]Playing audio ({len(audio) / sample_rate:.1f}s)...[/cyan]"
        )
        subprocess.run(["afplay", tmp_path], check=True)
        Path(tmp_path).unlink(missing_ok=True)
        console.print("[green]Done.[/green]")
    else:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_path), audio, sample_rate)
        console.print(f"[green]Saved:[/green] {out_path.resolve()} ({sample_rate} Hz)")


def info():
    """Print one-line JSON with model metadata."""
    import json

    print(
        json.dumps(
            {
                "name": "OmniVoice",
                "repo": REPO,
                "backend": "torch",
                "languages": [
                    "en", "zh", "ja", "ko", "de", "fr", "es", "pt",
                    "it", "pl", "nl", "tr", "ru", "ar", "600+",
                ],
                "sample_rate_hz": 24000,
                "modes": ["ivc"],
                "capabilities": ["zero-shot-voice-cloning", "multilingual"],
            }
        )
    )


if __name__ == "__main__":
    import fire

    fire.Fire({"prep": prep, "say": say, "info": info})
