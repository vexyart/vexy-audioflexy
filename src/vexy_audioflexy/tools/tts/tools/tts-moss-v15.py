#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = ["fire", "rich", "soundfile", "torch", "torchaudio", "torchcodec", "transformers>=5.0.0rc0", "huggingface-hub"]
# ///
# this_file: src/vexy_audioflexy/tools/tts/tools/tts-moss-v15.py
"""MOSS-TTS v1.5 voice-cloning TTS via Transformers AutoModel/AutoProcessor. Backend: PyTorch (fp32/bfloat16); languages: en, pl; mode: ivc (instant voice cloning); sample rate from model config (~24 kHz); supports inline pause tags e.g. [pause 1.2s]."""

from pathlib import Path
import importlib.util
import json
import os
import sys


def _ensure_ffmpeg_on_dyld_path() -> None:
    """Re-exec once with Homebrew's lib dir on the dyld fallback path.

    torchaudio 2.9 dropped its own decoders and now loads audio through
    torchcodec, whose core libraries link against FFmpeg 4/5/6 (``libavutil.56``
    and friends). Those live in Homebrew's prefix, which is not on the default
    dyld search path, so the dlopen fails and every read of the reference WAV
    dies with "Could not load this library". dyld reads its environment once at
    process start, so setting the variable in-process is too late — re-exec.
    """
    hint = "/opt/homebrew/lib"
    if not Path(hint).is_dir():
        return
    current = os.environ.get("DYLD_FALLBACK_LIBRARY_PATH", "")
    if hint in current.split(":"):
        return
    os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = ":".join(
        [p for p in (current, hint) if p]
    )
    os.execv(sys.executable, [sys.executable, *sys.argv])

VARIANTS: dict[str, str] = {
    "v1.5": "OpenMOSS-Team/MOSS-TTS-v1.5",
    "laion-voiceacting-4.55b": (
        "laion/moss-tts-local-transformer-4.55b-voice-acting-v2-sft3"
    ),
}
DEFAULT_VARIANT = "v1.5"
REPO = VARIANTS[DEFAULT_VARIANT]


def _repo(variant: str) -> str:
    try:
        return VARIANTS[variant]
    except KeyError:
        raise SystemExit(
            f"unknown variant '{variant}'. Choose from: {', '.join(VARIANTS)}"
        ) from None
VERSION = "1.0.0"


def _resolve_attn_implementation(device: str, dtype) -> str:
    import torch

    if (
        device == "cuda"
        and importlib.util.find_spec("flash_attn") is not None
        and dtype in {torch.float16, torch.bfloat16}
    ):
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            return "flash_attention_2"
    if device == "cuda":
        return "sdpa"
    return "eager"


def _skip_existing(output: str, force: bool, play: bool) -> bool:
    out_path = Path(output)
    return not play and not force and out_path.exists() and out_path.stat().st_size > 0


def prep(variant: str = DEFAULT_VARIANT, force: bool = False):
    """Download + cache all model weights/assets WITHOUT synthesizing. Idempotent."""
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from transformers import AutoModel, AutoProcessor

    console = Console()
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as progress:
        task = progress.add_task("Downloading processor...", total=None)
        repo = _repo(variant)
        AutoProcessor.from_pretrained(repo, trust_remote_code=True)
        progress.update(task, description="Downloading model weights...")
        AutoModel.from_pretrained(repo, trust_remote_code=True)
    print(f"[prep] ready: {repo}")


def say(
    text: str,
    output: str = "output.wav",
    ref: str | None = None,
    language: str | None = None,
    variant: str = DEFAULT_VARIANT,
    tokens: int | None = None,
    max_new_tokens: int = 4096,
    play: bool = False,
    force: bool = False,
):
    """Generate speech with OpenMOSS-Team/MOSS-TTS-v1.5.

    Args:
        text: Text to synthesize. MOSS supports inline pause tags like [pause 1.2s].
        output: Output WAV path.
        ref: Optional reference audio path/URL for voice cloning.
        language: Optional model-card language hint, e.g. English, French, Polish.
        variant: Which MOSS-TTS-Local checkpoint to run (`v1.5` or
            `laion-voiceacting-4.55b`, LAION's expressive voice-acting fine-tune).
        tokens: Optional duration control token count.
        max_new_tokens: Maximum generated tokens.
        play: Play with afplay instead of only saving.
        force: Overwrite output if it already exists.
    """
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn

    console = Console()
    if _skip_existing(output, force, play):
        console.print(f"[yellow]Skipped:[/yellow] {output} already exists (use --force)")
        return

    if ref and "://" not in ref and not Path(ref).exists():
        console.print(f"[red]Reference audio not found:[/red] {ref}")
        raise SystemExit(1)

    import torch
    import torchaudio
    from transformers import AutoModel, AutoProcessor

    if torch.cuda.is_available():
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    attn_implementation = _resolve_attn_implementation(device, dtype)

    repo = _repo(variant)
    console.print(f"[dim]Model: {repo}[/dim]")
    console.print(f"[dim]Device: {device}; attention: {attn_implementation}[/dim]")

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as progress:
        task = progress.add_task("Loading processor...", total=None)
        processor = AutoProcessor.from_pretrained(repo, trust_remote_code=True)
        if hasattr(processor, "audio_tokenizer"):
            processor.audio_tokenizer = processor.audio_tokenizer.to(device)

        progress.update(task, description="Loading model...")
        model = AutoModel.from_pretrained(
            repo,
            trust_remote_code=True,
            attn_implementation=attn_implementation,
            torch_dtype=dtype,
        ).to(device)
        model.eval()

        build_kwargs = {"text": text}
        if language:
            build_kwargs["language"] = language
        if ref:
            build_kwargs["reference"] = [ref]
        if tokens is not None:
            build_kwargs["tokens"] = tokens

        progress.update(task, description="Generating audio...")
        conversation = [[processor.build_user_message(**build_kwargs)]]
        batch = processor(conversation, mode="generation")
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
            )

        decoded = list(processor.decode(outputs))

    if not decoded or not decoded[0].audio_codes_list:
        console.print("[red]MOSS-TTS returned no audio.[/red]")
        raise SystemExit(1)

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    audio = decoded[0].audio_codes_list[0].cpu()
    # v1.5 hands back a bare sample vector; the LAION 4.55B fine-tune already
    # includes a channel dimension. torchaudio.save takes 1D or 2D only.
    audio = audio.reshape(-1) if audio.dim() > 2 else audio
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    sample_rate = int(processor.model_config.sampling_rate)
    torchaudio.save(str(out_path), audio, sample_rate)

    if play:
        import subprocess

        subprocess.run(["afplay", str(out_path)], check=True)

    console.print(f"[green]Saved:[/green] {out_path.resolve()} ({sample_rate} Hz)")


def info():
    """Print one-line JSON: name, repo, backend, languages, sample_rate_hz, modes, capabilities."""
    print(json.dumps({
        "name": "MOSS TTS v1.5",
        "repo": REPO,
        "variants": VARIANTS,
        "backend": "transformers",
        "languages": ["en", "pl"],
        "sample_rate_hz": 24000,
        "modes": ["ivc"],
        "capabilities": ["voice_cloning", "pause_tags"],
    }))


if __name__ == "__main__":
    _ensure_ffmpeg_on_dyld_path()
    import fire
    fire.Fire({"prep": prep, "say": say, "info": info})
