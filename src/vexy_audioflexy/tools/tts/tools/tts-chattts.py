#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "fire", "rich", "chattts", "torch", "torchaudio", "soundfile", "numpy",
#   "requests", "vector-quantize-pytorch", "vocos",
# ]
# ///
# this_file: src/vexy_audioflexy/tools/tts/tools/tts-chattts.py
"""ChatTTS. Backend: torch/chattts, languages: en/zh, 24 kHz mono, modes: auto, capabilities: none.

A conversational model: it samples a speaker embedding rather than offering presets
or cloning one, so the same text twice gives two different voices unless the sample
is seeded. `--seed` fixes the speaker, which is what makes it usable in a benchmark
that compares runs.

`refine` is on by default upstream — a text-refinement pass that inserts prosody
tokens (`[uv_break]`, `[laugh]`). It rewrites the prompt, which is wrong for a
harness scoring what was asked for against what came out, so it is off here and
opt-in via `--refine`.

Four dependencies the PyPI package needs and does not declare: `requests`,
`vector-quantize-pytorch`, `vocos` and `torchaudio`. Without them the import fails
with `ModuleNotFoundError: No module named 'requests'` from inside its own asset
downloader.

English and Chinese only — no Polish.
"""

from __future__ import annotations

import json
from pathlib import Path

VERSION = "1.0.0"
LANGS = ["en", "zh"]
REPO = "2noise/ChatTTS"


def _chat(device: str = "cpu"):
    import ChatTTS

    chat = ChatTTS.Chat()
    if not chat.load(compile=False, device=device):
        raise SystemExit("ChatTTS.load() failed — assets missing or corrupt")
    return chat


def prep(force: bool = False) -> None:
    """Download + cache the model assets WITHOUT synthesizing. Idempotent.

    Args:
        force: Unused; cached assets are never re-fetched.
    """
    _chat()
    print(f"[prep] ready: {REPO}")


def say(
    text: str,
    output: str = "output.wav",
    seed: int = 42,
    temperature: float = 0.3,
    top_p: float = 0.7,
    top_k: int = 20,
    refine: bool = False,
    lang: str = "en",
    ref: str = "",
    device: str = "cpu",
    play: bool = False,
    force: bool = False,
) -> None:
    """Synthesize `text` with a seeded random speaker.

    Args:
        text: Text to synthesize.
        output: Output WAV path (24 kHz mono).
        seed: Speaker seed. The same seed gives the same voice; change it for a
            different one.
        temperature: Sampling temperature for the acoustic codes.
        top_p: Nucleus sampling cutoff.
        top_k: Top-k sampling cutoff.
        refine: Run the upstream text-refinement pass, which inserts prosody tokens
            and rewrites the prompt. Off by default; see the module docstring.
        lang: `en` or `zh`. Checked, not forwarded — the model has no language flag.
        ref: Rejected. ChatTTS samples a speaker; it does not clone one.
        device: `cpu` (default) or `mps`.
        play: Play the result with afplay instead of just saving it.
        force: Overwrite an existing output file.
    """
    import time

    import numpy as np
    import soundfile as sf
    import torch
    from rich.console import Console
    from rich.panel import Panel

    console = Console()
    out_path = Path(output).resolve()
    if not force and out_path.exists() and out_path.stat().st_size > 0:
        console.print(f"[yellow]Skipped:[/yellow] {output} already exists (use --force)")
        return
    if ref:
        raise SystemExit(
            "ChatTTS samples a speaker embedding; it cannot clone "
            f"{ref}. Use --seed to pick a reproducible voice."
        )
    if lang not in LANGS:
        raise SystemExit(f"ChatTTS speaks {'/'.join(LANGS)}, not '{lang}'")

    console.print(
        Panel.fit(
            f"[bold cyan]ChatTTS[/bold cyan] v{VERSION}  seed=[yellow]{seed}[/yellow]  "
            f"lang=[yellow]{lang}[/yellow]  refine=[yellow]{refine}[/yellow]",
            border_style="cyan",
        )
    )

    chat = _chat(device)
    torch.manual_seed(seed)
    speaker = chat.sample_random_speaker()
    params = chat.InferCodeParams(
        spk_emb=speaker, temperature=temperature, top_P=top_p, top_K=top_k
    )

    t0 = time.monotonic()
    wavs = chat.infer(
        [text], params_infer_code=params, skip_refine_text=not refine, use_decoder=True
    )
    wall = time.monotonic() - t0

    if not wavs:
        console.print("[red]ChatTTS returned no audio[/red]")
        raise SystemExit(1)
    samples = np.asarray(wavs[0], dtype="float32").reshape(-1)
    if samples.size == 0 or not np.any(samples):
        console.print("[red]ChatTTS produced silence[/red]")
        raise SystemExit(1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), samples, 24000, subtype="PCM_16")
    duration = samples.size / 24000
    console.print(
        f"[green]Saved[/green] [bold]{out_path}[/bold] "
        f"({duration:.1f}s @ 24000 Hz, rtf={wall / duration:.2f})"
    )
    if play:
        import subprocess

        subprocess.run(["afplay", str(out_path)], check=True)


def info() -> None:
    """Print one-line JSON with model metadata."""
    print(json.dumps(
        {
            "name": "chattts",
            "repo": REPO,
            "stack": "torch",
            "runtime": "ChatTTS (PyTorch, CPU)",
            "quant": "fp32",
            "mode": "auto",
            "voice_clone": False,
            "langs": LANGS,
            "sample_rate": 24000,
            "note": "speaker is sampled from a seed, not chosen from presets",
        },
        separators=(",", ":"),
    ))


if __name__ == "__main__":
    import fire

    fire.Fire({"prep": prep, "say": say, "info": info})
