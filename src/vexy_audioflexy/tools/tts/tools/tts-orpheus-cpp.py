#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "fire", "rich", "numpy>=2.0", "soundfile", "onnxruntime", "huggingface-hub",
#   "llama-cpp-python",
# ]
# ///
# this_file: src/vexy_audioflexy/tools/tts/tools/tts-orpheus-cpp.py
"""Orpheus TTS via llama.cpp GGUF plus an ONNX SNAC decoder.

Variants (`--variant`):
  ft          isaiahbjork/orpheus-3b-0.1-ft-Q4_K_M-GGUF — the released Orpheus 3B
              fine-tune (default; verified intelligible)
  voiceacting mradermacher/orpheus-voiceacting-grpo-GGUF (Q8_0), a static quant of
              mrfakename/orpheus-voiceacting-grpo. See the caveat below.

English. 24 kHz mono. Mode: auto (preset voices; no cloning).
Emotive tags in text: <laugh>, <sigh>, <chuckle>, <gasp>, ...

Caveat on `voiceacting`: under the documented Orpheus prompt format it never emits
an end token and its audio codes decode to near-silence, with `tara`, `leah`,
`narrator` and a voiceless prompt all behaving the same way. The same code path
renders the `ft` checkpoint correctly, so this is the checkpoint, not the wrapper;
its repo publishes no inference template or voice list to correct the prompt with.

Orpheus is an autoregressive LM that emits `<custom_token_N>` ids rather than audio.
Every 7 consecutive ids form one SNAC frame across three codebooks, and the frames
are decoded to waveform by SNAC 24 kHz. This script runs the LM with
llama-cpp-python and the decoder with onnx-community/snac_24khz-ONNX, which is the
same split the upstream `orpheus-cpp` package uses — but pointed at an arbitrary
GGUF instead of that package's hardcoded per-language repos.

Note: CrispASR also has an `orpheus` backend, but it binds `talker.*`-prefixed
tensors from cstr's own conversion and cannot load a plain llama.cpp quant.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

VERSION = "2.0.0"
VARIANTS: dict[str, dict[str, str]] = {
    "ft": {
        "repo": "isaiahbjork/orpheus-3b-0.1-ft-Q4_K_M-GGUF",
        "file": "orpheus-3b-0.1-ft-q4_k_m.gguf",
        "quant": "q4_k_m",
    },
    "voiceacting": {
        "repo": "mradermacher/orpheus-voiceacting-grpo-GGUF",
        "file": "orpheus-voiceacting-grpo.Q8_0.gguf",
        "quant": "q8_0",
    },
}
DEFAULT_VARIANT = "ft"
REPO = VARIANTS[DEFAULT_VARIANT]["repo"]


def _variant(name: str) -> dict[str, str]:
    try:
        return VARIANTS[name]
    except KeyError:
        raise SystemExit(
            f"unknown variant '{name}'. Choose from: {', '.join(VARIANTS)}"
        ) from None
SNAC_REPO = "onnx-community/snac_24khz-ONNX"
SNAC_FILE = "onnx/decoder_model.onnx"

CUSTOM_TOKEN_PREFIX = "<custom_token_"
SAMPLE_RATE = 24_000
# One SNAC frame is 7 ids; the decoder is fed a sliding window of 4 frames.
IDS_PER_FRAME = 7
WINDOW_FRAMES = 4


def _model_path(variant: str = DEFAULT_VARIANT) -> Path:
    from huggingface_hub import hf_hub_download

    spec = _variant(variant)
    return Path(hf_hub_download(spec["repo"], spec["file"]))


def _snac_path() -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(SNAC_REPO, SNAC_FILE))


def prep(variant: str = DEFAULT_VARIANT, force: bool = False) -> None:
    """Download and cache the Orpheus GGUF and the SNAC decoder. Idempotent."""
    model = _model_path(variant)
    _snac_path()
    print(f"[prep] ready: {_variant(variant)['repo']} ({model.name}) + {SNAC_REPO}")


def _token_to_id(token_text: str, index: int) -> int | None:
    """Map one `<custom_token_N>` to its codebook-relative SNAC id."""
    token_string = token_text.strip()
    start = token_string.rfind(CUSTOM_TOKEN_PREFIX)
    if start == -1:
        return None
    last = token_string[start:]
    if not last.endswith(">"):
        return None
    try:
        # Ids are offset by 10, then strided by 4096 per position in the frame.
        return int(last[len(CUSTOM_TOKEN_PREFIX) : -1]) - 10 - ((index % IDS_PER_FRAME) * 4096)
    except ValueError:
        return None


def _frames_to_codes(ids: list[int]):
    """Split a flat id stream into SNAC's three codebooks."""
    import numpy as np

    frames = len(ids) // IDS_PER_FRAME
    flat = ids[: frames * IDS_PER_FRAME]
    c0, c1, c2 = [], [], []
    for j in range(frames):
        i = IDS_PER_FRAME * j
        c0.append(flat[i])
        c1.extend([flat[i + 1], flat[i + 4]])
        c2.extend([flat[i + 2], flat[i + 3], flat[i + 5], flat[i + 6]])
    codes = [
        np.array([c0], dtype=np.int64),
        np.array([c1], dtype=np.int64),
        np.array([c2], dtype=np.int64),
    ]
    if any((c < 0).any() or (c > 4096).any() for c in codes):
        return None
    return codes


def say(
    text: str,
    output: str = "output.wav",
    variant: str = DEFAULT_VARIANT,
    voice: str = "tara",
    max_tokens: int = 2048,
    temperature: float = 0.8,
    top_p: float = 0.95,
    top_k: int = 40,
    min_p: float = 0.05,
    n_gpu_layers: int = -1,
    play: bool = False,
    force: bool = False,
) -> None:
    """Generate speech with Orpheus.

    Args:
        text: Text to synthesize; supports emotive tags such as <laugh> and <sigh>.
        output: Output WAV path (24 kHz mono).
        variant: Which published Orpheus GGUF to run (see module docstring).
        voice: Preset voice name baked into the checkpoint (Orpheus default `tara`).
        max_tokens: Cap on generated LM tokens.
        temperature: LM sampling temperature.
        top_p: Nucleus sampling limit.
        top_k: Top-k sampling limit.
        min_p: Minimum-probability sampling floor.
        n_gpu_layers: Layers offloaded to Metal; -1 offloads all.
        play: Play the result with afplay instead of reporting the path.
        force: Overwrite an existing output file.
    """
    import numpy as np
    import onnxruntime as ort
    import soundfile as sf
    from llama_cpp import Llama
    from rich.console import Console
    from rich.panel import Panel

    console = Console()
    out_path = Path(output)
    if not force and out_path.exists() and out_path.stat().st_size > 0:
        console.print(
            f"[yellow]Skipped:[/yellow] {output} already exists (use --force to overwrite)"
        )
        return

    console.print(
        Panel.fit(
            f"[bold cyan]Orpheus (llama.cpp + SNAC)[/bold cyan] v{VERSION}\n"
            f"repo=[yellow]{_variant(variant)['repo']}[/yellow]  "
            f"voice=[yellow]{voice}[/yellow]",
            border_style="cyan",
        )
    )

    with console.status("[cyan]Loading model...[/cyan]"):
        llm = Llama(
            model_path=str(_model_path(variant)),
            n_ctx=0,
            verbose=False,
            n_gpu_layers=n_gpu_layers,
            batch_size=1,
        )
        snac = ort.InferenceSession(str(_snac_path()), providers=["CPUExecutionProvider"])
    snac_inputs = [i.name for i in snac.get_inputs()]

    prompt = f"<|audio|>{voice}: {text}<|eot_id|><custom_token_4>"
    ids: list[int] = []
    with console.status("[cyan]Generating audio tokens...[/cyan]"):
        stream = llm(
            prompt,
            max_tokens=max_tokens,
            stream=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
        )
        for chunk in stream:
            token_id = _token_to_id(chunk["choices"][0]["text"], len(ids))
            if token_id is not None and token_id > 0:
                ids.append(token_id)

    if len(ids) < IDS_PER_FRAME * WINDOW_FRAMES:
        console.print(f"[red]Orpheus produced only {len(ids)} audio tokens — no audio.[/red]")
        raise SystemExit(1)

    # Decode in sliding windows and keep each window's middle segment, matching
    # the upstream streaming decoder's 2048:4096 sample slice.
    pieces: list[np.ndarray] = []
    window = IDS_PER_FRAME * WINDOW_FRAMES
    for end in range(window, len(ids) + 1, IDS_PER_FRAME):
        codes = _frames_to_codes(ids[end - window : end])
        if codes is None:
            continue
        audio = snac.run(None, dict(zip(snac_inputs, codes)))[0]
        pieces.append(np.asarray(audio, dtype=np.float32).reshape(-1)[2048:4096])

    if not pieces:
        console.print("[red]SNAC produced no audio frames.[/red]")
        raise SystemExit(1)

    audio = np.concatenate(pieces)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), audio, SAMPLE_RATE)

    if play:
        subprocess.run(["afplay", str(out_path)], check=True)
    else:
        console.print(
            f"[green]Saved[/green] [bold]{out_path.resolve()}[/bold] "
            f"({SAMPLE_RATE} Hz, {len(audio) / SAMPLE_RATE:.1f}s)"
        )


def info(variant: str = DEFAULT_VARIANT) -> None:
    """Print one-line JSON with model metadata."""
    spec = _variant(variant)
    print(json.dumps({
        "name": f"orpheus-{variant}",
        "repo": spec["repo"],
        "backend": "gguf",
        "quant": spec["quant"],
        "languages": ["en"],
        "sample_rate_hz": SAMPLE_RATE,
        "modes": ["auto"],
        "capabilities": ["preset-voices", "emotive-tags"],
    }, separators=(",", ":")))


if __name__ == "__main__":
    import fire

    fire.Fire({"prep": prep, "say": say, "info": info})
