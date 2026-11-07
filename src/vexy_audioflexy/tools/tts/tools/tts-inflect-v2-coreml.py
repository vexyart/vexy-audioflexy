#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = ["fire", "rich", "huggingface-hub", "soundfile", "numpy", "coremltools", "phonemizer", "espeakng-loader", "num2words", "Unidecode"]
# ///
# this_file: src/vexy_audioflexy/tools/tts/tools/tts-inflect-v2-coreml.py
"""Inflect v2 CoreML wrapper (FluidInference/inflect-v2-coreml).
Supports both Micro (9.4M) and Nano (4.0M) CoreML model variants on Apple Silicon.
Backend: coreml (Apple Neural Engine / GPU via coremltools).
Languages: en. Sample rate: 24 kHz mono. Modes: auto. Capabilities: lightweight_tts.
"""

from pathlib import Path
import json
import os
import re
import sys

import fire

# Fallback HF_HOME if default ~/.cache/huggingface is a broken symlink
hf_cache = Path.home() / ".cache" / "huggingface"
if not os.environ.get("HF_HOME") and (hf_cache.is_symlink() and not hf_cache.exists()):
    os.environ["HF_HOME"] = str(Path.home() / ".cache" / "hf_local")

MODEL_REPO = "FluidInference/inflect-v2-coreml"
SAMPLE_RATE = 24000


def _get_model_dir() -> Path:
    from huggingface_hub import snapshot_download

    snapshot_path = snapshot_download(MODEL_REPO, token=False)
    return Path(snapshot_path)


def prep(force: bool = False) -> None:
    """Download and cache all model weights/assets without synthesizing. Idempotent."""
    from rich.console import Console

    console = Console()
    console.print(f"[cyan]Downloading/verifying {MODEL_REPO}…[/cyan]")
    model_dir = _get_model_dir()
    console.print(f"[green][prep] ready:[/green] {MODEL_REPO} ({model_dir})")


def say(
    text: str,
    output: str = "output.wav",
    variant: str = "micro",
    speed: float = 1.0,
    variation: float = 0.667,
    seed: int = 0,
    ref: str | None = None,
    play: bool = False,
    force: bool = False,
) -> None:
    """Generate speech using Inflect v2 CoreML (micro or nano).

    Args:
        text: Text to synthesize.
        output: Output WAV path.
        variant: "micro" or "nano".
        speed: Speed multiplier.
        variation: Noise scale variation.
        seed: Random seed.
        ref: Optional reference WAV (ignored for VITS preset).
        play: Play WAV via afplay after generation.
        force: Overwrite output file if it exists.
    """
    from rich.console import Console

    console = Console()
    out_path = Path(output)
    if not force and out_path.exists() and out_path.stat().st_size > 0:
        console.print(f"[yellow]Skipped:[/yellow] {output} already exists (use --force)")
        return

    model_dir = _get_model_dir()
    variant_dir = model_dir / variant
    if not variant_dir.exists():
        console.print(f"[red]Variant directory not found:[/red] {variant_dir}")
        raise SystemExit(1)

    import coremltools as ct
    import numpy as np
    import soundfile as sf
    from phonemizer.backend import EspeakBackend

    # Load symbol table
    symbols_file = model_dir / "symbols.json"
    if symbols_file.exists():
        loaded = json.loads(symbols_file.read_text())
        # The published table is an object with a "symbols" list, not a bare
        # list; enumerating the object yields its keys, which silently maps
        # almost every phoneme to the pad id and produces noise.
        symbols = loaded["symbols"] if isinstance(loaded, dict) else loaded
    else:
        # Keithito fallback symbols
        _pad = "_"
        _punctuation = "!'(),-.:;? "
        _letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        _ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɚʁʀʂʃʈʧʉʊʋⱱʌɣɤχʎʏʑʐʒʔʡʕʢǀǁǂǃ"
        symbols = [_pad] + list(_punctuation) + list(_letters) + list(_ipa)

    symbol_to_id = {s: i for i, s in enumerate(symbols)}

    # Phonemize text
    backend = EspeakBackend("en-us", preserve_punctuation=True, with_stress=True)

    def _load_coreml_model(model_path: Path):
        if model_path.suffix == ".mlpackage":
            compiled_dir = ct.models.MLModel.compileModel(str(model_path))
            return ct.models.CompiledMLModel(compiled_dir)
        return ct.models.CompiledMLModel(str(model_path))

    encoder_path = variant_dir / "encoder.mlmodelc"
    if not encoder_path.exists():
        encoder_path = variant_dir / "encoder.mlpackage"
    encoder_model = _load_coreml_model(encoder_path)

    synth_cache: dict[int, object] = {}

    def _synth_for(bucket: int):
        if bucket not in synth_cache:
            path = variant_dir / f"synthesizer_f{bucket}.mlmodelc"
            if not path.exists():
                path = variant_dir / f"synthesizer_f{bucket}.mlpackage"
            if not path.exists():
                path = variant_dir / "synthesizer_f1024.mlmodelc"
                if not path.exists():
                    path = variant_dir / "synthesizer_f1024.mlpackage"
            synth_cache[bucket] = _load_coreml_model(path)
        return synth_cache[bucket]

    # The encoder's token input is a fixed 512 wide, so a paragraph has to be
    # split or it is silently cut off mid-sentence. Chunks break on sentence
    # ends, and each is checked against the real token budget after
    # phonemization rather than guessed from character counts.
    def _tokens_for(chunk: str) -> np.ndarray:
        phonemes = backend.phonemize([chunk])[0]
        sequence = [symbol_to_id.get(char, 0) for char in phonemes]
        tokens = np.zeros(len(sequence) * 2 + 1, dtype=np.int32)
        tokens[1::2] = sequence
        return tokens

    sentences = [t for t in re.split(r"(?<=[.!?])\s+", text.strip()) if t]
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        candidate = f"{current} {sentence}".strip()
        if current and len(_tokens_for(candidate)) > 512:
            chunks.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        chunks.append(current)

    rng = np.random.default_rng(seed)
    length_scale = 1.0 / speed if speed else 1.0
    pieces: list[np.ndarray] = []

    for chunk in chunks:
        tokens = _tokens_for(chunk)
        length = min(len(tokens), 512)
        padded_tokens = np.zeros((1, 512), dtype=np.int32)
        padded_tokens[0, :length] = tokens[:length]
        x_mask = np.zeros((1, 1, 512), dtype=np.float32)
        x_mask[0, 0, :length] = 1.0

        encoder_outputs = encoder_model.predict(
            {"tokens": padded_tokens.astype(np.int32), "x_mask": x_mask}
        )
        m_p = encoder_outputs["m_p"]
        logs_p = encoder_outputs["logs_p"]
        logw = encoder_outputs["logw"]

        # Length regulation. The encoder emits token-level statistics plus a
        # log-duration per token; the synthesizer wants frame-level input.
        # Feeding m_p straight through skips this and yields exactly 512 frames
        # of unintelligible audio whatever the text says.
        durations = np.maximum(
            np.ceil(np.exp(logw[0, 0, :length]) * length_scale).astype(np.int32), 1
        )
        token_index = np.repeat(np.arange(length, dtype=np.int32), durations)
        m_p = m_p[:, :, :length][:, :, token_index]
        logs_p = logs_p[:, :, :length][:, :, token_index]

        zp_noise = rng.standard_normal(m_p.shape, dtype=np.float32)
        z_p = m_p + np.exp(logs_p) * zp_noise * variation

        num_frames = z_p.shape[-1]
        buckets = [256, 384, 512, 640, 768, 896, 1024, 2048]
        chosen_bucket = next((b for b in buckets if b >= num_frames), 2048)
        num_frames = min(num_frames, chosen_bucket)

        if z_p.shape[-1] < chosen_bucket:
            padded_zp = np.pad(
                z_p, ((0, 0), (0, 0), (0, chosen_bucket - z_p.shape[-1]))
            )
        else:
            padded_zp = z_p[:, :, :chosen_bucket]
        y_mask = np.zeros((1, 1, chosen_bucket), dtype=np.float32)
        y_mask[0, 0, :num_frames] = 1.0

        synth_outputs = _synth_for(chosen_bucket).predict(
            {"z_p": padded_zp.astype(np.float32), "y_mask": y_mask}
        )
        chunk_audio = list(synth_outputs.values())[0].flatten()
        pieces.append(chunk_audio[: min(len(chunk_audio), num_frames * 256)])

    waveform = np.concatenate(pieces) if len(pieces) > 1 else pieces[0]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), waveform.astype(np.float32), SAMPLE_RATE)
    console.print(f"[green]Saved:[/green] {out_path.resolve()} ({SAMPLE_RATE} Hz)")

    if play:
        import subprocess

        subprocess.run(["afplay", str(out_path)], check=True)


def info() -> None:
    """Print one-line JSON metadata for this engine."""
    print(
        json.dumps(
            {
                "name": "Inflect v2 CoreML (Micro & Nano)",
                "repo": MODEL_REPO,
                "backend": "coreml",
                "languages": ["en"],
                "sample_rate_hz": SAMPLE_RATE,
                "modes": ["auto"],
                "capabilities": ["lightweight_tts"],
            }
        )
    )


if __name__ == "__main__":
    fire.Fire({"prep": prep, "say": say, "info": info})
