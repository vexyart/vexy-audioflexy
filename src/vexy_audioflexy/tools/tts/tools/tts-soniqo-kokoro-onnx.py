#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = ["fire", "huggingface-hub", "numpy", "onnxruntime", "phonemizer", "rich", "soundfile"]
# ///
# this_file: src/vexy_audioflexy/tools/tts/tools/tts-soniqo-kokoro-onnx.py
"""Kokoro-82M ONNX by Soniqo (soniqo/Kokoro-82M-ONNX). Backend: onnx/onnxruntime, languages: en, 24 kHz mono, modes: auto (preset voices). Lightweight CPU inference; uses espeak-ng phonemizer + e2e ONNX graph. Voices: af_heart, am_adam, etc."""

from pathlib import Path
import json
import os

import fire

VERSION = "1.0.0"
MODEL = "soniqo/Kokoro-82M-ONNX"
SAMPLE_RATE = 24000
DEFAULT_MAX_TOKENS = 128


def _skip_existing(output: str, force: bool, play: bool) -> bool:
    out_path = Path(output)
    return not play and not force and out_path.exists() and out_path.stat().st_size > 0


def _find_file(root: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        matches = list(root.rglob(name))
        if matches:
            return matches[0]
    return None


def _configure_espeak_library() -> None:
    if os.environ.get("PHONEMIZER_ESPEAK_LIBRARY"):
        return

    for candidate in (
        "/opt/homebrew/lib/libespeak-ng.dylib",
        "/usr/local/lib/libespeak-ng.dylib",
        "/opt/homebrew/lib/libespeak.dylib",
        "/usr/local/lib/libespeak.dylib",
    ):
        if Path(candidate).exists():
            os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = candidate
            return


def _phonemize(text: str, language: str) -> str:
    _configure_espeak_library()
    try:
        from phonemizer import phonemize
    except ImportError as exc:
        raise RuntimeError(
            "Kokoro ONNX needs text converted to phonemes before the single e2e "
            "ONNX graph can run. Install phonemizer/espeak-ng or pass --phonemes "
            "with precomputed phonemes."
        ) from exc
    return phonemize(text, language=language, backend="espeak", strip=True)


def _ids_from_phonemes(phonemes: str, vocab_path: Path) -> list[int]:
    vocab = json.loads(vocab_path.read_text())
    token_to_id = vocab.get("vocab", vocab)
    ids = [int(token_to_id[p]) for p in phonemes if p in token_to_id]
    if not ids:
        raise RuntimeError(
            "No Kokoro token IDs were produced from phonemes. Check --language, "
            "--phonemes, and the downloaded vocab/config assets."
        )
    return ids


def _max_tokens_from_inputs(inputs) -> int:
    for name, input_info in inputs.items():
        lower = name.lower()
        if "token" not in lower and "input_ids" not in lower:
            continue
        shape = getattr(input_info, "shape", ())
        if len(shape) >= 2 and isinstance(shape[1], int):
            return int(shape[1])
    return DEFAULT_MAX_TOKENS


def _build_soniqo_e2e_feed(
    input_names: list[str],
    token_ids: list[int],
    style,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    speed: float = 1.0,
    random_seed: int | None = None,
):
    import numpy as np

    usable_tokens = max_tokens - 2
    if len(token_ids) > usable_tokens:
        raise RuntimeError(
            f"Soniqo Kokoro ONNX accepts at most {usable_tokens} phoneme tokens "
            f"for a {max_tokens}-token graph; got {len(token_ids)}."
        )

    sequence = [0, *token_ids, 0]
    input_ids = np.zeros((1, max_tokens), dtype=np.int64)
    input_ids[0, : len(sequence)] = sequence
    attention_mask = np.zeros((1, max_tokens), dtype=np.int64)
    attention_mask[0, : len(sequence)] = 1
    style_array = np.asarray(style, dtype=np.float32).reshape(1, 256)

    rng = np.random.default_rng(random_seed)
    random_phases = rng.random((1, 9)).astype(np.float32)

    feed = {}
    for name in input_names:
        lower = name.lower()
        if "token" in lower or "input_ids" in lower:
            feed[name] = input_ids
        elif "attention" in lower:
            feed[name] = attention_mask
        elif lower == "ref_s" or "style" in lower or "voice" in lower or "speaker" in lower:
            feed[name] = style_array
        elif "speed" in lower:
            feed[name] = np.asarray([speed], dtype=np.float32)
        elif "phase" in lower:
            feed[name] = random_phases
    return feed


def _chunk_token_ids(token_ids: list[int], max_tokens: int) -> list[list[int]]:
    usable_tokens = max_tokens - 2
    if usable_tokens <= 0:
        raise RuntimeError(f"Soniqo Kokoro ONNX max_tokens must be at least 3; got {max_tokens}.")
    return [
        token_ids[index : index + usable_tokens]
        for index in range(0, len(token_ids), usable_tokens)
    ]


def prep(force: bool = False):
    """Download + cache all model weights/assets WITHOUT synthesizing. Idempotent."""
    from huggingface_hub import snapshot_download

    snapshot_download(MODEL)
    print(f"[prep] ready: {MODEL}")


def say(
    text: str,
    output: str = "output.wav",
    voice: str = "af_heart",
    language: str = "en-us",
    model_dir: str | None = None,
    phonemes: str | None = None,
    speed: float = 1.0,
    random_seed: int | None = None,
    play: bool = False,
    force: bool = False,
):
    """Generate speech with soniqo/Kokoro-82M-ONNX if its e2e graph assets exist.

    Args:
        text: Text to synthesize.
        output: Output WAV path.
        voice: Voice preset key, e.g. af_heart.
        language: espeak language used by phonemizer when --phonemes is absent.
        model_dir: Existing checkout/snapshot. Defaults to Hugging Face download.
        phonemes: Precomputed Kokoro phoneme string.
        speed: Speech speed multiplier.
        random_seed: Optional deterministic seed for ONNX random_phases input.
        play: Play with afplay after saving.
        force: Overwrite output if it already exists.
    """
    from huggingface_hub import snapshot_download
    import numpy as np
    import onnxruntime as ort
    from rich.console import Console
    import soundfile as sf

    console = Console()
    if _skip_existing(output, force, play):
        console.print(f"[yellow]Skipped:[/yellow] {output} already exists (use --force)")
        return

    root = Path(model_dir) if model_dir else Path(snapshot_download(MODEL))
    if not root.exists():
        console.print(f"[red]Kokoro ONNX checkout not found:[/red] {root}")
        raise SystemExit(1)

    onnx_path = _find_file(root, ("kokoro-e2e.onnx", "model.onnx", "model_q8.onnx", "kokoro.onnx"))
    voice_path = root / "voices" / f"{voice}.bin"
    voices_path = voice_path if voice_path.exists() else _find_file(root, ("voices.bin", "voices.npz", "voices.npy"))
    vocab_path = _find_file(root, ("vocab_index.json", "vocab.json", "tokens.json", "config.json"))

    missing = [
        name
        for name, path in {
            "ONNX graph": onnx_path,
            "voice tensor file": voices_path,
            "phoneme vocab/config": vocab_path,
        }.items()
        if path is None
    ]
    if missing:
        console.print(
            "[red]Kokoro ONNX assets are incomplete:[/red] "
            + ", ".join(missing)
            + f"\nDownload {MODEL} or pass --model-dir to a complete checkout."
        )
        raise SystemExit(1)

    try:
        effective_phonemes = phonemes or _phonemize(text, language)
        token_ids = _ids_from_phonemes(effective_phonemes, vocab_path)
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        raise SystemExit(1) from exc

    try:
        if voices_path.suffix == ".bin":
            style = np.fromfile(voices_path, dtype=np.float32).reshape(1, 256)
        else:
            voices = np.load(voices_path, allow_pickle=True)
            style = voices[voice] if hasattr(voices, "__getitem__") and voice in voices else None
    except Exception as exc:
        console.print(
            f"[red]Could not load voice '{voice}' from {voices_path}:[/red] {exc}\n"
            "Use a voice key present in the Soniqo Kokoro voice asset."
        )
        raise SystemExit(1) from exc
    if style is None:
        console.print(f"[red]Voice '{voice}' was not found in {voices_path}.[/red]")
        raise SystemExit(1)

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inputs = {i.name: i for i in session.get_inputs()}
    max_tokens = _max_tokens_from_inputs(inputs)

    audio_parts = []
    for chunk_index, chunk_ids in enumerate(_chunk_token_ids(token_ids, max_tokens)):
        try:
            feed = _build_soniqo_e2e_feed(
                input_names=list(inputs),
                token_ids=chunk_ids,
                style=style,
                max_tokens=max_tokens,
                speed=speed,
                random_seed=None if random_seed is None else random_seed + chunk_index,
            )
        except RuntimeError as exc:
            console.print(f"[red]{exc}[/red]")
            raise SystemExit(1) from exc

        missing_inputs = sorted(set(inputs) - set(feed))
        if missing_inputs:
            console.print(
                "[red]Cannot run Kokoro ONNX graph automatically; unknown inputs:[/red] "
                + ", ".join(missing_inputs)
                + "\nThis wrapper only runs the single e2e graph when tokenizer/voice "
                "input names match known Kokoro exports."
            )
            raise SystemExit(1)

        outputs = session.run(None, feed)
        chunk_audio = np.asarray(outputs[0]).reshape(-1)
        if len(outputs) > 1:
            valid_samples = int(np.asarray(outputs[1]).reshape(-1)[0])
            chunk_audio = chunk_audio[:valid_samples]
        audio_parts.append(chunk_audio)

    audio = np.concatenate(audio_parts)
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), audio, SAMPLE_RATE)

    if play:
        import subprocess

        subprocess.run(["afplay", str(out_path)], check=True)

    console.print(f"[green]Saved:[/green] {out_path.resolve()} ({SAMPLE_RATE} Hz)")


def info():
    """Print one-line JSON with model metadata."""
    print(json.dumps({
        "name": "Kokoro 82M ONNX by Soniqo",
        "repo": MODEL,
        "backend": "onnx",
        "languages": ["en"],
        "sample_rate_hz": SAMPLE_RATE,
        "modes": ["auto"],
        "capabilities": ["preset_voices"],
    }))


if __name__ == "__main__":
    fire.Fire({"prep": prep, "say": say, "info": info})
