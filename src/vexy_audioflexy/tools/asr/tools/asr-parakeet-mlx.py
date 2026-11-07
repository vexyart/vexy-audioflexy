#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = ["fire>=0.7,<0.8", "mlx-audio==0.5.7", "soundfile"]
# ///
# this_file: src/vexy_audioflexy/tools/asr/tools/asr-parakeet-mlx.py
"""Local 8-bit Parakeet TDT ASR, including models whose repository names mislead discovery."""

import json
import time
from pathlib import Path


def model_config(model_path: str) -> tuple[Path, dict]:
    """Require the local Parakeet TDT layout before selecting its explicit loader."""
    directory = Path(model_path).expanduser().resolve()
    for name in ("config.json", "model.safetensors", "tokenizer.model"):
        if not (directory / name).is_file():
            raise ValueError(f"Local Parakeet model is missing {name}")
    config = json.loads((directory / "config.json").read_text())
    if (not config.get("encoder", {}).get("_target_", "").endswith("ConformerEncoder")
            or not config.get("decoder", {}).get("_target_", "").endswith("RNNTDecoder")
            or config.get("model_defaults", {}).get("tdt_durations") != [0, 1, 2, 3, 4]
            or config.get("quantization", {}).get("bits") != 8
            or config.get("sample_rate") != 16000):
        raise ValueError("Require a 16 kHz, 8-bit Parakeet TDT model configuration")
    return directory, config


def transcribe(audio: str, model_path: str, text_out: str = "", force: bool = False) -> None:
    """Load the supplied ASR model explicitly as Parakeet; never infer type from its name."""
    source = Path(audio).expanduser().resolve()
    output = Path(text_out).expanduser().resolve() if text_out else source.with_suffix(".txt")
    if not source.is_file() or source == output:
        raise ValueError("Require existing audio and a distinct transcript path")
    if output.exists() and not force:
        raise FileExistsError("Transcript exists; use --force")
    directory, _ = model_config(model_path)
    import soundfile as sf
    from mlx_audio.stt.utils import load

    duration = sf.info(source).duration
    if duration <= 0:
        raise ValueError("Input audio is empty")
    model = load(str(directory), model_type="parakeet")
    start = time.monotonic()
    result = model.generate(str(source))
    wall = time.monotonic() - start
    text = getattr(result, "text", None)
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Parakeet returned no recognized text")
    text = text.strip()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text + "\n", encoding="utf-8")
    print("asr.result=" + json.dumps({"text": text, "text_out": str(output),
          "wall_s": wall, "audio_duration_s": duration, "rtf": wall / duration,
          "model_path": str(directory), "model_type": "parakeet", "quant": "8bit"},
          ensure_ascii=False))


if __name__ == "__main__":
    import fire

    fire.Fire({"transcribe": transcribe})
