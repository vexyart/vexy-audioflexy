#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.13"
# dependencies = ["fire", "huggingface-hub", "soundfile", "onnxruntime", "mlx-speech @ git+https://github.com/appautomaton/mlx-speech@4577ad28e6bef2ae2a11b8fa76a6cdb23bb2a9cc"]
# ///
# this_file: src/vexy_audioflexy/tools/asr/tools/asr-mikhailo.py
"""Exact Mikhailo FP16 encoder on MLX, published decoder/joiner on CPU ONNX."""
from __future__ import annotations

import importlib.metadata
import json
import subprocess
import tempfile
import time
from pathlib import Path

from mikhailo_layout import decode, remap

ROOT = Path(__file__).resolve().parents[1]
REPO = "Mikhailo/nemotron-3.5-asr-streaming-0.6b-mlx-fp16"
REVISION = "2c5f17156b246e2e3061b8c014d961313c4a64dc"
RUNTIME = "4577ad28e6bef2ae2a11b8fa76a6cdb23bb2a9cc"


def prep(repo: str = REPO) -> str:
    """Pin all published hybrid-runtime assets without another model's weights."""
    from huggingface_hub import snapshot_download

    if repo != REPO:
        raise SystemExit(f"Require exact conversion {REPO}")
    path = snapshot_download(repo, revision=REVISION, cache_dir=None,
        allow_patterns=["encoder.safetensors", "decoder.onnx", "joiner.onnx", "tokens.txt", "nemotron_onnx_config.json", "melfilters.f32"])
    print(f"[prep] ready: {repo} -> {path}")
    return path


def load_encoder(base: Path, config: dict):
    """Map all 642 FP16 tensors, preserving every parameter and strict shapes."""
    import mlx.core as mx
    import numpy as np
    from mlx import nn
    from mlx_speech.models.nemotron_asr.checkpoint import load_state_dict_strict
    from mlx_speech.models.nemotron_asr.config import (
        ConformerArgs,
        PreprocessArgs,
        PromptArgs,
    )
    from mlx_speech.models.nemotron_asr.encoder import FastConformerEncoder
    from mlx_speech.models.nemotron_asr.feature_extraction import NemotronPreprocessor
    from mlx_speech.models.nemotron_asr.prompt import build_prompt_kernel

    assert config["encoder"]["hidden_size"] == 1024 and config["encoder"]["num_hidden_layers"] == 24
    assert config["encoder"]["num_attention_heads"] == 8 and config["encoder"]["subsampling_factor"] == 8

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = FastConformerEncoder(ConformerArgs(default_att_context_size=(56, 3)))
            self.prompt_kernel = build_prompt_kernel(1024, PromptArgs(prompt_dictionary=config["prompt_dictionary"]))
            self.encoder_projector = nn.Linear(1024, 640)

    model = Encoder()
    source = mx.load(str(base / "encoder.safetensors"))
    if len(source) != 642 or any(v.dtype != mx.float16 for v in source.values()):
        raise SystemExit("Require all 642 published FP16 encoder tensors")
    weights = {}
    mappings = []
    for key, value in source.items():
        target = remap(key)
        if target in weights:
            raise SystemExit(f"Duplicate target: {target}")
        if target.endswith("weight") and ".conv." in target and value.ndim in (3, 4):
            value = value.transpose(0, 2, 1) if value.ndim == 3 else value.transpose(0, 2, 3, 1)
        weights[target] = value
        mappings.append({"source": key, "target": target, "shape": list(value.shape)})
    load_state_dict_strict(model, weights)
    model.eval()
    mx.eval(model.parameters())
    frontend = NemotronPreprocessor(PreprocessArgs())
    filters = np.fromfile(base / "melfilters.f32", dtype="<f4").reshape(128, 257)
    frontend.featurizer.fb = mx.array(filters[None])
    return model, frontend, mappings


def onnx_sessions(base: Path):
    """Select CPU explicitly and keep tiny decoder calls single-threaded."""
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return tuple(ort.InferenceSession(str(base / name), sess_options=options,
        providers=["CPUExecutionProvider"]) for name in ["decoder.onnx", "joiner.onnx"])


def greedy(frames, decoder, joiner, config: dict) -> list[int]:
    """Carry predictor state across frames; blank advances audio only."""
    import numpy as np

    state = np.zeros((2, 1, 640), dtype=np.float32)
    prediction, hidden, cell = decoder.run(None, {"token": np.array([[config["blank_token_id"]]], dtype=np.int64),
        "h_in": state, "c_in": state.copy()})
    tokens = []
    for frame in frames:
        for _ in range(config["max_symbols_per_step"]):
            logits = joiner.run(None, {"encoder_frame": frame[None].astype(np.float32), "decoder_out": prediction})[0]
            if not np.isfinite(logits).all():
                raise SystemExit("Nonfinite ONNX joiner logits")
            token = int(logits.argmax())
            if token == config["blank_token_id"]:
                break
            tokens.append(token)
            prediction, hidden, cell = decoder.run(None, {"token": np.array([[token]], dtype=np.int64),
                "h_in": hidden, "c_in": cell})
    return tokens


def transcribe(audio: str, text_out: str = "", repo: str = REPO, lang: str = "en",
               force: bool = False, timeout: int = 900) -> None:
    """Full-file PCM through stateful 320 ms encoder chunks and CPU greedy decode."""
    import mlx.core as mx
    import numpy as np
    import soundfile as sf
    from mlx_speech.models.nemotron_asr.config import PromptArgs
    from mlx_speech.models.nemotron_asr.prompt import apply_language_prompt
    from mlx_speech.models.nemotron_asr.streaming import StreamingEncoder

    source = Path(audio).resolve()
    output = Path(text_out).resolve() if text_out else source.with_suffix(".txt")
    if not source.is_file() or source == output or timeout <= 0:
        raise SystemExit("Require audio, distinct transcript path and positive timeout")
    if output.exists() and not force:
        raise SystemExit(f"Transcript exists: {output}; use --force")
    base = Path(prep(repo))
    config = json.loads((base / "nemotron_onnx_config.json").read_text())
    selected = lang or "auto"
    if selected not in config["prompt_dictionary"]:
        raise SystemExit(f"Unknown language prompt: {selected}")
    start = time.monotonic()
    model, frontend, mappings = load_encoder(base, config)
    decoder, joiner = onnx_sessions(base)
    load_s = time.monotonic() - start
    with tempfile.TemporaryDirectory() as temporary:
        wav = Path(temporary) / "input.wav"
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(source), "-ar", "16000", "-ac", "1", str(wav)],
            check=True, timeout=timeout)
        pcm, rate = sf.read(wav, dtype="float32")
    duration = len(pcm) / rate
    if duration <= 0:
        raise SystemExit("Empty audio")
    start = time.monotonic()
    features, lengths = frontend(pcm)
    stream = StreamingEncoder(model.encoder, att_context_size=(56, 3))
    chunks = stream.feed(features[:, :int(lengths[0].item())], final=True)
    encoded = mx.concatenate(chunks, axis=1)
    prompted = apply_language_prompt(encoded, selected, PromptArgs(prompt_dictionary=config["prompt_dictionary"]), model.prompt_kernel)
    projected = model.encoder_projector(prompted)
    mx.eval(projected)
    frames = np.asarray(projected[0], dtype=np.float32)
    tokens = greedy(frames, decoder, joiner, config)
    mx.synchronize()
    wall = time.monotonic() - start
    pieces = [line.split("\t", 1)[1] for line in (base / "tokens.txt").read_text().splitlines()]
    text = decode(tokens, pieces, config["prompt_dictionary"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text + "\n")
    receipt = {"text": text, "text_out": str(output), "wall_s": wall, "load_s": load_s,
        "audio_duration_s": duration, "rtf": wall / duration, "repo": REPO, "weights_revision": REVISION,
        "runtime_commit": RUNTIME, "mlx_default_device": str(mx.default_device()), "tokens": tokens, "language": selected, "chunk_frames": 4,
        "encoded_frames": len(frames), "tensor_mappings": mappings, "decoder_provider": "CPUExecutionProvider",
        "runtime_versions": {name: importlib.metadata.version(name) for name in ["mlx", "mlx-speech", "onnxruntime", "numpy", "huggingface-hub"]}}
    output.with_suffix(".native.json").write_text(json.dumps(receipt, indent=2) + "\n")
    np.save(output.with_suffix(".encoder.npy"), frames)
    print("asr.result=" + json.dumps({k: v for k, v in receipt.items() if k != "tensor_mappings"}))


if __name__ == "__main__":
    import fire

    fire.Fire({"prep": prep, "transcribe": transcribe})
