# this_file: tests/tools/asr/test_mikhailo.py
"""Exact HF-to-NeMo names and greedy prediction-state initialization."""
import importlib.util
from pathlib import Path

import pytest


def layout():
    spec = importlib.util.spec_from_file_location("mikhailo_layout", Path(__file__).resolve().parents[3] / "src/vexy_audioflexy/tools/asr" / "tools/mikhailo_layout.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_encoder_names_map_without_dropping_projection_or_normalization():
    module = layout()
    assert module.remap("encoder.layers.0.self_attn.q_proj.weight") == "encoder.layers.0.self_attn.linear_q.weight"
    assert module.remap("encoder.layers.23.conv.norm.bias") == "encoder.layers.23.conv.batch_norm.bias"
    assert module.remap("encoder.subsampling.layers.1.pointwise_conv.weight") == "encoder.pre_encode.conv.6.weight"
    assert module.remap("prompt_projector.linear_2.weight") == "prompt_kernel.2.weight"
    assert module.remap("encoder_projector.weight") == "encoder_projector.weight"
    with pytest.raises(ValueError):
        module.remap("not_a_model.weight")


def test_token_decoding_preserves_unknown_but_removes_only_known_language_tag():
    module = layout()
    pieces = ["<en-US>", "▁Hello", "▁world", "<unk>"]
    assert module.decode([0, 1, 2], pieces, {"en-US": 0}) == "Hello world"
    assert module.decode([1, 3], pieces, {"en-US": 0}) == "Hello<unk>"


def test_greedy_initializes_zero_blank_embedding_and_keeps_state_on_blank(monkeypatch):
    import numpy as np

    tools = Path(__file__).resolve().parents[3] / "src/vexy_audioflexy/tools/asr" / "tools"
    monkeypatch.syspath_prepend(str(tools))
    spec = importlib.util.spec_from_file_location("mikhailo", tools / "asr-mikhailo.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Decoder:
        def __init__(self):
            self.calls = []

        def run(self, names, values):
            self.calls.append(values)
            return np.zeros((1, 640), dtype=np.float32), values["h_in"] + 1, values["c_in"] + 1

    class Joiner:
        chosen = iter([5, 13087, 13087])

        def run(self, names, values):
            out = np.zeros((1, 13088), dtype=np.float32)
            out[0, next(self.chosen)] = 1
            return [out]

    decoder = Decoder()
    assert module.greedy(np.zeros((2, 640), dtype=np.float32), decoder, Joiner(),
        {"blank_token_id": 13087, "max_symbols_per_step": 10}) == [5]
    assert len(decoder.calls) == 2, "Blank must not advance predictor state"
    assert decoder.calls[0]["token"].item() == 13087
    assert decoder.calls[1]["h_in"][0, 0, 0] == 1
