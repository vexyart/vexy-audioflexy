# this_file: tests/tools/asr/test_qwen301_export.py
"""HF-layout conversion preserves the exact requested quantized Qwen tensors."""

import importlib.util
from pathlib import Path

import pytest


def wrapper():
    path = Path(__file__).resolve().parents[3] / "src/vexy_audioflexy/tools/asr" / "tools/qwen301_export.py"
    spec = importlib.util.spec_from_file_location("qwen301_export", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "source,target",
    [
        ("model.audio_tower.layers.0.fc1.weight", "audio_tower.layers.0.fc1.weight"),
        ("model.language_model.embed_tokens.scales", "model.embed_tokens.scales"),
        ("model.multi_modal_projector.linear_1.weight", "audio_tower.proj1.weight"),
        ("model.multi_modal_projector.linear_2.bias", "audio_tower.proj2.bias"),
    ],
)
def test_requested_tensor_name_when_hf_layout_then_maps_without_dropping(
    source, target
):
    assert wrapper().target_name(source) == target, (
        "Preserve every trained tensor in its runtime slot"
    )


def test_unknown_tensor_when_mapping_then_rejects_architecture_drift():
    with pytest.raises(ValueError):
        wrapper().target_name("model.unexpected.weight")
