# this_file: src/vexy_audioflexy/tools/asr/tools/qwen301_export.py
"""Adapt the exact Majentik HF tensor layout to the existing MLX Qwen runtime."""

import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from _audioflexy_paths import models_root

POLICY = "majentik-hf-to-mlx-audio-0.5.7-v1"


def digest(path: Path) -> str:
    """Hash large weights without materializing a second full copy in memory."""
    result = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def target_name(name: str) -> str:
    """Map inspected official HF encoder/projector/text namespaces injectively."""
    for original, target in [
        ("model.audio_tower.", "audio_tower."),
        ("model.language_model.", "model."),
        ("model.multi_modal_projector.linear_1.", "audio_tower.proj1."),
        ("model.multi_modal_projector.linear_2.", "audio_tower.proj2."),
    ]:
        if name.startswith(original):
            return target + name.removeprefix(original)
    raise ValueError(f"Unexpected requested-checkpoint tensor: {name}")


def prepare(source: str) -> str:
    """Preserve raw weights; record exact hashes, key mapping and convolution layout."""
    snapshot = Path(source)
    signature = {
        "policy": POLICY,
        "source_weights_sha256": digest(snapshot / "model.safetensors"),
        "source_config_sha256": digest(snapshot / "config.json"),
    }
    root = models_root() / "conversions/majentik-mlx-audio"
    destination = root / signature["source_weights_sha256"][:16]
    receipt = destination / "conversion.json"
    if receipt.exists():
        old = json.loads(receipt.read_text())
        if (
            all(old.get(k) == v for k, v in signature.items())
            and digest(destination / "model.safetensors")
            == old["runtime_weights_sha256"]
        ):
            return str(destination)
        raise ValueError(
            "Prepared Qwen runtime differs from its source/conversion receipt"
        )
    import mlx.core as mx

    weights = mx.load(str(snapshot / "model.safetensors"))
    converted, mapping = {}, {}
    for original, tensor in weights.items():
        name = target_name(original)
        if name in converted:
            raise ValueError("Tensor namespace conversion produced a collision")
        transpose = name.startswith("audio_tower.conv2d") and name.endswith(".weight")
        converted[name] = tensor.transpose(0, 2, 3, 1) if transpose else tensor
        mapping[original] = {
            "target": name,
            "shape": list(tensor.shape),
            "conv_layout_transpose": transpose,
        }
    if len(converted) != len(weights):
        raise ValueError("Tensor conversion dropped trained weights")
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root) as temporary:
        stage = Path(temporary) / "runtime"
        stage.mkdir()
        for path in snapshot.iterdir():
            if path.is_file() and path.suffix in {".json", ".txt", ".model"}:
                shutil.copyfile(path, stage / path.name)
        config = json.loads((stage / "config.json").read_text())
        config["text_config"]["rope_theta"] = config["text_config"]["rope_parameters"][
            "rope_theta"
        ]
        (stage / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        mx.save_safetensors(str(stage / "model.safetensors"), converted)
        info = {
            **signature,
            "source_snapshot": str(snapshot),
            "tensor_count": len(weights),
            "runtime_weights_sha256": digest(stage / "model.safetensors"),
            "mapping": mapping,
            "numeric_changes": "none; quantized values retained, conv axis permutation only",
        }
        (stage / "conversion.json").write_text(json.dumps(info, indent=2) + "\n")
        stage.rename(destination)
    return str(destination)
