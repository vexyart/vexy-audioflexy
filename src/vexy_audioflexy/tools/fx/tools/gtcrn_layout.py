# this_file: src/vexy_audioflexy/tools/fx/tools/gtcrn_layout.py
"""Bind the requested GTCRN MLX tensors to their pinned source graph structure."""
from __future__ import annotations


import os
from pathlib import Path
from platformdirs import user_cache_path, user_data_path
RUNTIME_ROOT = Path(os.environ.get("VAFX_AUDIO_RUNTIME_ROOT", user_data_path("vexy-audioflexy") / "audio" )).expanduser()
CACHE_ROOT = Path(os.environ.get("VAFX_AUDIO_CACHE_ROOT", user_cache_path("vexy-audioflexy") / "audio")).expanduser()
import hashlib
from pathlib import Path

import numpy as np

SOURCE_SHA = "e77603ac0c23dac3227dd2d7135b3a585cbee2679048aecfa886657d3ae1b534"
WEIGHTS_SHA = "d503ed83a35b91f66af97a6a1b07a76425988db7d9744556d0565c336a7e6e3b"


def restore(value: np.ndarray, shape: tuple, operation: str = "", group: int = 1) -> np.ndarray:
    """Invert the published OHWI layout, including grouped transposed kernels."""
    if value.dtype != np.float32 or not np.isfinite(value).all():
        raise ValueError("GTCRN weights must be finite F32")
    if operation == "Conv":
        value = value.transpose(0, 3, 1, 2)
    elif operation == "ConvTranspose":
        output, height, width, inputs = value.shape
        value = value.reshape(group, output // group, height, width, inputs)
        value = value.transpose(0, 4, 1, 2, 3)
    return np.ascontiguousarray(value.reshape(shape)) if shape else value.reshape(shape)


def bind(graph: Path, weights: Path) -> tuple[bytes, dict]:
    """Replace every float initializer; retain only integer structural constants."""
    import onnx
    from onnx import numpy_helper
    from safetensors.numpy import load_file

    for path, expected in [(graph, SOURCE_SHA), (weights, WEIGHTS_SHA)]:
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Pinned GTCRN checksum mismatch: {path}")
    model = onnx.load(graph)
    tensors = load_file(weights)
    learned = {item.name for item in model.graph.initializer if item.data_type == onnx.TensorProto.FLOAT}
    if len(tensors) != 179 or set(tensors) != learned:
        raise ValueError("Requested artifact must replace exactly all 179 learned initializers")
    convolutions = {node.input[1]: (node.op_type, next((a.i for a in node.attribute if a.name == "group"), 1))
                    for node in model.graph.node if node.op_type in {"Conv", "ConvTranspose"}}
    identical = 0
    for item in model.graph.initializer:
        if item.name not in learned:
            if item.data_type != onnx.TensorProto.INT64:
                raise ValueError("Unexpected nonstructural source initializer")
            continue
        original = numpy_helper.to_array(item)
        operation, group = convolutions.get(item.name, ("", 1))
        replacement = restore(tensors[item.name], tuple(original.shape), operation, group)
        identical += int(np.array_equal(replacement.view(np.uint32), original.view(np.uint32)))
        item.CopyFrom(numpy_helper.from_array(replacement, item.name))
    onnx.checker.check_model(model)
    serialized = model.SerializeToString()
    return serialized, {"weights_sha256": WEIGHTS_SHA, "source_graph_sha256": SOURCE_SHA,
        "bound_graph_sha256": hashlib.sha256(serialized).hexdigest(), "replaced_float_initializers": 179,
        "retained_int64_initializers": len(model.graph.initializer) - 179,
        "bitwise_equal_source_tensors": identical}
