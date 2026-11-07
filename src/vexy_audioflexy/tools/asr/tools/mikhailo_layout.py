# this_file: src/vexy_audioflexy/tools/asr/tools/mikhailo_layout.py
"""Name-only adaptation of the exact Mikhailo HF encoder to NeMo MLX primitives."""


def remap(key: str) -> str:
    """Reject unrelated weights; strict native loading proves final key coverage."""
    if key.startswith("encoder.layers."):
        replacements = {".self_attn.q_proj.": ".self_attn.linear_q.",
            ".self_attn.k_proj.": ".self_attn.linear_k.", ".self_attn.v_proj.": ".self_attn.linear_v.",
            ".self_attn.o_proj.": ".self_attn.linear_out.", ".self_attn.relative_k_proj.": ".self_attn.linear_pos.",
            ".self_attn.bias_u": ".self_attn.pos_bias_u", ".self_attn.bias_v": ".self_attn.pos_bias_v",
            ".conv.norm.": ".conv.batch_norm."}
        for old, new in replacements.items():
            key = key.replace(old, new)
        return key
    prefixes = {"encoder.subsampling.conv_in.": "encoder.pre_encode.conv.0.",
        "encoder.subsampling.layers.0.depthwise_conv.": "encoder.pre_encode.conv.2.",
        "encoder.subsampling.layers.0.pointwise_conv.": "encoder.pre_encode.conv.3.",
        "encoder.subsampling.layers.1.depthwise_conv.": "encoder.pre_encode.conv.5.",
        "encoder.subsampling.layers.1.pointwise_conv.": "encoder.pre_encode.conv.6.",
        "encoder.subsampling.linear.": "encoder.pre_encode.out.",
        "prompt_projector.linear_1.": "prompt_kernel.0.", "prompt_projector.linear_2.": "prompt_kernel.2."}
    for old, new in prefixes.items():
        if key.startswith(old):
            return new + key[len(old):]
    if key.startswith("encoder_projector."):
        return key
    raise ValueError(f"Unmapped exact encoder tensor: {key}")


def decode(ids: list[int], pieces: list[str], languages: dict) -> str:
    """Join SentencePiece spellings, stripping only declared language markers."""
    selected = [pieces[i] for i in ids]
    selected = [p for p in selected if not (p.startswith("<") and p.endswith(">") and p[1:-1] in languages)]
    return "".join(selected).replace("▁", " ").strip()
