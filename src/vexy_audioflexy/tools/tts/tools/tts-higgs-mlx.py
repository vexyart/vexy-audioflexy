#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = ["fire", "rich", "mlx-audio>=0.3.1", "mlx-lm>=0.22", "soundfile", "numpy", "huggingface_hub"]
# [tool.uv]
# prerelease = "allow"
# ///
# this_file: src/vexy_audioflexy/tools/tts/tools/tts-higgs-mlx.py
"""Higgs Audio v3 TTS via MLX.
Merges the base model (bosonai/higgs-audio-v3-tts-4b) with the quantized weights (Reza2kn/Higgs-Audio-v3-TTS-4bit-MLX).
"""

from pathlib import Path
import os

VERSION = "1.0.0"
REPO = "Reza2kn/Higgs-Audio-v3-TTS-4bit-MLX"
BASE_REPO = "bosonai/higgs-audio-v3-tts-4b"

def _monkeypatch():
    import mlx_audio.utils
    orig_load_config = mlx_audio.utils.load_config

    def custom_load_config(model_path, **kwargs):
        config = orig_load_config(model_path, **kwargs)
        if config.get("model_type") == "higgs_multimodal_qwen3" and "quantization" not in config:
            config["quantization"] = {
                "group_size": 64,
                "bits": 4
            }
        return config

    mlx_audio.utils.load_config = custom_load_config
    
    orig_load_weights = mlx_audio.utils.load_weights

    def custom_load_weights(model_path):
        weights = orig_load_weights(model_path)
        if "Reza2kn" in str(model_path) or "Higgs-Audio-v3-TTS-4bit-MLX" in str(model_path):
            from huggingface_hub import snapshot_download
            import glob
            import mlx.core as mx
            base_path = snapshot_download(BASE_REPO, allow_patterns=["*.safetensors"])
            base_weights = {}
            for f in glob.glob(f"{base_path}/*.safetensors"):
                base_weights.update(mx.load(f))
            
            merged_weights = dict(base_weights)
            import numpy as np
            for k, v in weights.items():
                fixed_k = k
                
                # Fix uint8 to uint32 for qweights
                if k.endswith(".weight.qweight"):
                    v_np = np.array(v)
                    if v_np.dtype == np.uint8:
                        v_np = v_np.view(np.uint32)
                        scales_key = k.replace(".weight.qweight", ".weight.scales")
                        if scales_key in weights:
                            scales_shape = weights[scales_key].shape
                            out_dim = scales_shape[0]
                            in_dim = scales_shape[1] * 64
                            v_np = v_np.reshape(out_dim, in_dim // 8)
                        v = mx.array(v_np)
                        
                fixed_k = fixed_k.replace(".weight.qweight", ".weight")
                fixed_k = fixed_k.replace(".weight.scales", ".scales")
                fixed_k = fixed_k.replace(".weight.biases", ".biases")
                merged_weights[fixed_k] = v

            # Fix up zero-points that were passed as biases
            # Reza2kn provided zero_points in the 'biases' tensor (as int16)
            # MLX expects biases = -zero_point * scales
            for k in list(merged_weights.keys()):
                if k.endswith(".biases"):
                    b = merged_weights[k]
                    if b.dtype == mx.int16:
                        scales_key = k.replace(".biases", ".scales")
                        if scales_key in merged_weights:
                            s = merged_weights[scales_key]
                            # compute MLX bias
                            mlx_b = - (b.astype(mx.float32) * s.astype(mx.float32)).astype(s.dtype)
                            merged_weights[k] = mlx_b
                
            return merged_weights
        return weights

    mlx_audio.utils.load_weights = custom_load_weights

    from mlx_audio.tts.models.higgs_audio_v3.model import Model as HiggsAudioV3Model
    orig_post_load_hook = HiggsAudioV3Model.post_load_hook

    @classmethod
    def custom_post_load_hook(cls, model, model_path):
        if "Reza2kn" in str(model_path) or "Higgs-Audio-v3-TTS-4bit-MLX" in str(model_path):
            from huggingface_hub import snapshot_download
            base_path = snapshot_download(BASE_REPO, allow_patterns=["*.safetensors", "*.json"])
            return orig_post_load_hook(model, base_path)
        return orig_post_load_hook(model, model_path)

    HiggsAudioV3Model.post_load_hook = custom_post_load_hook

    import safetensors
    orig_safe_open = safetensors.safe_open
    class MockSafeOpen:
        def __init__(self, path, framework=None):
            import mlx.core as mx
            self.weights = mx.load(str(path))
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass
        def get_tensor(self, key):
            return self.weights[key]
    safetensors.safe_open = MockSafeOpen


def prep(force: bool = False):
    """Download + cache all model weights/assets WITHOUT synthesizing. Idempotent."""
    from huggingface_hub import snapshot_download
    from mlx_audio.tts.utils import load_model
    
    # Pre-download base model since it's 8.5 GB
    print(f"[prep] Downloading base model weights from {BASE_REPO}...")
    snapshot_download(BASE_REPO, allow_patterns=["*.safetensors"])
    
    _monkeypatch()
    print(f"[prep] Downloading quantized weights from {REPO} and caching model...")
    load_model(REPO)
    print(f"[prep] ready: {REPO} (and base {BASE_REPO})")


def say(
    text: str,
    output: str = "output.wav",
    play: bool = False,
    force: bool = False,
    ref: str = None,
    ref_text: str = None,
):
    """Generate speech using Higgs Audio v3 (MLX).
    """
    import numpy as np
    import soundfile as sf
    from mlx_audio.tts.utils import load_model
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn

    console = Console()

    if not force and os.path.exists(output) and os.path.getsize(output) > 0:
        console.print(f"[yellow]Skipped:[/yellow] {output} already exists")
        return

    console.print(f"[dim]Model: {REPO}[/dim]")
    
    _monkeypatch()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading Higgs Audio (MLX)...", total=None)
        model = load_model(REPO)
        progress.update(task, description="Generating speech...")

        results = list(
            model.generate(
                text=text,
                ref_audio=ref,
                ref_text=ref_text,
            )
        )
        progress.update(task, description="Done.")

    audio_chunks = [np.array(r.audio) for r in results]
    audio = (
        np.concatenate(audio_chunks, axis=0)
        if len(audio_chunks) > 1
        else audio_chunks[0]
    )
    sample_rate = results[0].sample_rate

    if play:
        import subprocess
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, audio, sample_rate)
            tmp_path = tmp.name

        console.print(
            f"[cyan]Playing audio ({len(audio) / sample_rate:.1f}s)...[/cyan]"
        )
        subprocess.run(["afplay", tmp_path], check=True)
        Path(tmp_path).unlink(missing_ok=True)
        console.print("[green]Done.[/green]")
    else:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_path), audio, sample_rate)
        console.print(f"[green]Saved:[/green] {out_path.resolve()} ({sample_rate} Hz)")


if __name__ == "__main__":
    import fire
    fire.Fire({"prep": prep, "say": say})
