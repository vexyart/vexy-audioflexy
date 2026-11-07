# this_file: src/vexy_audioflexy/tools/fx/tools/gtcrn_audio.py
"""Full-file periodic sqrt-Hann frontend for the original GTCRN STFT contract."""

import os
from pathlib import Path
from platformdirs import user_cache_path, user_data_path
RUNTIME_ROOT = Path(os.environ.get("VAFX_AUDIO_RUNTIME_ROOT", user_data_path("vexy-audioflexy") / "audio" )).expanduser()
CACHE_ROOT = Path(os.environ.get("VAFX_AUDIO_CACHE_ROOT", user_cache_path("vexy-audioflexy") / "audio")).expanduser()
import numpy as np

WINDOW = np.sqrt(np.hanning(513)[:-1]).astype(np.float32)


def analyze(samples: np.ndarray) -> np.ndarray:
    """Center frames with reflection, covering even a final partial hop."""
    if samples.ndim != 1 or not samples.size or not np.isfinite(samples).all():
        raise ValueError("Require nonempty finite mono samples")
    padded = np.pad(samples, (256, 512), mode="reflect" if samples.size > 1 else "edge")
    frames = np.stack([padded[i:i + 512] * WINDOW for i in range(0, samples.size + 1, 256)])
    return np.fft.rfft(frames, axis=-1).astype(np.complex64)


def synthesize(spectra: np.ndarray, length: int) -> np.ndarray:
    """Weighted overlap-add cancels the centered padding and preserves length."""
    if spectra.ndim != 2 or spectra.shape[1] != 257 or length <= 0 or not np.isfinite(spectra).all():
        raise ValueError("Require finite 257-bin frames and positive target length")
    frames = np.fft.irfft(spectra, n=512, axis=-1) * WINDOW
    output = np.zeros((len(frames) - 1) * 256 + 512)
    divisor = np.zeros_like(output)
    for i, frame in enumerate(frames):
        offset = i * 256
        output[offset:offset + 512] += frame
        divisor[offset:offset + 512] += WINDOW ** 2
    output, divisor = output[256:256 + length], divisor[256:256 + length]
    if output.size != length or (divisor <= 0).any():
        raise ValueError("Insufficient frames to reconstruct requested timeline")
    return (output / divisor).astype(np.float32)
