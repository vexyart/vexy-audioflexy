#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = ["platformdirs>=4", "fire", "numpy"]
# ///
# this_file: src/vexy_audioflexy/tools/fx/tools/make_noisy_fixture.py
"""Build `testdata/noisy-speech.wav` — the input the denoisers are judged on.

Half the models in this project remove noise, and none of the original fixtures
contain any: fed a clean clip, a denoiser has nothing to do and every score says
so. This makes the missing fixture by adding noise to the clean source, which
keeps the clean file as an exact reference for the intrusive metrics — a recorded
noisy clip would have no such reference.

The noise is pink (1/sqrt(f)) with 50 Hz mains hum on top, mixed at a stated SNR.
Pink because white noise is the easy case every denoiser handles, and hum because
it is narrowband and survives models that only learn broadband suppression. The
seed is fixed, so the fixture is reproducible rather than merely repeatable.

    ./tools/make_noisy_fixture.py                 # 10 dB SNR, the checked-in fixture
    ./tools/make_noisy_fixture.py --snr_db 5      # a harder one
"""

from __future__ import annotations


import os
from pathlib import Path
from platformdirs import user_cache_path, user_data_path
RUNTIME_ROOT = Path(os.environ.get("VAFX_AUDIO_RUNTIME_ROOT", user_data_path("vexy-audioflexy") / "audio" )).expanduser()
CACHE_ROOT = Path(os.environ.get("VAFX_AUDIO_CACHE_ROOT", user_cache_path("vexy-audioflexy") / "audio")).expanduser()
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main(
    source: str,
    output: str,
    snr_db: float = 10.0,
    hum_amplitude: float = 0.15,
    seed: int = 42,
) -> None:
    """Mix noise into a clean clip at a fixed SNR and write the result.

    Args:
        source: Clean mono PCM WAV.
        output: Where the noisy fixture goes.
        snr_db: Signal-to-noise ratio of the mix, in dB.
        hum_amplitude: 50 Hz hum level relative to the pink noise, before the
            mix is scaled to *snr_db*.
        seed: Noise seed; changing it changes the fixture.
    """
    import numpy as np

    with wave.open(str(Path(source).resolve())) as w:
        if w.getsampwidth() != 2 or w.getnchannels() != 1:
            raise SystemExit(f"{source}: expected 16-bit mono PCM")
        rate, frames = w.getframerate(), w.readframes(w.getnframes())
    clean = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

    if clean.size < 2 or not np.isfinite(snr_db) or not np.isfinite(hum_amplitude):
        raise SystemExit("Require at least two PCM samples and finite noise parameters")
    rng = np.random.default_rng(seed)
    spectrum = np.fft.rfft(rng.standard_normal(clean.size))
    freqs = np.fft.rfftfreq(clean.size, 1.0 / rate)
    freqs[0] = freqs[1]  # DC would divide by zero and dominate the mix
    noise = np.fft.irfft(spectrum / np.sqrt(freqs), n=clean.size)
    noise /= np.sqrt(np.mean(noise**2))
    t = np.arange(clean.size) / rate
    noise = noise + hum_amplitude * np.sin(2 * np.pi * 50 * t)
    noise /= np.sqrt(np.mean(noise**2))

    gain = np.sqrt(np.mean(clean**2)) / (10 ** (snr_db / 20))
    noisy = clean + gain * noise
    peak = float(np.max(np.abs(noisy)))
    if peak > 0.99:
        # Clipping would be a second, undeclared degradation on top of the noise.
        noisy *= 0.99 / peak

    out_path = Path(output).resolve()
    with wave.open(str(out_path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((noisy * 32767).astype(np.int16).tobytes())

    measured = 10 * np.log10(np.mean(clean**2) / np.mean((noisy - clean) ** 2))
    print(f"wrote {out_path} — {rate} Hz mono, measured SNR {measured:.2f} dB")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
