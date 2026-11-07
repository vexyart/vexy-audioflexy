#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = ["platformdirs>=4", 
#   "fire",
#   "rich",
#   "soundfile",
#   "numpy",
#   "scipy",
#   "speechonnxmetrics @ git+https://github.com/TigreGotico/speechonnxmetrics@dev",
# ]
# ///
# this_file: src/vexy_audioflexy/tools/fx/tools/aud-metrics.py
"""Objective audio quality measurement — the acceptance test for enhancement.

`aud_all.py` already refuses to call a well-formed silent file "support". That is
necessary and nowhere near sufficient here: a denoiser that passes its input
through untouched clears an RMS check while doing nothing. So every enhancement
run is scored, input against output, and the numbers go into `out/report.json`.

Two kinds of number, plus one this project computes itself:

- **No-reference MOS** (`utmos`, `dnsmos`, `dnsmos_p808`, `sigmos`, `nisqa`) —
  neural quality predictors from `speechonnxmetrics`, ONNX graphs on numpy +
  onnxruntime, no torch. They need only the clip being judged, so they work on
  voice conversion and speech-to-speech output where no reference exists.
- **Intrusive** (`stoi`, `estoi`, `si_sdr`, `lsd`, `mcd`) — comparisons against a
  clean reference. Meaningful for denoising and super-resolution, where the
  reference is the undegraded original; meaningless for voice conversion, whose
  whole job is to differ from the input.
- **`input_corr`** — waveform correlation between the model's output and the clip
  that went in. It is how an *editing* model is checked: a route that quietly
  degrades to passthrough scores better on every MOS metric than one that works,
  and only a correlation near 1.0 exposes it. Near 0 means the audio was
  genuinely resynthesised.

  Read it against the operation, not on its own. Band extension *should* score
  near 1.0 — it keeps the waveform and adds a high band, and the measured values
  sit at 0.995-0.999. Voice conversion, speech editing and speech-to-speech
  should score near 0, and do. A `vc` run at 0.99 would be the bug this catches.
- **`hf8k_ratio`** — the fraction of signal energy above 8 kHz. Not a MOS: a flat
  statement about bandwidth, and the honest test of whether a super-resolution
  model invented a high band or merely resampled. Computed here in numpy.

`nisqa`'s weights are NonCommercial (see the upstream model card); the scores are
recorded for judging this project's own output.

    ./tools/aud-metrics.py score --audio out/dn__dpdfnet.wav --ref testdata/speech.wav
    ./tools/aud-metrics.py score --audio out/vc__meanvc2.wav          # no-reference only
    ./tools/aud-metrics.py prep                                        # cache the MOS weights
"""

from __future__ import annotations


import os
from pathlib import Path
from platformdirs import user_cache_path, user_data_path
RUNTIME_ROOT = Path(os.environ.get("VAFX_AUDIO_RUNTIME_ROOT", user_data_path("vexy-audioflexy") / "audio" )).expanduser()
CACHE_ROOT = Path(os.environ.get("VAFX_AUDIO_CACHE_ROOT", user_cache_path("vexy-audioflexy") / "audio")).expanduser()
import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # numpy is imported inside the functions that need it
    import numpy as np

VERSION = "1.0.0"

# Neural MOS predictors: no reference needed, weights downloaded on first use.
NOREF = ("utmos", "dnsmos", "dnsmos_p808", "sigmos", "nisqa")
# Reference-based metrics: pure numpy, but useless without a clean original.
INTRUSIVE = ("stoi", "estoi", "si_sdr", "lsd", "mcd")


def _read_mono(path: str) -> tuple[np.ndarray, int]:
    """Return (float32 mono samples, sample rate)."""
    import numpy as np
    import soundfile as sf

    data, rate = sf.read(str(Path(path).resolve()), dtype="float32", always_2d=True)
    return np.mean(data, axis=1), int(rate)


def hf_energy_ratio(path: str, cutoff_hz: float = 8000.0) -> float | None:
    """Fraction of the clip's energy above *cutoff_hz*, in [0, 1].

    A 16 kHz source has nothing above 8 kHz by construction, so this is 0 for the
    degraded input and non-zero exactly when a model put something up there. It
    says nothing about whether the invented band sounds right — only that it exists.

    Returns None — not 0.0 — when the clip's Nyquist frequency is at or below the
    cutoff, because a 16 kHz output *cannot* carry that band. Reporting 0 there
    would read as "the model destroyed the high frequencies" next to a genuine
    measurement, when the truth is that the question does not apply.
    """
    import numpy as np

    audio, rate = _read_mono(path)
    if rate <= 2 * cutoff_hz:
        return None
    if audio.size == 0:
        return 0.0
    spectrum = np.abs(np.fft.rfft(audio * np.hanning(audio.size))) ** 2
    freqs = np.fft.rfftfreq(audio.size, 1.0 / rate)
    total = float(spectrum.sum())
    if total <= 0.0:
        return 0.0
    return float(spectrum[freqs > cutoff_hz].sum() / total)


def input_correlation(audio: str, source: str, rate: int = 16000) -> float | None:
    """Pearson correlation between *audio* and *source*, both resampled to *rate*.

    A model that returned its input unchanged scores ~1.0 here. That is the only
    automatic check that catches a passthrough on operations where no reference
    exists and where higher MOS is not evidence of anything — speech editing
    above all, since a copy of clean input scores better than a real edit.

    Returns None when either clip is empty. Clips of different lengths are
    compared over their overlap, which is what a same-length edit needs; a model
    that changes duration will score low regardless, and that is not a false
    negative — it did resynthesise.
    """
    from math import gcd

    import numpy as np
    from scipy.signal import resample_poly

    def _at_rate(path: str) -> np.ndarray:
        samples, in_rate = _read_mono(path)
        if in_rate == rate or samples.size == 0:
            return samples
        g = gcd(int(in_rate), rate)
        return resample_poly(samples, rate // g, int(in_rate) // g)

    a, b = _at_rate(audio), _at_rate(source)
    n = min(a.size, b.size)
    if n == 0:
        return None
    a, b = a[:n].astype(float), b[:n].astype(float)
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def prep(force: bool = False) -> None:
    """Download and warm every no-reference MOS model.

    Args:
        force: accepted for symmetry with the other wrappers; the HuggingFace
            cache already makes repeat downloads free.
    """
    import numpy as np
    import soundfile as sf
    import speechonnxmetrics as som

    tmp = Path("/tmp/aud-metrics-probe.wav")
    rng = np.random.default_rng(0)
    sf.write(str(tmp), (rng.standard_normal(16000) * 0.05).astype("float32"), 16000)
    for metric in NOREF:
        som.score(str(tmp), [metric])
        print(f"[prep] ready: {metric}")
    tmp.unlink(missing_ok=True)


def selected_metrics(metrics: str | list[str] | tuple[str, ...]) -> list[str]:
    """Normalize Fire's CSV string/tuple forms before selecting metric functions."""
    values = metrics.split(",") if isinstance(metrics, str) else metrics
    return [m.strip() for m in values if m.strip()] if metrics else list(NOREF + INTRUSIVE)


def score(
    audio: str = "",
    ref: str = "",
    source: str = "",
    metrics: str | list[str] | tuple[str, ...] = "",
    pretty: bool = False,
) -> None:
    """Score one clip and print the results as JSON on one line.

    Args:
        audio: The clip to judge.
        ref: Clean reference. Given, the intrusive metrics run too and both
            clips are measured, so a report can show input vs output.
        source: The clip that went into the model. Given, `input_corr` is
            computed — the passthrough check.
        metrics: Comma-separated metric names, overriding the defaults.
        pretty: Indent the JSON instead of emitting one line.
    """
    import speechonnxmetrics as som

    if not audio:
        raise SystemExit("--audio is required")
    wanted = selected_metrics(metrics)
    noref = [m for m in (wanted or NOREF) if m in NOREF]
    intrusive = [m for m in (wanted or INTRUSIVE) if m in INTRUSIVE] if ref else []

    out: dict[str, object] = {"audio": audio, "ref": ref or "", "source": source or ""}
    scores: dict[str, float | str | None] = {}
    # Each metric is scored on its own: one unloadable model must not take the
    # rest of the report down with it.
    for metric in noref:
        try:
            scores.update(som.score(audio, [metric]))
        except Exception as exc:  # noqa: BLE001 - a failed metric is data, not a crash
            scores[f"{metric}.error"] = str(exc)[:120]
    for metric in intrusive:
        try:
            scores.update(som.score(audio, [metric], ref=ref))
        except Exception as exc:  # noqa: BLE001
            scores[f"{metric}.error"] = str(exc)[:120]

    scores["hf8k_ratio"] = hf_energy_ratio(audio)
    if ref:
        scores["ref.hf8k_ratio"] = hf_energy_ratio(ref)
    if source:
        scores["input_corr"] = input_correlation(audio, source)
    out["scores"] = scores
    print(json.dumps(out, separators=(",", ":"), indent=2 if pretty else None))


def compare(before: str = "", after: str = "", ref: str = "") -> None:
    """Score two clips and print the delta — the shape enhancement is judged in.

    Args:
        before: The degraded input.
        after: The model's output.
        ref: Optional clean reference for the intrusive metrics.
    """
    import speechonnxmetrics as som

    if not before or not after:
        raise SystemExit("--before and --after are both required")
    rows: dict[str, dict[str, float | str | None]] = {}
    for label, path in (("before", before), ("after", after)):
        values: dict[str, float | str | None] = {}
        for metric in NOREF:
            try:
                values.update(som.score(path, [metric]))
            except Exception as exc:  # noqa: BLE001
                values[f"{metric}.error"] = str(exc)[:120]
        if ref:
            for metric in INTRUSIVE:
                try:
                    values.update(som.score(path, [metric], ref=ref))
                except Exception as exc:  # noqa: BLE001
                    values[f"{metric}.error"] = str(exc)[:120]
        values["hf8k_ratio"] = hf_energy_ratio(path)
        rows[label] = values

    delta = {
        k: float(rows["after"][k]) - float(rows["before"][k])
        for k in rows["before"]
        if isinstance(rows["before"].get(k), (int, float))
        and isinstance(rows["after"].get(k), (int, float))
    }
    print(
        json.dumps(
            {"before": rows["before"], "after": rows["after"], "delta": delta},
            separators=(",", ":"),
        )
    )


def metrics_list() -> None:
    """List every metric the installed speechonnxmetrics exposes."""
    import speechonnxmetrics as som

    for entry in som.list_metrics():
        print(
            f"{entry.name:14} {entry.kind:6} "
            f"{'intrusive' if entry.intrusive else 'no-ref':10} "
            f"{'download' if entry.requires_download else 'numpy':9} "
            f"{entry.description[:70]}"
        )


if __name__ == "__main__":
    import fire

    fire.Fire({"prep": prep, "score": score, "compare": compare, "list": metrics_list})
