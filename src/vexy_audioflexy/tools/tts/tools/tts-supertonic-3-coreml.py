#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = ["fire", "rich", "coremltools>=7.0", "numpy", "soundfile", "huggingface-hub"]
# ///
# this_file: src/vexy_audioflexy/tools/tts/tools/tts-supertonic-3-coreml.py
"""Supertonic-3 TTS via CoreML on macOS (44.1 kHz output). Backend: coreml (coremltools fp16
four-stage pipeline: DurationPredictor, TextEncoder, VectorEstimator, Vocoder). 31 languages
(en ko ja ar bg cs da de el es et fi fr hi hr hu id it lt lv nl pl pt ro ru sk sl sv tr uk vi na).
10 preset voices (M1-M5 male, F1-F5 female). Mode: auto (no voice cloning). macOS/Apple
Silicon only via CoreML ANE/GPU/CPU compute units."""

from __future__ import annotations

import json
import hashlib
import shutil
from _paths import runtime_path
import platform
import re
import subprocess
from pathlib import Path
from unicodedata import normalize

VERSION = "1.1.0"

# Two independent CoreML conversions of the same Supertonic-3 four-stage pipeline.
VARIANTS: dict[str, dict[str, str]] = {
    "fluid": {
        "repo": "FluidInference/supertonic-3-coreml",
        "cache": "supertonic-3-coreml",
        # Neural Engine + CPU is the fastest path for this export.
        "compute_units": "cpu_and_ne",
    },
    "aufklarer-fp16": {
        "repo": "aufklarer/Supertonic-3-CoreML-FP16",
        "cache": "supertonic-3-coreml-aufklarer",
        # This export leaves a dynamic axis in VectorEstimator that the MPS graph
        # rejects ("shape for TensorData is not static"); the CPU path is exact.
        "compute_units": "cpu",
    },
}
DEFAULT_VARIANT = "fluid"
MODEL_REPO = VARIANTS[DEFAULT_VARIANT]["repo"]
CACHE_SUBDIR = VARIANTS[DEFAULT_VARIANT]["cache"]


def _variant(name: str) -> dict[str, str]:
    try:
        return VARIANTS[name]
    except KeyError:
        raise SystemExit(
            f"unknown variant '{name}'. Choose from: {', '.join(VARIANTS)}"
        ) from None

# Languages supported by Supertonic-3 v1.7.3.
SUPPORTED_LANGS = [
    "en", "ko", "ja", "ar", "bg", "cs", "da", "de", "el", "es",
    "et", "fi", "fr", "hi", "hr", "hu", "id", "it", "lt", "lv",
    "nl", "pl", "pt", "ro", "ru", "sk", "sl", "sv", "tr", "uk",
    "vi", "na",
]

VALID_VOICES = ["M1", "M2", "M3", "M4", "M5", "F1", "F2", "F3", "F4", "F5"]

# CoreML shape pins.
TEXT_T_FIXED = 128
VEC_EST_L_MIN = 17

_EMOJI_RE = re.compile(
    "[\U0001f600-\U0001f64f\U0001f300-\U0001f5ff\U0001f680-\U0001f6ff"
    "\U0001f700-\U0001f77f\U0001f780-\U0001f7ff\U0001f800-\U0001f8ff"
    "\U0001f900-\U0001f9ff\U0001fa00-\U0001fa6f\U0001fa70-\U0001faff"
    "☀-⛿✀-➿\U0001f1e6-\U0001f1ff]+",
    flags=re.UNICODE,
)
_CHAR_REPL = {
    "–": "-", "‑": "-", "—": "-", "_": " ",
    "“": '"', "”": '"', "‘": "'", "’": "'",
    "´": "'", "`": "'",
    "[": " ", "]": " ", "|": " ", "/": " ", "#": " ", "→": " ", "←": " ",
}


def _link_cached(source: Path, target: Path) -> Path:
    """Expose immutable provider assets without copying or replacing owner files."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if target.resolve() != source.resolve():
            raise FileExistsError(f"Model compatibility path conflicts: {target}")
    else:
        target.symlink_to(source, target_is_directory=source.is_dir())
    return target


def _download_bundle(repo: str, cache: str = CACHE_SUBDIR) -> Path:
    from huggingface_hub import snapshot_download

    snapshot = Path(snapshot_download(repo_id=repo, allow_patterns=[
        "*.mlpackage/**", "tts.json", "unicode_indexer.json",
        "voice_styles/*.json", "README.md", "manifest.json",
    ]))
    root = runtime_path(cache) / snapshot.name
    root.mkdir(parents=True, exist_ok=True)
    for entry in snapshot.iterdir():
        if entry.name == 'voice_styles' and entry.is_dir():
            for style in entry.iterdir():
                _link_cached(style, root / 'voice_styles' / style.name)
        else:
            _link_cached(entry, root / entry.name)
    return root


def _download_voice_style(voice: str, local_dir: Path, repo: str = MODEL_REPO) -> Path:
    from huggingface_hub import hf_hub_download
    voice_file = f"voice_styles/{voice}.json"
    local_file = local_dir / voice_file
    if local_file.exists():
        return local_file
    try:
        source = hf_hub_download(repo_id=repo, filename=voice_file)
    except Exception:
        # Some conversions omit styles; preserve the upstream fallback.
        source = hf_hub_download(repo_id="Supertone/supertonic-3", filename=voice_file)
    return _link_cached(Path(source), local_file)


def _load_coreml_model(package: Path, compute_units):
    """Keep compiled models in writable runtime storage, keyed by provider asset."""
    import coremltools as ct
    identity = hashlib.sha256(str(package.resolve()).encode()).hexdigest()[:16]
    compiled = runtime_path('coreml-compiled') / identity / f'{package.stem}.mlmodelc'
    if not compiled.is_dir():
        compiled.parent.mkdir(parents=True, exist_ok=True)
        temporary = ct.models.utils.compile_model(str(package))
        shutil.move(temporary, compiled)
    return ct.models.CompiledMLModel(str(compiled), compute_units=compute_units)


def _compute_units(value: str):
    import coremltools as ct

    mapping = {
        "all": ct.ComputeUnit.ALL,
        "cpu": ct.ComputeUnit.CPU_ONLY,
        "cpu_and_gpu": ct.ComputeUnit.CPU_AND_GPU,
        "cpu_and_ne": ct.ComputeUnit.CPU_AND_NE,
    }
    if value.lower() not in mapping:
        raise ValueError(f"unsupported compute_units '{value}'")
    return mapping[value.lower()]


def preprocess_text(text: str, lang: str) -> str:
    text = normalize("NFKD", text)
    text = _EMOJI_RE.sub("", text)
    for k, v in _CHAR_REPL.items():
        text = text.replace(k, v)
    text = re.sub(r"\s+", " ", text).strip()
    if not re.search(r"[.!?;:,'\"')\]}…。」』】〉》›»]$", text):
        text += "."
    if lang not in SUPPORTED_LANGS:
        raise ValueError(f"Unsupported lang '{lang}'. Available: {SUPPORTED_LANGS}")
    return f"<{lang}>" + text + f"</{lang}>"


def tokenize(text: str, lang: str, indexer: list):
    import numpy as np
    s = preprocess_text(text, lang)
    ids = np.zeros((1, TEXT_T_FIXED), dtype=np.int32)
    mask = np.zeros((1, 1, TEXT_T_FIXED), dtype=np.float32)
    codepoints = [ord(c) for c in s][:TEXT_T_FIXED]
    for i, cp in enumerate(codepoints):
        ids[0, i] = indexer[cp] if cp < len(indexer) else 0
    mask[0, 0, : len(codepoints)] = 1.0
    return ids, mask


def load_voice_style(path: Path):
    import numpy as np
    with open(path) as f:
        cfg = json.load(f)
    ttl_d = cfg["style_ttl"]["dims"]
    dp_d = cfg["style_dp"]["dims"]
    ttl = np.array(cfg["style_ttl"]["data"], dtype=np.float32).reshape(1, ttl_d[1], ttl_d[2])
    dp = np.array(cfg["style_dp"]["data"], dtype=np.float32).reshape(1, dp_d[1], dp_d[2])
    return ttl, dp


def sample_noisy_latent(
    duration_sec: float, sample_rate: int, base_chunk_size: int,
    chunk_compress_factor: int, latent_dim: int, rng,
):
    import numpy as np
    wav_len = int(duration_sec * sample_rate)
    chunk_size = base_chunk_size * chunk_compress_factor
    L = (wav_len + chunk_size - 1) // chunk_size
    noisy = rng.standard_normal((1, latent_dim * chunk_compress_factor, L)).astype(np.float32)
    latent_mask = np.zeros((1, 1, L), dtype=np.float32)
    latent_mask[0, 0, :L] = 1.0
    return noisy * latent_mask, latent_mask


def pad_last(arr, target: int):
    if arr.shape[-1] >= target:
        return arr
    import numpy as np
    pad = [(0, 0)] * arr.ndim
    pad[-1] = (0, target - arr.shape[-1])
    return np.pad(arr, pad, constant_values=0.0)


def _split_text(text: str, max_len: int) -> list[str]:
    cap = min(max_len, 110)
    source = re.sub(r"\s+", " ", text.strip())
    if len(source) <= cap:
        return [source]

    sentences = re.split(r"(?<=[.!?。！？])\s+", source)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        tentative = f"{current} {sentence}".strip()
        if current and len(tentative) > cap:
            chunks.append(current)
            current = sentence
        else:
            current = tentative
    if current:
        chunks.append(current)

    if not chunks:
        chunks = [source]

    final_chunks: list[str] = []
    for chunk in chunks:
        if len(chunk) <= cap:
            final_chunks.append(chunk)
            continue
        for start in range(0, len(chunk), cap):
            final_chunks.append(chunk[start : start + cap].strip())
    return [chunk for chunk in final_chunks if chunk]


def _peak_normalize(audio):
    import numpy as np
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak <= 0.0:
        return audio.astype(np.float32)
    return (audio / peak * 0.95).astype(np.float32)


class Supertonic3TTS:
    def __init__(self, model_dir: Path, compute_units=None):
        import coremltools as ct
        import numpy as np
        if compute_units is None:
            compute_units = ct.ComputeUnit.CPU_AND_NE
        with open(model_dir / "tts.json") as f:
            cfg = json.load(f)
        self.sample_rate = int(cfg["ae"]["sample_rate"])
        self.base_chunk_size = int(cfg["ae"]["base_chunk_size"])
        self.ccf = int(cfg["ttl"]["chunk_compress_factor"])
        self.ldim = int(cfg["ttl"]["latent_dim"])

        with open(model_dir / "unicode_indexer.json") as f:
            self.indexer = json.load(f)

        def _load(name: str):
            return _load_coreml_model(model_dir / f"{name}.mlpackage", compute_units)

        self.dp = _load("DurationPredictor")
        self.te = _load("TextEncoder")
        self.ve = _load("VectorEstimator")
        self.vc = _load("Vocoder")
        self.rng = np.random.default_rng()

    def synthesize(self, text: str, voice_style_path: Path, lang: str = "en",
                   total_step: int = 8, speed: float = 1.05):
        import numpy as np
        ttl, dp_style = load_voice_style(voice_style_path)
        text_ids, text_mask = tokenize(text, lang, self.indexer)

        # 1. Duration.
        dp_out = self.dp.predict({
            "text_ids": text_ids, "style_dp": dp_style, "text_mask": text_mask,
        })
        duration = float(np.asarray(dp_out["duration"], dtype=np.float32)[0]) / speed

        # 2. Text embedding.
        te_out = self.te.predict({
            "text_ids": text_ids, "style_ttl": ttl, "text_mask": text_mask,
        })
        text_emb = np.asarray(te_out["text_emb"], dtype=np.float32)

        # 3. Noisy latent.
        noisy, latent_mask = sample_noisy_latent(
            duration, self.sample_rate, self.base_chunk_size, self.ccf, self.ldim, self.rng,
        )
        L_true = noisy.shape[-1]
        L_use = max(L_true, VEC_EST_L_MIN)
        noisy = pad_last(noisy, L_use)
        latent_mask = pad_last(latent_mask, L_use)

        # 4. 8-step flow-matching diffusion.
        xt = noisy
        total_t = np.array([float(total_step)], dtype=np.float32)
        for step in range(total_step):
            cur_t = np.array([float(step)], dtype=np.float32)
            ve_out = self.ve.predict({
                "noisy_latent": xt, "text_emb": text_emb, "style_ttl": ttl,
                "latent_mask": latent_mask, "text_mask": text_mask,
                "current_step": cur_t, "total_step": total_t,
            })
            xt = np.asarray(ve_out["denoised_latent"], dtype=np.float32)

        # 5. Vocoder → 44.1 kHz wav.
        vc_out = self.vc.predict({"latent": xt})
        wav = np.asarray(vc_out["wav"], dtype=np.float32)
        wav = wav[:, : (self.base_chunk_size * self.ccf) * L_true]  # trim pad
        wav = wav[0, : int(self.sample_rate * duration)]            # trim per-sample
        return wav, duration


def prep(variant: str = DEFAULT_VARIANT, force: bool = False):
    """Download and cache all Supertonic-3 CoreML model weights without synthesizing. Idempotent."""
    spec = _variant(variant)
    bundle_root = _download_bundle(spec["repo"], spec["cache"])
    # Also cache all voice styles.
    for voice in VALID_VOICES:
        _download_voice_style(voice, bundle_root, spec["repo"])
    print(f"[prep] ready: {spec['repo']}")


def say(
    text: str,
    output: str = "output.wav",
    variant: str = DEFAULT_VARIANT,
    voice: str = "M1",
    lang_code: str = "en",
    steps: int = 8,
    speed: float = 1.05,
    silence_seconds: float = 0.3,
    compute_units: str = "",
    play: bool = False,
    force: bool = False,
):
    """Generate speech using Supertonic-3 (CoreML).

    10 built-in voice styles (M1-M5 male, F1-F5 female). 44.1kHz output.
    Supports 31 languages.

    Args:
        text: Text to synthesize.
        output: Output WAV file path (44.1kHz).
        variant: Which CoreML conversion to run (`fluid` or `aufklarer-fp16`).
        voice: Voice style: M1-M5 (male), F1-F5 (female).
        lang_code: Language ISO code (e.g. en, ko, ja, es, fr, de, etc.).
        steps: Flow-matching steps (default: 8).
        speed: Speed factor (default: 1.05).
        silence_seconds: Silence pad between chunks (default: 0.3).
        compute_units: CoreML compute units (all, cpu, cpu_and_gpu, cpu_and_ne).
            Defaults to whichever path the chosen variant actually runs on.
        play: Play audio via afplay instead of saving (macOS).
        force: Overwrite output file if it exists.
    """
    import numpy as np
    import soundfile as sf
    from rich.console import Console
    from rich.panel import Panel

    console = Console()
    out_path = Path(output)
    spec = _variant(variant)

    if platform.system() != "Darwin":
        console.print("[red]Supertonic CoreML requires macOS.[/red]")
        raise SystemExit(1)

    if voice not in VALID_VOICES:
        console.print(f"[red]Error:[/red] Invalid voice. Choose from {', '.join(VALID_VOICES)}")
        raise SystemExit(1)

    if lang_code not in SUPPORTED_LANGS:
        console.print(f"[red]Error:[/red] Invalid lang_code. Choose from {', '.join(SUPPORTED_LANGS)}")
        raise SystemExit(1)

    if (
        not play
        and not force
        and out_path.exists()
        and out_path.stat().st_size > 0
    ):
        console.print(
            f"[yellow]Skipped:[/yellow] {output} already exists (use --force to overwrite)"
        )
        return

    console.print(
        Panel.fit(
            f"[bold cyan]Supertonic 3 CoreML[/bold cyan] v{VERSION}\n"
            f"repo=[yellow]{spec['repo']}[/yellow]\n"
            f"voice=[yellow]{voice}[/yellow]  lang=[yellow]{lang_code}[/yellow]  steps=[yellow]{steps}[/yellow]",
            border_style="cyan",
        )
    )

    with console.status("[cyan]Downloading Supertonic 3 bundle...[/cyan]"):
        bundle_root = _download_bundle(spec["repo"], spec["cache"])
        voice_style_path = _download_voice_style(voice, bundle_root, spec["repo"])

    with console.status("[cyan]Loading CoreML models...[/cyan]"):
        units = compute_units or spec.get("compute_units", "all")
        tts = Supertonic3TTS(bundle_root, _compute_units(units))

    chunks = _split_text(text, 110)
    audio_chunks: list[np.ndarray] = []

    for idx, chunk in enumerate(chunks):
        with console.status(f"[cyan]Synthesizing chunk {idx+1}/{len(chunks)}...[/cyan]"):
            wav, dur = tts.synthesize(chunk, voice_style_path, lang_code, steps, speed)
            if silence_seconds > 0.0 and idx < len(chunks) - 1:
                silence = np.zeros(int(silence_seconds * tts.sample_rate), dtype=np.float32)
                wav = np.concatenate([wav, silence])
            audio_chunks.append(wav)

    final_audio = np.concatenate(audio_chunks) if len(audio_chunks) > 1 else audio_chunks[0]
    final_audio = _peak_normalize(final_audio)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), final_audio, tts.sample_rate)

    if play:
        console.print(f"[cyan]Playing[/cyan] [bold]{out_path}[/bold]")
        subprocess.run(["afplay", str(out_path)], check=True)
    else:
        duration_seconds = len(final_audio) / tts.sample_rate
        console.print(
            f"[green]Saved[/green] [bold]{out_path.resolve()}[/bold] "
            f"({tts.sample_rate} Hz, {duration_seconds:.1f}s)"
        )


def info(variant: str = DEFAULT_VARIANT):
    """Print one-line JSON with model metadata."""
    import json as _json
    spec = _variant(variant)
    print(_json.dumps({
        "name": f"supertonic-3-coreml ({variant})",
        "repo": spec["repo"],
        "backend": "coreml",
        "languages": SUPPORTED_LANGS,
        "sample_rate_hz": 44100,
        "modes": ["auto"],
        "capabilities": ["preset-voices", "multilingual", "flow-matching-diffusion"],
    }, separators=(",", ":")))


if __name__ == "__main__":
    import fire
    fire.Fire({"prep": prep, "say": say, "info": info})
