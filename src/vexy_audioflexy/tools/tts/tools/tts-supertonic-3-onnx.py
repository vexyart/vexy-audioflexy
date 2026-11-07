#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = ["fire", "rich", "onnxruntime", "numpy", "soundfile", "huggingface-hub"]
# ///
# this_file: src/vexy_audioflexy/tools/tts/tools/tts-supertonic-3-onnx.py
"""Supertonic-3 TTS via ONNX Runtime INT8 (q3146dq4/Supertonic-3-ONNX-INT8).

Backend: onnx. The same four-stage pipeline the CoreML build runs — DurationPredictor,
TextEncoder, VectorEstimator (flow-matching denoiser), Vocoder — but the ONNX export
keeps every axis dynamic, so text and latent tensors are used at their true lengths
instead of the CoreML build's fixed 128-token / 17-frame pins.

44.1 kHz output. 31 languages, 10 preset voices (M1-M5 male, F1-F5 female).
Mode: auto (preset voices; this model takes no speaker reference).
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from unicodedata import normalize

VERSION = "1.0.0"
MODEL_REPO = "q3146dq4/Supertonic-3-ONNX-INT8"

SUPPORTED_LANGS = [
    "en", "ko", "ja", "ar", "bg", "cs", "da", "de", "el", "es",
    "et", "fi", "fr", "hi", "hr", "hu", "id", "it", "lt", "lv",
    "nl", "pl", "pt", "ro", "ru", "sk", "sl", "sv", "tr", "uk",
    "vi", "na",
]
VALID_VOICES = ["M1", "M2", "M3", "M4", "M5", "F1", "F2", "F3", "F4", "F5"]

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


def _bundle() -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(MODEL_REPO))


def preprocess_text(text: str, lang: str) -> str:
    """Normalize, strip emoji, end-punctuate, and wrap in the model's language tags."""
    if lang not in SUPPORTED_LANGS:
        raise SystemExit(f"Unsupported lang '{lang}'. Available: {SUPPORTED_LANGS}")
    text = normalize("NFKD", text)
    text = _EMOJI_RE.sub("", text)
    for k, v in _CHAR_REPL.items():
        text = text.replace(k, v)
    text = re.sub(r"\s+", " ", text).strip()
    if not re.search(r"[.!?;:,'\"')\]}…。」』】〉》›»]$", text):
        text += "."
    return f"<{lang}>{text}</{lang}>"


def tokenize(text: str, lang: str, indexer: list):
    """Codepoint-indexed ids at their true length — the ONNX graph takes any length."""
    import numpy as np

    s = preprocess_text(text, lang)
    codepoints = [ord(c) for c in s]
    # The ONNX export declares int64 text ids (the CoreML build uses int32).
    ids = np.array(
        [[indexer[cp] if cp < len(indexer) else 0 for cp in codepoints]], dtype=np.int64
    )
    mask = np.ones((1, 1, len(codepoints)), dtype=np.float32)
    return ids, mask


def load_voice_style(path: Path):
    import numpy as np

    cfg = json.loads(path.read_text())
    ttl_d = cfg["style_ttl"]["dims"]
    dp_d = cfg["style_dp"]["dims"]
    ttl = np.array(cfg["style_ttl"]["data"], dtype=np.float32).reshape(1, ttl_d[1], ttl_d[2])
    dp = np.array(cfg["style_dp"]["data"], dtype=np.float32).reshape(1, dp_d[1], dp_d[2])
    return ttl, dp


def _split_text(text: str, cap: int = 300) -> list[str]:
    source = re.sub(r"\s+", " ", text.strip())
    if len(source) <= cap:
        return [source]
    chunks: list[str] = []
    current = ""
    for sentence in re.split(r"(?<=[.!?。！？])\s+", source):
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
    return chunks or [source]


def _peak_normalize(audio):
    import numpy as np

    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak <= 0.0:
        return audio.astype(np.float32)
    return (audio / peak * 0.95).astype(np.float32)


class Supertonic3Onnx:
    def __init__(self, root: Path, threads: int = 0):
        import numpy as np
        import onnxruntime as ort

        onnx_dir = root / "onnx"
        cfg = json.loads((onnx_dir / "tts.json").read_text())
        self.sample_rate = int(cfg["ae"]["sample_rate"])
        self.base_chunk_size = int(cfg["ae"]["base_chunk_size"])
        self.ccf = int(cfg["ttl"]["chunk_compress_factor"])
        self.ldim = int(cfg["ttl"]["latent_dim"])
        self.indexer = json.loads((onnx_dir / "unicode_indexer.json").read_text())

        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads

        def _load(name: str):
            return ort.InferenceSession(
                str(onnx_dir / f"{name}.onnx"),
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )

        self.dp = _load("duration_predictor")
        self.te = _load("text_encoder")
        self.ve = _load("vector_estimator")
        self.vc = _load("vocoder")
        self.rng = np.random.default_rng()

    def synthesize(self, text: str, style_path: Path, lang: str, steps: int, speed: float):
        import numpy as np

        ttl, dp_style = load_voice_style(style_path)
        text_ids, text_mask = tokenize(text, lang, self.indexer)

        duration = float(
            np.asarray(
                self.dp.run(
                    None, {"text_ids": text_ids, "style_dp": dp_style, "text_mask": text_mask}
                )[0],
                dtype=np.float32,
            ).reshape(-1)[0]
        ) / speed

        text_emb = np.asarray(
            self.te.run(
                None, {"text_ids": text_ids, "style_ttl": ttl, "text_mask": text_mask}
            )[0],
            dtype=np.float32,
        )

        chunk = self.base_chunk_size * self.ccf
        frames = max(1, (int(duration * self.sample_rate) + chunk - 1) // chunk)
        latent_mask = np.ones((1, 1, frames), dtype=np.float32)
        xt = self.rng.standard_normal(
            (1, self.ldim * self.ccf, frames)
        ).astype(np.float32) * latent_mask

        total_t = np.array([float(steps)], dtype=np.float32)
        for step in range(steps):
            xt = np.asarray(
                self.ve.run(
                    None,
                    {
                        "noisy_latent": xt,
                        "text_emb": text_emb,
                        "style_ttl": ttl,
                        "latent_mask": latent_mask,
                        "text_mask": text_mask,
                        "current_step": np.array([float(step)], dtype=np.float32),
                        "total_step": total_t,
                    },
                )[0],
                dtype=np.float32,
            )

        wav = np.asarray(self.vc.run(None, {"latent": xt})[0], dtype=np.float32)
        wav = wav.reshape(-1)[: int(self.sample_rate * duration)]
        return wav, duration


def prep(force: bool = False) -> None:
    """Download and cache the ONNX bundle without synthesizing. Idempotent."""
    root = _bundle()
    print(f"[prep] ready: {MODEL_REPO} ({root})")


def say(
    text: str,
    output: str = "output.wav",
    voice: str = "M1",
    lang_code: str = "en",
    steps: int = 8,
    speed: float = 1.05,
    silence_seconds: float = 0.3,
    threads: int = 0,
    play: bool = False,
    force: bool = False,
) -> None:
    """Generate speech using Supertonic-3 (ONNX INT8).

    Args:
        text: Text to synthesize.
        output: Output WAV path (44.1 kHz).
        voice: Preset voice: M1-M5 (male), F1-F5 (female).
        lang_code: Language ISO code (en, pl, de, fr, ...).
        steps: Flow-matching denoising steps.
        speed: Speech speed multiplier.
        silence_seconds: Silence inserted between long-form chunks.
        threads: ONNX Runtime intra-op threads; 0 leaves the runtime default.
        play: Play the result with afplay instead of reporting the path.
        force: Overwrite an existing output file.
    """
    import numpy as np
    import soundfile as sf
    from rich.console import Console
    from rich.panel import Panel

    console = Console()
    out_path = Path(output)

    if voice not in VALID_VOICES:
        console.print(f"[red]Error:[/red] Invalid voice. Choose from {', '.join(VALID_VOICES)}")
        raise SystemExit(1)

    if not play and not force and out_path.exists() and out_path.stat().st_size > 0:
        console.print(
            f"[yellow]Skipped:[/yellow] {output} already exists (use --force to overwrite)"
        )
        return

    console.print(
        Panel.fit(
            f"[bold cyan]Supertonic 3 ONNX INT8[/bold cyan] v{VERSION}\n"
            f"repo=[yellow]{MODEL_REPO}[/yellow]\n"
            f"voice=[yellow]{voice}[/yellow]  lang=[yellow]{lang_code}[/yellow]  "
            f"steps=[yellow]{steps}[/yellow]",
            border_style="cyan",
        )
    )

    with console.status("[cyan]Downloading bundle...[/cyan]"):
        root = _bundle()
    style_path = root / "voice_styles" / f"{voice}.json"

    with console.status("[cyan]Loading ONNX sessions...[/cyan]"):
        tts = Supertonic3Onnx(root, threads)

    chunks = _split_text(text)
    pieces: list[np.ndarray] = []
    for idx, chunk in enumerate(chunks):
        with console.status(f"[cyan]Synthesizing chunk {idx + 1}/{len(chunks)}...[/cyan]"):
            wav, _ = tts.synthesize(chunk, style_path, lang_code, steps, speed)
            if silence_seconds > 0.0 and idx < len(chunks) - 1:
                wav = np.concatenate(
                    [wav, np.zeros(int(silence_seconds * tts.sample_rate), dtype=np.float32)]
                )
            pieces.append(wav)

    audio = _peak_normalize(np.concatenate(pieces) if len(pieces) > 1 else pieces[0])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), audio, tts.sample_rate)

    if play:
        subprocess.run(["afplay", str(out_path)], check=True)
    else:
        console.print(
            f"[green]Saved[/green] [bold]{out_path.resolve()}[/bold] "
            f"({tts.sample_rate} Hz, {len(audio) / tts.sample_rate:.1f}s)"
        )


def info() -> None:
    """Print one-line JSON with model metadata."""
    print(json.dumps({
        "name": "supertonic-3-onnx-int8",
        "repo": MODEL_REPO,
        "backend": "onnx",
        "quant": "int8",
        "languages": SUPPORTED_LANGS,
        "sample_rate_hz": 44100,
        "modes": ["auto"],
        "capabilities": ["preset-voices", "multilingual", "flow-matching-diffusion"],
    }, separators=(",", ":")))


if __name__ == "__main__":
    import fire

    fire.Fire({"prep": prep, "say": say, "info": info})
