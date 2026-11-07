#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = ["fire", "rich", "funasr", "torch", "torchaudio", "soundfile"]
# ///
# this_file: src/vexy_audioflexy/tools/asr/tools/asr-funasr.py
"""FunASR (ModelScope): SenseVoice and Paraformer from the toolkit that trained them.

`asr` already runs SenseVoice and Paraformer as sherpa-onnx ONNX exports. This is
the same weights through their own PyTorch toolkit, which is the point: it is the
reference the exports are compared against, and it adds the pieces the exports drop
— the VAD-chunked long-form pipeline, and the CT-Transformer punctuation model.

**No Polish.** SenseVoice covers zh/en/ja/ko/yue and Paraformer-en is English only.
Both are registered `en` only; asked for Polish they would still emit confident
text, which is why the runner is not told they speak it.

Weights come from ModelScope, not HuggingFace, and cache under
`~/.cache/modelscope`. `MODELSCOPE_CACHE` moves that.

SenseVoice returns its transcript wrapped in tags —
`<|en|><|NEUTRAL|><|Speech|><|withitn|>…` — which are language, emotion, event and
inverse-text-normalisation labels, not speech. `rich_transcription_postprocess`
strips them; the raw string is kept in the result line as `tagged`.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

VERSION = "1.0.0"

PACKAGES: dict[str, dict] = {
    "sensevoice-small": {
        "model": "iic/SenseVoiceSmall",
        "langs": ["en"],
        "declared": ["zh", "en", "ja", "ko", "yue"],
        "lang_flag": True,
        "quant": "fp32",
    },
    "paraformer-en": {
        "model": "iic/speech_paraformer_asr-en-16k-vocab4199-pytorch",
        "langs": ["en"],
        "declared": ["en"],
        "lang_flag": False,
        "quant": "fp32",
    },
    "paraformer-zh": {
        "model": "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        "langs": [],  # zh only: registered nowhere, reachable from the CLI
        "declared": ["zh"],
        "lang_flag": False,
        "quant": "fp32",
    },
}

# Optional pipeline pieces, off by default. They cost a second model download each.
VAD_MODEL = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
PUNC_MODEL = "iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch"


def _spec(package: str) -> dict:
    try:
        return PACKAGES[package]
    except KeyError:
        raise SystemExit(
            f"unknown package '{package}'. Known: {', '.join(PACKAGES)}"
        ) from None


def _model(package: str, vad: bool = False, punc: bool = False):
    from funasr import AutoModel

    spec = _spec(package)
    kwargs: dict[str, object] = {
        "model": spec["model"],
        "device": "cpu",
        "disable_update": True,
    }
    if vad:
        kwargs["vad_model"] = VAD_MODEL
    if punc:
        kwargs["punc_model"] = PUNC_MODEL
    return AutoModel(**kwargs)


def _duration(path: Path) -> float:
    import soundfile as sf

    try:
        info = sf.info(str(path))
        return info.frames / info.samplerate
    except Exception:
        return 0.0


def prep(package: str = "sensevoice-small", vad: bool = False, punc: bool = False,
         force: bool = False) -> None:
    """Download the model from ModelScope WITHOUT transcribing. Idempotent.

    Args:
        package: Registered package name; `packages` lists them.
        vad: Also fetch the FSMN VAD used for long-form chunking.
        punc: Also fetch the CT-Transformer punctuation model.
        force: Unused; cached weights are never re-fetched.
    """
    spec = _spec(package)
    _model(package, vad=vad, punc=punc)
    print(f"[prep] ready: {package} ({spec['model']})")


def transcribe(
    audio: str,
    package: str = "sensevoice-small",
    text_out: str = "",
    lang: str = "",
    vad: bool = False,
    punc: bool = False,
    force: bool = False,
) -> None:
    """Transcribe one audio file with FunASR.

    Args:
        audio: Input audio file. FunASR resamples to 16 kHz itself.
        package: Registered package name.
        text_out: Transcript path (default: *audio* with a `.txt` suffix).
        lang: Language hint, honoured by SenseVoice only; Paraformer checkpoints are
            single-language and reject it.
        vad: Chunk with the FSMN VAD first — needed for clips over ~30 s.
        punc: Restore punctuation with the CT-Transformer model.
        force: Overwrite an existing transcript.
    """
    from rich.console import Console

    console = Console()
    src = Path(audio)
    if not src.exists():
        raise SystemExit(f"audio not found: {src}")
    spec = _spec(package)
    if lang and lang not in spec["declared"]:
        raise SystemExit(
            f"package '{package}' covers {'/'.join(spec['declared'])}, not '{lang}'"
        )
    out_path = Path(text_out) if text_out else src.with_suffix(".txt")
    if not force and out_path.exists() and out_path.stat().st_size > 0:
        console.print(f"[yellow]Skipped:[/yellow] {out_path} already exists")
        return

    console.print(f"[cyan]FunASR[/cyan] v{VERSION} package=[yellow]{package}[/yellow]")
    model = _model(package, vad=vad, punc=punc)
    kwargs: dict[str, object] = {"input": str(src.resolve())}
    if spec["lang_flag"]:
        kwargs["language"] = lang or "auto"
        kwargs["use_itn"] = True

    t0 = time.monotonic()
    results = model.generate(**kwargs)
    wall = time.monotonic() - t0

    if not results:
        console.print("[red]empty transcript[/red]")
        raise SystemExit(1)
    tagged = str(results[0].get("text", ""))

    from funasr.utils.postprocess_utils import rich_transcription_postprocess

    transcript = rich_transcription_postprocess(tagged).strip()
    if not transcript:
        console.print("[red]empty transcript[/red]")
        raise SystemExit(1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(transcript + "\n", encoding="utf-8")
    audio_s = _duration(src)
    console.print(f"[green]Transcript[/green] {out_path.resolve()}\n  {transcript}")
    print("asr.result=" + json.dumps(
        {
            "text": transcript,
            "text_out": str(out_path),
            "wall_s": round(wall, 4),
            "audio_duration_s": round(audio_s, 4),
            "rtf": round(wall / audio_s, 6) if audio_s else 0.0,
            "package": package,
            "tagged": tagged,
        },
        ensure_ascii=False, separators=(",", ":"),
    ))


def info(package: str = "sensevoice-small") -> None:
    """Print one-line JSON with package metadata."""
    spec = _spec(package)
    print(json.dumps(
        {
            "name": package,
            "family": "funasr",
            "repo": f"modelscope/FunASR ({spec['model']})",
            "stack": "funasr",
            "runtime": "FunASR (PyTorch, CPU)",
            "quant": spec["quant"],
            "langs": spec["declared"],
            "tasks": ["asr", "vad", "punctuation"],
        },
        separators=(",", ":"),
    ))


def packages() -> None:
    """List the registered packages: name, languages, benchmarked languages."""
    for name, spec in PACKAGES.items():
        bench = "/".join(spec["langs"]) or "(cli only)"
        print(f"{name:20} declared={'/'.join(spec['declared']):18} benchmarked={bench}")


if __name__ == "__main__":
    import fire

    fire.Fire({"prep": prep, "transcribe": transcribe, "info": info, "packages": packages})
