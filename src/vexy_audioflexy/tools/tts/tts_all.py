#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = ["fire", "rich"]
# ///
# this_file: src/vexy_audioflexy/tools/tts/tts_all.py
"""Batch TTS tool — preps then runs ALL supported models, collects results, reports.

Each model is invoked via subprocess (its `prep` then `say` Fire subcommand) so
dependency environments never conflict.

Default behavior: **prep every model (download/cache weights), then say every model.**
Just run `./tts_all.py` and come back later — `out/` will hold every clip plus a
sortable `out/report.html`. Matching request/output receipts permit output reuse;
unbound or changed outputs require explicit `--force`. Reports retain failures.

Usage:
    ./tts_all.py                              # prep ALL, then say ALL (en+pl)
    ./tts_all.py --models f5tts,kokoro-bf16-mlx        # specific models
    ./tts_all.py --langs en                   # English only
    ./tts_all.py --prep=False                 # skip prep, lazy-download during say
    ./tts_all.py --prep-only                  # just download/cache everything, no synthesis
    ./tts_all.py --force                      # redo even if outputs already exist
    ./tts_all.py --outdir /tmp/tts_out        # custom output dir
    ./tts_all.py --verbose                    # show subprocess output
    ./tts_all.py --timeout 600                # longer per-model say timeout
"""

import datetime
import html as html_mod
import json
import os
import runpy
import subprocess
import sys
import tempfile
import time
from pathlib import Path

_identity = runpy.run_path(str(Path(__file__).resolve().parents[1] / "_batch_identity.py"))
from typing import Any

import fire
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from tools._paths import runtime_path

console = Console()

# ---------------------------------------------------------------------------
# Ignored engines — permanently broken, kept in report.json but never executed
# ---------------------------------------------------------------------------

# Upstream projects reviewed on 2026-09-03 (issues/2026-09-02-repo-triage.md) and
# deliberately not wrapped, with the reason. They are not engines this harness can
# score; recorded so the next pass does not re-litigate them.
OUT_OF_SCOPE: dict[str, str] = {
    "mockingbird": (
        "babysor/MockingBird is Chinese SV2TTS. It has no English or Polish path, so "
        "there is nothing here to score it on, and its checkpoints are published on "
        "Chinese cloud drives rather than a package index."
    ),
    "whisperlivekit": (
        "QuentinFuxa/WhisperLiveKit is a streaming ASR server — audio in, text out, "
        "and a service at that. ASR belongs in ../asr, where its models already run."
    ),
    "auto-subs": (
        "tmoroney/auto-subs is a DaVinci Resolve / Premiere plugin, not a synthesis engine."
    ),
    "openless": (
        "Open-Less/openless is a Rust desktop dictation app: speech to text, at the cursor."
    ),
    "dograh": (
        "dograh-hq/dograh is a voice-agent platform. It calls TTS services; it is not one."
    ),
    "voicestudio": (
        "debpalash/VoiceStudio is an end-user application wrapping other people's models "
        "— the same purpose as this project, not an engine for it."
    ),
    "amphion-toolkit": (
        "open-mmlab/Amphion as a whole is a training toolkit with no inference package. "
        "Its one piece that fits a harness is FACodec, wrapped in ../fx as facodec-*."
    ),
}

IGNORED_ENGINES: frozenset[str] = frozenset(
    {
        "higgs",
        "dia",
        "dia-gguf",
        "dia-mlx",
        "f5tts",
        "klonaudio",
        "kokoro-q8-gguf",
        "longcat-1b-4bit",
        "longcat-1b-8bit",
        "longcat-1b-bf16",
        "longcat-1b-mxfp4",
        "longcat-1b-mxfp8",
        "longcat-3.5b-4bit",
        "longcat-3.5b-8bit",
        "longcat-3.5b-mxfp4",
        "miotts-gguf",
        "mlx-tada",
        "moss-tts",
        "neutts-gguf",
        "openaudio-gguf",
        "outetss-mlx",
        "parler-gguf",
        "sesame-csm-gguf",
        "sesame-csm-mlx",
        "soprano-mlx",
        "vibevoice",
        "vibevoice-gguf",
        "vibevoice-mlx",
        "voxtral-mlx",
        "melotts-mlx",
        "supertonic-mlx",
        "kimi-audio",
        "omnivoice-quant-mlx",
        "omnivoice-bf16-mlx",
        "fishs2pro-8bit-mlx",
        "kitten-nano",
        "kitten-mini",
        "kitten-micro",
        "scenema-audio",
        "magpie-mlx",
        "ming-omni-bf16-mlx",
        "magpie-coreml",
        "supertonic-coreml",
        "chatterbox",
        "cosyvoice3-mlx",
        "dramabox",
        "kokoro-fp16-coreml",
        "neutts",
        "outetss",
        "spark-mlx",
        "supertonic-3-mlx",
        "voxcpm-8bit",
        "voxcpm2-8bit",
    }
)
IGNORED_REASON = "Permanently broken on Apple Silicon — kept for record-keeping"

# ---------------------------------------------------------------------------
# Paths (all relative to this script's directory)
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent
TEXT_EN = runtime_path('testdata/text/text-en.md')
TEXT_PL = runtime_path('testdata/text/text-pl.md')

# The canonical speaker dataset: 24 kHz mono Int16 WAV clips, each paired with a
# .lab plain-text transcript. Files are prefixed by language code ("en_", "pl_").
DATASET_DIR = runtime_path('references')

TEXT_FILES = {"en": TEXT_EN, "pl": TEXT_PL}


def _wav_duration(path: Path) -> float:
    """Duration in seconds of a PCM WAV via stdlib (no external deps)."""
    import wave

    try:
        with wave.open(str(path), "rb") as wf:
            return wf.getnframes() / wf.getframerate()
    except Exception:
        return 0.0


def select_reference(lang: str) -> tuple[Path | None, str]:
    """Pick the longest ``{lang}_*.wav`` in the dataset that has a sibling ``.lab``.

    Returns ``(wav_path, transcript)``; a longer clip gives instant voice cloning
    (IVC) models more speaker context. Falls back to ``(None, "")`` if the dataset
    is missing or holds no matched pair for *lang*.
    """
    candidates: list[tuple[Path, Path]] = []
    for wav in sorted(DATASET_DIR.glob(f"{lang}_*.wav")):
        lab = wav.with_suffix(".lab")
        if lab.exists():
            candidates.append((wav, lab))
    if not candidates:
        return None, ""
    wav, lab = max(candidates, key=lambda pair: _wav_duration(pair[0]))
    try:
        text = lab.read_text(encoding="utf-8").strip()
    except OSError:
        text = ""
    return wav, text


# Resolve references once (deterministic: longest matched clip per language).
_REF_EN, _REF_TEXT_EN = None, ""
_REF_PL, _REF_TEXT_PL = None, ""

REFS: dict[str, Path | None] = {"en": _REF_EN, "pl": _REF_PL}

# Transcripts of the chosen reference audio (from .lab) — needed by IVC models
# such as F5-TTS that condition on the reference text.
REF_TEXTS = {"en": _REF_TEXT_EN, "pl": _REF_TEXT_PL}

# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------
# args_fn(text, ref_path_str, out_path_str, lang_code) -> list[str]
# These are positional/keyword args passed AFTER the script path.
# ---------------------------------------------------------------------------

# sherpa-onnx serves several TTS families from one wrapper, selected by
# --package; none of them clones, so `ref` is dropped rather than forwarded.
CHATTTS = "tools/tts-chattts.py"
ARGMAX = "tools/tts-argmax-coreml.py"
EMOTIVOICE = "tools/tts-emotivoice.py"
ESPNET = "tools/tts-espnet.py"
SILERO = "tools/tts-silero.py"
MELOTTS = "tools/tts-melotts.py"
XTTS = "tools/tts-xtts.py"
SHERPAONNX = "tools/tts-sherpaonnx.py"


def _sherpaonnx_args(package: str):
    """args_fn for a sherpa-onnx package: text, output, package."""
    return lambda text, _ref, out, _lang="en": [
        text,
        "--output",
        out,
        "--package",
        package,
    ]


MODELS: list[dict[str, Any]] = [
    # Audio8 TTS Preview 0.6B ONNX INT4
    {
        "name": "audio8-06b-onnx-int4",
        "script": "tools/tts-audio8-onnx.py",
        "repo": "Audio8/Audio8-TTS-Preview-0.6B-ONNX-INT4",
        "voice_clone": True,
        "langs": ["en", "pl"],
        "args_fn": lambda text, ref, out, lang="en": [
            text,
            "--output",
            out,
            "--ref",
            ref,
            "--language",
            lang,
        ] if ref else [text, "--output", out, "--language", lang],
    },
    # Inflect Micro v2 ONNX
    {
        "name": "inflect-micro-v2-onnx",
        "script": "tools/tts-inflect-onnx.py",
        "repo": "owensong/Inflect-Micro-v2-ONNX",
        "voice_clone": False,
        "langs": ["en"],
        "args_fn": lambda text, _ref, out, _lang="en": [
            text,
            "--model",
            "micro",
            "--output",
            out,
        ],
    },
    # Inflect v2 CoreML
    {
        "name": "inflect-v2-coreml",
        "script": "tools/tts-inflect-v2-coreml.py",
        "repo": "FluidInference/inflect-v2-coreml",
        "voice_clone": False,
        "langs": ["en"],
        "args_fn": lambda text, _ref, out, _lang="en": [
            text,
            "--variant",
            "micro",
            "--output",
            out,
        ],
    },
    # Inflect Micro v2 GGUF
    {
        "name": "inflect-micro-v2-gguf",
        "script": "tools/tts-inflect-gguf.py",
        "repo": "remixerdec/Inflect-Micro-v2-GGUF",
        "voice_clone": False,
        "langs": ["en"],
        "args_fn": lambda text, _ref, out, _lang="en": [
            text,
            "--model",
            "micro",
            "--output",
            out,
        ],
    },
    # Inflect Nano v2 ONNX
    {
        "name": "inflect-nano-v2-onnx",
        "script": "tools/tts-inflect-onnx.py",
        "repo": "owensong/Inflect-Nano-v2-ONNX",
        "voice_clone": False,
        "langs": ["en"],
        "args_fn": lambda text, _ref, out, _lang="en": [
            text,
            "--model",
            "nano",
            "--output",
            out,
        ],
    },
    # Inflect Nano v2 GGUF
    {
        "name": "inflect-nano-v2-gguf",
        "script": "tools/tts-inflect-gguf.py",
        "repo": "remixerdec/Inflect-Nano-v2-GGUF",
        "voice_clone": False,
        "langs": ["en"],
        "args_fn": lambda text, _ref, out, _lang="en": [
            text,
            "--model",
            "nano",
            "--output",
            out,
        ],
    },
    # Inflect Nano v2 ONNX by csukuangfj
    {
        "name": "inflect-nano-v2-onnx-csukuangfj",
        "script": "tools/tts-inflect-onnx.py",
        "repo": "csukuangfj/Inflect-Nano-v2-ONNX",
        "voice_clone": False,
        "langs": ["en"],
        "args_fn": lambda text, _ref, out, _lang="en": [
            text,
            "--model",
            "nano-csukuangfj",
            "--output",
            out,
        ],
    },
    # Fish Audio S2 Pro GGUF
    {
        "name": "s2-pro-gguf",
        "lang_agnostic": True,
        "script": "tools/tts-fishs2pro-gguf.py",
        "repo": "rodrigomt/s2-pro-gguf",
        "voice_clone": True,
        "langs": ["en", "pl"],
        "args_fn": lambda text, ref, out, _lang="en": [
            text,
            "--output",
            out,
            "--ref",
            ref,
        ] if ref else [text, "--output", out],
    },
    # Fish Audio S2 Pro 8-bit MLX
    {
        "name": "fish-audio-s2-pro-8bit-mlx",
        "lang_agnostic": True,
        "script": "tools/tts-fishs2pro-mlx.py",
        "repo": "mlx-community/fish-audio-s2-pro-8bit",
        "voice_clone": True,
        "langs": ["en", "pl"],
        "args_fn": lambda text, ref, out, _lang="en": [
            text,
            "--output",
            out,
            "--model",
            "8bit",
            "--ref",
            ref,
        ] if ref else [text, "--output", out, "--model", "8bit"],
    },
    # Fun-CosyVoice3 0.5B GGUF
    {
        "name": "cosyvoice3-0.5b-2512-gguf",
        "script": "tools/tts-cosyvoice3-gguf.py",
        "repo": "cstr/cosyvoice3-0.5b-2512-GGUF",
        "voice_clone": True,
        "langs": ["en", "pl"],
        "args_fn": lambda text, ref, out, lang="en": [
            text,
            "--output",
            out,
            "--ref",
            ref,
            "--language",
            lang,
        ] if ref else [text, "--output", out, "--language", lang],
    },
    # Fun-CosyVoice3 0.5B ONNX
    {
        "name": "fun-cosyvoice3-0.5b-2512-onnx",
        "script": "tools/tts-cosyvoice3-onnx.py",
        "repo": "Lourdle/Fun-CosyVoice3-0.5B-2512_ONNX",
        "voice_clone": False,
        "langs": ["en"],
        "args_fn": lambda text, _ref, out, _lang="en": [
            text,
            "--output",
            out,
        ],
    },
    # ===================================================================
    # Tier 1: Lightweight / CPU / GGUF (no GPU memory pressure)
    # ===================================================================
    # Kokoro ONNX (CPU inference, no GPU)
    {
        "name": "kokoro-fp16-onnx",
        "script": "tools/tts-kokoro-fp16-onnx.py",
        "voice_clone": False,
        "langs": ["en"],
        "args_fn": lambda text, _ref, out, _lang="en": [
            text,
            "--output",
            out,
            "--voice",
            "am_adam",
        ],
    },
    # Kokoro ONNX by Soniqo (CPU inference, preset voices)
    {
        "name": "kokoro-soniqo-onnx",
        "script": "tools/tts-soniqo-kokoro-onnx.py",
        "repo": "soniqo/Kokoro-82M-ONNX",
        "voice_clone": False,
        "langs": ["en"],
        "args_fn": lambda text, _ref, out, _lang="en": [
            text,
            "--output",
            out,
            "--voice",
            "am_adam",
        ],
    },
    # Kokoro GGUF (TTS.cpp, CPU/Metal — lightweight)
    {
        "name": "kokoro-q8-gguf",
        "script": "tools/tts-kokoro-q8-gguf.py",
        "voice_clone": False,
        "langs": ["en"],
        "args_fn": lambda text, _ref, out, _lang="en": [
            text,
            "--output",
            out,
            "--voice",
            "am_adam",
        ],
    },
    # Kokoro CoreML (Apple Neural Engine, no GPU memory)
    # kokoro-fp16-coreml: precision assumed (CoreML default for ANE-deployed Kokoro)
    {
        "name": "kokoro-fp16-coreml",
        "script": "tools/tts-kokoro-fp16-coreml.py",
        "voice_clone": False,
        "langs": ["en"],
        "args_fn": lambda text, _ref, out, _lang="en": [
            text,
            "--output",
            out,
            "--voice",
            "am_adam",
        ],
    },
    # Magpie GGUF (magpie-tts.cpp, CPU/Metal — multilingual)
    {
        "name": "magpie-gguf",
        "script": "tools/tts-magpie-gguf.py",
        "voice_clone": False,
        "langs": ["en", "es", "de", "fr", "vi", "it", "zh"],
        "args_fn": lambda text, _ref, out, lang="en": [
            text,
            "--output",
            out,
            "--speaker",
            "0",
            "--lang_code",
            lang,
        ],
    },
    # Magpie CoreML (Apple Neural Engine / GPU, multilingual)
    {
        "name": "magpie-coreml",
        "script": "tools/tts-magpie-coreml.py",
        "voice_clone": False,
        "langs": ["en", "es", "de", "fr", "it", "vi", "zh", "hi"],
        "args_fn": lambda text, _ref, out, lang="en": [
            text,
            "--output",
            out,
            "--speaker",
            "0",
            "--lang_code",
            lang,
        ],
    },
    # Magpie MLX (Swift/C++ only, ignored)
    {
        "name": "magpie-mlx",
        "script": "tools/tts-magpie-mlx.py",
        "voice_clone": False,
        "langs": ["en"],
        "args_fn": lambda text, _ref, out, _lang="en": [
            text,
            "--output",
            out,
        ],
    },
    # Supertonic CoreML (Apple Neural Engine / GPU, multilingual)
    {
        "name": "supertonic-coreml",
        "script": "tools/tts-supertonic-coreml.py",
        "voice_clone": False,
        "langs": ["en", "ko", "es", "pt", "fr"],
        "args_fn": lambda text, _ref, out, lang="en": [
            text,
            "--output",
            out,
            "--voice",
            "M1",
            "--lang_code",
            lang,
        ],
    },
    # Supertonic-3 CoreML (Apple Neural Engine / GPU, multilingual)
    {
        "name": "supertonic-3-coreml",
        "script": "tools/tts-supertonic-3-coreml.py",
        "voice_clone": False,
        "langs": [
            "en",
            "ko",
            "ja",
            "ar",
            "bg",
            "cs",
            "da",
            "de",
            "el",
            "es",
            "et",
            "fi",
            "fr",
            "hi",
            "hr",
            "hu",
            "id",
            "it",
            "lt",
            "lv",
            "nl",
            "pt",
            "ro",
            "ru",
            "sk",
            "sl",
            "sv",
            "tr",
            "uk",
            "vi",
        ],
        "args_fn": lambda text, _ref, out, lang="en": [
            text,
            "--output",
            out,
            "--voice",
            "M1",
            "--lang_code",
            lang,
        ],
    },
    # Parler GGUF (TTS.cpp — known Metal crash, but low memory)
    {
        "name": "parler-gguf",
        "script": "tools/tts-parler-gguf.py",
        "voice_clone": False,
        "langs": ["en"],
        "args_fn": lambda text, _ref, out, _lang="en": [text, "--output", out],
    },
    # Dia GGUF (TTS.cpp — known Metal crash, but low memory)
    {
        "name": "dia-gguf",
        "script": "tools/tts-dia-gguf.py",
        "voice_clone": False,
        "langs": ["en"],
        "args_fn": lambda text, _ref, out, _lang="en": [
            "[S1] " + text,
            "--output",
            out,
        ],
    },
    # Dia 1.6B GGUF via CrispASR wrapper
    {
        "name": "dia-cstr-gguf",
        "script": "tools/tts-crispasr-gguf.py",
        "repo": "cstr/dia-1.6b-GGUF",
        "voice_clone": False,
        "langs": ["en"],
        "args_fn": lambda text, _ref, out, _lang="en": [
            "[S1] " + text,
            "--model",
            "dia",
            "--output",
            out,
        ],
    },
    # Pocket TTS GGUF via CrispASR wrapper
    {
        "name": "pocket-tts-cstr-gguf",
        "script": "tools/tts-crispasr-gguf.py",
        "repo": "cstr/pocket-tts-GGUF",
        "voice_clone": True,
        "langs": ["en"],
        "args_fn": lambda text, ref, out, _lang="en": [
            text,
            "--model",
            "pocket-tts",
            "--output",
            out,
            "--ref",
            ref,
        ],
    },
    # Orpheus TTS 3B via llama.cpp + ONNX SNAC decoder (preset voices)
    {
        "name": "orpheus-cpp",
        "script": "tools/tts-orpheus-cpp.py",
        "repo": "isaiahbjork/orpheus-3b-0.1-ft-Q4_K_M-GGUF",
        "stack": "gguf",
        "quant": "q4_k",
        "timeout": 1800,
        "voice_clone": False,
        "langs": ["en"],
        "args_fn": lambda text, _ref, out, _lang="en": [
            text,
            "--output",
            out,
        ],
    },
    # Sesame CSM 1B GGUF (llama.cpp — needs custom binary)
    {
        "name": "sesame-csm-gguf",
        "script": "tools/tts-sesame-csm-gguf.py",
        "voice_clone": False,
        "langs": ["en"],
        "args_fn": lambda text, _ref, out, _lang="en": [
            text,
            "--output",
            out,
        ],
    },
    # MioTTS GGUF (needs custom binary)
    {
        "name": "miotts-gguf",
        "script": "tools/tts-miotts-gguf.py",
        "voice_clone": False,
        "langs": ["en"],
        "args_fn": lambda text, _ref, out, _lang="en": [
            text,
            "--output",
            out,
            "--voice",
            "en_female",
        ],
    },
    # NeuTTS GGUF (voice cloning, lightweight)
    {
        "name": "neutts",
        "script": "tools/tts-neutts.py",
        "voice_clone": True,
        "langs": ["en"],
        "args_fn": lambda text, ref, out, lang="en": [
            text,
            "--ref",
            ref,
            "--ref-text",
            REF_TEXTS.get(lang, ""),
            "--output",
            out,
        ],
    },
    # OuteTTS via outetts pip + llama.cpp (hangs sometimes)
    {
        "name": "outetss",
        "script": "tools/tts-outetss.py",
        "voice_clone": True,
        "langs": ["en"],
        "args_fn": lambda text, ref, out, _lang="en": [
            text,
            "--ref_audio",
            ref,
            "--out_file",
            out,
        ],
    },
    # OpenAudio s1-mini GGUF (Fish-Speech engine — GGUF incompatible)
    {
        "name": "openaudio-gguf",
        "script": "tools/tts-openaudio-gguf.py",
        "voice_clone": True,
        "langs": ["en"],
        "args_fn": lambda text, ref, out, lang="en": (
            [
                text,
                out,
                "--ref_audio",
                ref,
            ]
            if ref
            else [text, out]
        ),
    },
    # MOSS-TTS (transformers AutoModel)
    {
        "name": "moss-tts",
        "script": "tools/tts-moss.py",
        "voice_clone": True,
        "langs": ["en"],
        "args_fn": lambda text, ref, out, lang="en": [
            "--text",
            text,
            "--out_file",
            out,
            "--ref_audio",
            ref,
            "--ref_text",
            REF_TEXTS.get(lang, ""),
        ],
    },
    # MOSS-TTS v1.5 GGUF
    {
        "name": "moss-v15-gguf",
        "script": "tools/tts-moss-v15-gguf.py",
        "repo": "smcleod/MOSS-TTS-v1.5-GGUF",
        "voice_clone": True,
        "langs": ["en", "pl"],
        "timeout": 600,
        "args_fn": lambda text, ref, out, lang="en": [
            text,
            "--output",
            out,
            "--ref",
            ref,
            "--language",
            {"en": "en", "pl": "pl"}.get(lang, "en"),
        ],
    },
    # VoxCPM2 ONNX by Soniqo
    {
        "name": "voxcpm2-soniqo-onnx",
        "script": "tools/tts-soniqo-voxcpm2-onnx.py",
        "repo": "soniqo/VoxCPM2-ONNX",
        "voice_clone": True,
        "langs": ["en"],
        "timeout": 600,
        "args_fn": lambda text, ref, out, _lang="en": [
            text,
            "--output",
            out,
            "--ref",
            ref,
        ],
    },
    # Qwen3 0.6B CustomVoice ONNX
    {
        "name": "qwen3-06b-customvoice-onnx",
        "script": "tools/tts-qwen3-06b-customvoice-onnx.py",
        "repo": "tonythethompson/Qwen3-TTS-12Hz-0.6B-CustomVoice-ONNX",
        "voice_clone": False,
        "langs": ["en"],
        "timeout": 600,
        "args_fn": lambda text, _ref, out, _lang="en": [
            text,
            "--output",
            out,
            "--speaker",
            "ryan",
            "--language",
            "english",
        ],
    },
    # neutts-air-q8-gguf (NeuTTS Python package)
    {
        "name": "neutts-gguf",
        "script": "tools/tts-neutts-gguf.py",
        "voice_clone": True,
        "langs": ["en"],
        "args_fn": lambda text, ref, out, lang="en": [
            "--text",
            text,
            "--out_file",
            out,
            "--ref_audio",
            ref,
            "--ref_text",
            REF_TEXTS.get(lang, ""),
        ],
    },
    # ===================================================================
    # Tier 2: MLX-native (small models, fast, moderate GPU memory)
    # ===================================================================
    # Kokoro MLX (82M, fastest)
    {
        "name": "kokoro-bf16-mlx",
        "script": "tools/tts-kokoro-bf16-mlx.py",
        "voice_clone": False,
        "langs": ["en"],
        "args_fn": lambda text, _ref, out, _lang="en": [
            text,
            "--output",
            out,
            "--voice",
            "am_adam",
        ],
    },
    # Soprano MLX (80M, tiny)
    {
        "name": "soprano-mlx",
        "script": "tools/tts-soprano-mlx.py",
        "voice_clone": False,
        "langs": ["en"],
        "args_fn": lambda text, _ref, out, _lang="en": [
            text,
            "--output",
            out,
        ],
    },
    # OuteTTS MLX (0.6B)
    {
        "name": "outetss-mlx",
        "script": "tools/tts-outetss-mlx.py",
        "voice_clone": True,
        "langs": ["en"],
        "args_fn": lambda text, ref, out, _lang="en": [
            text,
            "--ref",
            ref,
            "--output",
            out,
        ],
    },
    # Spark MLX (0.5B)
    {
        "name": "spark-mlx",
        "script": "tools/tts-spark-mlx.py",
        "voice_clone": True,
        "langs": ["en"],
        "args_fn": lambda text, ref, out, _lang="en": [
            text,
            "--ref",
            ref,
            "--output",
            out,
        ],
    },
    {
        "name": "voxtral-mlx",
        "script": "tools/tts-voxtral-mlx.py",
        "voice_clone": False,
        "langs": ["en"],
        "args_fn": lambda text, _ref, out, _lang="en": [
            text,
            "--output",
            out,
        ],
    },
    # Sesame CSM MLX (1B)
    {
        "name": "sesame-csm-mlx",
        "script": "tools/tts-sesame-csm-mlx.py",
        "voice_clone": True,
        "langs": ["en"],
        "args_fn": lambda text, ref, out, lang="en": [
            text,
            "--ref",
            ref,
            "--ref-text",
            REF_TEXTS.get(lang, ""),
            "--output",
            out,
        ],
    },
    # Chatterbox MLX (base multilingual 8-bit via mlx-audio)
    {
        "name": "chatterbox-mlx",
        "script": "tools/tts-chatterbox-mlx.py",
        "repo": "mlx-community/chatterbox-8bit",
        "quant": "8bit",
        "voice_clone": True,
        "langs": ["en", "pl"],
        # Explicit language prevents incorrect multilingual pronunciation.
        "args_fn": lambda text, ref, out, lang="en": [
            text,
            "--ref",
            ref,
            "--output",
            out,
            "--lang",
            lang,
        ],
    },
    # Dia MLX (1.6B, dialogue via mlx-audio, 44.1kHz)
    {
        "name": "dia-mlx",
        "script": "tools/tts-dia-mlx.py",
        "voice_clone": True,
        "langs": ["en"],
        "args_fn": lambda text, ref, out, lang="en": [
            "[S1] ... " + text,
            "--ref",
            ref,
            "--ref-text",
            "[S1] " + REF_TEXTS.get(lang, ""),
            "--output",
            out,
        ],
    },
    # Qwen3 TTS VoiceDesign MLX (1.7B, text-described voices → desi1/desi2/desi3)
    {
        "name": "qwen3-voicedesign-mlx",
        "script": "tools/tts-qwen3-voicedesign-mlx.py",
        "voice_clone": False,
        "design": True,
        "langs": ["en"],
        "args_fn": lambda text, _ref, out, _lang="en": [
            text,
            "--output",
            out,
        ],
        "design_args_fn": lambda text, instruct, out, lang="en": [
            text,
            "--output",
            out,
            "--instruct",
            instruct,
            "--language",
            {"en": "English", "pl": "Polish"}.get(lang, "English"),
        ],
    },
    # Qwen3 TTS Base/CustomVoice MLX (1.7B, voice cloning)
    {
        "name": "qwen3-mlx",
        "script": "tools/tts-qwen3-mlx.py",
        "voice_clone": True,
        "langs": ["en"],
        "args_fn": lambda text, ref, out, _lang="en": [
            text,
            "--ref",
            ref,
            "--ref-text",
            REF_TEXTS.get(_lang, ""),
            "--output",
            out,
        ],
    },
    # Holler 0.6B MLX (Qwen3-TTS variant, preset voice)
    {
        "name": "holler-0.6b",
        "script": "tools/tts-qwen3-mlx.py",
        "voice_clone": False,
        "langs": ["en"],
        "args_fn": lambda text, _ref, out, _lang="en": [
            text,
            "--output",
            out,
            "--model",
            "holler",
            "--voice",
            "kit",
        ],
    },
    # CosyVoice3 0.5B MLX (voice cloning)
    {
        "name": "cosyvoice3-mlx",
        "script": "tools/tts-cosyvoice3-mlx.py",
        "voice_clone": True,
        "langs": ["en"],
        "args_fn": lambda text, ref, out, _lang="en": [
            text,
            "--ref",
            ref,
            *(
                ["--ref-text", REF_TEXTS.get(_lang, "")]
                if REF_TEXTS.get(_lang, "")
                else []
            ),
            "--output",
            out,
        ],
    },
    # mlx-tada-3b
    {
        "name": "mlx-tada",
        "script": "tools/tts-mlxtada.py",
        "voice_clone": True,
        "langs": ["en"],
        "args_fn": lambda text, ref, out, lang="en": [
            "--text",
            text,
            "--out_file",
            out,
            "--ref_audio",
            ref,
            "--ref_text",
            REF_TEXTS.get(lang, ""),
        ],
    },
    # Kitten-TTS nano (14.6M, fastest/smallest)
    {
        "name": "kitten-nano",
        "script": "tools/tts-kitten-mlx.py",
        "voice_clone": False,
        "langs": ["en"],
        "args_fn": lambda text, _ref, out, _lang="en": [
            text,
            "--model",
            "nano",
            "--output",
            out,
        ],
    },
    # Kitten-TTS mini
    {
        "name": "kitten-mini",
        "script": "tools/tts-kitten-mlx.py",
        "voice_clone": False,
        "langs": ["en"],
        "args_fn": lambda text, _ref, out, _lang="en": [
            text,
            "--model",
            "mini",
            "--output",
            out,
        ],
    },
    # Kitten-TTS micro
    {
        "name": "kitten-micro",
        "script": "tools/tts-kitten-mlx.py",
        "voice_clone": False,
        "langs": ["en"],
        "args_fn": lambda text, _ref, out, _lang="en": [
            text,
            "--model",
            "micro",
            "--output",
            out,
        ],
    },
    {
        "name": "higgs",
        "script": "tools/tts-higgs-mlx.py",
        "voice_clone": True,
        "langs": ["en", "pl"],
        "args_fn": lambda text, ref, out, lang="en": [
            text,
            "--output",
            out,
            "--ref",
            ref,
            "--ref-text",
            REF_TEXTS.get(lang, ""),
        ],
    },
    # MisoTTS 8bit MLX
    {
        "name": "miso-8bit-mlx",
        "script": "tools/tts-miso-mlx.py",
        "repo": "mlx-community/MisoLabs-MisoTTS-8bit",
        "voice_clone": True,
        "langs": ["en"],
        "timeout": 600,
        "args_fn": lambda text, ref, out, _lang="en": [
            text,
            "--model",
            "8bit",
            "--output",
            out,
            "--ref",
            ref,
        ],
    },
    # MisoTTS bf16 MLX
    {
        "name": "miso-bf16-mlx",
        "script": "tools/tts-miso-mlx.py",
        "repo": "mlx-community/MisoLabs-MisoTTS-bf16",
        "voice_clone": True,
        "langs": ["en"],
        "timeout": 600,
        "args_fn": lambda text, ref, out, _lang="en": [
            text,
            "--model",
            "bf16",
            "--output",
            out,
            "--ref",
            ref,
        ],
    },
    # Zonos transformer bf16 MLX
    {
        "name": "zonos-transformer-bf16-mlx",
        "script": "tools/tts-zonos-mlx.py",
        "repo": "mlx-community/Zonos-v0.1-transformer-bf16",
        "voice_clone": True,
        "langs": ["en"],
        "timeout": 600,
        "args_fn": lambda text, ref, out, _lang="en": [
            text,
            "--output",
            out,
            "--ref",
            ref,
        ],
    },
    {
        "name": "celune-upbeat-mlx",
        "script": "tools/tts-celune-mlx.py",
        "voice_clone": False,
        "langs": ["en"],
        "args_fn": lambda text, _ref, out, _lang="en": [
            text,
            "--variant",
            "Upbeat",
            "--output",
            out,
        ],
    },
    {
        "name": "celune-energetic-mlx",
        "script": "tools/tts-celune-mlx.py",
        "voice_clone": False,
        "langs": ["en"],
        "args_fn": lambda text, _ref, out, _lang="en": [
            text,
            "--variant",
            "Energetic",
            "--output",
            out,
        ],
    },
    {
        "name": "celune-calm-mlx",
        "script": "tools/tts-celune-mlx.py",
        "voice_clone": False,
        "langs": ["en"],
        "args_fn": lambda text, _ref, out, _lang="en": [
            text,
            "--variant",
            "Calm",
            "--output",
            out,
        ],
    },
    {
        "name": "celune-neutral-mlx",
        "script": "tools/tts-celune-mlx.py",
        "voice_clone": False,
        "langs": ["en"],
        "args_fn": lambda text, _ref, out, _lang="en": [
            text,
            "--variant",
            "Neutral",
            "--output",
            out,
        ],
    },
    # MeloTTS English v1 MLX (VITS2, 207M, 44.1kHz, 5 accent variants)
    {
        "name": "melotts-mlx",
        "script": "tools/tts-melotts-mlx.py",
        "voice_clone": False,
        "langs": ["en"],
        "args_fn": lambda text, _ref, out, _lang="en": [
            text,
            "--speaker",
            "EN-US",
            "--output",
            out,
        ],
    },
    # MeloTTS English v3 MLX (VITS2, improved quality, single voice)
    {
        "name": "melotts-v3-mlx",
        "script": "tools/tts-melotts-mlx.py",
        "voice_clone": False,
        "langs": ["en"],
        "args_fn": lambda text, _ref, out, _lang="en": [
            text,
            "--model",
            "v3",
            "--speaker",
            "EN-Newest",
            "--output",
            out,
        ],
    },
    # Supertonic-2 (44.1kHz, 10 voice styles M1-M5/F1-F5)
    {
        "name": "supertonic-mlx",
        "script": "tools/tts-supertonic-mlx.py",
        "voice_clone": False,
        "langs": ["en"],
        "args_fn": lambda text, _ref, out, _lang="en": [
            text,
            "--voice",
            "M1",
            "--output",
            out,
        ],
    },
    # Supertonic-3 (44.1kHz, 10 voice styles M1-M5/F1-F5) - MLX variant (ignored)
    {
        "name": "supertonic-3-mlx",
        "script": "tools/tts-supertonic-3-mlx.py",
        "voice_clone": False,
        "langs": ["en"],
        "args_fn": lambda text, _ref, out, _lang="en": [
            text,
            "--voice",
            "M1",
            "--output",
            out,
        ],
    },
    {
        "name": "vibevoice-mlx",
        "script": "tools/tts-vibevoice-mlx.py",
        "voice_clone": True,
        "langs": ["en"],
        "args_fn": lambda text, ref, out, _lang="en": [
            text,
            "--ref",
            ref,
            "--output",
            out,
        ],
    },
    # LongCat AudioDiT 3.5B (MLX) — only bf16 and mxfp8 kept
    {
        "name": "longcat-3.5b-bf16",
        "script": "tools/tts-longcat-mlx.py",
        "voice_clone": True,
        "langs": ["en"],
        "timeout": 600,
        "args_fn": lambda text, ref, out, lang="en": [
            text,
            "--ref",
            ref,
            "--ref-text",
            REF_TEXTS.get(lang, ""),
            "--model",
            "3.5b-bf16",
            "--output",
            out,
        ],
    },
    {
        "name": "longcat-3.5b-mxfp8",
        "script": "tools/tts-longcat-mlx.py",
        "voice_clone": True,
        "langs": ["en"],
        "args_fn": lambda text, ref, out, lang="en": [
            text,
            "--ref",
            ref,
            "--ref-text",
            REF_TEXTS.get(lang, ""),
            "--model",
            "3.5b-mxfp8",
            "--output",
            out,
        ],
    },
    # VoxCPM1.5 fp16 (0.6B, 44.1kHz, bilingual en+zh, voice cloning)
    {
        "name": "voxcpm-fp16",
        "script": "tools/tts-voxcpm-mlx.py",
        "voice_clone": True,
        "langs": ["en", "zh"],
        "args_fn": lambda text, ref, out, lang="en": [
            text,
            *(
                ["--ref", ref, "--ref-text", REF_TEXTS.get(lang, "")]
                if REF_TEXTS.get(lang, "")
                else []
            ),
            "--model",
            "fp16",
            "--output",
            out,
        ],
    },
    # VoxCPM1.5 8bit (0.6B quantized, 44.1kHz, bilingual en+zh, voice cloning)
    {
        "name": "voxcpm-8bit",
        "script": "tools/tts-voxcpm-mlx.py",
        "voice_clone": True,
        "langs": ["en", "zh"],
        "args_fn": lambda text, ref, out, lang="en": [
            text,
            *(
                ["--ref", ref, "--ref-text", REF_TEXTS.get(lang, "")]
                if REF_TEXTS.get(lang, "")
                else []
            ),
            "--model",
            "8bit",
            "--output",
            out,
        ],
    },
    # VoxCPM2 bf16 (2B, 48kHz, 30 languages, voice cloning)
    {
        "name": "voxcpm2-bf16",
        "script": "tools/tts-voxcpm2-mlx.py",
        "voice_clone": True,
        "langs": ["en", "pl"],
        "args_fn": lambda text, ref, out, lang="en": [
            text,
            *(["--ref", ref] if ref else []),
            *(
                ["--ref-text", REF_TEXTS.get(lang, "")]
                if REF_TEXTS.get(lang, "")
                else []
            ),
            "--model",
            "bf16",
            "--output",
            out,
        ],
    },
    # VoxCPM2 bf16 aufklarer (2B, 48kHz, 30 languages, voice cloning)
    {
        "name": "voxcpm2-bf16-aufklarer",
        "script": "tools/tts-voxcpm2-mlx.py",
        "voice_clone": True,
        "langs": ["en", "pl"],
        "args_fn": lambda text, ref, out, lang="en": [
            text,
            *(["--ref", ref] if ref else []),
            *(
                ["--ref-text", REF_TEXTS.get(lang, "")]
                if REF_TEXTS.get(lang, "")
                else []
            ),
            "--model",
            "bf16-aufklarer",
            "--output",
            out,
        ],
    },
    # VoxCPM2 8bit (2B quantized, 48kHz, 30 languages, ~2x faster)
    {
        "name": "voxcpm2-8bit",
        "script": "tools/tts-voxcpm2-mlx.py",
        "voice_clone": True,
        "langs": ["en", "pl"],
        "args_fn": lambda text, ref, out, lang="en": [
            text,
            *(["--ref", ref] if ref else []),
            *(
                ["--ref-text", REF_TEXTS.get(lang, "")]
                if REF_TEXTS.get(lang, "")
                else []
            ),
            "--model",
            "8bit",
            "--output",
            out,
        ],
    },
    # Ming-omni-tts (16.8B-A3B MoE, EN+ZH, voice cloning) — needs 64+ GB RAM
    {
        "name": "ming-omni-bf16-mlx",
        "script": "tools/tts-ming-omni-bf16-mlx.py",
        "voice_clone": True,
        "langs": ["en"],
        "args_fn": lambda text, ref, out, lang="en": [
            text,
            *(["--ref", ref] if ref else []),
            *(
                ["--ref-text", REF_TEXTS.get(lang, "")]
                if REF_TEXTS.get(lang, "")
                else []
            ),
            "--model",
            "bf16",
            "--output",
            out,
        ],
    },
    # Fish Audio S2 Pro bf16 (5B dual-AR, 11GB, 77 languages, voice cloning)
    {
        "name": "fishs2pro-bf16-mlx",
        "script": "tools/tts-fishs2pro-mlx.py",
        "voice_clone": True,
        "langs": ["en", "pl"],
        "timeout": 600,
        "args_fn": lambda text, ref, out, lang="en": [
            text,
            *(["--ref", ref] if ref else []),
            *(
                ["--ref-text", REF_TEXTS.get(lang, "")]
                if REF_TEXTS.get(lang, "")
                else []
            ),
            "--model",
            "bf16",
            "--output",
            out,
        ],
    },
    # Fish Audio S2 Pro 8bit (5B dual-AR, 6.7GB, 77 languages, voice cloning)
    {
        "name": "fishs2pro-8bit-mlx",
        "script": "tools/tts-fishs2pro-mlx.py",
        "voice_clone": True,
        "langs": ["en", "pl"],
        "timeout": 600,
        "args_fn": lambda text, ref, out, lang="en": [
            text,
            *(["--ref", ref] if ref else []),
            *(
                ["--ref-text", REF_TEXTS.get(lang, "")]
                if REF_TEXTS.get(lang, "")
                else []
            ),
            "--model",
            "8bit",
            "--output",
            out,
        ],
    },
    # OmniVoice bf16 (0.6B backbone, 1.37GB, 24kHz, 600+ languages)
    # NOTE: voice cloning broken in current MLX conversion (missing encode weights)
    {
        "name": "omnivoice-bf16-mlx",
        "lang_agnostic": True,
        "script": "tools/tts-omnivoice-mlx.py",
        "voice_clone": False,
        "langs": ["en", "pl"],
        "args_fn": lambda text, _ref, out, _lang="en": [
            text,
            "--model",
            "bf16",
            "--output",
            out,
        ],
    },
    # OmniVoice quantized — model files incomplete (missing model.safetensors)
    {
        "name": "omnivoice-quant-mlx",
        "lang_agnostic": True,
        "script": "tools/tts-omnivoice-mlx.py",
        "voice_clone": False,
        "langs": ["en", "pl"],
        "args_fn": lambda text, _ref, out, _lang="en": [
            text,
            "--model",
            "quantized",
            "--output",
            out,
        ],
    },
    # OmniVoice 8bit Commandeaw (0.6B quantized, 24kHz, 600+ languages)
    {
        "name": "omnivoice-8bit-commandeaw",
        "script": "tools/tts-omnivoice-mlx.py",
        "voice_clone": False,
        "langs": ["en"],
        "args_fn": lambda text, _ref, out, _lang="en": [
            text,
            "--model",
            "8bit-commandeaw",
            "--output",
            out,
        ],
    },
    # OmniVoice bf16 Commandeaw (0.6B, 24kHz, 600+ languages)
    {
        "name": "omnivoice-bf16-commandeaw",
        "lang_agnostic": True,
        "script": "tools/tts-omnivoice-mlx.py",
        "voice_clone": False,
        "langs": ["en", "pl"],
        "args_fn": lambda text, _ref, out, _lang="en": [
            text,
            "--model",
            "bf16-commandeaw",
            "--output",
            out,
        ],
    },
    # Breeze-TTS-2 on MLX. The engine pins its own Python and MLX, so
    # tools/tts-breeze-mlx.py shells out to its checkout rather than importing
    # it. Both mlx-community conversions clone from a reference plus transcript.
    {
        "name": "breeze-tts-2-mlx",
        "script": "tools/tts-breeze-mlx.py",
        "repo": "mlx-community/Breeze-TTS-2-mlx",
        "stack": "mlx",
        "quant": "bf16",
        "voice_clone": True,
        "langs": ["en"],
        "timeout": 3600,
        "prep_args_fn": lambda v="bf16": ["--variant", v],
        "args_fn": lambda text, ref, out, lang="en": [
            text,
            "--variant",
            "bf16",
            "--ref",
            ref,
            "--ref_text",
            REF_TEXTS.get(lang, ""),
            "--output",
            out,
        ],
    },
    {
        "name": "breeze-tts-2-mlx-8bit",
        "script": "tools/tts-breeze-mlx.py",
        "repo": "mlx-community/Breeze-TTS-2-mlx-8bit",
        "stack": "mlx",
        "quant": "8bit",
        "voice_clone": True,
        "langs": ["en"],
        "timeout": 3600,
        "prep_args_fn": lambda v="8bit": ["--variant", v],
        "args_fn": lambda text, ref, out, lang="en": [
            text,
            "--variant",
            "8bit",
            "--ref",
            ref,
            "--ref_text",
            REF_TEXTS.get(lang, ""),
            "--output",
            out,
        ],
    },
    # OmniVoice bf16 Commandeaw, cloning the speaker instead of using the
    # model's own voice. The card documents --ref_audio as optional zero-shot
    # cloning, so this is the same weights in ivc mode; the auto entry above is
    # kept because its default voice is worth having too.
    {
        "name": "omnivoice-bf16-commandeaw-ivc",
        "lang_agnostic": True,
        "script": "tools/tts-omnivoice-mlx.py",
        "voice_clone": True,
        "langs": ["en", "pl"],
        "args_fn": lambda text, ref, out, lang="en": [
            text,
            "--model",
            "bf16-commandeaw",
            "--ref",
            ref,
            "--ref_text",
            REF_TEXTS.get(lang, ""),
            "--output",
            out,
        ],
    },
    # OmniVoice Q8_G64 Lightsofapollo (0.6B quantized, 24kHz, 600+ languages)
    {
        "name": "omnivoice-q8-g64-lightsofapollo",
        "lang_agnostic": True,
        "script": "tools/tts-omnivoice-mlx.py",
        "voice_clone": False,
        "langs": ["en", "pl"],
        "args_fn": lambda text, _ref, out, _lang="en": [
            text,
            "--model",
            "q8-g64-lightsofapollo",
            "--output",
            out,
        ],
    },
    # ===================================================================
    # Tier 3: MPS / PyTorch (heavy GPU memory, run last)
    # ===================================================================
    # F5-TTS (MPS, torch 2.5.1) — EN+ZH only
    {
        "name": "f5tts",
        "script": "tools/tts-f5tts.py",
        "voice_clone": True,
        "langs": ["en"],
        "args_fn": lambda text, ref, out, lang="en": [
            text,
            ref,
            "--output",
            out,
            "--ref-text",
            REF_TEXTS.get(lang, ""),
        ],
    },
    # Chatterbox (MPS, voice cloning)
    {
        "name": "chatterbox",
        "script": "tools/tts-chatterbox.py",
        "voice_clone": True,
        "langs": ["en"],
        "args_fn": lambda text, ref, out, _lang="en": [
            "... " + text,
            "--ref",
            ref,
            "--output",
            out,
        ],
    },
    # Chatterbox Turbo (MPS, faster) — gated model, needs HF_TOKEN
    {
        "name": "chatterbox-turbo",
        "script": "tools/tts-chatterbox-turbo.py",
        "repo": "mlx-community/chatterbox-turbo-8bit",
        "stack": "mlx",
        "quant": "8bit",
        "voice_clone": True,
        # Turbo's model card lists Polish, but it does not deliver it: even with
        # lang_code=pl its Polish transcribes back as "Fontla Bayt, i ditos fontu
        # de la maca i vindorza" while its English is clean. It is a distilled,
        # English-leaning checkpoint; the base chatterbox-8bit is the one whose
        # Polish the lang_code fix actually rescues. English only here.
        "langs": ["en"],
        "env": {
            "HF_TOKEN": os.environ.get(
                "HUGGINGFACE_API_KEY", os.environ.get("HF_TOKEN", "")
            )
        },
        "args_fn": lambda text, ref, out, lang="en": [
            text,
            "--ref",
            ref,
            "--output",
            out,
            "--lang",
            lang,
        ],
    },
    # Qwen3-TTS (multilingual, voice cloning via qwen-tts, heavy)
    {
        "name": "qwen3",
        "script": "tools/tts-qwen3.py",
        "voice_clone": True,
        "langs": ["en"],
        "args_fn": lambda text, ref, out, lang="en": [
            text,
            "--ref",
            ref,
            "--ref-text",
            REF_TEXTS.get(lang, ""),
            "--output",
            out,
            "--language",
            {"en": "English"}.get(lang, "English"),
        ],
    },
    # OmniVoice PyTorch (0.6B, MPS, 24kHz, 600+ languages, voice cloning)
    {
        "name": "omnivoice-fp32-torch",
        "script": "tools/tts-omnivoice-torch.py",
        "voice_clone": True,
        "langs": ["en", "pl"],
        "args_fn": lambda text, ref, out, lang="en": [
            text,
            *(["--ref", ref] if ref else []),
            *(
                ["--ref-text", REF_TEXTS.get(lang, "")]
                if REF_TEXTS.get(lang, "")
                else []
            ),
            "--output",
            out,
        ],
    },
    # VibeVoice (multi-speaker, source install, heavy)
    {
        "name": "vibevoice",
        "script": "tools/tts-vibevoice.py",
        "voice_clone": True,
        "langs": ["en"],
        "args_fn": lambda text, ref, out, _lang="en": [
            text,
            "--ref",
            ref,
            "--output",
            out,
        ],
    },
    # VibeVoice GGUF (dequant + yvoice, heavy MPS)
    {
        "name": "vibevoice-gguf",
        "script": "tools/tts-vibevoice-gguf.py",
        "voice_clone": True,
        "langs": ["en"],
        "args_fn": lambda text, ref, out, _lang="en": [
            text,
            "--ref",
            ref,
            "--output",
            out,
        ],
    },
    # Dia (PyTorch MPS, dialogue, heavy)
    {
        "name": "dia",
        "script": "tools/tts-dia.py",
        "voice_clone": True,
        "langs": ["en"],
        "ext": "mp3",
        "timeout": 600,
        "args_fn": lambda text, ref, out, lang="en": [
            "[S1] ... " + text,
            "--ref",
            ref,
            "--ref-text",
            "[S1] " + REF_TEXTS.get(lang, ""),
            "--output",
            out,
        ],
    },
    # Kimi-Audio 7B (22.6 GB, EN+ZH, heaviest MLX model)
    {
        "name": "kimi-audio",
        "script": "tools/tts-kimi-mlx.py",
        "voice_clone": False,
        "langs": ["en"],
        "timeout": 600,
        "args_fn": lambda text, _ref, out, _lang="en": [
            text,
            "--output",
            out,
        ],
    },
    # KlonAudio (7B, heaviest model — needs long timeout)
    {
        "name": "klonaudio",
        "lang_agnostic": True,
        "script": "tools/tts-klonaudio.py",
        "voice_clone": True,
        "langs": ["en", "pl"],
        "timeout": 600,
        "args_fn": lambda text, ref, out, _lang="en": [
            text,
            "--ref",
            ref,
            "--output",
            out,
        ],
    },
    # Scenema Audio (ignored CUDA model)
    {
        "name": "scenema-audio",
        "script": "tools/tts-scenema-audio.py",
        "voice_clone": False,
        "langs": ["en"],
        "args_fn": lambda text, _ref, out, _lang="en": [
            text,
            "--output",
            out,
        ],
    },
    # DramaBox (MLX model)
    {
        "name": "dramabox",
        "script": "tools/tts-dramabox.py",
        "voice_clone": True,
        "langs": ["en"],
        "args_fn": lambda text, ref, out, _lang="en": [
            text,
            "--output",
            out,
            *(
                [
                    "--ref",
                    ref,
                ]
                if ref
                else []
            ),
        ],
    },
    # Higgs Audio v3 TTS 4B
    {
        "name": "higgs-v3",
        "script": "tools/tts-higgs-v3.py",
        "repo": "bosonai/higgs-audio-v3-tts-4b",
        "voice_clone": True,
        "langs": ["en", "pl"],
        "timeout": 600,
        "args_fn": lambda text, ref, out, lang="en": [
            text,
            "--output",
            out,
            "--ref",
            ref,
            "--language",
            lang,
            *(
                ["--ref-text", REF_TEXTS.get(lang, "")]
                if REF_TEXTS.get(lang, "")
                else []
            ),
        ],
    },
    # Confucius4 TTS
    {
        "name": "confucius4",
        "script": "tools/tts-confucius4.py",
        "repo": "netease-youdao/Confucius4-TTS",
        "voice_clone": True,
        "langs": ["en"],
        "timeout": 600,
        "args_fn": lambda text, ref, out, _lang="en": [
            text,
            "--output",
            out,
            "--ref",
            ref,
            "--lang",
            "en",
        ],
    },
    # MOSS-TTS v1.5
    {
        "name": "moss-v15",
        "script": "tools/tts-moss-v15.py",
        "repo": "OpenMOSS-Team/MOSS-TTS-v1.5",
        "voice_clone": True,
        "langs": ["en", "pl"],
        "timeout": 900,
        "args_fn": lambda text, ref, out, lang="en": [
            text,
            "--output",
            out,
            "--ref",
            ref,
            "--language",
            {"en": "English", "pl": "Polish"}.get(lang, "English"),
        ],
    },
    # --- issue 221: engines added beside the audio.cpp import ---
    # Audio8 TTS 0.1B ONNX INT8 (Mamba/attention hybrid export)
    {
        "name": "audio8-01b-onnx-int8",
        "script": "tools/tts-audio8-onnx.py",
        "repo": "Audio8/audio8-TTS-0.1B-ONNX-INT8",
        "stack": "onnx",
        "quant": "int8",
        "voice_clone": True,
        "langs": ["en", "pl"],
        "timeout": 1800,
        "args_fn": lambda text, ref, out, lang="en": [
            text,
            "--variant",
            "0.1b-int8",
            "--output",
            out,
            "--language",
            lang,
        ] + (["--ref", ref] if ref else []),
    },
    # Supertonic-3 CoreML FP16, aufklarer conversion
    {
        "name": "supertonic-3-coreml-aufklarer",
        "script": "tools/tts-supertonic-3-coreml.py",
        "repo": "aufklarer/Supertonic-3-CoreML-FP16",
        "stack": "coreml",
        "quant": "fp16",
        "voice_clone": False,
        "langs": ["en", "pl"],
        "args_fn": lambda text, _ref, out, lang="en": [
            text,
            "--variant",
            "aufklarer-fp16",
            "--output",
            out,
            "--lang-code",
            lang,
        ],
    },
    # Supertonic-3 ONNX INT8 — see issues/221-audit.md on output quality
    {
        "name": "supertonic-3-onnx-int8",
        "script": "tools/tts-supertonic-3-onnx.py",
        "repo": "q3146dq4/Supertonic-3-ONNX-INT8",
        "stack": "onnx",
        "quant": "int8",
        "voice_clone": False,
        "langs": ["en", "pl"],
        "args_fn": lambda text, _ref, out, lang="en": [
            text,
            "--output",
            out,
            "--lang-code",
            lang,
        ],
    },
    # Chatterbox Nano GGUF via CrispASR
    {
        "name": "chatterbox-nano-gguf",
        "script": "tools/tts-crispasr-gguf.py",
        "repo": "cstr/chatterbox-nano-GGUF",
        "stack": "gguf",
        "quant": "q8",
        "voice_clone": True,
        "langs": ["en"],
        "timeout": 1800,
        "args_fn": lambda text, ref, out, _lang="en": [
            text,
            "--model",
            "chatterbox-nano",
            "--output",
            out,
            "--ref",
            ref,
        ],
    },
    # Chatterbox Multilingual T3 GGUF via codec.cpp
    {
        "name": "chatterbox-mtl-codeccpp",
        "script": "tools/tts-chatterbox-mtl-codeccpp.py",
        "repo": "BricksDisplay/Chatterbox-Multilingual-TTS-GGUF",
        "stack": "codeccpp",
        "quant": "q8_0",
        "voice_clone": True,
        # The upstream card lists 23 languages including Polish, but through
        # codec.cpp the Polish output is Thai script ("Fontlub Oy กันซี ทุกทุพี")
        # while the English is clean. tts-cli exposes no language flag at all, so
        # there is no selector being missed — the model infers from the text and
        # gets it wrong. English only until that host gains one.
        "langs": ["en"],
        "timeout": 1800,
        "args_fn": lambda text, ref, out, lang="en": [
            text,
            "--output",
            out,
            "--ref",
            ref,
            "--ref-text",
            REF_TEXTS.get(lang, ""),
        ],
    },
    # IndexTTS-2 MLX
    {
        "name": "indextts2-mlx",
        "script": "tools/tts-indextts2-mlx.py",
        "repo": "mlx-community/IndexTTS-2-MLX",
        "stack": "mlx",
        "quant": "fp32",
        "voice_clone": True,
        "langs": ["en"],
        "timeout": 1800,
        "args_fn": lambda text, ref, out, _lang="en": [
            text,
            "--output",
            out,
            "--ref",
            ref,
        ],
    },
    # --- issue 222: audio.cpp capabilities the default packages do not reach ---
    # Qwen3 TTS VoiceDesign — a separate package, driven through the vdes route
    {
        "name": "audiocpp-qwen3-voicedesign",
        "script": "tools/tts-audiocpp.py",
        "repo": "audio-cpp/audio.cpp-gguf",
        "package": "qwen3_tts_1_7b_voicedesign_q8_0",
        "prep_args_fn": lambda p="qwen3_tts_1_7b_voicedesign_q8_0": ["--package", p],
        "stack": "audiocpp",
        "quant": "q8_0",
        "voice_clone": False,
        "design": True,
        "langs": ["en"],
        "timeout": 3600,
        "args_fn": lambda text, _ref, out, lang="en": [
            text,
            "--package",
            "qwen3_tts_1_7b_voicedesign_q8_0",
            "--extra=--max-tokens 2048",
            "--output",
            out,
            "--task",
            "vdes",
            "--instruct",
            "A warm adult narrator with clear, natural delivery.",
        ],
        "design_args_fn": lambda text, instruct, out, lang="en": [
            text,
            "--package",
            "qwen3_tts_1_7b_voicedesign_q8_0",
            "--extra=--max-tokens 2048",
            "--output",
            out,
            "--task",
            "vdes",
            "--instruct",
            instruct,
        ],
    },
    # Qwen3 TTS CustomVoice — packaged speakers through --speaker, no reference WAV
    {
        "name": "audiocpp-qwen3-customvoice",
        "script": "tools/tts-audiocpp.py",
        "repo": "audio-cpp/audio.cpp-gguf",
        "package": "qwen3_tts_1_7b_customvoice_q8_0",
        "prep_args_fn": lambda p="qwen3_tts_1_7b_customvoice_q8_0": ["--package", p],
        "stack": "audiocpp",
        "quant": "q8_0",
        "voice_clone": False,
        "langs": ["en"],
        "timeout": 3600,
        "args_fn": lambda text, _ref, out, lang="en": [
            text,
            "--package",
            "qwen3_tts_1_7b_customvoice_q8_0",
            "--extra=--max-tokens 2048",
            "--output",
            out,
            "--task",
            "tts",
            "--speaker",
            "Vivian",
        ],
    },
    # Qwen3 TTS Base 0.6B — the smaller backbone beside the registered 1.7B
    {
        "name": "audiocpp-qwen3-tts-0.6b",
        "script": "tools/tts-audiocpp.py",
        "repo": "audio-cpp/audio.cpp-gguf",
        "package": "qwen3_tts_0_6b_base_q8_0",
        "prep_args_fn": lambda p="qwen3_tts_0_6b_base_q8_0": ["--package", p],
        "stack": "audiocpp",
        "quant": "q8_0",
        "voice_clone": True,
        "langs": ["en"],
        "timeout": 3600,
        "args_fn": lambda text, ref, out, lang="en": [
            text,
            "--package",
            "qwen3_tts_0_6b_base_q8_0",
            "--extra=--max-tokens 2048",
            "--output",
            out,
            "--task",
            "tts",
            "--ref",
            ref,
            "--ref-text",
            REF_TEXTS.get(lang, ""),
        ],
    },
    # IndexTTS2 with explicit emotion conditioning
    {
        "name": "audiocpp-indextts2-emotion",
        "script": "tools/tts-audiocpp.py",
        "repo": "audio-cpp/audio.cpp-gguf",
        "package": "index_tts2_q8_0",
        "prep_args_fn": lambda p="index_tts2_q8_0": ["--package", p],
        "stack": "audiocpp",
        "quant": "q8_0",
        "voice_clone": True,
        "langs": ["en"],
        "timeout": 3600,
        "args_fn": lambda text, ref, out, lang="en": [
            text,
            "--package",
            "index_tts2_q8_0",
            "--output",
            out,
            "--task",
            "clon",
            "--ref",
            ref,
            "--emotion",
            "This is wonderful news, I am delighted to hear it!",
            "--request-options",
            "emotion_alpha=0.6",
            "--lang",
            lang,
        ],
    },
    # DotTTS MeanFlow package beside the registered SOAR one
    {
        "name": "audiocpp-dots-mf",
        "script": "tools/tts-audiocpp.py",
        "repo": "audio-cpp/audio.cpp-gguf",
        "package": "dots_tts_mf_q8_0",
        "prep_args_fn": lambda p="dots_tts_mf_q8_0": ["--package", p],
        "stack": "audiocpp",
        "quant": "q8_0",
        "voice_clone": True,
        "langs": ["en", "pl"],
        "timeout": 3600,
        "args_fn": lambda text, ref, out, lang="en": [
            text,
            "--package",
            "dots_tts_mf_q8_0",
            "--output",
            out,
            "--task",
            "clon",
            "--ref",
            ref,
            "--ref-text",
            REF_TEXTS.get(lang, ""),
            "--request-options",
            "reference_duration_sec=5",
        ],
    },
    # FireRedTTS3 Base package beside the registered Instruct one
    {
        "name": "audiocpp-fireredtts3-base",
        "script": "tools/tts-audiocpp.py",
        "repo": "audio-cpp/audio.cpp-gguf",
        "package": "fireredtts3_base_q8_0",
        "prep_args_fn": lambda p="fireredtts3_base_q8_0": ["--package", p],
        "stack": "audiocpp",
        "quant": "q8_0",
        "voice_clone": True,
        # See the note on the instruct entry: language tags are accepted but
        # non-English output is documented as weaker, and Polish bears it out.
        "langs": ["en"],
        "timeout": 3600,
        "args_fn": lambda text, ref, out, lang="en": [
            text,
            "--package",
            "fireredtts3_base_q8_0",
            "--output",
            out,
            "--task",
            "clon",
            "--ref",
            ref,
            "--request-options",
            "language=" + {"en": "English", "pl": "Polish"}.get(lang, "English"),
        ],
    },
    # FireRedTTS3 voice design — its own request template, no speaker reference
    {
        "name": "audiocpp-fireredtts3-design",
        "script": "tools/tts-audiocpp.py",
        "repo": "audio-cpp/audio.cpp-gguf",
        "package": "fireredtts3_instruct_q8_0",
        "prep_args_fn": lambda p="fireredtts3_instruct_q8_0": ["--package", p],
        "stack": "audiocpp",
        "quant": "q8_0",
        "voice_clone": False,
        "design": True,
        "langs": ["en"],
        "timeout": 3600,
        "args_fn": lambda text, _ref, out, lang="en": [
            text,
            "--package",
            "fireredtts3_instruct_q8_0",
            "--output",
            out,
            "--task",
            "vdes",
            "--request-options",
            "template_name=voice_design;instruction=A warm adult narrator with clear delivery.",
        ],
        "design_args_fn": lambda text, instruct, out, lang="en": [
            text,
            "--package",
            "fireredtts3_instruct_q8_0",
            "--output",
            out,
            "--task",
            "vdes",
            "--request-options",
            f"template_name=voice_design;instruction={instruct}",
        ],
    },
    # FireRedAudio voice design — same family as the registered clone entry
    {
        "name": "audiocpp-firered-audio-design",
        "script": "tools/tts-audiocpp.py",
        "repo": "audio-cpp/audio.cpp-gguf",
        "package": "firered_audio_q8_0",
        "prep_args_fn": lambda p="firered_audio_q8_0": ["--package", p],
        "stack": "audiocpp",
        "quant": "q8_0",
        "voice_clone": False,
        "design": True,
        "langs": ["en"],
        "timeout": 3600,
        "args_fn": lambda text, _ref, out, lang="en": [
            text,
            "--package",
            "firered_audio_q8_0",
            "--output",
            out,
            "--task",
            "vdes",
            "--request-options",
            "template_name=voice_design;instruction=A warm adult narrator with clear delivery.",
        ],
        "design_args_fn": lambda text, instruct, out, lang="en": [
            text,
            "--package",
            "firered_audio_q8_0",
            "--output",
            out,
            "--task",
            "vdes",
            "--request-options",
            f"template_name=voice_design;instruction={instruct}",
        ],
    },
    # OmniVoice voice design — the instruct path the clone entry never takes
    {
        "name": "audiocpp-omnivoice-design",
        # OmniVoice validates each comma-separated item in --instruct against a
        # fixed vocabulary and fails the request on anything outside it, so the
        # prose in DESIGN_PROMPTS cannot be used here.
        "design_prompts": [
            "male, young adult, moderate pitch",
            "male, middle-aged, low pitch",
            "female, young adult, high pitch",
        ],
        "script": "tools/tts-audiocpp.py",
        "repo": "audio-cpp/audio.cpp-gguf",
        "package": "omnivoice_q8_0",
        "prep_args_fn": lambda p="omnivoice_q8_0": ["--package", p],
        "stack": "audiocpp",
        "quant": "q8_0",
        "voice_clone": False,
        "design": True,
        "langs": ["en"],
        "timeout": 3600,
        "args_fn": lambda text, _ref, out, lang="en": [
            text,
            "--package",
            "omnivoice_q8_0",
            "--output",
            out,
            "--task",
            "tts",
            "--instruct",
            "A warm adult narrator with clear, natural delivery.",
        ],
        "design_args_fn": lambda text, instruct, out, lang="en": [
            text,
            "--package",
            "omnivoice_q8_0",
            "--output",
            out,
            "--task",
            "tts",
            "--instruct",
            instruct,
        ],
    },
    # NeuTTS with a second built-in speaker and an emotion token
    {
        "name": "audiocpp-neutts-2e-dave-happy",
        "script": "tools/tts-audiocpp.py",
        "repo": "audio-cpp/audio.cpp-gguf",
        "package": "neutts_2e_orig",
        "prep_args_fn": lambda p="neutts_2e_orig": ["--package", p],
        "stack": "audiocpp",
        "quant": "orig",
        "voice_clone": False,
        "langs": ["en"],
        "timeout": 3600,
        "args_fn": lambda text, _ref, out, _lang="en": [
            text,
            "--package",
            "neutts_2e_orig",
            "--output",
            out,
            "--task",
            "tts",
            "--request-options",
            "voice_id=dave;emotion=happy",
        ],
    },
    # MiDashengLM-Gen — the `gen` route: a structured scene prompt, not plain TTS
    {
        "name": "audiocpp-midashenglm-gen",
        "script": "tools/tts-audiocpp.py",
        "repo": "audio-cpp/audio.cpp-gguf",
        "package": "midashenglm_gen_q8_0",
        "prep_args_fn": lambda p="midashenglm_gen_q8_0": ["--package", p],
        "stack": "audiocpp",
        "quant": "q8_0",
        "voice_clone": False,
        "langs": ["en"],
        "timeout": 3600,
        # This family reads MiDashengLM caption tags rather than a plain
        # paragraph, so the batch text is wrapped in a scene prompt.
        "args_fn": lambda text, _ref, out, _lang="en": [
            "<|caption|> A calm narrator reads a product description in a quiet room. "
            "<|asr|> " + text[:280] + " <|speech|> calm warm male narrator, close microphone "
            "<|music|> <|unknown|> <|sfx|> <|unknown|> <|env|> small quiet room",
            "--package",
            "midashenglm_gen_q8_0",
            "--output",
            out,
            "--task",
            "gen",
            "--request-options",
            "duration_sec=12;guidance_scale=2.0",
        ],
    },
    # --- audio.cpp GGUF families (audio-cpp/audio.cpp-gguf, one C++ engine) ---
    # audiocpp-chatterbox
    {
        "name": "audiocpp-chatterbox",
        "script": "tools/tts-audiocpp.py",
        "repo": "audio-cpp/audio.cpp-gguf",
        "package": "chatterbox_q8_0",
        "prep_args_fn": lambda p="chatterbox_q8_0": ["--package", p],
        "stack": "audiocpp",
        "quant": "q8_0",
        "voice_clone": True,
        # Its docs warn that "non-English and non-Chinese generation can be
        # weaker even though the language tags are accepted", and Polish bears
        # that out: with language=Polish passed correctly, the instruct package
        # renders Greek and the base package renders Polish-shaped noise.
        "langs": ['en', 'pl'],  # Polish verified by transcript, not by the model card
        "timeout": 3600,
        "args_fn": lambda text, ref, out, lang="en": [
            text,
            "--package",
            "chatterbox_q8_0",
            "--output",
            out,
            "--task",
            "clon",
            "--lang",
            lang,
            "--ref",
            ref,
        ],
    },
    # audiocpp-confucius4
    {
        "name": "audiocpp-confucius4",
        "script": "tools/tts-audiocpp.py",
        "repo": "audio-cpp/audio.cpp-gguf",
        "package": "confucius4_tts_orig",
        "prep_args_fn": lambda p="confucius4_tts_orig": ["--package", p],
        "stack": "audiocpp",
        "quant": "orig",
        "voice_clone": True,
        "langs": ['en'],
        "timeout": 3600,
        "args_fn": lambda text, ref, out, lang="en": [
            text,
            "--package",
            "confucius4_tts_orig",
            "--output",
            out,
            "--task",
            "clon",
            "--lang",
            lang,
            "--ref",
            ref,
        ],
    },
    # audiocpp-dots-soar
    {
        "name": "audiocpp-dots-soar",
        "script": "tools/tts-audiocpp.py",
        "repo": "audio-cpp/audio.cpp-gguf",
        "package": "dots_tts_soar_q8_0",
        "prep_args_fn": lambda p="dots_tts_soar_q8_0": ["--package", p],
        "stack": "audiocpp",
        "quant": "q8_0",
        "voice_clone": True,
        # Its docs warn that "non-English and non-Chinese generation can be
        # weaker even though the language tags are accepted", and Polish bears
        # that out: with language=Polish passed correctly, the instruct package
        # renders Greek and the base package renders Polish-shaped noise.
        "langs": ["en"],
        "timeout": 3600,
        "args_fn": lambda text, ref, out, lang="en": [
            text,
            "--package",
            "dots_tts_soar_q8_0",
            "--output",
            out,
            "--task",
            "clon",
            "--extra=--request-option reference_duration_sec=5",
            "--ref",
            ref,
            "--ref-text",
            REF_TEXTS.get(lang, ""),
        ],
    },
    # audiocpp-dramabox
    {
        "name": "audiocpp-dramabox",
        "script": "tools/tts-audiocpp.py",
        "repo": "audio-cpp/audio.cpp-gguf",
        "package": "dramabox_q8_0",
        "prep_args_fn": lambda p="dramabox_q8_0": ["--package", p],
        "stack": "audiocpp",
        "quant": "q8_0",
        "voice_clone": True,
        "langs": ['en'],
        "timeout": 3600,
        "args_fn": lambda text, ref, out, lang="en": [
            text,
            "--package",
            "dramabox_q8_0",
            "--output",
            out,
            "--task",
            "clon",
            "--ref",
            ref,
            # DramaBox sizes its output from an estimate of the prompt
            # duration, and that estimate reads this paragraph at roughly a
            # quarter of its real length: the default scale of 1.1 yields an
            # 8s clip of a 33s text. Scaling the estimate keeps the model
            # pacing itself; pinning duration_sec instead makes it stretch
            # content it does not have and mangles the opening. 5.2 overshoots
            # and degrades badly, so this is close to the usable ceiling.
            "--request-options",
            "duration_scale=4.5",
        ],
    },
    # audiocpp-echo
    {
        "name": "audiocpp-echo",
        "script": "tools/tts-audiocpp.py",
        "repo": "audio-cpp/audio.cpp-gguf",
        "package": "echo_tts_q8_0",
        "prep_args_fn": lambda p="echo_tts_q8_0": ["--package", p],
        "stack": "audiocpp",
        "quant": "q8_0",
        "voice_clone": True,
        "langs": ['en'],
        "timeout": 3600,
        "args_fn": lambda text, ref, out, lang="en": [
            text,
            "--package",
            "echo_tts_q8_0",
            "--output",
            out,
            "--task",
            "clon",
            "--ref",
            ref,
        ],
    },
    # audiocpp-f5tts-habibi
    {
        "name": "audiocpp-f5tts-habibi",
        "script": "tools/tts-audiocpp.py",
        "repo": "audio-cpp/audio.cpp-gguf",
        "package": "habibi_unified",
        "prep_args_fn": lambda p="habibi_unified": ["--package", p],
        "stack": "audiocpp",
        "quant": "orig",
        "voice_clone": True,
        "langs": ['en'],
        # Metal RTF measured at 105x real time: 2.1 s of audio took ~220 s of
        # wall clock, so the test paragraph needs well over an hour.
        "timeout": 7200,
        "args_fn": lambda text, ref, out, lang="en": [
            text,
            "--package",
            "habibi_unified",
            "--output",
            out,
            "--task",
            "clon",
            "--ref",
            ref,
            "--ref-text",
            REF_TEXTS.get(lang, ""),
        ],
    },
    # audiocpp-firered-audio
    {
        "name": "audiocpp-firered-audio",
        "script": "tools/tts-audiocpp.py",
        "repo": "audio-cpp/audio.cpp-gguf",
        "package": "firered_audio_q8_0",
        "prep_args_fn": lambda p="firered_audio_q8_0": ["--package", p],
        "stack": "audiocpp",
        "quant": "q8_0",
        "voice_clone": True,
        "langs": ['en'],
        "timeout": 3600,
        "args_fn": lambda text, ref, out, lang="en": [
            text,
            "--package",
            "firered_audio_q8_0",
            "--output",
            out,
            "--task",
            "clon",
            "--ref",
            ref,
            # The clone path is a template, not just a task: without
            # template_name and a reference transcript the model free-runs.
            "--ref_text",
            REF_TEXTS.get(lang, ""),
            "--extra=--request-option template_name=tts_clone",
        ],
    },
    # audiocpp-fireredtts3
    {
        "name": "audiocpp-fireredtts3",
        "script": "tools/tts-audiocpp.py",
        "repo": "audio-cpp/audio.cpp-gguf",
        "package": "fireredtts3_instruct_q8_0",
        "prep_args_fn": lambda p="fireredtts3_instruct_q8_0": ["--package", p],
        "stack": "audiocpp",
        "quant": "q8_0",
        "voice_clone": True,
        # Its docs warn that "non-English and non-Chinese generation can be
        # weaker even though the language tags are accepted", and Polish bears
        # that out: with language=Polish passed correctly, the instruct package
        # renders Greek and the base package renders Polish-shaped noise.
        "langs": ["en"],
        "timeout": 3600,
        # FireRedTTS3 takes a language *name*, not an ISO code, and defaults to
        # Chinese when none is given — so this entry has to pass one explicitly
        # or it synthesizes Chinese no matter which language it was asked for.
        "args_fn": lambda text, ref, out, lang="en": [
            text,
            "--package",
            "fireredtts3_instruct_q8_0",
            "--output",
            out,
            "--task",
            "clon",
            "--ref",
            ref,
            "--ref-text",
            REF_TEXTS.get(lang, ""),
            "--request-options",
            "language=" + {"en": "English", "pl": "Polish"}.get(lang, "English"),
        ],
    },
    # audiocpp-fish-s2-pro
    {
        "name": "audiocpp-fish-s2-pro",
        "script": "tools/tts-audiocpp.py",
        "repo": "audio-cpp/audio.cpp-gguf",
        "package": "fish_audio_s2_pro_q8_0",
        "prep_args_fn": lambda p="fish_audio_s2_pro_q8_0": ["--package", p],
        "stack": "audiocpp",
        "quant": "q8_0",
        "voice_clone": True,
        "langs": ['en', 'pl'],
        "timeout": 3600,
        "args_fn": lambda text, ref, out, lang="en": [
            text,
            "--package",
            "fish_audio_s2_pro_q8_0",
            "--output",
            out,
            "--task",
            "tts",
            "--ref",
            ref,
            "--ref-text",
            REF_TEXTS.get(lang, ""),
        ],
    },
    # audiocpp-glm-tts
    {
        "name": "audiocpp-glm-tts",
        "script": "tools/tts-audiocpp.py",
        "repo": "audio-cpp/audio.cpp-gguf",
        "package": "glm_tts_q8_0",
        "prep_args_fn": lambda p="glm_tts_q8_0": ["--package", p],
        "stack": "audiocpp",
        "quant": "q8_0",
        "voice_clone": True,
        "langs": ['en'],
        "timeout": 3600,
        "args_fn": lambda text, ref, out, lang="en": [
            text,
            "--package",
            "glm_tts_q8_0",
            "--output",
            out,
            "--task",
            "clon",
            "--ref",
            ref,
            "--ref-text",
            REF_TEXTS.get(lang, ""),
        ],
    },
    # audiocpp-higgs-v3-4b
    {
        "name": "audiocpp-higgs-v3-4b",
        "script": "tools/tts-audiocpp.py",
        "repo": "audio-cpp/audio.cpp-gguf",
        "package": "higgs_audio_tts_4b_q8_0",
        "prep_args_fn": lambda p="higgs_audio_tts_4b_q8_0": ["--package", p],
        "stack": "audiocpp",
        "quant": "q8_0",
        "voice_clone": True,
        "langs": ['en', 'pl'],
        "timeout": 3600,
        "args_fn": lambda text, ref, out, lang="en": [
            text,
            "--package",
            "higgs_audio_tts_4b_q8_0",
            "--output",
            out,
            "--task",
            "tts",
            "--ref",
            ref,
            "--ref-text",
            REF_TEXTS.get(lang, ""),
            # Higgs errors out rather than truncating when a chunk needs more
            # than --max-tokens AR steps; the default pair (1024 chars, 2048
            # tokens) is not enough for the test paragraph.
            "--extra=--text-chunk-size 300 --max-tokens 4096",
        ],
    },
    # audiocpp-indextts2
    {
        "name": "audiocpp-indextts2",
        "script": "tools/tts-audiocpp.py",
        "repo": "audio-cpp/audio.cpp-gguf",
        "package": "index_tts2_q8_0",
        "prep_args_fn": lambda p="index_tts2_q8_0": ["--package", p],
        "stack": "audiocpp",
        "quant": "q8_0",
        "voice_clone": True,
        "langs": ['en'],
        "timeout": 3600,
        "args_fn": lambda text, ref, out, lang="en": [
            text,
            "--package",
            "index_tts2_q8_0",
            "--output",
            out,
            "--task",
            "clon",
            "--lang",
            lang,
            "--ref",
            ref,
        ],
    },
    # audiocpp-indextts2.5
    {
        "name": "audiocpp-indextts2.5",
        "script": "tools/tts-audiocpp.py",
        "repo": "audio-cpp/audio.cpp-gguf",
        "package": "index_tts2_5_q8_0",
        "prep_args_fn": lambda p="index_tts2_5_q8_0": ["--package", p],
        "stack": "audiocpp",
        "quant": "q8_0",
        "voice_clone": True,
        "langs": ['en'],
        "timeout": 3600,
        "args_fn": lambda text, ref, out, lang="en": [
            text,
            "--package",
            "index_tts2_5_q8_0",
            "--output",
            out,
            "--task",
            "clon",
            "--lang",
            lang,
            "--ref",
            ref,
        ],
    },
    # audiocpp-inflect-micro-v2
    {
        "name": "audiocpp-inflect-micro-v2",
        "script": "tools/tts-audiocpp.py",
        "repo": "audio-cpp/audio.cpp-gguf",
        "package": "inflect_micro_v2_orig",
        "prep_args_fn": lambda p="inflect_micro_v2_orig": ["--package", p],
        "stack": "audiocpp",
        "quant": "orig",
        "voice_clone": False,
        "langs": ['en'],
        "timeout": 3600,
        "args_fn": lambda text, _ref, out, _lang="en": [
            text,
            "--package",
            "inflect_micro_v2_orig",
            "--output",
            out,
            "--task",
            "tts",
            # The family phonemizes through espeak-ng and refuses to start
            # without being told where its data lives; Homebrew's copy is the
            # only one on this machine.
            "--session-options",
            "inflect_v2.espeak_library_path=/opt/homebrew/lib/libespeak-ng.dylib;"
            "inflect_v2.espeak_data_path=/opt/homebrew/share/espeak-ng-data",
        ],
    },
    # audiocpp-magpie-357m
    {
        "name": "audiocpp-magpie-357m",
        "script": "tools/tts-audiocpp.py",
        "repo": "audio-cpp/audio.cpp-gguf",
        "package": "magpie_tts_q8_0",
        "prep_args_fn": lambda p="magpie_tts_q8_0": ["--package", p],
        "stack": "audiocpp",
        "quant": "q8_0",
        "voice_clone": False,
        "langs": ['en'],
        "timeout": 3600,
        "args_fn": lambda text, ref, out, lang="en": [
            text,
            "--package",
            "magpie_tts_q8_0",
            "--output",
            out,
            "--task",
            "tts",
            "--extra=--request-option voice_id=Sofia",
            "--lang",
            lang,
        ],
    },
    # audiocpp-miotts-1.7b
    {
        "name": "audiocpp-miotts-1.7b",
        "script": "tools/tts-audiocpp.py",
        "repo": "audio-cpp/audio.cpp-gguf",
        "package": "miotts_1_7b_q8_0",
        "prep_args_fn": lambda p="miotts_1_7b_q8_0": ["--package", p],
        "stack": "audiocpp",
        "quant": "q8_0",
        "voice_clone": True,
        "langs": ['en'],
        "timeout": 3600,
        "args_fn": lambda text, ref, out, lang="en": [
            text,
            "--package",
            "miotts_1_7b_q8_0",
            "--output",
            out,
            "--task",
            "tts",
            "--deps",
            "miocodec_q8_0",
            "--extra=--session-option miotts.codec_model_path=models/MioCodec-25Hz-44.1kHz-v2-GGUF",
            "--ref",
            ref,
        ],
    },
    # audiocpp-moss-local-v1.5
    {
        "name": "audiocpp-moss-local-v1.5",
        "script": "tools/tts-audiocpp.py",
        "repo": "audio-cpp/audio.cpp-gguf",
        "package": "moss_tts_local_v1_5_q8_0",
        "prep_args_fn": lambda p="moss_tts_local_v1_5_q8_0": ["--package", p],
        "stack": "audiocpp",
        "quant": "q8_0",
        "voice_clone": True,
        "langs": ['en', 'pl'],
        "timeout": 3600,
        "args_fn": lambda text, ref, out, lang="en": [
            text,
            "--package",
            "moss_tts_local_v1_5_q8_0",
            "--output",
            out,
            "--task",
            "clon",
            "--lang",
            lang,
            "--ref",
            ref,
            "--ref-text",
            REF_TEXTS.get(lang, ""),
        ],
    },
    # audiocpp-moss-nano-100m
    {
        "name": "audiocpp-moss-nano-100m",
        "script": "tools/tts-audiocpp.py",
        "repo": "audio-cpp/audio.cpp-gguf",
        "package": "moss_tts_nano_100m_q8_0",
        "prep_args_fn": lambda p="moss_tts_nano_100m_q8_0": ["--package", p],
        "stack": "audiocpp",
        "quant": "q8_0",
        "voice_clone": True,
        "langs": ['en', 'pl'],
        "timeout": 3600,
        "args_fn": lambda text, ref, out, lang="en": [
            text,
            "--package",
            "moss_tts_nano_100m_q8_0",
            "--output",
            out,
            "--task",
            "clon",
            "--lang",
            lang,
            "--ref",
            ref,
            "--ref-text",
            REF_TEXTS.get(lang, ""),
        ],
    },
    # audiocpp-neutts-2e
    {
        "name": "audiocpp-neutts-2e",
        "script": "tools/tts-audiocpp.py",
        "repo": "audio-cpp/audio.cpp-gguf",
        "package": "neutts_2e_orig",
        "prep_args_fn": lambda p="neutts_2e_orig": ["--package", p],
        "stack": "audiocpp",
        "quant": "orig",
        "voice_clone": False,
        "langs": ['en'],
        "timeout": 3600,
        "args_fn": lambda text, _ref, out, _lang="en": [
            text,
            "--package",
            "neutts_2e_orig",
            "--output",
            out,
            "--task",
            "tts",
            "--extra=--request-option voice_id=emily --request-option emotion=neutral",
        ],
    },
    # audiocpp-omnivoice
    {
        "name": "audiocpp-omnivoice",
        "script": "tools/tts-audiocpp.py",
        "repo": "audio-cpp/audio.cpp-gguf",
        "package": "omnivoice_q8_0",
        "prep_args_fn": lambda p="omnivoice_q8_0": ["--package", p],
        "stack": "audiocpp",
        "quant": "q8_0",
        "voice_clone": True,
        "langs": ['en', 'pl'],
        "timeout": 3600,
        "args_fn": lambda text, ref, out, lang="en": [
            text,
            "--package",
            "omnivoice_q8_0",
            "--output",
            out,
            "--task",
            "tts",
            "--ref",
            ref,
            "--ref-text",
            REF_TEXTS.get(lang, ""),
        ],
    },
    # audiocpp-outetts-1b
    {
        "name": "audiocpp-outetts-1b",
        "script": "tools/tts-audiocpp.py",
        "repo": "audio-cpp/audio.cpp-gguf",
        "package": "outetts_1_0_1b_q8_0",
        "prep_args_fn": lambda p="outetts_1_0_1b_q8_0": ["--package", p],
        "stack": "audiocpp",
        "quant": "q8_0",
        "voice_clone": True,
        "langs": ['en', 'pl'],
        "timeout": 3600,
        "args_fn": lambda text, ref, out, lang="en": [
            text,
            "--package",
            "outetts_1_0_1b_q8_0",
            "--output",
            out,
            "--task",
            "clon",
            "--extra=--max-tokens 1024",
            "--ref",
            ref,
            "--ref-text",
            REF_TEXTS.get(lang, ""),
        ],
    },
    # audiocpp-pocket-tts-en
    {
        "name": "audiocpp-pocket-tts-en",
        "script": "tools/tts-audiocpp.py",
        "repo": "audio-cpp/audio.cpp-gguf",
        "package": "pocket_tts_english_q8_0",
        "prep_args_fn": lambda p="pocket_tts_english_q8_0": ["--package", p],
        "stack": "audiocpp",
        "quant": "q8_0",
        "voice_clone": True,
        "langs": ['en'],
        "timeout": 3600,
        "args_fn": lambda text, ref, out, lang="en": [
            text,
            "--package",
            "pocket_tts_english_q8_0",
            "--output",
            out,
            "--task",
            "tts",
            "--ref",
            ref,
        ],
    },
    # audiocpp-qwen3-tts-1.7b
    {
        "name": "audiocpp-qwen3-tts-1.7b",
        "script": "tools/tts-audiocpp.py",
        "repo": "audio-cpp/audio.cpp-gguf",
        "package": "qwen3_tts_1_7b_base_q8_0",
        "prep_args_fn": lambda p="qwen3_tts_1_7b_base_q8_0": ["--package", p],
        "stack": "audiocpp",
        "quant": "q8_0",
        "voice_clone": True,
        "langs": ['en'],
        "timeout": 3600,
        "args_fn": lambda text, ref, out, lang="en": [
            text,
            "--package",
            "qwen3_tts_1_7b_base_q8_0",
            "--extra=--max-tokens 2048",
            "--output",
            out,
            "--task",
            "tts",
            "--ref",
            ref,
            "--ref_text",
            REF_TEXTS.get(lang, ""),
        ],
    },
    # audiocpp-soprano-80m
    {
        "name": "audiocpp-soprano-80m",
        "script": "tools/tts-audiocpp.py",
        "repo": "audio-cpp/audio.cpp-gguf",
        "package": "soprano_1_1_80m_q8_0",
        "prep_args_fn": lambda p="soprano_1_1_80m_q8_0": ["--package", p],
        "stack": "audiocpp",
        "quant": "q8_0",
        "voice_clone": False,
        "langs": ['en'],
        "timeout": 3600,
        "args_fn": lambda text, _ref, out, _lang="en": [
            text,
            "--package",
            "soprano_1_1_80m_q8_0",
            "--output",
            out,
            "--task",
            "tts",
            # Segfaults under Metal (exit 139, no diagnostic); CPU is fine and
            # the model is 80M, so the speed cost is negligible.
            "--backend",
            "cpu",
        ],
    },
    # audiocpp-supertonic-3
    {
        "name": "audiocpp-supertonic-3",
        "script": "tools/tts-audiocpp.py",
        "repo": "audio-cpp/audio.cpp-gguf",
        "package": "supertonic_3_orig",
        "prep_args_fn": lambda p="supertonic_3_orig": ["--package", p],
        "stack": "audiocpp",
        "quant": "orig",
        "voice_clone": False,
        "langs": ['en', 'pl'],
        "timeout": 3600,
        "args_fn": lambda text, ref, out, lang="en": [
            text,
            "--package",
            "supertonic_3_orig",
            "--output",
            out,
            "--task",
            "tts",
            "--extra=--voice-id M1",
            "--lang",
            lang,
        ],
    },
    # audiocpp-vibevoice-1.5b
    {
        "name": "audiocpp-vibevoice-1.5b",
        "script": "tools/tts-audiocpp.py",
        "repo": "audio-cpp/audio.cpp-gguf",
        "package": "vibevoice_1_5b_q8_0",
        "prep_args_fn": lambda p="vibevoice_1_5b_q8_0": ["--package", p],
        "stack": "audiocpp",
        "quant": "q8_0",
        "voice_clone": True,
        "langs": ['en'],
        "timeout": 3600,
        # VibeVoice is a dialogue model: it rejects a prompt with no
        # "Speaker N:" line rather than reading it as a single turn.
        "args_fn": lambda text, ref, out, lang="en": [
            f"Speaker 1: {text}",
            "--package",
            "vibevoice_1_5b_q8_0",
            "--output",
            out,
            "--task",
            "tts",
            "--ref-mode",
            "voice_samples",
            "--ref",
            ref,
        ],
    },
    # audiocpp-vibevoice-7b
    {
        "name": "audiocpp-vibevoice-7b",
        "script": "tools/tts-audiocpp.py",
        "repo": "audio-cpp/audio.cpp-gguf",
        "package": "vibevoice_7b_q8_0",
        "prep_args_fn": lambda p="vibevoice_7b_q8_0": ["--package", p],
        "stack": "audiocpp",
        "quant": "q8_0",
        "voice_clone": True,
        "langs": ['en'],
        "timeout": 3600,
        # Same dialogue-model prompt requirement as the 1.5B entry.
        "args_fn": lambda text, ref, out, lang="en": [
            f"Speaker 1: {text}",
            "--package",
            "vibevoice_7b_q8_0",
            "--output",
            out,
            "--task",
            "tts",
            "--ref-mode",
            "voice_samples",
            "--ref",
            ref,
        ],
    },
    # audiocpp-vietneu-v3 — REMOVED, engine build cannot stop generating.
    # VieNeu-TTS v3 Turbo requires a voice reference (it refuses to run without
    # one), and with any reference — ours or the engine's own
    # assets/resources/a.wav — English text yields Mandarin that runs until the
    # token cap: 512 tokens gives 40.9 s, 2048 gives 163.8 s, exactly linear, so
    # no EOS is ever emitted. The model spec also shipped a broken
    # "strip_prefix": "." that made the package undownloadable; that part is
    # fixed in model_specs/vietneu_tts.json, the generation defect is not.
    # audiocpp-voxcpm1-0.5b
    {
        "name": "audiocpp-voxcpm1-0.5b",
        "script": "tools/tts-audiocpp.py",
        "repo": "audio-cpp/audio.cpp-gguf",
        "package": "voxcpm1_0.5b_q8_0",
        "prep_args_fn": lambda p="voxcpm1_0.5b_q8_0": ["--package", p],
        "stack": "audiocpp",
        "quant": "q8_0",
        "voice_clone": True,
        "langs": ['en'],
        "timeout": 3600,
        "args_fn": lambda text, ref, out, lang="en": [
            text,
            "--package",
            "voxcpm1_0.5b_q8_0",
            "--output",
            out,
            "--task",
            "tts",
            "--ref",
            ref,
            "--ref_text",
            REF_TEXTS.get(lang, ""),
        ],
    },
    # audiocpp-voxcpm2
    {
        "name": "audiocpp-voxcpm2",
        "script": "tools/tts-audiocpp.py",
        "repo": "audio-cpp/audio.cpp-gguf",
        "package": "voxcpm2_q8_0",
        "prep_args_fn": lambda p="voxcpm2_q8_0": ["--package", p],
        "stack": "audiocpp",
        "quant": "q8_0",
        "voice_clone": True,
        "langs": ['en', 'pl'],
        "timeout": 3600,
        "args_fn": lambda text, ref, out, lang="en": [
            text,
            "--package",
            "voxcpm2_q8_0",
            "--output",
            out,
            "--task",
            "tts",
            "--ref",
            ref,
            "--ref-text",
            REF_TEXTS.get(lang, ""),
        ],
    },
    # --- sherpa-onnx (one onnxruntime engine, family selected by config slot) ---
    # Fixed-speaker models: no cloning, so `ref` is never passed — the wrapper
    # rejects it rather than pretending a clone happened.
    {
        "name": "sherpaonnx-vits-piper-en",
        "script": SHERPAONNX,
        "repo": "k2-fsa/sherpa-onnx (vits-piper-en_US-amy-medium)",
        "package": "vits-piper-en",
        "prep_args_fn": lambda p="vits-piper-en": ["--package", p],
        "stack": "onnx",
        "quant": "f32",
        "voice_clone": False,
        "langs": ["en"],
        "args_fn": _sherpaonnx_args("vits-piper-en"),
    },
    {
        "name": "sherpaonnx-vits-piper-pl",
        "script": SHERPAONNX,
        "repo": "k2-fsa/sherpa-onnx (vits-piper-pl_PL-gosia-medium)",
        "package": "vits-piper-pl",
        "prep_args_fn": lambda p="vits-piper-pl": ["--package", p],
        "stack": "onnx",
        "quant": "f32",
        "voice_clone": False,
        "langs": ["pl"],
        "args_fn": _sherpaonnx_args("vits-piper-pl"),
    },
    {
        "name": "sherpaonnx-kokoro-v1_1",
        "script": SHERPAONNX,
        "repo": "k2-fsa/sherpa-onnx (kokoro-multi-lang-v1_1)",
        "package": "kokoro-multi-v1_1",
        "prep_args_fn": lambda p="kokoro-multi-v1_1": ["--package", p],
        "stack": "onnx",
        "quant": "f32",
        "voice_clone": False,
        "langs": ["en"],
        "args_fn": _sherpaonnx_args("kokoro-multi-v1_1"),
    },
    {
        "name": "sherpaonnx-kitten-nano",
        "script": SHERPAONNX,
        "repo": "k2-fsa/sherpa-onnx (kitten-nano-en-v0_1-fp16)",
        "package": "kitten-nano-en",
        "prep_args_fn": lambda p="kitten-nano-en": ["--package", p],
        "stack": "onnx",
        "quant": "fp16",
        "voice_clone": False,
        "langs": ["en"],
        "args_fn": _sherpaonnx_args("kitten-nano-en"),
    },
    {
        "name": "sherpaonnx-matcha-ljspeech",
        "script": SHERPAONNX,
        "repo": "k2-fsa/sherpa-onnx (matcha-icefall-en_US-ljspeech + vocos-22khz)",
        "package": "matcha-ljspeech-en",
        "prep_args_fn": lambda p="matcha-ljspeech-en": ["--package", p],
        "stack": "onnx",
        "quant": "f32",
        "voice_clone": False,
        "langs": ["en"],
        "args_fn": _sherpaonnx_args("matcha-ljspeech-en"),
    },
    # --- Coqui XTTS-v2: the only engine here that clones into Polish ---------
    # CPML weights (non-commercial). torch/torchaudio pinned to 2.5.1 to stay off
    # torchcodec; see the wrapper docstring.
    {
        "name": "xtts-v2",
        "script": XTTS,
        "repo": "coqui-ai/TTS (tts_models/multilingual/multi-dataset/xtts_v2)",
        "prep_args_fn": lambda m="xtts-v2": ["--model", m],
        "stack": "torch",
        "quant": "fp32",
        "voice_clone": True,
        "langs": ["en", "pl"],
        "timeout": 1800,
        "args_fn": lambda text, ref, out, lang="en": [
            text,
            "--output",
            out,
            "--model",
            "xtts-v2",
            "--ref",
            ref,
            "--lang",
            lang,
        ],
    },
    # --- ChatTTS: seeded random speaker, no cloning, en/zh only --------------
    {
        "name": "chattts",
        "script": CHATTTS,
        "repo": "2noise/ChatTTS",
        "stack": "torch",
        "quant": "fp32",
        "voice_clone": False,
        "langs": ["en"],
        "timeout": 1800,
        "args_fn": lambda text, _ref, out, _lang="en": [
            text,
            "--output",
            out,
            "--seed",
            "42",
        ],
    },
    # --- MeloTTS (upstream PyTorch, CPU). The MLX attempt stays ignored. -----
    {
        "name": "melotts-en-us",
        "script": MELOTTS,
        "repo": "myshell-ai/MeloTTS (EN)",
        "prep_args_fn": lambda lang="en": ["--lang", lang],
        "stack": "torch",
        "quant": "fp32",
        "voice_clone": False,
        "langs": ["en"],
        "timeout": 900,
        "args_fn": lambda text, _ref, out, _lang="en": [
            text,
            "--output",
            out,
            "--lang",
            "en",
            "--speaker",
            "EN-US",
        ],
    },
    # --- Silero TTS (torch.hub JIT, ~50 MB, CPU) — 119 English voices --------
    {
        "name": "silero-en",
        "script": SILERO,
        "repo": "snakers4/silero-models (v3_en)",
        "prep_args_fn": lambda lang="en": ["--lang", lang],
        "stack": "torch",
        "quant": "fp32",
        "voice_clone": False,
        "langs": ["en"],
        "timeout": 900,
        "args_fn": lambda text, _ref, out, _lang="en": [
            text, "--output", out, "--lang", "en", "--speaker", "en_0",
        ],
    },
    # --- ESPnet2 recipes: the training side of models this project runs -------
    {
        "name": "espnet-ljspeech-vits",
        "script": ESPNET,
        "repo": "espnet/espnet (espnet/kan-bayashi_ljspeech_vits)",
        "prep_args_fn": lambda m="ljspeech-vits": ["--model", m],
        "stack": "torch",
        "quant": "fp32",
        "voice_clone": False,
        "langs": ["en"],
        "timeout": 1800,
        "args_fn": lambda text, _ref, out, _lang="en": [
            text, "--output", out, "--model", "ljspeech-vits",
        ],
    },
    {
        "name": "espnet-ljspeech-fastspeech2",
        "script": ESPNET,
        "repo": "espnet/espnet (espnet/kan-bayashi_ljspeech_fastspeech2)",
        "prep_args_fn": lambda m="ljspeech-fastspeech2": ["--model", m],
        "stack": "torch",
        "quant": "fp32",
        "voice_clone": False,
        "langs": ["en"],
        "timeout": 1800,
        "args_fn": lambda text, _ref, out, _lang="en": [
            text, "--output", out, "--model", "ljspeech-fastspeech2",
        ],
    },
    # --- EmotiVoice: voice design by natural-language prompt (mode desi) ------
    {
        "name": "emotivoice",
        "script": EMOTIVOICE,
        "repo": "netease-youdao/EmotiVoice",
        "stack": "torch",
        "quant": "fp32",
        "voice_clone": False,
        "design": True,
        "langs": ["en"],
        "timeout": 1800,
        "args_fn": lambda text, _ref, out, _lang="en": [
            text, "--output", out, "--prompt", "neutral", "--speaker", "8051",
        ],
        "design_args_fn": lambda text, instruct, out, _lang="en": [
            text, "--output", out, "--prompt", instruct, "--speaker", "8051",
        ],
    },
    # --- Qwen3-TTS on CoreML/ANE via the argmax Swift CLI --------------------
    {
        "name": "argmax-qwen3-tts-coreml",
        "script": ARGMAX,
        "repo": "argmaxinc/argmax-oss-swift (Qwen3-TTS CoreML)",
        "stack": "coreml",
        "quant": "f16",
        "voice_clone": False,
        "langs": ["en"],
        "timeout": 1800,
        "args_fn": lambda text, _ref, out, _lang="en": [
            text, "--output", out, "--speaker", "aiden", "--lang", "en",
        ],
    },
]

# Drama 3 uses different GGUF layouts in these two native engines.
MODELS.extend([
    {
        "name": "audiocpp-vibevoice-q4-ios", "script": "tools/tts-audiocpp.py",
        "repo": "audio-cpp/audio.cpp-gguf", "package": "vibevoice_1_5b_q4_ios",
        "stack": "audiocpp", "quant": "q4", "voice_clone": True,
        "langs": ["en"], "timeout": 3600,
        "env": {
            "AUDIOCPP_ROOT": os.environ.get("AUDIOCPP_NEW_ROOT", str(runtime_path('audio.cpp-20261030'))),
            "AUDIOCPP_BIN": os.environ.get("AUDIOCPP_FULL_BIN", str(Path(os.environ.get("AUDIOCPP_NEW_ROOT", str(runtime_path('audio.cpp-20261030')))) / "build/macos-metal-full/bin/audiocpp_cli")),
        },
        "prep_args_fn": lambda: ["--package", "vibevoice_1_5b_q4_ios"],
        "args_fn": lambda text, ref, out, lang="en": [
            f"Speaker 1: {text}", "--package", "vibevoice_1_5b_q4_ios", "--output", out,
            "--ref-mode", "voice_samples", "--ref", ref, "--seed", "42",
        ],
    },
    {
        "name": "audiocpp-kitten2-tq2", "script": "tools/tts-audiocpp.py",
        "repo": "audio-cpp/audio.cpp-gguf", "package": "kitten_tts2_20261021_tq2_1",
        "stack": "audiocpp", "quant": "tq2_1", "voice_clone": False,
        "langs": ["en"], "timeout": 3600,
        "env": {
            "AUDIOCPP_ROOT": os.environ.get("AUDIOCPP_NEW_ROOT", str(runtime_path('audio.cpp-20261030'))),
            "AUDIOCPP_BIN": os.environ.get("AUDIOCPP_FULL_BIN", str(Path(os.environ.get("AUDIOCPP_NEW_ROOT", str(runtime_path('audio.cpp-20261030')))) / "build/macos-metal-full/bin/audiocpp_cli")),
        },
        "prep_args_fn": lambda: ["--package", "kitten_tts2_20261021_tq2_1"],
        "args_fn": lambda text, _ref, out, lang="en": [text, "--package", "kitten_tts2_20261021_tq2_1", "--output", out, "--voice-id", "Bruno", "--seed", "42"],
    },
    {
        "name": "audiocpp-kokoro-multilingual", "script": "tools/tts-audiocpp.py",
        "repo": "audio-cpp/audio.cpp-gguf", "package": "kokoro_82m_multilingual_q8_0",
        "stack": "audiocpp", "quant": "q8_0", "voice_clone": False,
        "langs": ["en"], "timeout": 3600,
        "env": {
            "AUDIOCPP_ROOT": os.environ.get("AUDIOCPP_NEW_ROOT", str(runtime_path('audio.cpp-20261030'))),
            "AUDIOCPP_BIN": os.environ.get("AUDIOCPP_FULL_BIN", str(Path(os.environ.get("AUDIOCPP_NEW_ROOT", str(runtime_path('audio.cpp-20261030')))) / "build/macos-metal-full/bin/audiocpp_cli")),
        },
        "prep_args_fn": lambda: ["--package", "kokoro_82m_multilingual_q8_0"],
        "args_fn": lambda text, _ref, out, lang="en": [text, "--package", "kokoro_82m_multilingual_q8_0", "--output", out, "--voice-id", "af_heart", "--lang", "en-us"],
    },
    {
        "name": "audiocpp-moss-tts-local-v15", "script": "tools/tts-audiocpp.py",
        "repo": "audio-cpp/audio.cpp-gguf", "package": "moss_tts_local_v1_5_q8_0",
        "stack": "audiocpp", "quant": "q8_0", "voice_clone": True,
        "langs": ["en", "pl"], "timeout": 3600,
        "env": {
            "AUDIOCPP_ROOT": os.environ.get("AUDIOCPP_NEW_ROOT", str(runtime_path('audio.cpp-20261030'))),
            "AUDIOCPP_BIN": os.environ.get("AUDIOCPP_FULL_BIN", str(Path(os.environ.get("AUDIOCPP_NEW_ROOT", str(runtime_path('audio.cpp-20261030')))) / "build/macos-metal-full/bin/audiocpp_cli")),
        },
        "prep_args_fn": lambda: ["--package", "moss_tts_local_v1_5_q8_0"],
        "args_fn": lambda text, ref, out, lang="en": [
            text, "--package", "moss_tts_local_v1_5_q8_0", "--output", out,
            "--task", "clon", "--ref", ref, "--lang", lang, "--seed", "42",
        ],
    },
    {
        "name": "audiocpp-lfm3-audio", "script": "tools/tts-audiocpp.py",
        "repo": "LiquidAI/LFM3-Audio-2B-GGUF", "package": "lfm3_audio_2b_q4_k_m",
        "stack": "audiocpp", "quant": "q4_k_m", "voice_clone": False,
        "langs": ["en", "pl"], "timeout": 3600,
        "env": {
            "AUDIOCPP_ROOT": os.environ.get("AUDIOCPP_NEW_ROOT", str(runtime_path('audio.cpp-20261030'))),
            "AUDIOCPP_BIN": os.environ.get("AUDIOCPP_FULL_BIN", str(Path(os.environ.get("AUDIOCPP_NEW_ROOT", str(runtime_path('audio.cpp-20261030')))) / "build/macos-metal-full/bin/audiocpp_cli")),
        },
        "prep_args_fn": lambda: ["--package", "lfm3_audio_2b_q4_k_m"],
        "args_fn": lambda text, _ref, out, lang="en": [
            text, "--package", "lfm3_audio_2b_q4_k_m", "--output", out,
            "--voice-id", "polish_female" if lang == "pl" else "american_english_female",
            "--seed", "42",
        ],
    },
    {
        "name": "audiocpp-audio8-realistic", "script": "tools/tts-audiocpp.py",
        "repo": "audio-cpp/Audio8-TTS-Realistic-0.6b-GGUF", "package": "audio8_tts_realistic_q8_0",
        "stack": "audiocpp", "quant": "q8_0", "voice_clone": True,
        "langs": ["en", "pl"], "lang_agnostic": True, "timeout": 3600,
        "env": {"AUDIOCPP_ROOT": os.environ.get("AUDIOCPP_NEW_ROOT", str(runtime_path('audio.cpp-20261030')))},
        "prep_args_fn": lambda: ["--package", "audio8_tts_realistic_q8_0"],
        "args_fn": lambda text, ref, out, lang="en": [
            text, "--package", "audio8_tts_realistic_q8_0", "--output", out,
            "--task", "clon", "--ref", ref, "--ref-text", REF_TEXTS.get(lang, ""),
            "--request-options", "text_chunk_size=60", "--seed", "42",
        ],
    },
    {
        "name": "audiocpp-ming3-audio", "script": "tools/tts-audiocpp.py",
        "repo": "audio-cpp/Ming-3.0-Audio-Tiny-GGUF", "package": "ming3_audio_tiny_q4_k",
        "stack": "audiocpp", "quant": "q4_k", "voice_clone": True,
        "langs": ["en"], "timeout": 3600,
        "env": {"AUDIOCPP_ROOT": os.environ.get("AUDIOCPP_NEW_ROOT", str(runtime_path('audio.cpp-20261030')))},
        "prep_args_fn": lambda: ["--package", "ming3_audio_tiny_q4_k"],
        "args_fn": lambda text, ref, out, lang="en": [
            text, "--package", "ming3_audio_tiny_q4_k", "--output", out,
            "--task", "clon", "--ref", ref, "--ref-text", REF_TEXTS.get(lang, ""),
            "--request-options", "language=en",
        ],
    },
    {
        "name": "audiocpp-fish-drama3", "script": "tools/tts-audiocpp.py",
        "repo": "Jmica/Drama-3-GGUF", "package": "fish_drama3_q4_experts_q8_backbone",
        "stack": "audiocpp", "quant": "q4_k", "voice_clone": True,
        "langs": ["en", "pl"], "lang_agnostic": True, "timeout": 3600,
        "env": {"AUDIOCPP_ROOT": os.environ.get("AUDIOCPP_DRAMA3_ROOT", str(runtime_path('audio.cpp-drama3')))},
        "prep_args_fn": lambda: ["--package", "fish_drama3_q4_experts_q8_backbone"],
        "args_fn": lambda text, ref, out, lang="en": [
            text, "--package", "fish_drama3_q4_experts_q8_backbone", "--output", out,
            "--task", "clon", "--ref", ref, "--ref-text", REF_TEXTS.get(lang, ""),
        ],
    },
    {
        "name": "speechcpp-drama3-q6", "script": "tools/tts-speechcpp.py",
        "repo": "sakasegawa/drama-3-GGUF", "stack": "speechcpp", "quant": "q6_k",
        "voice_clone": True, "langs": ["en", "pl"], "timeout": 3600,
        "args_fn": lambda text, ref, out, lang="en": [
            text, "--output", out, "--ref", ref, "--ref-text", REF_TEXTS.get(lang, ""), "--lang", lang,
        ],
    },
])

# Native qwentts.cpp has distinct Base, CustomVoice and VoiceDesign contracts.
for _qwen_package in ("0.6b-base", "0.6b-customvoice", "1.7b-base", "1.7b-customvoice", "1.7b-voicedesign"):
    _qwen_mode = _qwen_package.split("-", 1)[1]
    _qwen_entry = {
        "name": "qwenttscpp-" + _qwen_package,
        "script": "tools/tts-qwenttscpp.py",
        "repo": "ServeurpersoCom/qwentts.cpp (Serveurperso/Qwen3-TTS-GGUF)",
        "stack": "qwenttscpp", "quant": "q8_0",
        "voice_clone": _qwen_mode == "base", "langs": ["en"], "timeout": 900,
        "prep_args_fn": lambda p=_qwen_package: ["--package", p],
        "args_fn": lambda text, ref, out, lang="en", p=_qwen_package, m=_qwen_mode: (
            [text, "--output", out, "--package", p, "--lang", lang]
            + (["--ref", ref] if m == "base" else [])
            + (["--instruct", "A clear, warm English narrator"] if m == "voicedesign" else [])
        ),
    }
    if _qwen_mode == "voicedesign":
        _qwen_entry["design"] = True
        _qwen_entry["design_args_fn"] = lambda text, instruct, out, lang="en", p=_qwen_package: [
            text, "--output", out, "--package", p, "--lang", lang, "--instruct", instruct,
        ]
    MODELS.append(_qwen_entry)

for _small_name, _small_package, _small_quant in (
    ("audiocpp-kitten-mini-0.8", "kitten_tts_mini_0_8_orig", "orig"),
    ("audiocpp-kokoro-bf16", "kokoro_82m_bf16", "bf16"),
    ("audiocpp-kokoro-q8", "kokoro_82m_q8_0", "q8_0"),
):
    MODELS.append({
        "name": _small_name, "script": "tools/tts-smallgguf.py", "repo": "audio-cpp/audio.cpp-gguf",
        "package": _small_package, "stack": "audiocpp", "quant": _small_quant,
        "voice_clone": False, "langs": ["en"], "timeout": 600,
        "prep_args_fn": lambda p=_small_package: ["--package", p],
        "args_fn": lambda text, ref, out, lang="en", p=_small_package: [
            text, "--output", out, "--package", p, "--lang", lang, "--backend", "cpu",
        ],
    })

# ---------------------------------------------------------------------------
# Model metadata derivation (inference stack / quantization / cloning mode)
# ---------------------------------------------------------------------------
# These drive the self-describing output filenames and the HTML report so every
# generated clip records HOW it was produced. A model dict may set "stack",
# "quant" or "mode" explicitly; otherwise they are inferred from its name/script.
# ---------------------------------------------------------------------------

# Quantization tokens, longest-first so "mxfp8" wins over a stray "fp8" etc.
_QUANT_TOKENS = (
    "q8_0",
    "q4_k",
    "mxfp8",
    "mxfp4",
    "bf16",
    "fp16",
    "fp32",
    "int8",
    "8bit",
    "4bit",
    "q8",
    "q5",
    "q4",
)

_STACK_DEFAULT_QUANT = {
    "audiocpp": "q8_0",
    "onnx": "fp16",
    "coreml": "fp16",
    "gguf": "q8",
    "mlx": "bf16",
    "torch": "fp32",
}


def model_stack(model: dict) -> str:
    """Inference stack: audiocpp | mlx | coreml | onnx | gguf | torch."""
    if model.get("stack"):
        return str(model["stack"])
    s = model["script"].lower()
    name = model["name"].lower()
    if "coreml" in s or "coreml" in name:
        return "coreml"
    if "onnx" in s or "onnx" in name:
        return "onnx"
    if "-mlx" in s or "mlx" in name:
        return "mlx"
    if "gguf" in s or "cpp" in s or "cstr" in s or "gguf" in name:
        return "gguf"
    if "torch" in s:
        return "torch"
    return "torch"  # plain PyTorch/MPS scripts (chatterbox, f5tts, qwen3, ...)


def model_quant(model: dict) -> str:
    """Quantization label inferred from the model name, with a stack default."""
    if model.get("quant"):
        return str(model["quant"])
    name = model["name"].lower()
    for tok in _QUANT_TOKENS:
        if tok in name:
            return tok
    return _STACK_DEFAULT_QUANT.get(model_stack(model), "na")


def model_mode(model: dict) -> str:
    """Cloning mode: ivc (instant clone) | auto (prebuilt voice) | desi (voice design).

    PVC (LoRA fine-tune) is a separate offline pipeline and is never produced by a
    plain batch run, so it is not inferred here.
    """
    if model.get("design"):
        return "desi"
    return "ivc" if model["voice_clone"] else "auto"


def output_filename(run_name: str, model: dict, lang: str, mode: str, ext: str) -> str:
    """Self-describing flat filename: ``{lang}__{model}__{quant}__{stack}__{mode}.{ext}``."""
    return (
        f"{lang}__{run_name}__{model_quant(model)}__{model_stack(model)}__{mode}.{ext}"
    )


# ---------------------------------------------------------------------------
# Voice design — text-described voices (mode "desi"). Each design-capable model
# is rendered with several distinct voice descriptions named desi1, desi2, ...
# ---------------------------------------------------------------------------

DESIGN_PROMPTS: list[str] = ["A clear voice with steady pacing."]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def read_text(path: Path) -> str:
    """Read text file, strip whitespace."""
    return path.read_text(encoding="utf-8").strip()


def get_audio_duration(path: str) -> float:
    """Get audio duration in seconds. Uses macOS afinfo (handles all formats), falls back to stdlib wave."""
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return 0.0
    # Primary: macOS afinfo — handles WAV (PCM, Float32), MP3, AAC, etc.
    try:
        r = subprocess.run(
            ["afinfo", str(p)], capture_output=True, text=True, timeout=5
        )
        for line in r.stdout.splitlines():
            if "duration" in line.lower():
                for part in line.split():
                    try:
                        return float(part)
                    except ValueError:
                        continue
    except FileNotFoundError:
        pass  # afinfo not available (non-macOS) — fall through
    except Exception:
        pass
    # Fallback: stdlib wave (PCM WAV only)
    if p.suffix.lower() == ".wav":
        import wave

        try:
            with wave.open(str(p), "rb") as wf:
                return wf.getnframes() / wf.getframerate()
        except Exception:
            pass
    return 0.0


def _split_csv_arg(value: Any) -> list[str]:
    """Normalize Fire string/tuple CSV arguments."""
    raw_values = value if isinstance(value, (list, tuple)) else [value]
    return [
        part.strip()
        for raw_value in raw_values
        for part in str(raw_value).split(",")
        if part.strip()
    ]


def resolve_models(models_arg: str | list[str] | tuple[str, ...]) -> list[dict]:
    """Filter MODELS by name list or return all.  Silently excludes IGNORED_ENGINES."""
    names = _split_csv_arg(models_arg)
    if names == ["all"]:
        return [m for m in MODELS if m["name"] not in IGNORED_ENGINES]
    name_set = set(names)
    # Remove ignored names so they don't trigger the unknown-model warning
    name_set -= IGNORED_ENGINES
    selected = [m for m in MODELS if m["name"] in name_set]
    missing = name_set - {m["name"] for m in selected}
    if missing:
        console.print(
            f"[yellow]Warning: unknown model(s): {', '.join(sorted(missing))}[/yellow]"
        )
    return selected


def resolve_langs(langs_arg: str | list[str] | tuple[str, ...]) -> list[str]:
    """Parse lang filter."""
    lang_codes = _split_csv_arg(langs_arg)
    if lang_codes == ["all"]:
        return ["en", "pl"]
    return lang_codes


def run_model(
    model: dict,
    lang: str,
    text: str,
    ref: str,
    out: str,
    timeout: int,
    verbose: bool,
    force: bool = False,
    args: list | None = None,
) -> tuple[bool, float, str]:
    """Run one model for one language. Returns (success, duration_s, error_msg).

    *args* overrides the script arguments (used for voice-design jobs); when None
    the model's own ``args_fn(text, ref, out, lang)`` is used.
    """
    script_path = str(ROOT / model["script"])
    arglist = args if args is not None else model["args_fn"](text, ref, out, lang)
    # Tools are Fire CLIs with an explicit `say` subcommand (see refactor contract).
    cmd = ["uv", "run", "--script", script_path, "say"] + [str(a) for a in arglist]
    if force:
        cmd.append("--force")

    # Merge any model-specific env overrides
    env = os.environ.copy()
    for k, v in model.get("env", {}).items():
        env[k] = v

    t0 = time.monotonic()
    try:
        result = subprocess.run(
            cmd,
            stdout=None if verbose else subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            env=env,
            cwd=str(Path(out).resolve().parent),
        )
        duration = time.monotonic() - t0
        if result.returncode != 0:
            # Script errors go to stdout (rich console), uv messages go to stderr
            combined = _script_error_output(result.stdout, result.stderr)
            last_lines = "\n".join(combined.splitlines()[-5:])
            return False, duration, last_lines
        return True, duration, ""
    except subprocess.TimeoutExpired:
        duration = time.monotonic() - t0
        return False, duration, f"timeout after {timeout}s"
    except FileNotFoundError:
        duration = time.monotonic() - t0
        return False, duration, f"script not found: {model['script']}"
    except Exception as exc:
        duration = time.monotonic() - t0
        return False, duration, str(exc)


def prep_model(model: dict, timeout: int, verbose: bool) -> tuple[bool, float, str]:
    """Run a model's ``prep`` subcommand (download + cache weights). Idempotent.

    Returns (success, duration_s, error_msg). A model whose tool predates the
    prep/say refactor (no `prep` command) is treated as a soft skip.
    """
    script_path = str(ROOT / model["script"])
    # A wrapper that serves many models needs to be told which one to fetch.
    # tools/tts-audiocpp.py covers ~40 packages and defaults to PocketTTS, so
    # without this every audio.cpp prep returned instantly having downloaded
    # nothing, and the real download then ran inside `say` against the model's
    # inference timeout — which is how the 5 GB Higgs package timed out.
    cmd = ["uv", "run", "--script", script_path, "prep"]
    prep_args = model.get("prep_args_fn")
    if prep_args is not None:
        cmd += [str(a) for a in prep_args()]
    env = os.environ.copy()
    for k, v in model.get("env", {}).items():
        env[k] = v
    scratch = runtime_path("scratch")
    scratch.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    try:
        result = subprocess.run(
            cmd,
            stdout=None if verbose else subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            env=env,
            cwd=str(scratch),
        )
        duration = time.monotonic() - t0
        if result.returncode != 0:
            combined = _script_error_output(result.stdout, result.stderr)
            return False, duration, "\n".join(combined.splitlines()[-5:])
        return True, duration, ""
    except subprocess.TimeoutExpired:
        return False, time.monotonic() - t0, f"prep timeout after {timeout}s"
    except FileNotFoundError:
        return False, time.monotonic() - t0, f"script not found: {model['script']}"
    except Exception as exc:
        return False, time.monotonic() - t0, str(exc)


def _script_error_output(stdout: str | None, stderr: str | None) -> str:
    parts = [part.strip() for part in (stdout, stderr) if part and part.strip()]
    return "\n".join(parts) if parts else "(no output)"


# ---------------------------------------------------------------------------
# Result record
# ---------------------------------------------------------------------------


class Result:
    __slots__ = (
        "model",
        "lang",
        "success",
        "duration",
        "error",
        "output_path",
        "file_size_kb",
        "audio_duration_s",
        "rtf",
        "chars_per_sec",
        "text_len",
        "stack",
        "quant",
        "mode",
    )

    def __init__(
        self,
        model: str,
        lang: str,
        success: bool,
        duration: float,
        error: str,
        output_path: str = "",
        file_size_kb: float = 0.0,
        audio_duration_s: float = 0.0,
        rtf: float = 0.0,
        chars_per_sec: float = 0.0,
        text_len: int = 0,
        stack: str = "",
        quant: str = "",
        mode: str = "",
    ):
        self.model = model
        self.lang = lang
        self.success = success
        self.duration = duration
        self.error = error
        self.output_path = output_path
        self.file_size_kb = file_size_kb
        self.audio_duration_s = audio_duration_s
        self.rtf = rtf
        self.chars_per_sec = chars_per_sec
        self.text_len = text_len
        self.stack = stack
        self.quant = quant
        self.mode = mode

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        return {
            "model": self.model,
            "lang": self.lang,
            "success": self.success,
            "duration": self.duration,
            "error": self.error,
            "output_path": self.output_path,
            "file_size_kb": self.file_size_kb,
            "audio_duration_s": self.audio_duration_s,
            "rtf": self.rtf,
            "chars_per_sec": self.chars_per_sec,
            "text_len": self.text_len,
            "stack": self.stack,
            "quant": self.quant,
            "mode": self.mode,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Result":
        """Deserialize from a dict, with safe type coercion."""
        return cls(
            model=str(d.get("model", "")),
            lang=str(d.get("lang", "")),
            success=bool(d.get("success", False)),
            duration=float(d.get("duration", 0.0)),
            error=str(d.get("error", "")),
            output_path=str(d.get("output_path", "")),
            file_size_kb=float(d.get("file_size_kb", 0.0)),
            audio_duration_s=float(d.get("audio_duration_s", 0.0)),
            rtf=float(d.get("rtf", 0.0)),
            chars_per_sec=float(d.get("chars_per_sec", 0.0)),
            text_len=int(d.get("text_len", 0)),
            stack=str(d.get("stack", "")),
            quant=str(d.get("quant", "")),
            mode=str(d.get("mode", "")),
        )


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

ALL_LANGS = ["en", "pl"]


def print_summary(
    results: list[Result], _active_langs: list[str] | None = None
) -> None:
    """Print rich summary table with benchmarks."""
    table = Table(title="TTS Batch Results", show_lines=True)
    table.add_column("Model", style="bold cyan", no_wrap=True)
    table.add_column("Lang", justify="center")
    table.add_column("Status", justify="center")
    table.add_column("Gen Time", justify="right")
    table.add_column("Audio", justify="right")
    table.add_column("RTF", justify="right")
    table.add_column("Size", justify="right")
    table.add_column("Chars/s", justify="right")

    for r in results:
        status = "[green]✓[/green]" if r.success else "[red]✗[/red]"
        gen_time = f"{r.duration:.1f}s"
        audio = f"{r.audio_duration_s:.1f}s" if r.audio_duration_s > 0 else "—"
        rtf_str = f"{r.rtf:.2f}x" if r.rtf > 0 else "—"
        size = f"{r.file_size_kb:.0f}KB" if r.file_size_kb > 0 else "—"
        cps = f"{r.chars_per_sec:.0f}" if r.chars_per_sec > 0 else "—"

        # Color RTF: green if <1 (faster than real-time), yellow 1-3, red >3
        if r.rtf > 0:
            if r.rtf < 1:
                rtf_str = f"[green]{rtf_str}[/green]"
            elif r.rtf < 3:
                rtf_str = f"[yellow]{rtf_str}[/yellow]"
            else:
                rtf_str = f"[red]{rtf_str}[/red]"

        table.add_row(r.model, r.lang, status, gen_time, audio, rtf_str, size, cps)

    console.print()
    console.print(table)

    # Counts
    successes = sum(1 for r in results if r.success)
    failures = sum(1 for r in results if not r.success)
    console.print(
        f"\n[bold]Total:[/bold] {len(results)} runs — "
        f"[green]{successes} succeeded[/green], "
        f"[red]{failures} failed[/red]"
    )

    # Failure details
    failed = [r for r in results if not r.success]
    if failed:
        console.print("\n[bold red]Failure details:[/bold red]")
        for r in failed:
            console.print(
                f"  [cyan]{r.model}[/cyan]/[yellow]{r.lang}[/yellow]: {r.error}"
            )


def print_benchmarks(results: list[Result], batch_elapsed: float) -> None:
    """Print benchmark rankings sorted by speed (RTF)."""
    successful = [r for r in results if r.success and r.rtf > 0]
    if not successful:
        return

    console.print("\n")
    table = Table(title="Benchmark Rankings (by Real-Time Factor)", show_lines=True)
    table.add_column("#", justify="right", style="dim", width=3)
    table.add_column("Model", style="bold cyan", no_wrap=True)
    table.add_column("Lang", justify="center", width=4)
    table.add_column("RTF", justify="right", width=7)
    table.add_column("Speed", justify="left", width=30)
    table.add_column("Gen", justify="right", width=7)
    table.add_column("Audio", justify="right", width=7)

    # Sort by RTF ascending (fastest first)
    ranked = sorted(successful, key=lambda r: r.rtf)

    for i, r in enumerate(ranked, 1):
        # Bar width proportional to inverse RTF (fastest gets longest bar)
        bar_len = (
            max(1, int(25 * (1.0 / r.rtf) / (1.0 / ranked[0].rtf))) if r.rtf > 0 else 0
        )
        bar = "█" * bar_len

        if r.rtf < 1:
            color = "green"
        elif r.rtf < 3:
            color = "yellow"
        else:
            color = "red"

        rtf_str = f"{r.rtf:.2f}x"
        speed_bar = f"[{color}]{bar}[/{color}]"

        table.add_row(
            str(i),
            r.model,
            r.lang,
            rtf_str,
            speed_bar,
            f"{r.duration:.0f}s",
            f"{r.audio_duration_s:.1f}s",
        )

    console.print(table)
    console.print(f"\n[dim]Total batch wall-clock: {batch_elapsed:.0f}s[/dim]")


# ---------------------------------------------------------------------------
# JSON Report persistence
# ---------------------------------------------------------------------------


def _result_key(model: str, lang: str) -> str:
    """Canonical key for a model/lang pair."""
    return f"{model}/{lang}"


def load_report_json(outdir: Path) -> dict:
    """Load prior report.json from *outdir*; return {} on missing/malformed."""
    path = outdir / "report.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            console.print(
                "[yellow]⚠ report.json is not a dict — starting fresh[/yellow]"
            )
            return {}
        return data
    except (json.JSONDecodeError, OSError) as exc:
        console.print(
            f"[yellow]⚠ report.json malformed ({exc}) — starting fresh[/yellow]"
        )
        return {}


def _seed_ignored_records(merged: dict[str, dict], now: str) -> None:
    """Ensure every IGNORED_ENGINES entry exists in *merged* with ignored metadata."""
    for name in sorted(IGNORED_ENGINES):
        for lang in ALL_LANGS:
            key = _result_key(name, lang)
            rec = merged.get(key, {})
            rec.setdefault("model", name)
            rec.setdefault("lang", lang)
            rec["ignored"] = True
            rec["status"] = "ignored"
            rec["reason"] = IGNORED_REASON
            rec["updated_at"] = now
            # Do not overwrite real success/duration if engine once worked
            rec.setdefault("success", False)
            merged[key] = rec


def _build_report_data(
    results: list[Result],
    prior_data: dict,
    *,
    batch_elapsed: float | None = None,
    completed: int | None = None,
    total: int | None = None,
    total_models: int = 0,
    total_available: int = 0,
    selected_langs: list[str] | None = None,
    models_filter: str = "all",
) -> dict:
    """Merge *results* into *prior_data*, preserving entries not in current run."""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Start from prior results (preserve old entries)
    merged: dict[str, dict] = {}
    prior_results = prior_data.get("results", {})
    if isinstance(prior_results, dict):
        merged.update(prior_results)

    # Overlay current run results
    for r in results:
        merged[_result_key(r.model, r.lang)] = r.to_dict()

    # Seed / refresh ignored-engine records
    _seed_ignored_records(merged, now)

    # Stats exclude ignored engines
    active = {k: v for k, v in merged.items() if not v.get("ignored")}
    successes = sum(1 for v in active.values() if v.get("success"))
    failures = len(active) - successes

    progress_str = None
    if completed is not None and total is not None:
        progress_str = f"{completed}/{total}"

    data: dict[str, Any] = {
        "generated": now,
        "meta": {
            "models_filter": models_filter,
            "selected_langs": selected_langs or [],
            "total_models": total_models,
            "total_available": total_available,
        },
        "summary": {
            "total_runs": len(active),
            "successes": successes,
            "failures": failures,
            "batch_elapsed_s": batch_elapsed if batch_elapsed is not None else 0.0,
        },
        "results": merged,
    }
    if progress_str:
        data["summary"]["progress"] = progress_str
    return data


def _write_json(outdir: Path, data: dict) -> Path:
    """Atomic write of report.json (write to tmp then rename)."""
    report_path = outdir / "report.json"
    fd, tmp = tempfile.mkstemp(dir=str(outdir), suffix=".json.tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")
        Path(tmp).replace(report_path)
    except BaseException:
        try:
            Path(tmp).unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return report_path


def save_partial_report(
    results: list[Result],
    prior_data: dict,
    outdir: Path,
    completed: int,
    total: int,
    **kwargs: Any,
) -> None:
    """Write a partial JSON snapshot after each model completes."""
    data = _build_report_data(
        results, prior_data, completed=completed, total=total, **kwargs
    )
    _write_json(outdir, data)


def save_final_report(
    results: list[Result],
    prior_data: dict,
    outdir: Path,
    batch_elapsed: float,
    **kwargs: Any,
) -> tuple[Path, dict]:
    """Write the final JSON report.  Returns (path, data_dict)."""
    data = _build_report_data(
        results, prior_data, batch_elapsed=batch_elapsed, **kwargs
    )
    return _write_json(outdir, data), data


def save_dashboard(outdir: Path) -> Path | None:
    """Rebuild the interactive sample browser at ``out/index.html``.

    The builder lives in ``tools/build_dashboard.py`` so the dashboard can be
    regenerated from a recorded ``report.json`` without re-running inference.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "build_dashboard", ROOT / "tools/build_dashboard.py"
    )
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build(outdir)


def save_report_html(data: dict, outdir: Path) -> Path:
    """Generate a polished HTML report from the report data dict."""
    esc = html_mod.escape
    meta = data.get("meta", {})
    summary = data.get("summary", {})
    results = data.get("results", {})
    generated = esc(str(data.get("generated", "")))

    total_runs = summary.get("total_runs", 0)
    successes = summary.get("successes", 0)
    failures = summary.get("failures", 0)
    elapsed = summary.get("batch_elapsed_s", 0.0)
    progress = summary.get("progress", "")

    models_filter = esc(str(meta.get("models_filter", "all")))
    selected_langs = (
        ", ".join(esc(str(l)) for l in meta.get("selected_langs", [])) or "all"
    )
    total_models = meta.get("total_models", 0)
    total_available = meta.get("total_available", 0)

    # Separate successes and failures (skip ignored engines)
    ok_rows = []
    fail_rows = []
    for key in sorted(results.keys()):
        r = results[key]
        if r.get("ignored"):
            continue
        if r.get("success"):
            ok_rows.append(r)
        else:
            fail_rows.append(r)

    # Sort successes by RTF (best first)
    ok_rows.sort(key=lambda r: r.get("rtf") or 999)

    def _rtf_badge(rtf):
        if rtf is None:
            return '<span class="badge badge-ghost">—</span>'
        if rtf < 0.3:
            return f'<span class="badge badge-success font-mono">{rtf:.2f}</span>'
        if rtf < 1.0:
            return f'<span class="badge badge-info font-mono">{rtf:.2f}</span>'
        if rtf < 3.0:
            return f'<span class="badge badge-warning font-mono">{rtf:.2f}</span>'
        return f'<span class="badge badge-error font-mono">{rtf:.2f}</span>'

    def _dur(s):
        if s is None:
            return "—"
        return f"{s:.1f}s"

    def _size(kb):
        if kb is None:
            return "—"
        if kb >= 1024:
            return f"{kb / 1024:.1f} MB"
        return f"{kb:.0f} KB"

    _MODE_BADGE = {
        "ivc": "badge-info",
        "auto": "badge-ghost",
        "pvc": "badge-accent",
    }

    def _mode_badge(mode: str) -> str:
        m = esc(mode or "—")
        cls = _MODE_BADGE.get(
            mode, "badge-secondary" if mode.startswith("desi") else "badge-ghost"
        )
        return f'<span class="badge {cls} badge-sm font-mono">{m}</span>'

    def _audio_cell(r: dict) -> str:
        name = os.path.basename(str(r.get("output_path", "")))
        if not name:
            return "—"
        href = esc("./" + name)
        return f'<audio controls preload="none" src="{href}"></audio>'

    # Build results table rows. data-sort on numeric cells keeps click-sort numeric.
    result_rows_html = ""
    for i, r in enumerate(ok_rows, 1):
        model = esc(str(r.get("model", "")))
        lang = esc(str(r.get("lang", "")))
        mode = _mode_badge(str(r.get("mode", "")))
        stack = esc(str(r.get("stack", "")) or "—")
        quant = esc(str(r.get("quant", "")) or "—")
        quality = r.get("quality", "")
        q_badge = (
            f'<span class="badge badge-accent text-lg">{quality}</span>'
            if quality
            else "—"
        )
        dur_v = r.get("duration") or 0.0
        dur = _dur(r.get("duration"))
        rtf_v = r.get("rtf") or 0.0
        rtf = _rtf_badge(r.get("rtf"))
        cps_v = r.get("chars_per_sec") or 0.0
        cps = f"{cps_v:.0f}" if cps_v else "—"
        adur_v = r.get("audio_duration_s") or 0.0
        audio_dur = _dur(r.get("audio_duration_s"))
        fsize_v = r.get("file_size_kb") or 0.0
        fsize = _size(r.get("file_size_kb"))
        result_rows_html += f"""<tr>
          <td data-sort="{i}">{i}</td>
          <td class="font-semibold">{model}</td>
          <td><span class="badge badge-outline badge-sm">{lang}</span></td>
          <td>{mode}</td>
          <td class="font-mono text-sm">{stack}</td>
          <td class="font-mono text-sm">{quant}</td>
          <td><span class="badge badge-success badge-sm">✓</span></td>
          <td class="font-bold text-center" data-sort="{len(quality)}">{q_badge}</td>
          <td class="font-mono text-sm" data-sort="{dur_v:.3f}">{dur}</td>
          <td data-sort="{rtf_v:.4f}">{rtf}</td>
          <td class="font-mono text-sm" data-sort="{cps_v:.2f}">{cps}</td>
          <td class="font-mono text-sm" data-sort="{adur_v:.3f}">{audio_dur}</td>
          <td class="font-mono text-sm" data-sort="{fsize_v:.1f}">{fsize}</td>
          <td>{_audio_cell(r)}</td>
        </tr>
"""

    for r in fail_rows:
        model = esc(str(r.get("model", "")))
        lang = esc(str(r.get("lang", "")))
        mode = _mode_badge(str(r.get("mode", "")))
        stack = esc(str(r.get("stack", "")) or "—")
        quant = esc(str(r.get("quant", "")) or "—")
        dur = _dur(r.get("duration"))
        result_rows_html += f"""<tr class="text-error/70">
          <td data-sort="999999">—</td>
          <td class="font-semibold">{model}</td>
          <td><span class="badge badge-outline badge-sm">{lang}</span></td>
          <td>{mode}</td>
          <td class="font-mono text-sm">{stack}</td>
          <td class="font-mono text-sm">{quant}</td>
          <td><span class="badge badge-error badge-sm">✗</span></td>
          <td>—</td>
          <td class="font-mono text-sm" data-sort="{r.get("duration") or 0:.3f}">{dur}</td>
          <td data-sort="9999">—</td><td data-sort="0">—</td>
          <td data-sort="0">—</td><td data-sort="0">—</td><td>—</td>
        </tr>
"""

    # Build benchmark ranking (top successes by RTF)
    benchmark_html = ""
    for i, r in enumerate(ok_rows[:10], 1):
        model = esc(str(r.get("model", "")))
        lang = esc(str(r.get("lang", "")))
        rtf = r.get("rtf")
        cps = r.get("chars_per_sec")
        rtf_str = f"{rtf:.3f}" if rtf is not None else "—"
        cps_str = f"{cps:.0f}" if cps else "—"
        medal = ["🥇", "🥈", "🥉"][i - 1] if i <= 3 else f"#{i}"
        bar_w = max(5, int((1.0 / (rtf or 1)) * 100))
        bar_w = min(bar_w, 100)
        benchmark_html += f"""<tr>
          <td class="text-lg">{medal}</td>
          <td class="font-semibold">{model}</td>
          <td><span class="badge badge-outline badge-sm">{lang}</span></td>
          <td class="font-mono">{rtf_str}</td>
          <td class="font-mono">{cps_str}</td>
          <td class="w-48"><div class="bg-primary/20 rounded-full h-3"><div class="bg-primary rounded-full h-3" style="width:{bar_w}%"></div></div></td>
        </tr>
"""

    # Build failure details
    failure_html = ""
    if fail_rows:
        failure_items = ""
        for r in fail_rows:
            model = esc(str(r.get("model", "")))
            lang = esc(str(r.get("lang", "")))
            err = esc(str(r.get("error", "unknown error")))
            dur = _dur(r.get("duration"))
            failure_items += f"""<div class="collapse collapse-arrow bg-error/5 mb-2">
              <input type="checkbox" />
              <div class="collapse-title font-semibold">
                <span class="text-error">✗</span> {model} <span class="badge badge-outline badge-sm">{lang}</span> <span class="text-sm opacity-60">({dur})</span>
              </div>
              <div class="collapse-content">
                <pre class="text-xs whitespace-pre-wrap break-all bg-base-300 rounded-lg p-3 mt-2">{err}</pre>
              </div>
            </div>
"""
        failure_html = f"""<div class="mt-10">
          <h2 class="text-2xl font-bold text-error mb-4">⚠ Failures ({failures})</h2>
          {failure_items}
        </div>"""

    # Success rate for the ring
    rate = (successes / total_runs * 100) if total_runs else 0
    rate_color = (
        "text-success"
        if rate >= 80
        else ("text-warning" if rate >= 50 else "text-error")
    )

    elapsed_str = f"{elapsed:.1f}s" if elapsed < 120 else f"{elapsed / 60:.1f}m"

    page = f"""<!DOCTYPE html>
<html lang="en" data-theme="night">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>PVC TTS Batch Report</title>
  <link href="https://cdn.jsdelivr.net/npm/daisyui@5/dist/full.min.css" rel="stylesheet" />
  <script src="https://cdn.tailwindcss.com"></script>
  <link href="https://cdn.jsdelivr.net/gh/tofsjonas/sortable@latest/sortable.min.css" rel="stylesheet" />
  <script src="https://cdn.jsdelivr.net/gh/tofsjonas/sortable@latest/sortable.min.js"></script>
  <style>
    .stat-card {{ @apply stats shadow bg-base-200; }}
    table.sortable th {{ cursor: pointer; user-select: none; }}
    table.sortable th[aria-sort] {{ text-decoration: underline dotted; }}
    audio {{ height: 2rem; max-width: 13rem; }}
  </style>
</head>
<body class="min-h-screen bg-base-100">
  <div class="container mx-auto px-4 py-8 max-w-7xl">

    <!-- Header -->
    <div class="flex flex-col md:flex-row md:items-end justify-between mb-8 gap-4">
      <div>
        <h1 class="text-4xl font-extrabold tracking-tight">🎙️ PVC TTS Report</h1>
        <p class="text-base-content/60 mt-1">Generated {generated}</p>
      </div>
      <div class="flex gap-2 flex-wrap">
        <span class="badge badge-primary badge-lg">Models: {models_filter}</span>
        <span class="badge badge-secondary badge-lg">Langs: {selected_langs}</span>
        {
        f'<span class="badge badge-accent badge-lg">Progress: {esc(progress)}</span>'
        if progress
        else ""
    }
      </div>
    </div>

    <!-- Stats cards -->
    <div class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
      <div class="stat bg-base-200 rounded-2xl shadow">
        <div class="stat-title">Total Runs</div>
        <div class="stat-value text-primary">{total_runs}</div>
        <div class="stat-desc">{total_models} models / {total_available} available</div>
      </div>
      <div class="stat bg-base-200 rounded-2xl shadow">
        <div class="stat-title">Successes</div>
        <div class="stat-value text-success">{successes}</div>
        <div class="stat-desc {rate_color}">{rate:.0f}% success rate</div>
      </div>
      <div class="stat bg-base-200 rounded-2xl shadow">
        <div class="stat-title">Failures</div>
        <div class="stat-value text-error">{failures}</div>
        <div class="stat-desc">{
        "All good! 🎉" if failures == 0 else "See details below"
    }</div>
      </div>
      <div class="stat bg-base-200 rounded-2xl shadow">
        <div class="stat-title">Batch Time</div>
        <div class="stat-value">{elapsed_str}</div>
        <div class="stat-desc">{f"{elapsed:.1f} seconds total"}</div>
      </div>
    </div>

    <!-- Results table -->
    <div class="bg-base-200 rounded-2xl shadow p-6 mb-8">
      <h2 class="text-2xl font-bold mb-4">📊 Results</h2>
      <p class="text-base-content/60 text-sm mb-4">Click any column header to sort. RTF &lt; 1 = faster than real-time. Mode: <span class="badge badge-info badge-xs">ivc</span> instant clone · <span class="badge badge-ghost badge-xs">auto</span> preset voice · <span class="badge badge-secondary badge-xs">desi*</span> voice design.</p>
      <div class="overflow-x-auto">
        <table class="table table-zebra table-sm w-full sortable">
          <thead>
            <tr>
              <th>#</th>
              <th>Model</th>
              <th>Lang</th>
              <th>Mode</th>
              <th>Stack</th>
              <th>Quant</th>
              <th>Status</th>
              <th>Quality</th>
              <th>Gen</th>
              <th>RTF</th>
              <th>Chars/s</th>
              <th>Audio</th>
              <th>Size</th>
              <th class="no-sort">Play</th>
            </tr>
          </thead>
          <tbody>
            {result_rows_html}
          </tbody>
        </table>
      </div>
    </div>

    <!-- Benchmark ranking -->
    {
        f'''<div class="bg-base-200 rounded-2xl shadow p-6 mb-8">
      <h2 class="text-2xl font-bold mb-4">🏆 Speed Benchmark (by RTF)</h2>
      <p class="text-base-content/60 text-sm mb-4">Lower RTF = faster. RTF &lt; 1 means faster than real-time.</p>
      <div class="overflow-x-auto">
        <table class="table table-sm w-full">
          <thead>
            <tr><th>Rank</th><th>Model</th><th>Lang</th><th>RTF</th><th>Chars/s</th><th>Speed</th></tr>
          </thead>
          <tbody>
            {benchmark_html}
          </tbody>
        </table>
      </div>
    </div>'''
        if ok_rows
        else ""
    }

    <!-- Failure details -->
    {failure_html}

    <!-- Footer -->
    <div class="text-center text-base-content/40 text-sm mt-12 mb-4">
      PVC — Personal Voice Clone · report.json is the canonical data source
    </div>
  </div>
</body>
</html>"""

    html_path = outdir / "report.html"
    html_path.write_text(page, encoding="utf-8")
    return html_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(
    outdir: str = "",
    text_en: str = "",
    text_pl: str = "",
    references: str = "",
    design_prompts_file: str = "",
    models: str = "all",
    langs: str = "all",
    timeout: int = 300,
    verbose: bool = False,
    force: bool = False,
    prep: bool = True,
    prep_only: bool = False,
    prep_timeout: int = 7200,
) -> None:
    """Prep then synthesize for all requested models and languages.

    Default (no args) = **prep every model, then say every model**: run
    ``./tts_all.py`` once and, some time later, ``out/`` holds every clip plus a
    sortable ``out/report.html``. The run is resumable — matching request/output
    receipts permit reuse (use ``--force`` to redo) and ``report.json`` is updated after each model,
    so an interrupted run continues where it left off on the next invocation.

    Args:
        outdir:  Output root directory (default: out/)
        models:  Comma-separated model names, or "all"
        langs:   Comma-separated lang codes (en, pl), or "all"
        timeout: Per-model timeout in seconds for ``say`` (default: 300; heavy models
                 set their own longer per-model timeouts)
        verbose: Show subprocess stdout/stderr
        force:   Overwrite existing output files (passed as --force to each script)
        prep:    Prep (download/cache) every selected model before synthesizing.
                 ON by default; pass ``--prep=False`` to skip and rely on lazy download.
        prep_only: Only run ``prep`` for selected models, then exit (no synthesis)
        prep_timeout: Per-model timeout for the prep/download step (default: 7200s;
            the largest audio.cpp packages are 8-19 GB and exceeded the old 1800s)
    """
    global DATASET_DIR, TEXT_FILES, REFS, REF_TEXTS, DESIGN_PROMPTS
    if design_prompts_file:
        prompts = json.loads(Path(design_prompts_file).expanduser().read_text(encoding="utf-8"))
        if not isinstance(prompts, list) or not prompts or any(not isinstance(prompt, str) or not prompt.strip() for prompt in prompts):
            raise SystemExit("Design prompts must be a nonempty JSON list of nonempty strings")
        DESIGN_PROMPTS = prompts
    if not outdir:
        outdir = str(runtime_path("out"))
    TEXT_FILES = {lang: Path(path).expanduser() for lang, path in {"en": text_en, "pl": text_pl}.items() if path}
    if not TEXT_FILES and not prep_only:
        raise SystemExit("Supply --text-en and/or --text-pl with input text files")
    DATASET_DIR = (Path(references).expanduser() if references else runtime_path("references")).resolve()
    selected_refs = {lang: select_reference(lang) for lang in TEXT_FILES} if references else {}
    REFS = {lang: selected_refs.get(lang, (None, ""))[0] for lang in TEXT_FILES}
    REF_TEXTS = {lang: selected_refs.get(lang, (None, ""))[1] for lang in TEXT_FILES}
    # Validate text files exist
    for lang, path in TEXT_FILES.items():
        if not path.exists():
            console.print(f"[red]Error: text file not found: {path}[/red]")
            sys.exit(1)

    # Warn if reference audio files missing (non-fatal)
    for lang_code, ref_path in REFS.items():
        if ref_path is None or not ref_path.exists():
            console.print(
                f"[yellow]Warning: no reference clip found in {DATASET_DIR} "
                f"for '{lang_code}'.[/yellow]"
            )
            console.print(
                f"[yellow]Voice cloning (IVC) models for '{lang_code}' will fail.[/yellow]"
            )

    selected_models = resolve_models(models)
    selected_langs = resolve_langs(langs)
    if not prep_only:
        missing_texts = set(selected_langs) - TEXT_FILES.keys()
        if missing_texts:
            raise SystemExit(f"Supply text files for languages: {sorted(missing_texts)}")

    active_model_count = len(selected_models)

    texts = {
        lang: read_text(TEXT_FILES[lang])
        for lang in selected_langs
        if lang in TEXT_FILES
    }

    # Build work queue. Each job is (model, lang, run_name, mode, instruct).
    # Voice-design models fan out into desi1/desi2/desi3 with distinct prompts;
    # every other model yields a single job whose mode is ivc or auto.
    work: list[tuple[dict, str, str, str, str | None]] = []
    for model in selected_models:
        for lang in selected_langs:
            if lang not in model["langs"]:
                continue
            if model.get("design") and "design_args_fn" in model:
                # Voice-design conventions are family-specific: most take prose,
                # but OmniVoice takes a controlled vocabulary and rejects anything
                # outside it. A model may therefore carry its own prompt set.
                prompts = model.get("design_prompts") or DESIGN_PROMPTS
                for i, prompt in enumerate(prompts, 1):
                    work.append((model, lang, model["name"], f"desi{i}", prompt))
            else:
                work.append((model, lang, model["name"], model_mode(model), None))

    out_root = Path(outdir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    # Prep phase: download/cache model weights up front (idempotent). Runs once per
    # selected model (not per lang). --prep-only stops here.
    if prep or prep_only:
        console.print(
            f"\n[bold]Prep[/bold] — downloading/caching {len(selected_models)} model(s)\n"
        )
        prep_ok = 0
        for model in selected_models:
            name = model["name"]
            ok, dur, err = prep_model(model, prep_timeout, verbose)
            if ok:
                prep_ok += 1
                console.print(f"  [green]✓[/green] prep {name} ({dur:.0f}s)")
            else:
                short = err.splitlines()[-1][:80] if err else "unknown error"
                console.print(
                    f"  [red]✗[/red] prep {name} failed ({dur:.0f}s): {short}"
                )
        console.print(
            f"\n[bold]Prep done:[/bold] {prep_ok}/{len(selected_models)} ready"
        )
        if prep_only:
            return

    # Load prior report.json for merging
    prior_data = load_report_json(out_root)
    prior_results: dict[str, dict] = {}
    if prior_data:
        pr = prior_data.get("results", {})
        if isinstance(pr, dict):
            prior_results = pr
        console.print(
            f"[dim]Loaded {len(prior_results)} prior result(s) from report.json[/dim]"
        )

    report_kw: dict[str, Any] = {
        "total_models": active_model_count,
        "total_available": len([m for m in MODELS if m["name"] not in IGNORED_ENGINES]),
        "selected_langs": selected_langs,
        "models_filter": models,
    }

    if not selected_models or not work:
        report_path, report_data = save_final_report(
            [], prior_data, out_root, 0.0, **report_kw
        )
        html_path = save_report_html(report_data, out_root)
        dash_path = save_dashboard(out_root)
        console.print(f"\n[bold]Report saved:[/bold] {report_path}")
        console.print(f"[bold]HTML report:[/bold]  {html_path}")
        console.print(f"[bold]Dashboard:[/bold]    {dash_path}")
        return

    console.print(
        f"\n[bold]TTS Batch[/bold] — {len(work)} run(s): "
        f"{active_model_count} model(s) × up to {len(selected_langs)} lang(s)\n"
    )

    results: list[Result] = []
    batch_start = time.monotonic()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task("Generating...", total=len(work))

        for model, lang, run_name, mode, instruct in work:
            stack = model_stack(model)
            quant = model_quant(model)
            # Display/identity name is unique per design variant (desi1/2/3).
            disp = run_name if instruct is None else f"{run_name}:{mode}"
            ref_path = REFS.get(lang)
            ref = str(ref_path) if (model["voice_clone"] and ref_path) else ""
            text = texts.get(lang, "")
            if not text:
                progress.advance(task)
                continue

            # Self-describing flat output path: lang__model__quant__stack__mode.ext
            ext = "mp3"
            out_name = output_filename(run_name, model, lang, mode, ext)
            out_path = str(out_root / out_name)

            out_name_wav = output_filename(run_name, model, lang, mode, "wav")
            out_path_wav = str(out_root / out_name_wav)

            run_args = (model["design_args_fn"](text, instruct, out_path_wav, lang)
                        if instruct is not None else model["args_fn"](text, ref, out_path_wav, lang))
            request = _identity["request_identity"](
                model, ROOT / model["script"], run_args,
                [Path(ref)] if ref else [],
                {"lang": lang, "mode": mode, "text": text, "instruct": instruct,
                 "reference_text": REF_TEXTS.get(lang, "") if ref else ""})

            if (
                not force
                and _identity["reusable_output"](Path(out_path), request)
            ):
                progress.console.print(
                    f"  [yellow]⏭[/yellow] {disp}/{lang} skipped (exists: {out_name})"
                )

                # Check for prior metrics to preserve
                key = _result_key(disp, lang)
                prior = prior_results.get(key)
                if prior and prior.get("success") and prior.get("rtf", 0) > 0:
                    # Restore prior result with real metrics
                    restored = Result.from_dict(prior)
                    # Update output_path in case directory moved
                    restored.output_path = out_path
                    restored.stack = stack
                    restored.quant = quant
                    restored.mode = mode
                    results.append(restored)
                else:
                    audio_dur = get_audio_duration(out_path)
                    file_size_kb = Path(out_path).stat().st_size / 1024
                    results.append(
                        Result(
                            disp,
                            lang,
                            True,
                            0.0,
                            "",
                            out_path,
                            file_size_kb,
                            audio_dur,
                            0.0,
                            0.0,
                            len(text),
                            stack,
                            quant,
                            mode,
                        )
                    )
                save_partial_report(
                    results, prior_data, out_root, len(results), len(work), **report_kw
                )
                progress.advance(task)
                continue

            progress.update(
                task, description=f"[cyan]{disp}[/cyan]/[yellow]{lang}[/yellow]"
            )

            model_timeout = model.get("timeout", timeout)
            if not force and any(path.exists() or path.is_symlink() for path in (Path(out_path), Path(out_path_wav))):
                success, duration, error = False, 0.0, "Unbound or changed audio output; use --force to regenerate"
            else:
                success, duration, error = run_model(
                    model=model, lang=lang, text=text, ref=ref, out=out_path_wav,
                    timeout=model_timeout, verbose=verbose, force=force, args=run_args)

            if success:
                if Path(out_path_wav).is_file() and Path(out_path_wav).stat().st_size > 0:
                    try:
                        subprocess.run(
                            ["ffmpeg", "-y", "-i", out_path_wav, "-q:a", "2", out_path],
                            check=True,
                            capture_output=True,
                            text=True,
                        )
                        if Path(out_path).is_file() and Path(out_path).stat().st_size > 0:
                            Path(out_path_wav).unlink()
                        else:
                            success, error = False, "missing MP3 output"
                    except subprocess.CalledProcessError as e:
                        success = False
                        error = f"ffmpeg conversion failed: {e.stderr}"
                else:
                    success, error = False, "missing WAV output"

            if success:
                if Path(out_path).is_file() and Path(out_path).stat().st_size > 0:
                    _identity["record_output"](Path(out_path), request)
                else:
                    success, error = False, "missing MP3 output"

            if success:
                progress.console.print(
                    f"  [green]✓[/green] {disp}/{lang} → {out_name} ({duration:.1f}s)"
                )
            else:
                short = error.splitlines()[-1][:80] if error else "unknown error"
                progress.console.print(
                    f"  [red]✗[/red] {disp}/{lang} failed ({duration:.1f}s): {short}"
                )
                if verbose and error:
                    progress.console.print(f"    [dim]{error}[/dim]")

            file_size_kb = 0.0
            if success:
                try:
                    file_size_kb = Path(out_path).stat().st_size / 1024
                except OSError:
                    pass

            audio_dur = 0.0
            rtf = 0.0
            cps = 0.0
            text_len = len(text)
            if success:
                audio_dur = get_audio_duration(out_path)
                if audio_dur > 0:
                    rtf = duration / audio_dur
                if duration > 0:
                    cps = text_len / duration

            results.append(
                Result(
                    disp,
                    lang,
                    success,
                    duration,
                    error,
                    out_path,
                    file_size_kb,
                    audio_dur,
                    rtf,
                    cps,
                    text_len,
                    stack,
                    quant,
                    mode,
                )
            )
            save_partial_report(
                results, prior_data, out_root, len(results), len(work), **report_kw
            )
            progress.advance(task)

    batch_elapsed = time.monotonic() - batch_start
    print_summary(results, selected_langs)
    print_benchmarks(results, batch_elapsed)

    report_path, report_data = save_final_report(
        results, prior_data, out_root, batch_elapsed, **report_kw
    )
    html_path = save_report_html(report_data, out_root)
    dash_path = save_dashboard(out_root)
    console.print(f"\n[bold]Report saved:[/bold] {report_path}")
    console.print(f"[bold]HTML report:[/bold]  {html_path}")
    console.print(f"[bold]Dashboard:[/bold]    {dash_path}")


if __name__ == "__main__":
    fire.Fire(main)
