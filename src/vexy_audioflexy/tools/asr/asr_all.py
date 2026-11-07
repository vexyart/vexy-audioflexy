#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = ["fire", "rich", "jiwer", "jieba==0.42.1", "platformdirs"]
# ///
# this_file: src/vexy_audioflexy/tools/asr/asr_all.py
"""Batch ASR tool — preps then runs ALL supported models over the eval set, scores, reports.

Each model is invoked via subprocess (its `prep` then `transcribe` Fire subcommand)
so dependency environments never conflict.

ASR output is text, so every run is *scored*: each caller-supplied eval clip has
a sibling `.lab` holding its ground-truth transcript, and the runner reports WER and
CER per clip and aggregated per model/language, alongside RTF.

With explicit input/output directories: prep every selected model, then transcribe.
The run is resumable: matching request/output receipts permit transcript reuse.
Unbound or changed outputs require explicit `--force`; reports retain failures.

Usage:
    ./asr_all.py --eval-dir /path/to/eval --outdir /path/to/results
    ./asr_all.py --eval-dir /path/to/eval --outdir /path/to/results --models parakeet-tdt-q8,sensevoice-small-q8
    ./asr_all.py --eval-dir /path/to/eval --outdir /path/to/results --langs en                       # English only
    ./asr_all.py --eval-dir /path/to/eval --outdir /path/to/results --clips 3                        # first N clips per language
    ./asr_all.py --eval-dir /path/to/eval --outdir /path/to/results --prep-only                      # download/cache everything, no inference
    ./asr_all.py --eval-dir /path/to/eval --outdir /path/to/results --force                          # redo even if transcripts exist
    ./asr_all.py --eval-dir /path/to/eval --outdir /path/to/results --verbose --timeout 900
"""

import datetime
import html as html_mod
import json
import os
import re
import runpy
import subprocess
import sys
import tempfile
import time
import unicodedata
from pathlib import Path

_identity = runpy.run_path(str(Path(__file__).resolve().parents[1] / "_batch_identity.py"))

sys.path.insert(0, str(Path(__file__).resolve().parent / "tools"))

from _audioflexy_paths import runtime_root, models_root
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

console = Console()

# ---------------------------------------------------------------------------
# Blocked engines — no working Apple-Silicon runtime, kept in report.json with the
# specific reason so the finding survives, but never executed.
# ---------------------------------------------------------------------------

BLOCKED_ENGINES: dict[str, str] = {
    # Evaluated 2026-09-03 from the upstream-repo review (issues/2026-09-02-repo-triage.md).
    # Each is a working project; none is an offline batch ASR engine, which is the
    # only shape this harness can score. Recorded so the next pass does not re-try them.
    "whisperlivekit": (
        "Not a batch engine: QuentinFuxa/WhisperLiveKit is a streaming ASR *server* "
        "(FastAPI/WebSocket, OpenAI- and Deepgram-compatible endpoints). Its models "
        "are faster-whisper and whisper.cpp, which run here directly as "
        "whisperx-* and whispercpp-*."
    ),
    "auto-subs": (
        "Not a batch engine: tmoroney/auto-subs is a DaVinci Resolve / Premiere "
        "plugin written in TypeScript. It drives an ASR backend rather than being one."
    ),
    "openless": (
        "Not a batch engine: Open-Less/openless is a Rust desktop dictation app "
        "(hold a key, speak, text at the cursor) for macOS and Windows."
    ),
    "dograh": (
        "Not a batch engine: dograh-hq/dograh is a self-hosted voice-agent platform, "
        "a Vapi/Retell alternative. It orchestrates STT/TTS services."
    ),
    "voicestudio": (
        "Not a batch engine: debpalash/VoiceStudio is an end-user application over "
        "other people's models — the same purpose as this workspace, not an engine for it."
    ),
    "argmax-oss-swift-tts": (
        "Wrong project: argmaxinc/argmax-oss-swift's TTSKit is text-to-speech and "
        "belongs to ../tts. Its ASR half is wrapped as whisperkit-*."
    ),
    "higgs-audio-v2-3b-mlx-q8": (
        "Not an ASR model: mlx-community/higgs-audio-v2-3B-mlx-q8 is text-to-speech "
        "(pipeline_tag: text-to-speech). Listed on the ASR issue by mistake."
    ),
    "moss-tts-local-mlx-8bit": (
        "Not an ASR model: mlx-community/MOSS-TTS-Local-Transformer-MLX-8bit is "
        "text-to-speech (model_type moss_tts_delay). Confirmed, as the issue suspected."
    ),
    "granite-speech-5.0-470m-mlx": (
        "No Python runtime: iky1e/granite-speech-5.0-470m-turboctc-nc-mlx-fp16 is "
        "model_type granite_speech5_ctc, absent from mlx-audio's MODEL_REMAPPING; its "
        "only runtime is the `granite-mlx` Swift CLI. Use granite-nar-mlx instead."
    ),
    "delusion": (
        "Not an ASR runtime: BrokenSource/Delusion wraps audio.cpp but exposes only "
        "TTS, voice design and music generation — it has no STT/ASR class at all."
    ),
    "audio8-asr-gguf": (
        "Not installable: audio8_asr_0_1b_q8_0 / _f16 carry download kind 'unsupported' "
        "in model_specs/audio8_asr.json. The safetensors package is registered instead."
    ),
    "mms-forced-aligner-gguf": (
        "Not installable: mms_forced_aligner_300m_q8_0 / _f16 carry download kind "
        "'unsupported'. The safetensors package is the aligner default instead."
    ),
}

BLOCKED_STATUS = "blocked"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent
# Eval set: {lang}_*.wav paired with a same-stem .lab ground-truth transcript,
# supplied by the caller; no reference corpus is distributed.
EVAL_DIR = None
ALL_LANGS = ["en", "pl"]


def eval_clips(lang: str, limit: int = 0) -> list[tuple[Path, str]]:
    """Return ``(wav, reference_transcript)`` for every matched pair of *lang*.

    A clip counts only when both the ``.wav`` and its sibling ``.lab`` exist;
    ``.lab`` files without audio are dataset leftovers and are skipped.
    """
    if EVAL_DIR is None:
        raise ValueError("Supply eval_dir to the batch command")
    pairs: list[tuple[Path, str]] = []
    for wav in sorted(EVAL_DIR.glob(f"{lang}_*.wav")):
        lab = wav.with_suffix(".lab")
        if not lab.exists():
            continue
        try:
            ref = lab.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if ref:
            pairs.append((wav, ref))
    return pairs[:limit] if limit else pairs


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------
# args_fn(audio_path, out_path, lang) -> list[str], appended after `transcribe`.
# ---------------------------------------------------------------------------

AUDIOCPP = "tools/asr-audiocpp.py"


def _audiocpp_args(package: str, *, lang_flag: bool = False, instruct: str = ""):
    """Build an args_fn for one audio.cpp package.

    *lang_flag* is opt-in: audio.cpp forwards ``--language`` as a per-request option
    and families that do not declare one (parakeet_tdt) hard-fail on it.
    """

    def args_fn(audio: str, out: str, lang: str) -> list[str]:
        argv = ["--audio", audio, "--package", package, "--text-out", out]
        if lang_flag and lang:
            argv += ["--lang", lang]
        if instruct:
            argv += ["--text", instruct]
        return argv

    return args_fn


def _mlx_args(repo: str = "", *, lang_flag: bool = False):
    """Build an args_fn for an MLX wrapper that takes `--repo`."""

    def args_fn(audio: str, out: str, lang: str) -> list[str]:
        argv = ["--audio", audio, "--text-out", out]
        if repo:
            argv += ["--repo", repo]
        if lang_flag and lang:
            argv += ["--lang", lang]
        return argv

    return args_fn


SHERPAONNX = "tools/asr-sherpaonnx.py"


def _sherpaonnx_args(package: str, *, lang_flag: bool = False):
    """Build an args_fn for one sherpa-onnx package."""

    def args_fn(audio: str, out: str, lang: str) -> list[str]:
        argv = ["--audio", audio, "--package", package, "--text-out", out]
        if lang_flag and lang:
            argv += ["--lang", lang]
        return argv

    return args_fn


WHISPERKIT = "tools/asr-whisperkit.py"
SILERO = "tools/asr-silero.py"
FUNASR = "tools/asr-funasr.py"


def _funasr_args(package: str, *, lang_flag: bool = False):
    """Build an args_fn for one FunASR package."""

    def args_fn(audio: str, out: str, lang: str) -> list[str]:
        argv = ["--audio", audio, "--package", package, "--text-out", out]
        if lang_flag and lang:
            argv += ["--lang", lang]
        return argv

    return args_fn


WHISPERX = "tools/asr-whisperx.py"


def _whisperx_args(package: str):
    """Build an args_fn for one whisperX model. Word alignment is off in batch
    runs: it roughly doubles the wall time and the harness scores text, not
    timings. Use the wrapper directly with `--words` for those."""

    def args_fn(audio: str, out: str, lang: str) -> list[str]:
        argv = ["--audio", audio, "--package", package, "--text-out", out]
        if lang:
            argv += ["--lang", lang]
        return argv

    return args_fn


WHISPERCPP = "tools/asr-whispercpp.py"


def _whispercpp_args(package: str):
    """Build an args_fn for one whisper.cpp model. `--lang` is always forwarded:
    whisper auto-detects otherwise and guesses badly on short clips."""

    def args_fn(audio: str, out: str, lang: str) -> list[str]:
        argv = ["--audio", audio, "--package", package, "--text-out", out]
        if lang:
            argv += ["--lang", lang]
        return argv

    return args_fn


MODELS: list[dict[str, Any]] = [
    # --- audio.cpp (GGUF / safetensors via one C++ engine, Metal backend) ---
    {
        "name": "parakeet-tdt-q8",
        "script": AUDIOCPP,
        "package": "parakeet_tdt_q8_0",
        "repo": "audio-cpp/audio.cpp-gguf",
        "langs": ["en", "pl"],
        # parakeet_tdt rejects the `language` request option outright.
        "args_fn": _audiocpp_args("parakeet_tdt_q8_0"),
    },
    {
        "name": "sensevoice-small-q8",
        "script": AUDIOCPP,
        "package": "sensevoice_small_q8",
        "repo": "FunAudioLLM/SenseVoiceSmall-GGUF-audiocpp",
        "langs": ["en", "pl"],
        "args_fn": _audiocpp_args("sensevoice_small_q8", lang_flag=True),
    },
    {
        "name": "qwen3-asr-06b-q8",
        "script": AUDIOCPP,
        "package": "qwen3_asr_0_6b_q8_0",
        "repo": "audio-cpp/audio.cpp-gguf",
        "langs": ["en", "pl"],
        "args_fn": _audiocpp_args("qwen3_asr_0_6b_q8_0", lang_flag=True),
    },
    {
        "name": "qwen3-asr-17b-q8",
        "script": AUDIOCPP,
        "package": "qwen3_asr_1_7b_q8_0",
        "repo": "audio-cpp/audio.cpp-gguf",
        "langs": ["en", "pl"],
        "timeout": 900,
        "args_fn": _audiocpp_args("qwen3_asr_1_7b_q8_0", lang_flag=True),
    },
    {
        "name": "nemotron-asr-q8",
        "script": AUDIOCPP,
        "package": "nemotron_asr_q8_0",
        "repo": "audio-cpp/audio.cpp-gguf",
        "langs": ["en", "pl"],
        "args_fn": _audiocpp_args("nemotron_asr_q8_0", lang_flag=True),
    },
    {
        "name": "hviske-asr-q8",
        "script": AUDIOCPP,
        "package": "hviske_asr_q8_0",
        "repo": "mirek190/audio.cpp",
        "langs": ["en", "pl"],
        "args_fn": _audiocpp_args("hviske_asr_q8_0", lang_flag=True),
    },
    {
        "name": "fun-asr-nano-q8",
        "script": AUDIOCPP,
        "package": "fun_asr_nano_2512_q8_0",
        "repo": "FunAudioLLM/Fun-ASR-Nano-2512-GGUF",
        "langs": ["en"],
        "args_fn": _audiocpp_args("fun_asr_nano_2512_q8_0", lang_flag=True),
    },
    {
        "name": "firered-audio-q8",
        "script": AUDIOCPP,
        "package": "firered_audio_q8_0",
        "repo": "audio-cpp/audio.cpp-gguf",
        "langs": ["en"],
        "args_fn": _audiocpp_args("firered_audio_q8_0", lang_flag=True),
    },
    {
        "name": "kroko-asr-q8",
        "script": AUDIOCPP,
        "package": "kroko_asr_community_q8_0",
        "repo": "audio-cpp/audio.cpp-gguf",
        "langs": ["en"],
        "args_fn": _audiocpp_args("kroko_asr_community_q8_0", lang_flag=True),
    },
    {
        "name": "audio8-asr-01b-st",
        "script": AUDIOCPP,
        "package": "audio8_asr_0_1b_safetensors",
        "repo": "Audio8/Audio8-ASR-0.1B",
        "quant": "bf16",
        "langs": ["en"],
        "args_fn": _audiocpp_args("audio8_asr_0_1b_safetensors", lang_flag=True),
    },
    {
        "name": "citrinet-asr-q8",
        "script": AUDIOCPP,
        "package": "citrinet_asr_q8_0",
        "repo": "audio-cpp/audio.cpp-gguf",
        "langs": ["en"],
        "args_fn": _audiocpp_args("citrinet_asr_q8_0"),
    },
    {
        "name": "granite5asr-q8",
        "script": AUDIOCPP,
        "package": "granite5asr_q8_0",
        "repo": "audio-cpp/audio.cpp-gguf",
        "langs": ["en"],
        "args_fn": _audiocpp_args("granite5asr_q8_0"),
    },
    {
        "name": "higgs-audio-stt-q8",
        "script": AUDIOCPP,
        "package": "higgs_audio_stt_q8_0",
        "repo": "audio-cpp/audio.cpp-gguf",
        "langs": ["en"],
        "timeout": 900,
        # This family takes an instruction prompt even though --list-loaders
        # reports instructions_policy "none" for it.
        "args_fn": _audiocpp_args(
            "higgs_audio_stt_q8_0", instruct="Transcribe the speech."
        ),
    },
    {
        "name": "voxtral-realtime-q8",
        "script": AUDIOCPP,
        "package": "voxtral_realtime_q8_0",
        "repo": "audio-cpp/audio.cpp-gguf",
        "langs": ["en"],
        "timeout": 900,
        "args_fn": _audiocpp_args("voxtral_realtime_q8_0", lang_flag=True),
    },
    {
        "name": "vibevoice-asr-q8",
        "script": AUDIOCPP,
        "package": "vibevoice_asr_q8_0",
        "repo": "audio-cpp/audio.cpp-gguf",
        "langs": ["en", "pl"],
        "timeout": 900,
        "args_fn": _audiocpp_args("vibevoice_asr_q8_0", lang_flag=True),
    },
    # --- MLX ---
    {
        "name": "mlx-whisper-large-v3-turbo",
        "script": "tools/asr-mlxwhisper.py",
        "repo": "mlx-community/whisper-large-v3-turbo",
        "stack": "mlx",
        "quant": "f16",
        "langs": ["en", "pl"],
        "timeout": 900,
        "args_fn": _mlx_args("mlx-community/whisper-large-v3-turbo", lang_flag=True),
    },
    {
        "name": "qwen3-asr-17b-mlx-8bit",
        "script": "tools/asr-qwen3-mlx.py",
        "repo": "mlx-community/Qwen3-ASR-1.7B-8bit",
        "stack": "mlx",
        "quant": "8bit",
        "langs": ["en", "pl"],
        "timeout": 900,
        "args_fn": _mlx_args("mlx-community/Qwen3-ASR-1.7B-8bit", lang_flag=True),
    },
    {
        "name": "qwen3-asr-17b-mlx-bf16",
        "script": "tools/asr-qwen3-mlx.py",
        "repo": "mlx-community/Qwen3-ASR-1.7B-bf16",
        "stack": "mlx",
        "quant": "bf16",
        "langs": ["en", "pl"],
        "timeout": 900,
        "args_fn": _mlx_args("mlx-community/Qwen3-ASR-1.7B-bf16", lang_flag=True),
    },
    {
        "name": "granite-nar-mlx",
        "script": "tools/asr-granite-nar-mlx.py",
        "repo": "mlx-community/granite-speech-4.1-2b-nar-mlx",
        "stack": "mlx",
        "quant": "bf16",
        "langs": ["en"],
        "timeout": 900,
        "args_fn": _mlx_args(),
    },
    {
        "name": "cohere-transcribe-mlx-8bit",
        "script": "tools/asr-cohere-mlx.py",
        "repo": "mlx-community/cohere-transcribe-03-2026-mlx-8bit",
        "stack": "mlx",
        "quant": "8bit",
        "langs": ["en"],
        "timeout": 900,
        "args_fn": _mlx_args(lang_flag=True),
    },
    {
        "name": "ark-asr-06b-mlx",
        "script": "tools/asr-ark-mlx.py",
        "repo": "leope/ark-asr-0.6B-mlx",
        "stack": "mlx",
        "quant": "bf16",
        "langs": ["en"],
        "timeout": 900,
        "args_fn": _mlx_args("leope/ark-asr-0.6B-mlx"),
    },
    {
        "name": "ark-asr-3b-mlx",
        "script": "tools/asr-ark-mlx.py",
        "repo": "leope/ark-asr-3B-mlx",
        "stack": "mlx",
        "quant": "bf16",
        "langs": ["en"],
        "timeout": 1200,
        "args_fn": _mlx_args("leope/ark-asr-3B-mlx"),
    },
    # --- sherpa-onnx (one onnxruntime engine, family selected by package) ---
    # `--lang` is opt-in: whisper and sense-voice take a language, the
    # transducers are multilingual without one and the wrapper rejects it.
    {
        "name": "sherpaonnx-whisper-tiny",
        "script": SHERPAONNX,
        "package": "whisper-tiny",
        "repo": "openai/whisper-tiny (sherpa-onnx export)",
        "stack": "onnx",
        "quant": "int8",
        "langs": ['en', 'pl'],
        "timeout": 600,
        "args_fn": _sherpaonnx_args("whisper-tiny", lang_flag=True),
    },
    {
        "name": "sherpaonnx-whisper-small",
        "script": SHERPAONNX,
        "package": "whisper-small",
        "repo": "openai/whisper-small (sherpa-onnx export)",
        "stack": "onnx",
        "quant": "int8",
        "langs": ['en', 'pl'],
        "timeout": 900,
        "args_fn": _sherpaonnx_args("whisper-small", lang_flag=True),
    },
    {
        "name": "sherpaonnx-sense-voice",
        "script": SHERPAONNX,
        "package": "sense-voice",
        "repo": "FunAudioLLM/SenseVoiceSmall",
        "stack": "onnx",
        "quant": "int8",
        "langs": ['en'],
        "timeout": 600,
        "args_fn": _sherpaonnx_args("sense-voice", lang_flag=True),
    },
    {
        "name": "sherpaonnx-zipformer-en",
        "script": SHERPAONNX,
        "package": "zipformer-en",
        "repo": "k2-fsa icefall zipformer en 2023-06-26",
        "stack": "onnx",
        "quant": "int8",
        "langs": ['en'],
        "timeout": 600,
        "args_fn": _sherpaonnx_args("zipformer-en", lang_flag=False),
    },
    {
        "name": "sherpaonnx-nemo-fastconformer-eu",
        "script": SHERPAONNX,
        "package": "nemo-fast-conformer-eu",
        "repo": "nvidia stt_multilingual_fastconformer",
        "stack": "onnx",
        "quant": "f32",
        "langs": ['en', 'pl'],
        "timeout": 600,
        "args_fn": _sherpaonnx_args("nemo-fast-conformer-eu", lang_flag=False),
    },
    {
        "name": "sherpaonnx-parakeet-tdt-v3",
        "script": SHERPAONNX,
        "package": "parakeet-tdt-v3",
        "repo": "nvidia/parakeet-tdt-0.6b-v3",
        "stack": "onnx",
        "quant": "int8",
        "langs": ['en', 'pl'],
        "timeout": 900,
        "args_fn": _sherpaonnx_args("parakeet-tdt-v3", lang_flag=False),
    },
    {
        "name": "sherpaonnx-moonshine-base-en",
        "script": SHERPAONNX,
        "package": "moonshine-base-en",
        "repo": "UsefulSensors/moonshine-base",
        "stack": "onnx",
        "quant": "int8",
        "langs": ['en'],
        "timeout": 600,
        "args_fn": _sherpaonnx_args("moonshine-base-en", lang_flag=False),
    },
    # --- whisper.cpp (ggml, Metal) via pywhispercpp -------------------------
    {
        "name": "whispercpp-small",
        "script": WHISPERCPP,
        "package": "small",
        "repo": "ggerganov/whisper.cpp (ggml weights)",
        "stack": "whispercpp",
        "quant": "f16",
        "langs": ['en', 'pl'],
        "timeout": 600,
        "args_fn": _whispercpp_args("small"),
    },
    {
        "name": "whispercpp-small-q5_1",
        "script": WHISPERCPP,
        "package": "small-q5_1",
        "repo": "ggerganov/whisper.cpp (ggml weights)",
        "stack": "whispercpp",
        "quant": "q5_1",
        "langs": ['en', 'pl'],
        "timeout": 600,
        "args_fn": _whispercpp_args("small-q5_1"),
    },
    {
        "name": "whispercpp-medium-q5_0",
        "script": WHISPERCPP,
        "package": "medium-q5_0",
        "repo": "ggerganov/whisper.cpp (ggml weights)",
        "stack": "whispercpp",
        "quant": "q5_0",
        "langs": ['en', 'pl'],
        "timeout": 900,
        "args_fn": _whispercpp_args("medium-q5_0"),
    },
    {
        "name": "whispercpp-large-v3-turbo",
        "script": WHISPERCPP,
        "package": "large-v3-turbo",
        "repo": "ggerganov/whisper.cpp (ggml weights)",
        "stack": "whispercpp",
        "quant": "f16",
        "langs": ['en', 'pl'],
        "timeout": 1200,
        "args_fn": _whispercpp_args("large-v3-turbo"),
    },
    {
        "name": "whispercpp-large-v3-turbo-q5_0",
        "script": WHISPERCPP,
        "package": "large-v3-turbo-q5_0",
        "repo": "ggerganov/whisper.cpp (ggml weights)",
        "stack": "whispercpp",
        "quant": "q5_0",
        "langs": ['en', 'pl'],
        "timeout": 900,
        "args_fn": _whispercpp_args("large-v3-turbo-q5_0"),
    },
    # --- whisperX (faster-whisper / CTranslate2, CPU only) -------------------
    {
        "name": "whisperx-small",
        "script": WHISPERX,
        "package": "small",
        "repo": "m-bain/whisperX (faster-whisper weights)",
        "stack": "whisperx",
        "quant": "int8",
        "langs": ["en", "pl"],
        "timeout": 900,
        "args_fn": _whisperx_args("small"),
    },
    {
        "name": "whisperx-large-v3-turbo",
        "script": WHISPERX,
        "package": "large-v3-turbo",
        "repo": "m-bain/whisperX (faster-whisper weights)",
        "stack": "whisperx",
        "quant": "int8",
        "langs": ["en", "pl"],
        "timeout": 1800,
        "args_fn": _whisperx_args("large-v3-turbo"),
    },
    # --- FunASR (ModelScope PyTorch toolkit) — English only, no Polish -------
    {
        "name": "funasr-sensevoice-small",
        "script": FUNASR,
        "package": "sensevoice-small",
        "repo": "modelscope/FunASR (iic/SenseVoiceSmall)",
        "stack": "funasr",
        "quant": "fp32",
        "langs": ["en"],
        "timeout": 900,
        "args_fn": _funasr_args("sensevoice-small", lang_flag=True),
    },
    {
        "name": "funasr-paraformer-en",
        "script": FUNASR,
        "package": "paraformer-en",
        "repo": "modelscope/FunASR (iic/speech_paraformer_asr-en-16k-vocab4199-pytorch)",
        "stack": "funasr",
        "quant": "fp32",
        "langs": ["en"],
        "timeout": 1800,
        "args_fn": _funasr_args("paraformer-en"),
    },
    # --- Silero STT (torch.hub JIT, ~50 MB, CPU) — en/de/es/ua, no Polish ----
    {
        "name": "silero-stt-en",
        "script": SILERO,
        "package": "en",
        "repo": "snakers4/silero-models",
        "stack": "torch",
        "quant": "fp32",
        "langs": ["en"],
        "timeout": 600,
        "args_fn": lambda audio, out, lang: [
            "--audio", audio, "--package", "en", "--text-out", out,
        ],
    },
    # --- WhisperKit (Swift + CoreML, Neural Engine) --------------------------
    {
        "name": "whisperkit-small",
        "script": WHISPERKIT,
        "package": "small",
        "repo": "argmaxinc/argmax-oss-swift (whisperkit-coreml)",
        "stack": "coreml",
        "quant": "f16",
        "langs": ["en", "pl"],
        "timeout": 1800,
        "args_fn": lambda audio, out, lang: (
            ["--audio", audio, "--package", "small", "--text-out", out]
            + (["--lang", lang] if lang else [])
        ),
    },
    {
        "name": "whisperkit-large-v3-turbo",
        "script": WHISPERKIT,
        "package": "large-v3_turbo",
        "repo": "argmaxinc/argmax-oss-swift (whisperkit-coreml)",
        "stack": "coreml",
        "quant": "f16",
        "langs": ["en", "pl"],
        "timeout": 2400,
        "args_fn": lambda audio, out, lang: (
            ["--audio", audio, "--package", "large-v3_turbo", "--text-out", out]
            + (["--lang", lang] if lang else [])
        ),
    },
]

# ---------------------------------------------------------------------------
# transcribe.cpp catalog from the pinned upstream checkout (issue 201).
# Language hints are accepted by the CLI for all families; its catalog determines
# which language fixtures can be scored, including variants outside en/pl.
def _transcribecpp_args(package: str, quant: str):
    def args_fn(audio: str, out: str, lang: str) -> list[str]:
        return ["--audio", audio, "--package", package, "--quant", quant,
                "--text-out", out, "--lang", lang]
    return args_fn


MODELS.append({
    "name": "index-echo-s2tt-2b-q8", "script": "tools/asr-indexecho.py",
    "repo": "audio-cpp/Index-Echo-S2TT-GGUF", "langs": ["zh"],
    "quant": "q8_0", "stack": "audiocpp", "timeout": 1800,
    "args_fn": lambda audio, text_out, lang: ["--audio", audio, "--text-out", text_out, "--lang", lang],
})

MODELS.append({
    "name": "mikhailo-nemotron35-mlx-onnx", "script": "tools/asr-mikhailo.py",
    "repo": "Mikhailo/nemotron-3.5-asr-streaming-0.6b-mlx-fp16", "langs": ["en", "pl"],
    "quant": "fp16", "stack": "mlx-onnx", "timeout": 1800,
    "args_fn": lambda audio, text_out, lang: ["--audio", audio, "--text-out", text_out, "--lang", lang],
})

MODELS.append({
    "name": "audio8-infinite-mlx-swift", "script": "tools/asr-mlxswift.py",
    "repo": "selcukkubur/Audio8-ASR-Infinite-mlx", "langs": ["en", "zh"],
    "quant": "bf16", "stack": "mlx-swift", "timeout": 1800,
    "args_fn": lambda audio, text_out, lang: ["--audio", audio, "--text-out", text_out,
        "--lang", lang, "--repo", "selcukkubur/Audio8-ASR-Infinite-mlx"],
})

MODELS.append({
    "name": "parakeet-ultra-mlx-swift", "script": "tools/asr-mlxswift.py",
    "repo": "selcukkubur/parakeet-ultra-mlx",
    "langs": ['bg', 'cs', 'da', 'de', 'el', 'en', 'es', 'et', 'fi', 'fr', 'hr', 'hu', 'it', 'lt', 'lv', 'mt', 'nl', 'pl', 'pt', 'ro', 'ru', 'sk', 'sl', 'sv', 'uk'],
    "quant": "bf16", "stack": "mlx-swift", "timeout": 1800,
    "args_fn": lambda audio, text_out, lang: ["--audio", audio, "--text-out", text_out, "--lang", lang],
})

MODELS.append({
    "name": "appautomaton-qwen3-asr-int8", "script": "tools/asr-appautomaton.py",
    "repo": "appautomaton/qwen3-asr-1.7b-int8-mlx", "langs": ["en", "zh"],
    "quant": "int8", "stack": "mlx", "timeout": 1800,
    "args_fn": lambda audio, out, lang: ["--audio", audio, "--text-out", out, "--lang", lang],
})

MODELS.append({
    "name": "confucius4-r2t2-8bit", "script": "tools/asr-qwen3-mlx.py",
    "repo": "mlx-community/Confucius4-R2T2-8bit", "langs": ["en"],
    "quant": "8bit", "stack": "mlx", "timeout": 1800,
    "args_fn": lambda audio, out, lang: ["--audio", audio, "--text-out", out,
        "--repo", "mlx-community/Confucius4-R2T2-8bit", "--lang", "English"],
})

for _entry in json.loads((ROOT / "catalogs/issue201-audiocpp.json").read_text())["models"]:
    if _entry["family"] in {"moss_transcribe_diarize", "index_echo"}:
        # Structured text needs dedicated parsing: diarization for MOSS;
        # Chinese source/translated subtitle separation for Index-Echo.
        continue
    MODELS.append({
        "name": "issue201-" + _entry["package"].replace("_", "-"),
        "script": AUDIOCPP, "package": _entry["package"], "repo": _entry["repo"],
        "stack": "audiocpp", "quant": _entry["quant"],
        "langs": sorted({code.split("-")[0] for code in _entry["languages"]}),
        "env": {"AUDIOCPP_ROOT": os.environ.get("AUDIOCPP_ISSUE201_ROOT", str(runtime_root("audio.cpp")))},
        "args_fn": _audiocpp_args(_entry["package"], lang_flag=_entry["lang_flag"]),
        "timeout": 1800,
    })
    if _entry["family"] == "niagara_asr":
        MODELS[-1]["env"]["AUDIOCPP_BACKEND"] = "cpu"

for _package, _repo, _langs in [
    ("phonon2", "cstr/phonon2-GGUF", ["en"]),
    ("parakeet-redux", "cstr/parakeet-redux-GGUF", ["en", "pl"]),
]:
    MODELS.append({
        "name": "crispasr-" + _package, "script": "tools/asr-crispasr.py",
        "package": _package, "repo": _repo, "langs": _langs,
        "stack": "crispasr", "quant": "q8_0",
        "args_fn": _whispercpp_args(_package),
    })

for _entry in json.loads((ROOT / "catalogs/transcribecpp.json").read_text())["models"]:
    _downloads = _entry["downloads"]
    _quant = next((d["quant"] for d in _downloads if d["quant"] == "Q8_0"),
                  _downloads[0]["quant"])
    MODELS.append({
        "name": "transcribecpp-" + _entry["variant"],
        "script": "tools/asr-transcribecpp.py", "package": _entry["variant"],
        "repo": _entry["published_repo"], "family": _entry["family"],
        "stack": "transcribecpp", "quant": _quant.lower(),
        "langs": sorted(set(_entry["languages"]) | {code.split("-")[0] for code in _entry["languages"]}),
        "languages_declared": _entry["languages"], "timeout": 1800,
        "prep_args": ["--quant", _quant],
        "args_fn": _transcribecpp_args(_entry["variant"], _quant),
    })

# Model metadata derivation (inference stack / quantization)
# ---------------------------------------------------------------------------

_QUANT_TOKENS = ("q8_0", "q4_k", "bf16", "fp16", "int8", "8bit", "4bit", "q8", "q4")
_STACK_DEFAULT_QUANT = {
    "audiocpp": "q8_0", "mlx": "bf16", "onnx": "fp16",
    "whispercpp": "f16", "whisperx": "int8", "funasr": "fp32",
    "coreml": "f16", "torch": "fp32",
}


for _entry in json.loads((ROOT / "catalogs/nemospeechcpp.json").read_text())["models"]:
    MODELS.append({
        "name": "nemospeechcpp-" + _entry["package"],
        "script": "tools/asr-nemospeechcpp.py", "package": _entry["package"],
        "repo": _entry["repo"], "langs": _entry["langs"],
        "stack": "nemospeechcpp", "quant": "q8_0", "timeout": 900,
        "args_fn": lambda audio, out, lang="en", p=_entry["package"]: [
            audio, "--text_out", out, "--lang", lang, "--package", p,
        ],
    })


for _entry in json.loads((ROOT / "catalogs/openasr.json").read_text())["models"]:
    _langs = [lang for lang in ("en", "pl") if any(item.split("-")[0] == lang for item in _entry["langs"])]
    if not _langs:
        continue
    MODELS.append({
        "name": "openasr-" + _entry["package"],
        "script": "tools/asr-openasr.py", "package": _entry["package"],
        "repo": _entry["repo"], "langs": _langs,
        "stack": "openasr", "quant": _entry["quant"], "timeout": 900,
        "args_fn": lambda audio, out, lang="en", p=_entry["package"]: [
            audio, "--text_out", out, "--lang", lang, "--package", p,
        ],
    })


for _repo, _quant, _langs in [
    ("littoralai/cohere-transcribe-mlx-fp16", "f16", ["en", "pl"]),
    ("divydeep/granite-speech-4.1-2b-mlx-8bit", "8bit", ["en"]),
    ("majentik/Qwen3-ASR-1.7B-MLX-8bit", "8bit", ["en", "pl"]),
]:
    MODELS.append({
        "name": "issue301-" + _repo.replace("/", "-"),
        "script": "tools/asr-mlx301.py", "repo": _repo,
        "stack": "mlx", "quant": _quant, "langs": _langs,
        "args_fn": _mlx_args(_repo, lang_flag=True), "timeout": 1800,
    })


for _entry in json.loads((ROOT / "catalogs/issue301-audiocpp.json").read_text())["models"]:
    if _entry["task"] != "asr" or not _entry["package"]:
        continue
    if _entry["family"] in {"moss_transcribe_diarize", "index_echo"}:
        # These outputs require structured task consumers rather than plain WER.
        continue
    MODELS.append({
        "name": "issue301-" + _entry["package"].replace("_", "-"),
        "script": AUDIOCPP, "package": _entry["package"],
        "repo": _entry["download"].get("repo", "local conversion"),
        "stack": "audiocpp", "quant": _entry["quant"],
        "langs": sorted({code.split("-")[0] for code in _entry["languages"]}),
        "env": {
            "AUDIOCPP_ROOT": os.environ.get("AUDIOCPP_ISSUE301_ROOT", str(runtime_root("audio.cpp-issue301"))),
            "AUDIOCPP_SPECS_DIR": str(ROOT / "model_specs/issue301"),
            "AUDIOCPP_MODELS_ROOT": str(models_root() / "audio.cpp-issue301"),
            "AUDIOCPP_BACKEND": "cpu" if _entry["family"] == "niagara_asr" else "metal",
        },
        "args_fn": _audiocpp_args(_entry["package"], lang_flag=_entry["lang_flag"]),
        "timeout": 1800,
    })


def model_stack(model: dict) -> str:
    """Inference stack: audiocpp | whispercpp | whisperx | funasr | coreml | mlx | onnx | torch."""
    if model.get("stack"):
        return str(model["stack"])
    script = model["script"].lower()
    if "whisperkit" in script:
        return "coreml"
    if "funasr" in script:
        return "funasr"
    if "whisperx" in script:
        return "whisperx"
    if "whispercpp" in script:
        return "whispercpp"
    if "audiocpp" in script:
        return "audiocpp"
    if "mlx" in script:
        return "mlx"
    if "onnx" in script:
        return "onnx"
    return "torch"


def model_quant(model: dict) -> str:
    """Quantization label from the model dict, its name, or the stack default."""
    if model.get("quant"):
        return str(model["quant"])
    name = model["name"].lower()
    for tok in _QUANT_TOKENS:
        if tok in name:
            return tok
    return _STACK_DEFAULT_QUANT.get(model_stack(model), "na")


def output_filename(model: dict, lang: str, clip_stem: str) -> str:
    """Self-describing flat filename: ``{lang}__{model}__{quant}__{stack}__{clip}.txt``."""
    return f"{lang}__{model['name']}__{model_quant(model)}__{model_stack(model)}__{clip_stem}.txt"


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
# WER/CER come from `jiwer` (a maintained, widely used implementation — see
# DEPENDENCIES.md) so this file holds only the normalization policy, which is what
# actually decides whether a score is fair.
#
# Both reference and hypothesis go through the SAME normalization: NFC, casefold,
# punctuation and symbols stripped, whitespace collapsed. Diacritics are DELIBERATELY
# kept — folding them would hide real Polish errors (`łotrów` vs `lotrow`).
# ---------------------------------------------------------------------------

_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Case- and punctuation-insensitive form used for both sides of a score."""
    text = unicodedata.normalize("NFC", text).casefold()
    text = "".join(
        " " if unicodedata.category(ch)[0] in {"P", "S"} else ch for ch in text
    )
    return _WS.sub(" ", text).strip()


def score(reference: str, hypothesis: str, lang: str = "") -> dict[str, Any]:
    """Return normalized texts plus WER (word) and CER (character) rates."""
    import jiwer

    ref_n = normalize(reference)
    hyp_n = normalize(hypothesis)
    if lang == "zh":
        import jieba

        ref_n, hyp_n = ref_n.replace(" ", ""), hyp_n.replace(" ", "")
        ref_words = " ".join(jieba.lcut(ref_n, HMM=False))
        hyp_words = " ".join(jieba.lcut(hyp_n, HMM=False))
        return {"ref_norm": ref_n, "hyp_norm": hyp_n,
            "ref_wer_norm": ref_words, "hyp_wer_norm": hyp_words,
            "wer_tokenization": "jieba-0.42.1-accurate-HMM-false",
            "wer": float(jiwer.wer(ref_words, hyp_words)),
            "cer": float(jiwer.cer(ref_n, hyp_n))}
    if not ref_n:
        return {"ref_norm": ref_n, "hyp_norm": hyp_n, "wer": 0.0, "cer": 0.0}
    return {
        "ref_norm": ref_n,
        "hyp_norm": hyp_n,
        "wer": float(jiwer.wer(ref_n, hyp_n)),
        "cer": float(jiwer.cer(ref_n, hyp_n)),
    }


# ---------------------------------------------------------------------------
# Subprocess drivers
# ---------------------------------------------------------------------------


def _script_error_output(stdout: str | None, stderr: str | None) -> str:
    parts = [part.strip() for part in (stdout, stderr) if part and part.strip()]
    return "\n".join(parts) if parts else "(no output)"


def _parse_result_line(stdout: str) -> dict[str, Any]:
    """Extract the wrapper's machine-readable `asr.result={...}` tail line."""
    for line in reversed(stdout.splitlines()):
        if line.startswith("asr.result="):
            try:
                return json.loads(line[len("asr.result="):])
            except json.JSONDecodeError:
                return {}
    return {}


def run_model(
    model: dict, audio: Path, out: Path, lang: str, timeout: int, verbose: bool,
    force: bool = False,
) -> tuple[bool, float, str, dict]:
    """Transcribe one clip. Returns (success, wall_s, error, wrapper_result_dict)."""
    argv = model["args_fn"](str(audio), str(out), lang)
    cmd = ["uv", "run", "--script", str(ROOT / model["script"]), "transcribe"] + [str(a) for a in argv]
    if force:
        cmd.append("--force")
    env = os.environ.copy()
    env.update(model.get("env", {}))

    t0 = time.monotonic()
    try:
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=timeout, env=env, cwd=str(ROOT),
        )
        duration = time.monotonic() - t0
        if verbose and result.stdout:
            console.print(f"[dim]{result.stdout.strip()}[/dim]")
        if result.returncode != 0:
            combined = _script_error_output(result.stdout, result.stderr)
            return False, duration, "\n".join(combined.splitlines()[-5:]), {}
        return True, duration, "", _parse_result_line(result.stdout)
    except subprocess.TimeoutExpired:
        return False, time.monotonic() - t0, f"timeout after {timeout}s", {}
    except FileNotFoundError:
        return False, time.monotonic() - t0, f"script not found: {model['script']}", {}
    except Exception as exc:  # noqa: BLE001 - reported, not swallowed
        return False, time.monotonic() - t0, str(exc), {}


def prep_model(model: dict, timeout: int, verbose: bool) -> tuple[bool, float, str]:
    """Run a model's ``prep`` subcommand (download + cache weights). Idempotent."""
    cmd = ["uv", "run", "--script", str(ROOT / model["script"]), "prep"]
    if model.get("package"):
        cmd += ["--package", model["package"]]
    elif model.get("repo"):
        cmd += ["--repo", model["repo"]]
    cmd += model.get("prep_args", [])
    env = os.environ.copy()
    env.update(model.get("env", {}))
    t0 = time.monotonic()
    try:
        result = subprocess.run(
            cmd, stdout=None if verbose else subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=timeout, env=env, cwd=str(ROOT),
        )
        duration = time.monotonic() - t0
        if result.returncode != 0:
            combined = _script_error_output(result.stdout, result.stderr)
            return False, duration, "\n".join(combined.splitlines()[-5:])
        return True, duration, ""
    except subprocess.TimeoutExpired:
        return False, time.monotonic() - t0, f"prep timeout after {timeout}s"
    except Exception as exc:  # noqa: BLE001
        return False, time.monotonic() - t0, str(exc)


# ---------------------------------------------------------------------------
# Result record — one model × one language, aggregating its per-clip scores
# ---------------------------------------------------------------------------


class Result:
    __slots__ = (
        "model", "lang", "success", "duration", "error", "stack", "quant",
        "clips", "wer", "cer", "rtf", "audio_duration_s", "clips_ok", "clips_total",
    )

    def __init__(
        self, model: str, lang: str, success: bool, duration: float, error: str = "",
        stack: str = "", quant: str = "", clips: list[dict] | None = None,
        wer: float = 0.0, cer: float = 0.0, rtf: float = 0.0,
        audio_duration_s: float = 0.0, clips_ok: int = 0, clips_total: int = 0,
    ):
        self.model = model
        self.lang = lang
        self.success = success
        self.duration = duration
        self.error = error
        self.stack = stack
        self.quant = quant
        self.clips = clips or []
        self.wer = wer
        self.cer = cer
        self.rtf = rtf
        self.audio_duration_s = audio_duration_s
        self.clips_ok = clips_ok
        self.clips_total = clips_total

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__slots__}

    @classmethod
    def from_dict(cls, d: dict) -> "Result":
        return cls(
            model=str(d.get("model", "")), lang=str(d.get("lang", "")),
            success=bool(d.get("success", False)), duration=float(d.get("duration", 0.0)),
            error=str(d.get("error", "")), stack=str(d.get("stack", "")),
            quant=str(d.get("quant", "")), clips=list(d.get("clips", []) or []),
            wer=float(d.get("wer", 0.0)), cer=float(d.get("cer", 0.0)),
            rtf=float(d.get("rtf", 0.0)),
            audio_duration_s=float(d.get("audio_duration_s", 0.0)),
            clips_ok=int(d.get("clips_ok", 0)), clips_total=int(d.get("clips_total", 0)),
        )


def aggregate(clips: list[dict]) -> dict[str, float]:
    """Corpus-level WER/CER (total edits ÷ total length) plus a duration-weighted RTF.

    Averaging per-clip rates would over-weight short clips; jiwer is re-run over the
    concatenated normalized text so the reported rate is the real corpus rate.

    RTF is computed only over clips that carry a timing. A skipped clip contributes
    its text to the score but nothing to the speed, which is why the two are summed
    separately here.
    """
    import jiwer

    ok = [c for c in clips if c.get("success")]
    if not ok:
        return {"wer": 0.0, "cer": 0.0, "rtf": 0.0, "audio_duration_s": 0.0}
    refs = " ".join(c["ref_norm"] for c in ok)
    hyps = " ".join(c["hyp_norm"] for c in ok)
    timed = [c for c in ok if c.get("audio_duration_s") and "wall_s" in c]
    total_audio = sum(float(c["audio_duration_s"]) for c in timed)
    total_wall = sum(float(c["wall_s"]) for c in timed)
    return {
        "wer": float(jiwer.wer(" ".join(c.get("ref_wer_norm", c["ref_norm"]) for c in ok),
            " ".join(c.get("hyp_wer_norm", c["hyp_norm"]) for c in ok))) if refs else 0.0,
        "cer": float(jiwer.cer([c["ref_norm"] for c in ok], [c["hyp_norm"] for c in ok]))
            if any("wer_tokenization" in c for c in ok) else (float(jiwer.cer(refs, hyps)) if refs else 0.0),
        "rtf": (total_wall / total_audio) if total_audio else 0.0,
        "audio_duration_s": total_audio,
    }


# ---------------------------------------------------------------------------
# Selection helpers
# ---------------------------------------------------------------------------


def _split_csv_arg(value: Any) -> list[str]:
    raw_values = value if isinstance(value, (list, tuple)) else [value]
    return [
        part.strip()
        for raw_value in raw_values
        for part in str(raw_value).split(",")
        if part.strip()
    ]


def resolve_models(models_arg: str | list[str] | tuple[str, ...]) -> list[dict]:
    """Filter MODELS by name list or return all. Silently excludes BLOCKED_ENGINES."""
    names = _split_csv_arg(models_arg)
    if names == ["all"]:
        return [m for m in MODELS if m["name"] not in BLOCKED_ENGINES]
    name_set = set(names) - set(BLOCKED_ENGINES)
    selected = [m for m in MODELS if m["name"] in name_set]
    missing = name_set - {m["name"] for m in selected}
    if missing:
        console.print(f"[yellow]Warning: unknown model(s): {', '.join(sorted(missing))}[/yellow]")
    return selected


def resolve_langs(langs_arg: str | list[str] | tuple[str, ...]) -> list[str]:
    codes = _split_csv_arg(langs_arg)
    return list(ALL_LANGS) if codes == ["all"] else codes


# ---------------------------------------------------------------------------
# Summary tables
# ---------------------------------------------------------------------------


def print_summary(results: list[Result]) -> None:
    table = Table(title="ASR results (lower WER/CER is better)")
    table.add_column("Model", style="cyan")
    table.add_column("Lang")
    table.add_column("Clips", justify="right")
    table.add_column("WER %", justify="right")
    table.add_column("CER %", justify="right")
    table.add_column("RTF", justify="right")
    table.add_column("Status")
    for r in sorted(results, key=lambda x: (x.lang, x.wer if x.success else 9.9)):
        if r.success:
            status = "[green]ok[/green]"
            wer, cer, rtf = f"{r.wer * 100:.1f}", f"{r.cer * 100:.1f}", f"{r.rtf:.3f}"
        else:
            status = f"[red]{(r.error.splitlines() or ['failed'])[-1][:40]}[/red]"
            wer = cer = rtf = "—"
        table.add_row(
            r.model, r.lang, f"{r.clips_ok}/{r.clips_total}", wer, cer, rtf, status
        )
    console.print(table)


def print_benchmarks(results: list[Result], batch_elapsed: float) -> None:
    ok = [r for r in results if r.success]
    console.print(
        f"\n[bold]Batch:[/bold] {len(ok)}/{len(results)} model×lang runs succeeded "
        f"in {batch_elapsed:.0f}s"
    )
    if ok:
        best = min(ok, key=lambda r: r.wer)
        fastest = min(ok, key=lambda r: r.rtf)
        console.print(f"[bold]Lowest WER:[/bold] {best.model}/{best.lang} {best.wer * 100:.1f}%")
        console.print(f"[bold]Fastest:[/bold]    {fastest.model}/{fastest.lang} RTF {fastest.rtf:.3f}")


# ---------------------------------------------------------------------------
# JSON report persistence
# ---------------------------------------------------------------------------


def _result_key(model: str, lang: str) -> str:
    return f"{model}/{lang}"


def load_report_json(outdir: Path) -> dict:
    """Load prior report.json from *outdir*; return {} on missing/malformed."""
    path = outdir / "report.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError) as exc:
        console.print(f"[yellow]⚠ report.json malformed ({exc}) — starting fresh[/yellow]")
        return {}


def _seed_blocked_records(merged: dict[str, dict], now: str) -> None:
    """Keep every blocked engine in the report with its specific reason."""
    for name, reason in BLOCKED_ENGINES.items():
        for lang in ALL_LANGS:
            rec = merged.get(_result_key(name, lang), {})
            rec.update({
                "model": name, "lang": lang, "blocked": True,
                "status": BLOCKED_STATUS, "reason": reason, "updated_at": now,
            })
            rec.setdefault("success", False)
            merged[_result_key(name, lang)] = rec


def _build_report_data(
    results: list[Result], prior_data: dict, *,
    batch_elapsed: float | None = None, completed: int | None = None,
    total: int | None = None, total_models: int = 0, total_available: int = 0,
    selected_langs: list[str] | None = None, models_filter: str = "all",
) -> dict:
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    merged: dict[str, dict] = {}
    prior_results = prior_data.get("results", {})
    if isinstance(prior_results, dict):
        merged.update(prior_results)
    for r in results:
        merged[_result_key(r.model, r.lang)] = r.to_dict()
    _seed_blocked_records(merged, now)

    active = {k: v for k, v in merged.items() if not v.get("blocked")}
    successes = sum(1 for v in active.values() if v.get("success"))
    data: dict[str, Any] = {
        "generated": now,
        "meta": {
            "models_filter": models_filter,
            "selected_langs": selected_langs or [],
            "total_models": total_models,
            "total_available": total_available,
            "eval_dir": str(EVAL_DIR),
        },
        "summary": {
            "total_runs": len(active),
            "successes": successes,
            "failures": len(active) - successes,
            "batch_elapsed_s": batch_elapsed if batch_elapsed is not None else 0.0,
        },
        "results": merged,
    }
    if completed is not None and total is not None:
        data["summary"]["progress"] = f"{completed}/{total}"
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
        Path(tmp).unlink(missing_ok=True)
        raise
    return report_path


def save_partial_report(results, prior_data, outdir, completed, total, **kwargs) -> None:
    _write_json(outdir, _build_report_data(
        results, prior_data, completed=completed, total=total, **kwargs))


def save_final_report(results, prior_data, outdir, batch_elapsed, **kwargs):
    data = _build_report_data(results, prior_data, batch_elapsed=batch_elapsed, **kwargs)
    return _write_json(outdir, data), data


def save_dashboard(outdir: Path) -> Path | None:
    """Rebuild the reference-vs-hypothesis browser at ``out/index.html``."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "build_dashboard", ROOT / "tools/build_dashboard.py")
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build(outdir)


def save_report_html(data: dict, outdir: Path) -> Path:
    """Generate a sortable HTML report from the report data dict."""
    esc = html_mod.escape
    summary = data.get("summary", {})
    results = data.get("results", {})
    rows: list[str] = []
    for key in sorted(results):
        r = results[key]
        if r.get("blocked"):
            rows.append(
                f"<tr class=blocked><td>{esc(str(r.get('model', '')))}</td>"
                f"<td>{esc(str(r.get('lang', '')))}</td><td colspan=6>"
                f"blocked — {esc(str(r.get('reason', '')))}</td></tr>"
            )
            continue
        ok = r.get("success")
        cells = (
            f"<td data-v={r.get('wer', 0):.6f}>{r.get('wer', 0) * 100:.1f}</td>"
            f"<td data-v={r.get('cer', 0):.6f}>{r.get('cer', 0) * 100:.1f}</td>"
            f"<td data-v={r.get('rtf', 0):.6f}>{r.get('rtf', 0):.3f}</td>"
            if ok else "<td>—</td><td>—</td><td>—</td>"
        )
        rows.append(
            f"<tr class={'ok' if ok else 'fail'}>"
            f"<td>{esc(str(r.get('model', '')))}</td><td>{esc(str(r.get('lang', '')))}</td>"
            f"<td>{esc(str(r.get('stack', '')))}</td><td>{esc(str(r.get('quant', '')))}</td>"
            f"<td>{r.get('clips_ok', 0)}/{r.get('clips_total', 0)}</td>{cells}"
            f"<td>{esc(str(r.get('error', ''))[:120])}</td></tr>"
        )
    html = f"""<!doctype html><meta charset=utf-8><title>ASR report</title>
<style>body{{font:14px/1.5 system-ui;margin:2rem;background:#fafafa;color:#111}}
table{{border-collapse:collapse;width:100%}}th,td{{padding:.4rem .6rem;border-bottom:1px solid #ddd;text-align:left}}
th{{cursor:pointer;background:#eee;position:sticky;top:0}}tr.fail{{background:#fff2f2}}
tr.blocked{{background:#f4f4f4;color:#777;font-style:italic}}</style>
<h1>ASR report</h1>
<p>{summary.get('successes', 0)}/{summary.get('total_runs', 0)} runs succeeded ·
generated {esc(str(data.get('generated', '')))} ·
<a href="index.html">reference-vs-hypothesis dashboard</a></p>
<table id=t><thead><tr><th>Model<th>Lang<th>Stack<th>Quant<th>Clips<th>WER %<th>CER %<th>RTF<th>Error</tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<script>
document.querySelectorAll('#t th').forEach((th,i)=>th.onclick=()=>{{
 const tb=document.querySelector('#t tbody'),rows=[...tb.rows],dir=th.dataset.d=th.dataset.d==='1'?'':'1';
 const val=r=>{{const c=r.cells[i];if(!c)return'';const d=c.dataset.v;return d!==undefined?parseFloat(d):c.textContent}};
 rows.sort((a,b)=>{{const x=val(a),y=val(b);const c=(typeof x==='number'&&typeof y==='number')?x-y:String(x).localeCompare(String(y));return dir?-c:c}});
 rows.forEach(r=>tb.appendChild(r));}});
</script>"""
    path = outdir / "report.html"
    path.write_text(html, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(
    eval_dir: str,
    outdir: str,
    models: str = "all",
    langs: str = "all",
    clips: int = 0,
    timeout: int = 300,
    verbose: bool = False,
    force: bool = False,
    prep: bool = True,
    prep_only: bool = False,
    prep_timeout: int = 3600,
) -> None:
    """Prep, transcribe and score every requested model over the eval set.

    Args:
        eval_dir: Caller-supplied WAV/LAB evaluation corpus
        outdir: Caller-supplied output root directory
        models: Comma-separated model names, or "all"
        langs: Comma-separated lang codes (en, pl), or "all"
        clips: Max eval clips per language; 0 uses every matched pair
        timeout: Per-clip timeout in seconds (models may set a longer default)
        verbose: Show subprocess output
        force: Re-transcribe even when a transcript already exists
        prep: Download/cache every selected model first (ON by default)
        prep_only: Only run prep, then exit
        prep_timeout: Per-model timeout for the download step
    """
    global EVAL_DIR
    EVAL_DIR = Path(eval_dir).expanduser().resolve()
    if not EVAL_DIR.is_dir():
        console.print(f"[red]Error: eval set not found: {EVAL_DIR}[/red]")
        sys.exit(1)

    selected_models = resolve_models(models)
    selected_langs = resolve_langs(langs)
    clip_sets = {lang: eval_clips(lang, clips) for lang in selected_langs}
    for lang, pairs in clip_sets.items():
        if not pairs:
            console.print(f"[yellow]Warning: no matched .wav/.lab pairs for '{lang}'[/yellow]")
        else:
            console.print(f"[dim]{lang}: {len(pairs)} eval clip(s)[/dim]")

    out_root = Path(outdir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    if prep or prep_only:
        console.print(f"\n[bold]Prep[/bold] — caching {len(selected_models)} model(s)\n")
        prep_ok = 0
        for model in selected_models:
            ok, dur, err = prep_model(model, prep_timeout, verbose)
            if ok:
                prep_ok += 1
                console.print(f"  [green]✓[/green] prep {model['name']} ({dur:.0f}s)")
            else:
                short = (err.splitlines() or ["unknown error"])[-1][:100]
                console.print(f"  [red]✗[/red] prep {model['name']} failed ({dur:.0f}s): {short}")
        console.print(f"\n[bold]Prep done:[/bold] {prep_ok}/{len(selected_models)} ready")
        if prep_only:
            return

    prior_data = load_report_json(out_root)
    prior_results = prior_data.get("results", {})

    def prior_clip(model_name: str, lang: str, stem: str) -> dict[str, Any]:
        """Timings recorded for this clip by an earlier run, or {} if none."""
        rec = prior_results.get(_result_key(model_name, lang), {})
        for clip in rec.get("clips", []) or []:
            if clip.get("clip") == stem and clip.get("success"):
                return {
                    k: clip[k]
                    for k in ("wall_s", "audio_duration_s", "rtf", "engine_rtf")
                    if k in clip
                }
        return {}

    report_kw: dict[str, Any] = {
        "total_models": len(selected_models),
        "total_available": len([m for m in MODELS if m["name"] not in BLOCKED_ENGINES]),
        "selected_langs": selected_langs,
        "models_filter": models,
    }

    work = [
        (m, lang) for m in selected_models for lang in selected_langs
        if lang in m["langs"] and clip_sets.get(lang)
    ]
    if not work:
        report_path, report_data = save_final_report([], prior_data, out_root, 0.0, **report_kw)
        save_report_html(report_data, out_root)
        save_dashboard(out_root)
        console.print(f"\n[bold]Report saved:[/bold] {report_path}")
        return

    console.print(f"\n[bold]ASR Batch[/bold] — {len(work)} model×lang run(s)\n")
    results: list[Result] = []
    batch_start = time.monotonic()

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        BarColumn(), TaskProgressColumn(), TimeElapsedColumn(),
        console=console, transient=False,
    ) as progress:
        task = progress.add_task("Transcribing...", total=len(work))
        for model, lang in work:
            name = model["name"]
            progress.update(task, description=f"[cyan]{name}[/cyan]/[yellow]{lang}[/yellow]")
            stack, quant = model_stack(model), model_quant(model)
            model_timeout = int(model.get("timeout", timeout))
            clip_records: list[dict] = []
            first_error = ""

            for wav, reference in clip_sets[lang]:
                out_path = out_root / output_filename(model, lang, wav.stem)
                request = _identity["request_identity"](
                    model, ROOT / model["script"], model["args_fn"](str(wav), str(out_path), lang),
                    [wav, wav.with_suffix(".lab")], {"lang": lang, "reference": reference})
                reused = not force and _identity["reusable_output"](out_path, request)
                if reused:
                    ok, wall, err, payload = True, 0.0, "", {}
                elif not force and (out_path.exists() or out_path.is_symlink()):
                    ok, wall, err, payload = False, 0.0, "Unbound or changed transcript; use --force to regenerate", {}
                else:
                    ok, wall, err, payload = run_model(
                        model, wav, out_path, lang, model_timeout, verbose, force)
                record: dict[str, Any] = {
                    "clip": wav.stem,
                    "audio": str(wav),
                    "transcript_path": str(out_path),
                    "reference": reference,
                    "success": ok,
                    "subprocess_s": round(wall, 4),
                    "error": "" if ok else err,
                }
                # Three timings, deliberately distinct:
                #   engine_rtf   — what the model reports for inference alone
                #   rtf          — the wrapper's wall time (load + inference) / audio
                #   subprocess_s — adds `uv run` and interpreter startup on top
                # The report shows `rtf`: model loading is real cost, but uv's
                # bootstrap is an artifact of subprocess isolation, not of the model.
                if payload:
                    wrapper_wall = float(payload.get("wall_s", wall))
                    audio_s = float(payload.get("audio_duration_s", 0.0))
                    record["wall_s"] = round(wrapper_wall, 4)
                    record["audio_duration_s"] = round(audio_s, 4)
                    record["engine_rtf"] = round(float(payload.get("rtf", 0.0)), 6)
                    if audio_s:
                        record["rtf"] = round(wrapper_wall / audio_s, 6)
                elif reused:
                    # The wrapper skipped an existing transcript, so it measured
                    # nothing. Carry the previous run's timings forward rather than
                    # overwriting real numbers with zeros — and if there is no prior
                    # measurement, leave the fields absent, because "not measured"
                    # and "instant" must not look alike in the report.
                    record.update(prior_clip(name, lang, wav.stem))
                if ok:
                    hypothesis = payload.get("text") or (
                        out_path.read_text(encoding="utf-8").strip()
                        if out_path.exists() else "")
                    record["hypothesis"] = hypothesis
                    record.update(score(reference, hypothesis, lang=lang))
                    if not hypothesis:
                        record["success"] = ok = False
                        record["error"] = "empty transcript"
                    elif not out_path.is_file() or out_path.stat().st_size == 0:
                        record["success"] = ok = False
                        record["error"] = "missing transcript output"
                    elif record["ref_norm"] and not record["hyp_norm"]:
                        # Non-empty output that survives normalization as nothing —
                        # SenseVoice returns a bare "." on some Polish clips. It still
                        # scores (100% WER is the truthful measurement), but it is
                        # flagged so the dashboard does not show a blank cell that
                        # reads like a harness bug.
                        record["degenerate"] = True
                if record["success"]:
                    _identity["record_output"](out_path, request)
                if not record["success"] and not first_error:
                    first_error = record["error"]
                clip_records.append(record)

            ok_clips = [c for c in clip_records if c["success"]]
            agg = aggregate(clip_records)
            success = bool(ok_clips)
            results.append(Result(
                model=name, lang=lang, success=success,
                # A clip that failed before the wrapper returned timing has no
                # wall_s at all. Summing it blindly raises KeyError here and buries
                # the model's real error under a harness traceback — which is how
                # two MLX wrappers reported themselves as crashes rather than as
                # whatever actually went wrong.
                duration=sum(c.get("wall_s", 0.0) for c in clip_records),
                error="" if success else first_error, stack=stack, quant=quant,
                clips=clip_records, clips_ok=len(ok_clips), clips_total=len(clip_records),
                **agg,
            ))
            if success:
                progress.console.print(
                    f"  [green]✓[/green] {name}/{lang} — WER {agg['wer'] * 100:.1f}% "
                    f"CER {agg['cer'] * 100:.1f}% RTF {agg['rtf']:.3f} "
                    f"({len(ok_clips)}/{len(clip_records)} clips)"
                )
            else:
                progress.console.print(
                    f"  [red]✗[/red] {name}/{lang} failed: "
                    f"{(first_error.splitlines() or ['unknown'])[-1][:80]}"
                )
            save_partial_report(results, prior_data, out_root, len(results), len(work), **report_kw)
            progress.advance(task)

    batch_elapsed = time.monotonic() - batch_start
    print_summary(results)
    print_benchmarks(results, batch_elapsed)
    report_path, report_data = save_final_report(
        results, prior_data, out_root, batch_elapsed, **report_kw)
    html_path = save_report_html(report_data, out_root)
    dash_path = save_dashboard(out_root)
    console.print(f"\n[bold]Report saved:[/bold] {report_path}")
    console.print(f"[bold]HTML report:[/bold]  {html_path}")
    console.print(f"[bold]Dashboard:[/bold]    {dash_path}")


if __name__ == "__main__":
    fire.Fire(main)
