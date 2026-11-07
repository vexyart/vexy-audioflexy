#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = ["platformdirs>=4", "fire", "rich", "numpy", "soundfile"]
# ///
# this_file: src/vexy_audioflexy/tools/fx/aud_all.py
"""Batch audio processor — preps then runs every registered operation.

Each model is invoked via subprocess (its `prep` then `run` Fire subcommand) so
dependency environments never conflict, exactly as the sibling tts project does
for TTS. This project is the audio-in/audio-out counterpart: denoising,
super-resolution, voice conversion, singing voice conversion, speech editing and
speech-to-speech, across three inference stacks.

| stack      | wrapper                     | what runs it                     |
|------------|-----------------------------|----------------------------------|
| `audiocpp` | `tools/aud-audiocpp.py`     | the ggml `audiocpp_cli` binary   |
| `onnx`     | `tools/aud-audiosronnx.py`  | onnxruntime on the CPU           |
| `mlx`      | `tools/aud-mlxaudio.py`     | MLX on the Apple Silicon GPU     |

Every output is then scored by `tools/aud-metrics.py`, and so is the clip that
went in. For an enhancement model that pairing is the whole verification: a
denoiser that returns its input untouched is neither silent nor working, and only
the delta tells the two apart.

Default behavior: **prep every model (download weights), then run every model.**

    ./aud_all.py                                   # prep ALL, then run ALL
    ./aud_all.py --models meanvc2-f32,audiosr-basic-f32
    ./aud_all.py --prep=False                      # skip prep, lazy-download
    ./aud_all.py --prep-only                       # download everything, no inference
    ./aud_all.py --force                           # redo even if outputs exist
    ./aud_all.py --outdir /tmp/aud_out             # custom output dir
    ./aud_all.py --verbose                         # show subprocess output
    ./aud_all.py --timeout 1800                    # longer per-model timeout

Outputs are flat and self-describing:

    {op}__{model}__{quant}__{stack}__{source}.wav

where *op* is the operation performed (dn, sr, vc, svc, s2s, edit), *source* names the
input fixture. The run is resumable: existing outputs are skipped and report.json
is rewritten after every job.
"""


import os
from pathlib import Path
from platformdirs import user_cache_path, user_data_path
RUNTIME_ROOT = Path(os.environ.get("VAFX_AUDIO_RUNTIME_ROOT", user_data_path("vexy-audioflexy") / "audio" )).expanduser()
CACHE_ROOT = Path(os.environ.get("VAFX_AUDIO_CACHE_ROOT", user_cache_path("vexy-audioflexy") / "audio")).expanduser()
import datetime
import hashlib
import json
import os
import runpy
import subprocess
import sys
import tempfile
import time
from pathlib import Path
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

# Speech in this project measures RMS 0.05-0.27. PersonaPlex writes a correctly
# formed WAV at 0.0004 in both its precisions — inaudible, but not zero, so a
# `rms > 0` test called it non-silent. Anything under this floor is silence.
SILENCE_RMS = 0.001


ROOT = Path(__file__).parent
# ---------------------------------------------------------------------------
# Ignored engines — kept in report.json but never executed
# ---------------------------------------------------------------------------

IGNORED_ENGINES: frozenset[str] = frozenset()
IGNORED_REASON = "engine disabled in aud_all.py (see WORK.md)"

# ---------------------------------------------------------------------------
# Fixtures — the audio that goes IN
# ---------------------------------------------------------------------------
# "speech"   : source speech, a speaker, English, 24 kHz mono (from tts dataset)
# "lowres" : the same clip resampled to 16 kHz — the degraded input AudioSR restores
# "spkb"   : a DIFFERENT speaker, the target timbre for voice conversion
# ---------------------------------------------------------------------------

SOURCES: dict[str, Path] = {}
TARGETS: dict[str, Path] = {}
# The clean original behind each degraded fixture. Intrusive metrics (STOI,
# SI-SDR, LSD, MCD) compare an output against this; a source with no undegraded
# counterpart — or an operation whose job is to differ from its input, like voice
# conversion — is scored with the no-reference MOS predictors only.
REFERENCES: dict[str, Path] = {}
# Transcript of the "speech" source clip, for routes that condition on text.
SOURCE_TEXT = ""
TARGET_TEXT = ""

PERSONA_PROMPT = ""

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
# Every entry drives one subprocess call to tools/aud-audiocpp.py `run`.
#   op      : operation label baked into the output filename
#   source  : key into SOURCES  -> --audio
#   target  : key into TARGETS  -> --voice-ref (empty when the route needs none)
#   task    : audio.cpp --task, confirmed against `audiocpp_cli --list-loaders`
#   route   : audio.cpp --task-route (empty = the engine's documented default)
#   extra_fn: () -> list[str] of additional wrapper arguments
#   timeout : per-model inference timeout in seconds
# ---------------------------------------------------------------------------

# Upstream projects reviewed on 2026-09-03 (issues/2026-09-02-repo-triage.md) and
# deliberately not wrapped here, with the reason.
OUT_OF_SCOPE: dict[str, str] = {
    "whisperlivekit": "Streaming ASR server: audio in, text out. That is ../asr's job.",
    "auto-subs": "DaVinci Resolve / Premiere plugin, TypeScript. Not an audio-to-audio engine.",
    "openless": "Rust desktop dictation app. Speech to text, and an application at that.",
    "dograh": "Voice-agent platform. Orchestrates STT/TTS services rather than being one.",
    "voicestudio": "End-user application over other people's models.",
    "mockingbird": "Chinese SV2TTS — text to speech, and no en/pl path. Would belong to ../tts.",
    "espnet": "Research toolkit. Its TTS recipes are wrapped in ../tts; it has no enhancement model this harness scores.",
}

MODELS: list[dict[str, Any]] = [
    # --- Audio enhancement ------------------------------------------------
    {
        "name": "audiosr-basic-f32",
        "script": "tools/aud-audiocpp.py",
        "package": "audiosr_basic_f32",
        "family": "audiosr",
        "repo": "audio-cpp/audio.cpp-gguf",
        "op": "sr",
        "task": "s2s",
        "route": "",
        "source": "lowres",
        "target": "",
        "quant": "f32",
        "timeout": 3600,
        "extra_fn": lambda: [
            "--request_options",
            "num_inference_steps=50;guidance_scale=3.5;ddim_eta=1.0;seed=42",
        ],
    },
    # --- Voice conversion -------------------------------------------------
    {
        "name": "meanvc2-f32",
        "script": "tools/aud-audiocpp.py",
        "package": "meanvc2_120ms_40ms_f32",
        "family": "meanvc2",
        "repo": "audio-cpp/audio.cpp-gguf",
        "op": "vc",
        "task": "vc",
        "route": "",
        "source": "speech",
        "target": "spkb",
        "quant": "f32",
        "timeout": 1800,
    },
    {
        "name": "meanvc2-q4_k",
        "script": "tools/aud-audiocpp.py",
        "package": "meanvc2_120ms_40ms_q4_k",
        "family": "meanvc2",
        "repo": "audio-cpp/audio.cpp-gguf",
        "op": "vc",
        "task": "vc",
        "route": "",
        "source": "speech",
        "target": "spkb",
        "quant": "q4_k",
        "timeout": 1800,
    },
    {
        "name": "seedvc-v2-vc-q8_0",
        "script": "tools/aud-audiocpp.py",
        "package": "seed_vc_mlx_q8_0",
        "family": "seed_vc",
        "repo": "audio-cpp/audio.cpp-gguf",
        "op": "vc",
        "task": "vc",
        "route": "v2_vc",
        "source": "speech",
        "target": "spkb",
        "quant": "q8_0",
        "timeout": 2400,
    },
    {
        "name": "seedvc-v1-xlsr-hift-q8_0",
        "script": "tools/aud-audiocpp.py",
        "package": "seed_vc_mlx_q8_0",
        "family": "seed_vc",
        "repo": "audio-cpp/audio.cpp-gguf",
        "op": "vc",
        "task": "vc",
        "route": "v1_xlsr_hift_vc",
        "source": "speech",
        "target": "spkb",
        "quant": "q8_0",
        "timeout": 2400,
    },
    {
        "name": "seedvc-v1-svc-q8_0",
        "script": "tools/aud-audiocpp.py",
        "package": "seed_vc_mlx_q8_0",
        "family": "seed_vc",
        "repo": "audio-cpp/audio.cpp-gguf",
        "op": "svc",
        "task": "svc",
        "route": "v1_svc",
        "source": "speech",
        "target": "spkb",
        "quant": "q8_0",
        "timeout": 2400,
    },
    {
        "name": "vevo2-style-preserved-vc-q8_0",
        "script": "tools/aud-audiocpp.py",
        "package": "vevo2_q8_0",
        "family": "vevo2",
        "repo": "audio-cpp/audio.cpp-gguf",
        "op": "vc",
        "task": "vc",
        "route": "style_preserved_vc",
        "source": "speech",
        "target": "spkb",
        "quant": "q8_0",
        "timeout": 3600,
    },
    {
        "name": "vevo2-editing-q8_0",
        "script": "tools/aud-audiocpp.py",
        "package": "vevo2_q8_0",
        "family": "vevo2",
        "repo": "audio-cpp/audio.cpp-gguf",
        "op": "edit",
        "task": "s2s",
        "route": "editing",
        "source": "speech",
        "target": "spkb",
        "quant": "q8_0",
        "timeout": 3600,
        # The source clip's own transcript is the style reference; the target text
        # is the edited sentence the model has to speak instead.
        "extra_fn": lambda: [
            "--target_text",
            TARGET_TEXT,
            "--style_ref_text",
            SOURCE_TEXT,
        ],
    },
    # --- Speech to speech ---------------------------------------------------
    {
        "name": "personaplex-7b-q4_k",
        "script": "tools/aud-audiocpp.py",
        "package": "personaplex_7b_v1_q4_k",
        "family": "personaplex",
        "repo": "audio-cpp/audio.cpp-gguf",
        "op": "s2s",
        "task": "s2s",
        "route": "",
        "source": "speech",
        "target": "",
        "quant": "q4_k",
        "timeout": 3600,
        "extra_fn": lambda: ["--text", PERSONA_PROMPT, "--voice_id", "NATF2"],
    },
    {
        "name": "personaplex-7b-q8_0",
        "script": "tools/aud-audiocpp.py",
        "package": "personaplex_7b_v1_q8_0",
        "family": "personaplex",
        "repo": "audio-cpp/audio.cpp-gguf",
        "op": "s2s",
        "task": "s2s",
        "route": "",
        "source": "speech",
        "target": "",
        "quant": "q8_0",
        "timeout": 3600,
        "extra_fn": lambda: ["--text", PERSONA_PROMPT, "--voice_id", "NATM1"],
    },
]

# ---------------------------------------------------------------------------
# Audio improvement — one wrapper per stack, many engines each
# ---------------------------------------------------------------------------
# These two stacks hold 22 engines between them. Writing 22 near-identical dicts
# by hand would bury the two facts that actually differ per engine (what it does
# and how long it may take), so the varying columns are tabulated and the
# invariant ones filled in below. Everything downstream still sees plain MODELS
# entries.
#
#   op "dn" — denoising / restoration, fed the noisy fixture
#   op "sr" — bandwidth extension, fed the 16 kHz fixture
# ---------------------------------------------------------------------------

# alias -> (op, timeout seconds). Names match `aud-audiosronnx.py packages`.
AUDIOSRONNX_ENGINES: dict[str, tuple[str, int]] = {
    "dpdfnet": ("dn", 900),
    "mossformer2": ("dn", 1800),
    "frcrn": ("dn", 900),
    "mpsenet": ("dn", 900),
    "gtcrn": ("dn", 900),
    "cmgan": ("dn", 900),
    "metadenoiser": ("dn", 900),
    "mossformergan": ("dn", 1200),
    "voicefixer": ("dn", 1800),
    "deepfilternet": ("dn", 900),
    "lavasr": ("sr", 900),
    "novasr": ("sr", 900),
    "apbwe": ("sr", 1200),
    "hifiganbwe": ("sr", 1200),
    # Generative restorers: they resynthesise speech rather than filter it, and
    # cost far more per second of audio than the extenders above.
    "flowhigh": ("sr", 2400),
    "sidon": ("sr", 3600),
    "callenhancer": ("sr", 2400),
}

# package -> (weights repo, quant). Names match `aud-mlxaudio.py packages`.
MLXAUDIO_PACKAGES: dict[str, tuple[str, str]] = {
    "deepfilternet3": ("mlx-community/DeepFilterNet-mlx", "f32"),
    "deepfilternet2": ("mlx-community/DeepFilterNet-mlx", "f32"),
    "deepfilternet1": ("mlx-community/DeepFilterNet-mlx", "f32"),
    "deepfilternet3-iky1e": ("iky1e/DeepFilterNet3-MLX", "f32"),
    "mossformer2-se": ("starkdmi/MossFormer2-SE-fp16", "fp16"),
}

MODELS += [
    {
        "name": f"asronnx-{alias}",
        "script": "tools/aud-audiosronnx.py",
        "package": alias,
        "family": "",  # the engine alias is the whole address; there is no family flag
        "repo": f"TigreGotico/audiosronnx-{alias}",
        "op": op,
        "task": "denoise" if op == "dn" else "sr",
        "route": "",
        "source": "noisy" if op == "dn" else "lowres",
        "target": "",
        "stack": "onnx",
        "quant": "f32",
        "timeout": timeout,
    }
    for alias, (op, timeout) in AUDIOSRONNX_ENGINES.items()
] + [
    {
        "name": f"mlxaudio-{package}",
        "script": "tools/aud-mlxaudio.py",
        "package": package,
        "family": "",
        "repo": repo,
        "op": "dn",
        "task": "denoise",
        "route": "",
        "source": "noisy",
        "target": "",
        "stack": "mlx",
        "quant": quant,
        "timeout": 1800,
    }
    for package, (repo, quant) in MLXAUDIO_PACKAGES.items()
] + [
    {
        # Its own wrapper: the weights are on HuggingFace but the inference code
        # is a directory of scripts, not an importable package.
        "name": "mossformer2-sr-48k-mlx",
        "script": "tools/aud-mossformersr.py",
        "package": "mossformer2-sr-48k",
        "family": "",
        "repo": "starkdmi/MossFormer2_SR_48K_MLX",
        "op": "sr",
        "task": "sr",
        "route": "",
        "source": "lowres",
        "target": "",
        "stack": "mlx",
        "quant": "f32",
        "timeout": 2400,
    },
    {
        # The MLX conversion of nvidia/RE-USE, whose original needs CUDA kernels.
        "name": "re-use-semamba-mlx",
        "script": "tools/aud-mlxspeech.py",
        "package": "re-use-semamba",
        "family": "",
        "repo": "appautomaton/re-use-semamba-mlx",
        "op": "dn",
        "task": "denoise",
        "route": "",
        "source": "noisy",
        "target": "",
        "stack": "mlx",
        "quant": "f32",
        "timeout": 1800,
    },
    {
        # Re-speaks the clip with one attribute changed, so it needs the source
        # transcript the way VeVo2's editing route does.
        "name": "step-audio-editx-mlx",
        "script": "tools/aud-mlxspeech.py",
        "package": "step-audio-editx",
        "family": "",
        "repo": "appautomaton/step-audio-editx-8bit-mlx",
        "op": "edit",
        "task": "edit",
        "route": "",
        "source": "speech",
        "target": "",
        "stack": "mlx",
        "quant": "int8",
        "timeout": 3600,
        "extra_fn": lambda: [
            "--text", SOURCE_TEXT,
            "--edit_type", "emotion",
            "--edit_info", "happy",
        ],
    },
    {
        # Inference code ships inside the weights repo; see tools/aud-resemble.py.
        "name": "resemble-enhance-mlx",
        "script": "tools/aud-resemble.py",
        "package": "resemble-enhance",
        "family": "",
        "repo": "multimodalart/resemble-enhance-mlx",
        "op": "dn",
        "task": "denoise",
        "route": "",
        "source": "noisy",
        "target": "",
        "stack": "mlx",
        "quant": "f32",
        "timeout": 2400,
    },
    # --- sherpa-onnx speech enhancement (k2-fsa exports, C++ front-end) -----
    # gtcrn and dpdfnet also exist under `asronnx-*`; those are the
    # TigreGotico Python implementations. Both are registered so the scores
    # compare the two, not so one stands in for the other.
    {
        "name": "sherpaonnx-gtcrn",
        "script": "tools/aud-sherpaonnx.py",
        "package": "gtcrn",
        "family": "",  # the package name is the whole address
        "repo": "Xiaobin-Rong/gtcrn (sherpa-onnx export)",
        "op": "dn",
        "task": "denoise",
        "route": "",
        "source": "noisy",
        "target": "",
        "stack": "onnx",
        "quant": "f32",
        "timeout": 600,
    },
    {
        "name": "sherpaonnx-dpdfnet-baseline",
        "script": "tools/aud-sherpaonnx.py",
        "package": "dpdfnet-baseline",
        "family": "",  # the package name is the whole address
        "repo": "k2-fsa dpdfnet baseline",
        "op": "dn",
        "task": "denoise",
        "route": "",
        "source": "noisy",
        "target": "",
        "stack": "onnx",
        "quant": "f32",
        "timeout": 600,
    },
    {
        "name": "sherpaonnx-dpdfnet2",
        "script": "tools/aud-sherpaonnx.py",
        "package": "dpdfnet2",
        "family": "",  # the package name is the whole address
        "repo": "k2-fsa dpdfnet2",
        "op": "dn",
        "task": "denoise",
        "route": "",
        "source": "noisy",
        "target": "",
        "stack": "onnx",
        "quant": "f32",
        "timeout": 600,
    },
    {
        "name": "sherpaonnx-dpdfnet4",
        "script": "tools/aud-sherpaonnx.py",
        "package": "dpdfnet4",
        "family": "",  # the package name is the whole address
        "repo": "k2-fsa dpdfnet4",
        "op": "dn",
        "task": "denoise",
        "route": "",
        "source": "noisy",
        "target": "",
        "stack": "onnx",
        "quant": "f32",
        "timeout": 900,
    },
    {
        "name": "sherpaonnx-dpdfnet8",
        "script": "tools/aud-sherpaonnx.py",
        "package": "dpdfnet8",
        "family": "",  # the package name is the whole address
        "repo": "k2-fsa dpdfnet8",
        "op": "dn",
        "task": "denoise",
        "route": "",
        "source": "noisy",
        "target": "",
        "stack": "onnx",
        "quant": "f32",
        "timeout": 900,
    },
    {
        "name": "sherpaonnx-dpdfnet2-48k-hr",
        "script": "tools/aud-sherpaonnx.py",
        "package": "dpdfnet2-48k-hr",
        "family": "",  # the package name is the whole address
        "repo": "k2-fsa dpdfnet2 48 kHz high-resolution",
        "op": "dn",
        "task": "denoise",
        "route": "",
        "source": "noisy",
        "target": "",
        "stack": "onnx",
        "quant": "f32",
        "timeout": 900,
    },
    # --- OpenVoice v2 tone-colour conversion (local checkout) ----------------
    {
        "name": "openvoice-v2-vc",
        "script": "tools/aud-openvoice.py",
        "package": "openvoice-v2",
        "family": "",
        "repo": "myshell-ai/OpenVoice (checkpoints_v2 converter)",
        "op": "vc",
        "task": "vc",
        "route": "",
        "source": "speech",
        "target": "spkb",
        "stack": "torch",
        "quant": "fp32",
        "timeout": 1800,
    },
    # --- Amphion FACodec (NaturalSpeech 3) — codec resynthesis and VC --------
    # `codec` is its own op: the job is to return the input, and what is scored is
    # what the round trip costs. Fed the clean source, not the noisy fixture.
    {
        "name": "facodec-resynth",
        "script": "tools/aud-facodec.py",
        "package": "facodec",
        "family": "",
        "repo": "open-mmlab/Amphion (amphion/naturalspeech3_facodec)",
        "op": "codec",
        "task": "codec",
        "route": "",
        "source": "speech",
        "target": "",
        "stack": "torch",
        "quant": "fp32",
        "timeout": 1800,
    },
    {
        "name": "facodec-vc",
        "script": "tools/aud-facodec.py",
        "package": "facodec-vc",
        "family": "",
        "repo": "open-mmlab/Amphion (naturalspeech3_facodec redecoder)",
        "op": "vc",
        "task": "vc",
        "route": "",
        "source": "speech",
        "target": "spkb",
        "stack": "torch",
        "quant": "fp32",
        "timeout": 1800,
    },
]

# ---------------------------------------------------------------------------
MODELS.append({
    "name": "s2scpp-localvqe-f32", "script": "tools/aud-localvqe.py",
    "package": "localvqe-v1.3-f32", "family": "", "repo": "Serveurperso/LocalVQE-GGUF",
    "op": "dn", "task": "denoise", "route": "", "source": "noisy", "target": "",
    "stack": "s2scpp", "quant": "f32", "timeout": 900,
})

MODELS.append({
    "name": "speechbrain-metricgan-plus", "script": "tools/aud-speechbrain.py",
    "package": "metricgan-plus-voicebank", "family": "", "repo": "speechbrain/metricgan-plus-voicebank",
    "op": "dn", "task": "denoise", "route": "", "source": "noisy", "target": "",
    "stack": "torch", "quant": "fp32", "timeout": 900,
})

MODELS.append({
    "name": "faraday-re-use-mlx", "script": "tools/aud-speechlens.py",
    "package": "faraday-re-use-mlx", "family": "", "repo": "faraday/re-use-mlx",
    "op": "dn", "task": "denoise", "route": "", "source": "noisy", "target": "",
    "stack": "mlx-swift", "quant": "fp32", "timeout": 900,
})

MODELS.append({
    "name": "agentable-gtcrn-simple-mlx", "script": "tools/aud-gtcrnmlx.py",
    "package": "agentable-gtcrn-simple-mlx", "family": "", "repo": "agentable/gtcrn-simple-mlx",
    "op": "dn", "task": "denoise", "route": "", "source": "noisy", "target": "",
    "stack": "mlx-ort-hybrid", "quant": "fp32", "timeout": 900,
})

for _package, _family, _repo, _op, _source in [
    ("reuse_f32", "reuse", "audio-cpp/RE-USE-GGUF", "dn", "noisy"),
    ("sidon_f32", "sidon", "audio-cpp/audio.cpp-gguf", "dn", "noisy"),
    ("universr_speech_orig", "universr", "audio-cpp/audio.cpp-gguf", "sr", "lowres"),
    ("universr_audio_orig", "universr", "audio-cpp/audio.cpp-gguf", "sr", "lowres"),
]:
    MODELS.append({
        "name": "issue201-" + _package.replace("_", "-"),
        "script": "tools/aud-audiocpp.py" if _family == "reuse" else "tools/aud-restoration.py", "package": _package, "family": _family,
        "repo": _repo, "op": _op, "task": "s2s", "route": "",
        "source": _source, "target": "", "stack": "audiocpp", "quant": "f32", "timeout": 1800,
        "env": {"AUDIOCPP_ROOT": os.environ.get("AUDIOCPP_ISSUE201_ROOT", str(RUNTIME_ROOT / "engines/audio.cpp"))},
    })
    if _family == "reuse":
        # The compiled runtime explicitly disables RE-USE on Metal because of
        # GGML kernel bugs; CPU is a supported route for these exact weights.
        MODELS[-1]["env"]["AUDIOCPP_BACKEND"] = "cpu"

for _quant in ("f16", "q8_0", "q6_k", "q4_k"):
    MODELS.append({
        "name": "cstr-sidon-" + _quant.replace("_", "-"), "script": "tools/aud-sidoncrisp.py",
        "package": _quant, "family": "sidon", "repo": "cstr/Sidon-GGUF",
        "op": "dn", "task": "s2s", "route": "", "source": "noisy", "target": "",
        "stack": "crispasr", "quant": _quant, "timeout": 900,
    })

for _variant in ("v13", "v12"):
    for _chunk in ("16ms", "256ms"):
        for _task in ("denoise", "aec"):
            MODELS.append({
                "name": f"fluid-localvqe-{_variant}-{_chunk}-{_task}", "script": "tools/aud-localvqe-coreml.py",
                "package": f"{_variant}_{_chunk}", "family": "localvqe", "repo": "FluidInference/localvqe-coreml",
                "op": "aec" if _task == "aec" else "dn", "task": _task, "route": "",
                "source": "aec-mic" if _task == "aec" else "noisy", "target": "aec-far" if _task == "aec" else "",
                "stack": "coreml", "quant": "fp32", "timeout": 900,
            })

MODELS.append({
    "name": "df3-coreml-fp16", "script": "tools/aud-deepfilternet-coreml.py",
    "package": "df3_fp16", "family": "deepfilternet3", "repo": "iky1e/DeepFilterNet3-Streaming-CoreML",
    "op": "dn", "task": "denoise", "route": "", "source": "noisy", "target": "",
    "stack": "coreml", "quant": "fp16", "timeout": 900,
})

# Model metadata — drives the self-describing filenames and the report
# ---------------------------------------------------------------------------


def model_stack(model: dict) -> str:
    """Inference stack recorded per model, including native CrispASR, ONNX and MLX."""
    return str(model.get("stack") or "audiocpp")


def model_quant(model: dict) -> str:
    """Quantization label, from the registry or the package's precision suffix."""
    if model.get("quant"):
        return str(model["quant"])
    pkg = model["package"].lower()
    for tok in ("q8_0", "q4_k", "f32", "f16", "orig"):
        if pkg.endswith(tok):
            return tok
    return "na"


def model_op(model: dict) -> str:
    """Operation: sr (super-resolution) | vc | svc | s2s | edit."""
    return str(model.get("op") or model.get("task") or "vc")


def output_filename(model: dict, ext: str = "wav") -> str:
    """Flat self-describing name: ``{op}__{model}__{quant}__{stack}__{source}.{ext}``."""
    return (
        f"{model_op(model)}__{model['name']}__{model_quant(model)}"
        f"__{model_stack(model)}__{model['source']}.{ext}"
    )


# ---------------------------------------------------------------------------
# Audio measurement — the evidence that a run produced real audio
# ---------------------------------------------------------------------------


def audio_stats(path: str | Path) -> dict[str, float]:
    """Return {duration_s, sample_rate, rms, peak} for a PCM WAV.

    RMS is normalized to [0, 1]. A run that "succeeds" with rms == 0 emitted
    silence, which is not support — the report records the number so that shows.
    """
    import numpy as np
    import soundfile as sf

    p = Path(path)
    out = {"duration_s": 0.0, "sample_rate": 0.0, "rms": 0.0, "peak": 0.0}
    if not p.exists() or p.stat().st_size == 0:
        return out
    try:
        samples, rate = sf.read(p, dtype="float64", always_2d=True)
    except (sf.LibsndfileError, OSError):
        return out
    if samples.size == 0 or not np.isfinite(samples).all():
        return out
    return {"duration_s": len(samples) / rate, "sample_rate": float(rate),
            "rms": float(np.sqrt(np.mean(samples ** 2))),
            "peak": float(np.max(np.abs(samples)))}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
    """Filter MODELS by name list or return all. Silently excludes IGNORED_ENGINES."""
    names = _split_csv_arg(models_arg)
    if names == ["all"]:
        return [m for m in MODELS if m["name"] not in IGNORED_ENGINES]
    name_set = set(names) - IGNORED_ENGINES
    selected = [m for m in MODELS if m["name"] in name_set]
    missing = name_set - {m["name"] for m in selected}
    if missing:
        console.print(
            f"[yellow]Warning: unknown model(s): {', '.join(sorted(missing))}[/yellow]"
        )
    return selected


def build_args(model: dict, out: str) -> list[str]:
    """Wrapper arguments for one model's ``run`` subcommand."""
    args = [
        "--package", model["package"],
        "--output", out,
        "--task", model["task"],
    ]
    if model.get("family"):
        args += ["--family", model["family"]]
    if model.get("route"):
        args += ["--task_route", model["route"]]
    src = SOURCES.get(model["source"])
    if src:
        args += ["--audio", str(src)]
    tgt = TARGETS.get(model.get("target", ""))
    if tgt:
        args += ["--voice_ref", str(tgt)]
    extra_fn = model.get("extra_fn")
    if extra_fn:
        args += list(extra_fn())
    return args


def request_identity(model: dict) -> str:
    """Bind reuse to source/reference bytes, conditioned text, options and adapter."""
    paths = [SOURCES[model["source"]], TARGETS.get(model.get("target", "")), REFERENCES.get(model["source"])]
    inputs = []
    for path in paths:
        if path is None:
            inputs.append(None)
        else:
            with path.open("rb") as stream:
                inputs.append([str(path), hashlib.file_digest(stream, "sha256").hexdigest()])
    payload = {"inputs": inputs, "args": build_args(model, "__OUTPUT__"),
               "env": model.get("env", {}),
               "adapter": hashlib.sha256((ROOT / model["script"]).read_bytes()).hexdigest()}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def output_identity(path: Path) -> str:
    """Hash streamed output bytes for the resume receipt."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def reusable_output(path: Path, request: str) -> bool:
    """Legacy or malformed receipts never establish identity for a new request."""
    try:
        receipt = json.loads(path.with_suffix(".request.json").read_text())
        return receipt == {"request": request, "output_sha256": output_identity(path)}
    except (OSError, ValueError):
        return False


def _script_error_output(stdout: str | None, stderr: str | None) -> str:
    parts = [part.strip() for part in (stdout, stderr) if part and part.strip()]
    return "\n".join(parts) if parts else "(no output)"


def run_model(
    model: dict, out: str, timeout: int, verbose: bool, force: bool = False
) -> tuple[bool, float, str]:
    """Run one model. Returns (success, duration_s, error_msg)."""
    script_path = str(ROOT / model["script"])
    cmd = ["uv", "run", "--script", script_path, "run"] + build_args(model, out)
    if force:
        cmd.append("--force")

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
            cwd=str(Path.cwd()),
        )
        duration = time.monotonic() - t0
        if result.returncode != 0:
            combined = _script_error_output(result.stdout, result.stderr)
            return False, duration, "\n".join(combined.splitlines()[-5:])
        return True, duration, ""
    except subprocess.TimeoutExpired:
        return False, time.monotonic() - t0, f"timeout after {timeout}s"
    except FileNotFoundError:
        return False, time.monotonic() - t0, f"script not found: {model['script']}"
    except Exception as exc:
        return False, time.monotonic() - t0, str(exc)


def score_audio(
    audio: Path, ref: Path | None, timeout: int, source: Path | None = None
) -> dict[str, Any]:
    """Objective scores for one clip, via ``tools/aud-metrics.py``.

    RMS proves a file is not silence. It says nothing about whether a denoiser
    denoised: passing the input straight through scores exactly as well. These
    numbers are what separates the two, so a run that cannot be scored records
    the reason rather than an empty dict.
    """
    if not audio.exists() or audio.stat().st_size == 0:
        return {}
    cmd = ["uv", "run", "--script", str(ROOT / "tools/aud-metrics.py"), "score", "--audio", str(audio)]
    if ref is not None and ref.exists():
        cmd += ["--ref", str(ref)]
    if source is not None and source.exists():
        # The passthrough check: an edit or conversion that handed its input back
        # unchanged scores well on every other metric and near 1.0 on this one.
        cmd += ["--source", str(source)]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, cwd=str(Path.cwd())
        )
    except subprocess.TimeoutExpired:
        return {"error": f"metrics timeout after {timeout}s"}
    if result.returncode != 0:
        return {"error": _script_error_output(result.stdout, result.stderr).splitlines()[-1][:160]}
    # Fire prints nothing else on success, but uv can log to stdout; take the
    # last line that parses as the JSON document.
    for line in reversed(result.stdout.strip().splitlines()):
        try:
            return dict(json.loads(line).get("scores", {}))
        except json.JSONDecodeError:
            continue
    return {"error": "metrics produced no JSON"}


def prep_metrics(timeout: int, verbose: bool) -> tuple[bool, str]:
    """Download the MOS predictors once, before the batch needs them."""
    cmd = ["uv", "run", "--script", str(ROOT / "tools/aud-metrics.py"), "prep"]
    try:
        result = subprocess.run(
            cmd,
            stdout=None if verbose else subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            cwd=str(Path.cwd()),
        )
    except subprocess.TimeoutExpired:
        return False, f"prep timeout after {timeout}s"
    if result.returncode != 0:
        return False, _script_error_output(result.stdout, result.stderr).splitlines()[-1][:160]
    return True, ""


def prep_model(model: dict, timeout: int, verbose: bool) -> tuple[bool, float, str]:
    """Run a model's ``prep`` subcommand (download + cache weights). Idempotent."""
    script_path = str(ROOT / model["script"])
    cmd = ["uv", "run", "--script", script_path, "prep", "--package", model["package"]]
    env = os.environ.copy()
    env.update(model.get("env", {}))
    t0 = time.monotonic()
    try:
        result = subprocess.run(
            cmd,
            stdout=None if verbose else subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            env=env,
            cwd=str(Path.cwd()),
        )
        duration = time.monotonic() - t0
        if result.returncode != 0:
            combined = _script_error_output(result.stdout, result.stderr)
            return False, duration, "\n".join(combined.splitlines()[-5:])
        return True, duration, ""
    except subprocess.TimeoutExpired:
        return False, time.monotonic() - t0, f"prep timeout after {timeout}s"
    except Exception as exc:
        return False, time.monotonic() - t0, str(exc)


# ---------------------------------------------------------------------------
# Result record
# ---------------------------------------------------------------------------


class Result:
    """One processing job: what went in, what came out, and how it measured."""

    __slots__ = (
        "model", "op", "success", "duration", "error", "output_path",
        "source_path", "target_path", "file_size_kb", "audio_duration_s",
        "source_duration_s", "sample_rate", "source_sample_rate", "rms",
        "rtf", "stack", "quant", "family", "route", "metrics", "source_metrics",
    )

    def __init__(
        self,
        model: str,
        op: str,
        success: bool,
        duration: float,
        error: str,
        output_path: str = "",
        source_path: str = "",
        target_path: str = "",
        file_size_kb: float = 0.0,
        audio_duration_s: float = 0.0,
        source_duration_s: float = 0.0,
        sample_rate: float = 0.0,
        source_sample_rate: float = 0.0,
        rms: float = 0.0,
        rtf: float = 0.0,
        stack: str = "",
        quant: str = "",
        family: str = "",
        route: str = "",
        metrics: dict[str, Any] | None = None,
        source_metrics: dict[str, Any] | None = None,
    ):
        self.model = model
        self.op = op
        self.success = success
        self.duration = duration
        self.error = error
        self.output_path = output_path
        self.source_path = source_path
        self.target_path = target_path
        self.file_size_kb = file_size_kb
        self.audio_duration_s = audio_duration_s
        self.source_duration_s = source_duration_s
        self.sample_rate = sample_rate
        self.source_sample_rate = source_sample_rate
        self.rms = rms
        self.rtf = rtf
        self.stack = stack
        self.quant = quant
        self.family = family
        self.route = route
        # Objective scores for the output, and for the clip that went in — a
        # denoiser is judged on the difference, not on either number alone.
        self.metrics = metrics or {}
        self.source_metrics = source_metrics or {}

    def to_dict(self) -> dict[str, Any]:
        return {slot: getattr(self, slot) for slot in self.__slots__}

    @classmethod
    def from_dict(cls, d: dict) -> "Result":
        floats = {
            "duration", "file_size_kb", "audio_duration_s", "source_duration_s",
            "sample_rate", "source_sample_rate", "rms", "rtf",
        }
        kwargs: dict[str, Any] = {}
        for slot in cls.__slots__:
            value = d.get(slot)
            if slot == "success":
                kwargs[slot] = bool(value)
            elif slot in ("metrics", "source_metrics"):
                kwargs[slot] = dict(value) if isinstance(value, dict) else {}
            elif slot in floats:
                kwargs[slot] = float(value or 0.0)
            else:
                kwargs[slot] = str(value or "")
        return cls(**kwargs)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_summary(results: list[Result]) -> None:
    """Print a rich summary table."""
    table = Table(title="Audio Batch Results", show_lines=True)
    table.add_column("Model", style="bold cyan", no_wrap=True)
    table.add_column("Op", justify="center")
    table.add_column("Status", justify="center")
    table.add_column("Gen s", justify="right")
    table.add_column("Out s", justify="right")
    table.add_column("Rate", justify="right")
    table.add_column("RMS", justify="right")
    # DNSMOS overall is the single number that moves when a denoiser works; the
    # delta against the input clip is what says it was this model that moved it.
    # The band-energy ratio is a super-resolution number and would crowd this
    # table for every other op, so it lives in report.html instead.
    table.add_column("MOS", justify="right")
    table.add_column("ΔMOS", justify="right")
    table.add_column("Note", overflow="fold")

    for r in sorted(results, key=lambda r: (not r.success, r.model)):
        if r.success and r.rms > 0:
            status = "[green]ok[/green]"
        elif r.success:
            status = "[yellow]silent[/yellow]"
        else:
            status = "[red]fail[/red]"
        note = "" if r.success else (r.error.splitlines()[-1][:60] if r.error else "")
        if r.success and r.rms == 0:
            note = "output is silence"
        mos = r.metrics.get("dnsmos.ovrl")
        src_mos = r.source_metrics.get("dnsmos.ovrl")
        table.add_row(
            r.model, r.op, status, f"{r.duration:.1f}",
            f"{r.audio_duration_s:.1f}", f"{int(r.sample_rate) or '-'}",
            f"{r.rms:.4f}",
            f"{mos:.2f}" if isinstance(mos, (int, float)) else "-",
            f"{mos - src_mos:+.2f}"
            if isinstance(mos, (int, float)) and isinstance(src_mos, (int, float))
            else "-",
            note,
        )
    console.print(table)

    audible = sum(1 for r in results if r.success and r.rms >= SILENCE_RMS)
    silent = [r for r in results if r.success and r.rms < SILENCE_RMS]
    console.print(
        f"\n[bold]{audible}/{len(results)}[/bold] runs produced audible audio."
    )
    for r in silent:
        console.print(
            f"  [yellow]silent[/yellow] {r.model} — RMS {r.rms:.6f}, below the "
            f"{SILENCE_RMS} floor; the file is well formed and contains nothing"
        )


def _result_key(model: str) -> str:
    return model


def load_report_json(outdir: Path) -> dict:
    """Load prior report.json from *outdir*; return {} on missing/malformed."""
    path = outdir / "report.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError) as exc:
        console.print(f"[yellow]report.json malformed ({exc}) — starting fresh[/yellow]")
        return {}


def _build_report_data(
    results: list[Result],
    prior_data: dict,
    *,
    batch_elapsed: float | None = None,
    completed: int | None = None,
    total: int | None = None,
    total_models: int = 0,
    models_filter: str = "all",
) -> dict:
    """Merge *results* into *prior_data*, preserving entries not in this run."""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    merged: dict[str, dict] = {}
    prior_results = prior_data.get("results", {})
    if isinstance(prior_results, dict):
        merged.update(prior_results)
    for r in results:
        merged[_result_key(r.model)] = r.to_dict()

    for name in sorted(IGNORED_ENGINES):
        rec = merged.get(name, {"model": name})
        rec.update({"ignored": True, "status": "ignored", "reason": IGNORED_REASON})
        rec.setdefault("success", False)
        merged[name] = rec

    active = {k: v for k, v in merged.items() if not v.get("ignored")}
    successes = sum(1 for v in active.values() if v.get("success"))
    data: dict[str, Any] = {
        "generated": now,
        "meta": {
            "models_filter": models_filter,
            "total_models": total_models,
            "total_available": len([m for m in MODELS if m["name"] not in IGNORED_ENGINES]),
        },
        "summary": {
            "total_runs": len(active),
            "successes": successes,
            "failures": len(active) - successes,
            "batch_elapsed_s": batch_elapsed or 0.0,
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


def save_dashboard(outdir: Path) -> Path | None:
    """Rebuild ``out/index.html`` from the recorded report.json."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "build_dashboard", ROOT / "tools/build_dashboard.py"
    )
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build(outdir)


def _fmt_metric(rec: dict, key: str, fmt: str = "{:.2f}") -> str:
    """One score from a report record, or ``-`` when it was never measured."""
    value = (rec.get("metrics") or {}).get(key)
    return fmt.format(value) if isinstance(value, (int, float)) else "-"


def _fmt_delta(rec: dict, key: str) -> str:
    """Output score minus input score — the number enhancement is judged on."""
    after = (rec.get("metrics") or {}).get(key)
    before = (rec.get("source_metrics") or {}).get(key)
    if isinstance(after, (int, float)) and isinstance(before, (int, float)):
        return f"{after - before:+.2f}"
    return "-"


def save_report_html(data: dict, outdir: Path) -> Path:
    """Write a plain sortable HTML table of the run (report.html)."""
    import html as html_mod

    rows = []
    for rec in sorted(
        data.get("results", {}).values(), key=lambda r: str(r.get("model", ""))
    ):
        if rec.get("ignored"):
            status, cls = "ignored", "warn"
        elif rec.get("success") and float(rec.get("rms") or 0) > 0:
            status, cls = "ok", "ok"
        elif rec.get("success"):
            status, cls = "silent", "warn"
        else:
            status, cls = "failed", "bad"
        rows.append(
            "<tr>"
            + "".join(
                f"<td>{html_mod.escape(str(v))}</td>"
                for v in (
                    rec.get("model", ""),
                    rec.get("op", ""),
                    rec.get("family", ""),
                    rec.get("route", ""),
                    rec.get("quant", ""),
                )
            )
            + f'<td class="{cls}">{status}</td>'
            + "".join(
                f"<td>{v}</td>"
                for v in (
                    f"{float(rec.get('duration') or 0):.1f}",
                    f"{float(rec.get('audio_duration_s') or 0):.1f}",
                    int(float(rec.get("sample_rate") or 0)),
                    f"{float(rec.get('rms') or 0):.4f}",
                    _fmt_metric(rec, "dnsmos.ovrl"),
                    _fmt_delta(rec, "dnsmos.ovrl"),
                    _fmt_metric(rec, "hf8k_ratio", "{:.3f}"),
                    _fmt_metric(rec, "input_corr", "{:.3f}"),
                )
            )
            + f"<td>{html_mod.escape(str(rec.get('error', ''))[:200])}</td></tr>"
        )
    summary = data.get("summary", {})
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>fx — audio batch report</title>
<style>
body{{font:14px/1.5 -apple-system,BlinkMacSystemFont,sans-serif;margin:24px;color-scheme:light dark}}
table{{border-collapse:collapse;width:100%}}
th,td{{border-bottom:1px solid #8884;padding:6px 8px;text-align:left;font-variant-numeric:tabular-nums}}
th{{cursor:pointer;background:#8881}}
.ok{{color:#1f8a4c;font-weight:600}}.bad{{color:#c0392b;font-weight:600}}.warn{{color:#b8860b;font-weight:600}}
</style></head><body>
<h1>fx — audio batch report</h1>
<p>{summary.get("successes", 0)} of {summary.get("total_runs", 0)} runs succeeded;
last batch took {summary.get("batch_elapsed_s", 0):.0f}s. Generated {html_mod.escape(str(data.get("generated", "")))}.
Listen to the clips in <a href="index.html">index.html</a>.</p>
<table id="t"><thead><tr>
<th>Model</th><th>Op</th><th>Family</th><th>Route</th><th>Quant</th><th>Status</th>
<th>Gen s</th><th>Out s</th><th>Rate</th><th>RMS</th>
<th>DNSMOS</th><th>&Delta;DNSMOS</th><th>&gt;8&nbsp;kHz</th><th>corr(in)</th><th>Error</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table>
<script>
document.querySelectorAll('#t th').forEach((th,i)=>th.onclick=()=>{{
  const tb=document.querySelector('#t tbody');
  const rows=[...tb.rows];
  const num=rows.every(r=>!isNaN(parseFloat(r.cells[i].textContent))||!r.cells[i].textContent);
  th.asc=!th.asc;
  rows.sort((a,b)=>{{const x=a.cells[i].textContent,y=b.cells[i].textContent;
    return (num?parseFloat(x||0)-parseFloat(y||0):x.localeCompare(y))*(th.asc?1:-1);}});
  rows.forEach(r=>tb.appendChild(r));
}});
</script></body></html>
"""
    path = outdir / "report.html"
    path.write_text(page, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(
    manifest: str,
    outdir: str = "out",
    models: str = "all",
    timeout: int = 0,
    verbose: bool = False,
    force: bool = False,
    prep: bool = True,
    prep_only: bool = False,
    prep_timeout: int = 7200,
    metrics: bool = True,
    metrics_timeout: int = 900,
) -> None:
    """Prep then run every requested audio.cpp operation.

    Args:
        outdir: Output root directory (default: out/).
        models: Comma-separated model names, or "all".
        timeout: Inference timeout in seconds. 0 (the default) uses each model's
            own ``timeout`` from the registry; any other value overrides it for
            every selected model.
        verbose: Show subprocess stdout/stderr.
        force: Overwrite existing output files.
        prep: Download/cache weights for every selected model first (default on).
        prep_only: Only download, then exit.
        prep_timeout: Per-model timeout for the download step.
        metrics: Score every output (and the clip that went in) with
            ``tools/aud-metrics.py``. On by default: for enhancement models the
            scores are the only evidence that anything was enhanced.
        metrics_timeout: Per-clip timeout for the scoring step.
    """
    spec = json.loads(Path(manifest).expanduser().read_text())
    for name, target in (("sources", SOURCES), ("targets", TARGETS), ("references", REFERENCES)):
        target.clear()
        target.update({key: Path(value).expanduser().resolve() for key, value in spec.get(name, {}).items()})
    selected = resolve_models(models)
    global SOURCE_TEXT, TARGET_TEXT, PERSONA_PROMPT
    SOURCE_TEXT = str(spec.get("source_text", ""))
    TARGET_TEXT = str(spec.get("target_text", ""))
    PERSONA_PROMPT = str(spec.get("persona_prompt", ""))
    if any("" in list(m.get("extra_fn", lambda: [])()) for m in selected):
        raise SystemExit("manifest must supply source_text, target_text or persona_prompt for selected text-conditioned routes")
    required_sources = {m["source"] for m in selected}
    required_targets = {m["target"] for m in selected if m.get("target")}
    if not required_sources <= SOURCES.keys() or not required_targets <= TARGETS.keys():
        raise SystemExit("manifest must supply sources and targets for every selected model")
    missing = [str(p) for p in (*SOURCES.values(), *TARGETS.values()) if not p.exists()]
    if missing:
        console.print(f"[red]Missing fixture(s):[/red] {', '.join(missing)}")
        console.print("[yellow]Supply existing audio paths in the manifest.[/yellow]")
        sys.exit(1)

    selected = resolve_models(models)
    out_root = Path(outdir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    if prep or prep_only:
        # One download per package, not per registry entry (routes share weights).
        seen: set[str] = set()
        unique = [m for m in selected if not (m["package"] in seen or seen.add(m["package"]))]
        console.print(f"\n[bold]Prep[/bold] — caching {len(unique)} package(s)\n")
        for model in unique:
            ok, dur, err = prep_model(model, prep_timeout, verbose)
            mark = "[green]✓[/green]" if ok else "[red]✗[/red]"
            tail = "" if ok else f": {err.splitlines()[-1][:80] if err else 'unknown'}"
            console.print(f"  {mark} prep {model['package']} ({dur:.0f}s){tail}")
        if prep_only:
            return

    # Reference clip per fixture, and the fixture's own scores — measured once
    # and reused, since every denoiser in the batch is fed the same input.
    source_refs: dict[str, Path | None] = {
        key: REFERENCES.get(key) for key in SOURCES
    }
    _source_score_cache: dict[str, dict[str, Any]] = {}

    def source_scores(key: str) -> dict[str, Any]:
        """Scores for the input fixture *key*, computed at most once per batch."""
        if key not in _source_score_cache:
            src = SOURCES.get(key)
            _source_score_cache[key] = (
                score_audio(src, source_refs.get(key), metrics_timeout) if src else {}
            )
        return _source_score_cache[key]

    if metrics and not prep_only:
        ok, err = prep_metrics(prep_timeout, verbose)
        if not ok:
            console.print(
                f"[yellow]Metrics unavailable ({err}) — continuing without scores[/yellow]"
            )
            metrics = False

    prior_data = load_report_json(out_root)
    prior_results = prior_data.get("results", {})
    report_kw: dict[str, Any] = {
        "total_models": len(selected),
        "models_filter": models,
    }

    if not selected:
        data = _build_report_data([], prior_data, batch_elapsed=0.0, **report_kw)
        _write_json(out_root, data)
        console.print("[yellow]Nothing to run.[/yellow]")
        return

    console.print(f"\n[bold]Audio Batch[/bold] — {len(selected)} run(s)\n")
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
        task = progress.add_task("Processing...", total=len(selected))

        for model in selected:
            name = model["name"]
            out_name = output_filename(model)
            out_path = out_root / out_name
            src_path = SOURCES.get(model["source"], Path())
            tgt_path = TARGETS.get(model.get("target", ""), Path())
            src_stats = audio_stats(src_path)

            request = request_identity(model)
            stale_native = out_path.exists() and not reusable_output(out_path, request)
            if model.get("repo") in {"cstr/Sidon-GGUF", "FluidInference/localvqe-coreml", "iky1e/DeepFilterNet3-Streaming-CoreML"} and out_path.exists():
                gate = runpy.run_path(str(ROOT / model["script"]))["resume_ready"]
                gate_args = {"reference": str(tgt_path) if tgt_path.name else ""} if model["repo"] == "FluidInference/localvqe-coreml" else {}
                stale_native = stale_native or not gate(str(src_path), str(out_path), model["package"], **gate_args)

            if not force and not stale_native and out_path.exists() and out_path.stat().st_size > 0:
                progress.console.print(
                    f"  [yellow]⏭[/yellow] {name} skipped (exists: {out_name})"
                )
                prior = prior_results.get(_result_key(name))
                if prior and prior.get("success"):
                    restored = Result.from_dict(prior)
                    restored.output_path = str(out_path)
                    # Re-read format facts: older readers could mistake float
                    # WAV for silence. Cached timings/quality remain historical.
                    stats = audio_stats(out_path)
                    restored.audio_duration_s = stats["duration_s"]
                    restored.sample_rate = stats["sample_rate"]
                    restored.rms = stats["rms"]
                    restored.source_duration_s = src_stats["duration_s"]
                    restored.source_sample_rate = src_stats["sample_rate"]
                    restored.rtf = restored.duration / stats["duration_s"] if stats["duration_s"] else 0.0
                    # An output recorded before scoring existed still deserves
                    # scores; the audio is right there and re-running the model
                    # is the expensive half, not measuring it.
                    # "error" means the last attempt failed, not that the clip
                    # is unscoreable; a stuck download should not poison the
                    # record for every future run.
                    if metrics and (
                        not restored.metrics or "error" in restored.metrics
                    ):
                        restored.metrics = score_audio(
                            out_path,
                            source_refs.get(model["source"]),
                            metrics_timeout,
                            src_path,
                        )
                        restored.source_metrics = source_scores(model["source"])
                    results.append(restored)
                else:
                    stats = audio_stats(out_path)
                    results.append(
                        Result(
                            name, model_op(model), True, 0.0, "", str(out_path),
                            str(src_path), str(tgt_path) if tgt_path.name else "",
                            out_path.stat().st_size / 1024, stats["duration_s"],
                            src_stats["duration_s"], stats["sample_rate"],
                            src_stats["sample_rate"], stats["rms"], 0.0,
                            model_stack(model), model_quant(model),
                            model.get("family", ""), model.get("route", ""),
                            score_audio(
                                out_path,
                                source_refs.get(model["source"]),
                                metrics_timeout,
                                src_path,
                            )
                            if metrics
                            else {},
                            source_scores(model["source"]) if metrics else {},
                        )
                    )
                progress.advance(task)
                continue

            model_timeout = timeout or int(model.get("timeout", 1800))
            progress.console.print(f"  [cyan]▶[/cyan] {name} ({model_op(model)})")
            success, duration, error = run_model(
                model, str(out_path), model_timeout, verbose, force or stale_native
            )
            if not success and verbose and error:
                progress.console.print(f"    [dim]{error}[/dim]")

            if success:
                out_path.with_suffix(".request.json").write_text(json.dumps(
                    {"request": request, "output_sha256": output_identity(out_path)}, sort_keys=True) + "\n")
            stats = audio_stats(out_path) if success else audio_stats(Path())
            size_kb = out_path.stat().st_size / 1024 if out_path.exists() else 0.0
            rtf = duration / stats["duration_s"] if stats["duration_s"] > 0 else 0.0
            if success and stats["rms"] == 0.0:
                progress.console.print(
                    f"    [yellow]warning:[/yellow] {name} produced silence (rms=0)"
                )

            results.append(
                Result(
                    name, model_op(model), success, duration, error, str(out_path),
                    str(src_path), str(tgt_path) if tgt_path.name else "",
                    size_kb, stats["duration_s"], src_stats["duration_s"],
                    stats["sample_rate"], src_stats["sample_rate"], stats["rms"], rtf,
                    model_stack(model), model_quant(model),
                    model.get("family", ""), model.get("route", ""),
                    score_audio(
                        out_path, source_refs.get(model["source"]), metrics_timeout, src_path
                    )
                    if (metrics and success)
                    else {},
                    source_scores(model["source"]) if (metrics and success) else {},
                )
            )
            data = _build_report_data(
                results, prior_data, completed=len(results), total=len(selected),
                **report_kw,
            )
            _write_json(out_root, data)
            progress.advance(task)

    batch_elapsed = time.monotonic() - batch_start
    print_summary(results)

    data = _build_report_data(results, prior_data, batch_elapsed=batch_elapsed, **report_kw)
    report_path = _write_json(out_root, data)
    html_path = save_report_html(data, out_root)
    dash_path = save_dashboard(out_root)
    console.print(f"\n[bold]Report saved:[/bold] {report_path}")
    console.print(f"[bold]HTML report:[/bold]  {html_path}")
    console.print(f"[bold]Dashboard:[/bold]    {dash_path}")


if __name__ == "__main__":
    fire.Fire(main)
