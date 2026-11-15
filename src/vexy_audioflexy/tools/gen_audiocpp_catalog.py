#!/usr/bin/env python3
# this_file: src/vexy_audioflexy/tools/gen_audiocpp_catalog.py
"""Generate the shared audio.cpp family catalog from an engine's ``model_specs``.

One audio.cpp engine serves three registries (``tts/tts_all.py``,
``asr/asr_all.py``, ``fx/aud_all.py``), so the catalog lives beside them in
``tools/catalogs/`` and ``tools/_audiocpp_catalog.py`` turns it into registry
entries. Every route is written as ``pending runtime``: this script reads JSON
and source text only; it downloads and runs nothing.

    python src/vexy_audioflexy/tools/gen_audiocpp_catalog.py \
        --specs "$AUDIOCPP_ROOT/model_specs"

``--specs`` defaults to ``runtime_path('audio.cpp')/model_specs``; the engine
revision is read with ``git rev-parse HEAD`` in the specs' parent directory
unless ``--revision`` is given.
"""

from __future__ import annotations

import argparse
import json
import re
import runpy
import subprocess
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
STATUS = "pending runtime"
PENDING = "pending adapter"

# Spec task -> registry that has a generic audio.cpp adapter for it.
REGISTRY_FOR_TASK = {
    "tts": "tts_all",
    "clone": "tts_all",
    "design": "tts_all",
    "asr": "asr_all",
    "vc": "aud_all",
    "svc": "aud_all",
    "s2s": "aud_all",
    "sep": "separate_all",
}
# Spec tasks with no generic adapter in any registry, and why.
PENDING_REASONS = {
    "edit": "audiocpp_cli has no edit task, so aud-audiocpp cannot drive it",
    "music": "no registry harness supplies lyrics/tags/prompts for audio.cpp generation",
    "sfx": "no registry harness supplies sound-effect prompts for audio.cpp generation",
    "midi": "no registry consumes MIDI output",
    "vad": "detect_all.py pins a fixed package list at its own runtime commit; no generic adapter",
    "turn": "detect_all.py pins a fixed package list at its own runtime commit; no generic adapter",
    "wake_word": "no wake-word registry",
    "classify": "no audio-classification registry",
    "diar": "diarize_all.py does not wrap audio.cpp diarization packages generically",
    "align": "asr-audiocpp has an align subcommand but no registry drives it",
}
# Tasks `fx/tools/aud-audiocpp.py analyze` drives one clip at a time (no batch registry).
ANALYSIS_ROUTE = "aud_analyze"
ANALYSIS_TASKS = {"align", "vad", "diar", "wake_word", "classify"}
ALIGN_LANGUAGE = {"mms_forced_aligner": "eng", "qwen3_forced_aligner": "English"}
ANALYSIS_PENDING = {
    "sherpa_kws": "keyword spotting needs BPE-token keywords= rows the analyze command does not build",
}


def package_installable(spec: dict, package: str | None) -> bool:
    """True when prep can download *package* (its download kind is not 'unsupported')."""
    defaults = (spec.get("package_defaults") or {}).get("download") or {}
    entry = next((p for p in spec.get("packages") or [] if p["id"] == package), None)
    return entry is not None and {**defaults, **(entry.get("download") or {})}.get("kind") != "unsupported"


def analysis_route(spec: dict, task: str, package: str | None) -> dict:
    """Route an analysis task to the single-clip `analyze` command, or say why not."""
    family = spec["family"]
    if family in ANALYSIS_PENDING:
        return {"reason": ANALYSIS_PENDING[family]}
    if not package_installable(spec, package):
        return {"reason": "every package needs a local conversion (download kind 'unsupported')"}
    params = {"task": task}
    if task == "align":
        params["language"] = ALIGN_LANGUAGE.get(family, "")
    if family in ANALYSIS_OPTIONS:
        params["request_options"] = ANALYSIS_OPTIONS[family]
    return {"route": ANALYSIS_ROUTE, "package": package, "params": params, "reason": "single-clip analyze command; no batch registry"}


# Prompt-driven generation (`aud-audiocpp.py run --task gen|midi`), one clip per call.
GENERATION_ROUTE = "aud_generate"
GENERATION_TASKS = {"music", "sfx", "midi"}
GENERATION_PENDING = {
    "ace_step": "needs a --task-route the run command does not choose",
    "minimax_h3": "text-to-audio/video model; the run command handles audio only",
    "minimax_music3": "requires lyrics the run command does not supply by default",
    "mulacover": "needs melody and chord MIDI input files",
    "yue2": "requires a style option and a five-file package",
    "vevo2": "needs a reference recording",
    "latentsync": "needs a reference video",
    "liveavatar": "needs a reference image",
    "soulx_flashhead": "needs a reference image",
}
# Request options some analysis families cannot run without.
ANALYSIS_OPTIONS = {
    "lfm_d1_omni": 'questions={"sound":{"type":"choice","instructions":"What is the main sound?",'
                   '"criteria":{"speech":"A person speaking","music":"Music playing","noise":"Background noise"}}}',
}


def generation_route(spec: dict, task: str, package: str | None) -> dict:
    """Route music/sfx/midi to the single-clip run command, or say why not."""
    family = spec["family"]
    if family in GENERATION_PENDING:
        return {"reason": GENERATION_PENDING[family]}
    if not package_installable(spec, package):
        return {"reason": "every package needs a local conversion (download kind 'unsupported')"}
    params = {"task": "midi" if task == "midi" else "gen"}
    return {"route": GENERATION_ROUTE, "package": package, "params": params,
            "reason": "single-clip run command with a text prompt (midi: an audio input); no batch registry"}


# Files whose text counts as an existing registry reference, per registry.
REGISTRY_FILES = {
    "tts_all": ["tts/tts_all.py"],
    "asr_all": ["asr/asr_all.py", "asr/catalogs/*.json"],
    "aud_all": ["fx/aud_all.py", "fx/catalogs/*.json"],
    "diarize_all": ["asr/diarize_all.py"],
    "separate_all": ["fx/separate_all.py"],
    "detect_all": ["fx/detect_all.py"],
}
OVERLAY_DIRS = {"tts_all": "tts/model_specs", "asr_all": "asr/model_specs"}
SUPPORTED_REQUIRED = {"tts_all": {"reference_text", "instruct"}, "asr_all": {"language"}, "aud_all": set(), "separate_all": set()}
NOTABLE_OPTIONS = ("voice_id", "speaker", "reference_text", "language", "instruct", "lyrics", "tags", "prompt", "emotion")
SR_WORDS = ("super-resolution", "super resolution", "upsampl")
NOISY_WORDS = ("enhanc", "restor", "denois")
# Hand decisions the spec JSON cannot express, keyed by (family, registry).
PACKAGE_OVERRIDES = {
    # The recommended package is the 16 kHz separation model; s2s needs the 48 kHz enhancer.
    ("mossformer2", "aud_all"): "mossformer2_se_48k_f32",
    # voice_id=1 is a CustomVoice preset; the VoiceClone checkpoint rejects it (session.cpp:143).
    ("m5_tts", "tts_all"): "m5_tts_50m_customvoice_q8_0",
}
MANUAL_PENDING = {
    ("sam_audio", "aud_all"): "prompt-conditioned separation; the harness supplies no target-source prompt",
    ("vevo2", "tts_all"): "text-to-speech needs a target_voice reference that tts-audiocpp does not pass",
    ("samsone", "asr_all"): "audio captioner (default instruct 'describe the audio'), not a transcriber; verified 2026-11-15 it returns captions",
    ("minimax_h3", "tts_all"): "text-to-audio/video generation (--task gen, dit.gguf, video request options), not TTS; the CLI rejects --task tts (docs/models/minimax_h3.md)",
    ("vieneu_v3_turbo", "tts_all"): "reads SEA-G2P phonemes, not raw text, and needs a speaker_embedding_file; the CLI rejects --task clon (docs/models/vieneu_v3_turbo.md)",
    ("auk", "tts_all"): "multi-component model: each generator package also needs the Qwen2.5-Omni-3B and VAE packages; run 2026-11-15 failed on the missing qwen2.5-omni-3b-bf16.gguf",
}

ISO639_3 = {"en": "eng", "pl": "pol"}


def asr_params(spec: dict) -> dict:
    """``lang_flag`` when the spec takes ``language``; ``lang_map`` when its default is an ISO 639-3 code (OWSM)."""
    language = next((o for o in request_options(spec) if o["name"] == "language"), None)
    params: dict = {"lang_flag": language is not None}
    default = str((language or {}).get("default") or "")
    if len(default) == 3 and default.isalpha():
        params["lang_map"] = ISO639_3
    return params


def precision_rank(precision: str) -> tuple[int, str]:
    """Smaller number = smaller weights; unknown precisions sort last."""
    match = re.search(r"(\d+)", precision or "")
    return (int(match.group(1)) if match else 99, precision or "")


def pick_package(spec: dict) -> str | None:
    """recommended_package, else the default package, else the smallest precision."""
    defaults = (spec.get("package_defaults") or {}).get("download") or {}
    every = spec.get("packages") or []
    # Prefer packages prep can install; "unsupported" ones need a local conversion.
    installable = [p for p in every if {**defaults, **(p.get("download") or {})}.get("kind") != "unsupported"]
    packages = installable or every
    ids = {p["id"] for p in packages}
    recommended = (spec.get("ui") or {}).get("recommended_package")
    if recommended in ids:
        return recommended
    for package in packages:
        if package.get("default"):
            return package["id"]
    return min(packages, key=lambda p: precision_rank(p.get("precision", "")))["id"] if packages else None


def package_record(package: dict, defaults: dict) -> dict:
    download = {**defaults, **(package.get("download") or {})}
    return {
        "id": package["id"],
        "precision": package.get("precision"),
        "format": package.get("format"),
        "repo": download.get("repo"),
        "revision": download.get("revision"),
        "gated": bool(download.get("gated", False)),
        "files": package.get("files", []),
        "target_directory": package.get("target_directory"),
    }


def registry_langs(spec: dict) -> list[str]:
    """Benchmark languages (en/pl) the registries can feed this family."""
    iso3 = {"eng": "en", "pol": "pl"}
    langs = [iso3.get(str(code).lower(), str(code).lower()) for code in spec.get("languages") or []]
    agnostic = not langs or any(code in {"auto", "multilingual", "language_agnostic"} for code in langs)
    out = [code for code in ("en", "pl") if agnostic or any(c == code or c.startswith(code + "-") for c in langs)]
    return out


def request_options(spec: dict) -> list[dict]:
    return list(((spec.get("options") or {}).get("request")) or [])


def tts_params(spec: dict, tasks: set[str], required: list[str]) -> dict:
    ui = spec.get("ui") or {}
    caps = set((spec.get("capabilities") or {}).get("tts", []))
    options = {o["name"]: o for o in request_options(spec)}
    voice = ui.get("default_voice") or options.get("voice_id", {}).get("default") or next(iter(ui.get("builtin_voices") or []), None)
    if "reference_text" in required:
        mode = "clone"
    elif "instruct" in required:
        mode = "design"
    elif "tts" in tasks and ("built_in_voices" in caps or voice or "clone" not in tasks):
        mode = "tts"
    elif "clone" in tasks:
        mode = "clone"
    else:
        mode = "design"
    language = options.get("language", {})
    lang_flag = bool(language) and "not an iso" not in str(language.get("description", "")).lower()
    # A tts route without built-in voices but with speaker_reference conditions on the reference clip.
    ref = mode == "clone" or (mode == "tts" and not voice and "speaker_reference" in caps)
    return {
        "mode": mode,
        "task": {"tts": "tts", "clone": "clon", "design": "vdes"}[mode],
        "voice_id": voice if mode == "tts" else None,
        "ref": ref,
        "ref_text": ref and "reference_text" in options,
        "lang_flag": lang_flag,
    }


# Families whose engine rejects other input rates need a matching manifest source.
SOURCE_OVERRIDES = {"apollo": "noisy44k"}  # session.cpp: "Apollo requires 44.1 kHz input audio"


def aud_params(spec: dict, tasks: set[str]) -> dict:
    task = next(t for t in ("vc", "svc", "s2s") if t in tasks)
    text = f"{spec.get('description', '')} {spec.get('display_name', '')}".lower()
    caps = (spec.get("capabilities") or {}).get(task, [])
    if task in {"vc", "svc"} or "speaker_reference" in caps:
        source, target = "speech", "spkb"
    elif "audio_enhancement" in caps:
        source, target = "noisy", ""
    elif any(word in text for word in SR_WORDS):
        source, target = "lowres", ""
    elif any(word in text for word in NOISY_WORDS):
        source, target = "noisy", ""
    else:
        source, target = "speech", ""
    source = SOURCE_OVERRIDES.get(spec["family"], source)
    return {"op": task, "task": task, "source": source, "target": target}


def route_record(spec: dict, registry: str, tasks: set[str], covered: list[str], package: str | None) -> dict:
    record: dict = {"registry": registry, "tasks": sorted(tasks), "covered_by": covered, "status": STATUS}
    package = PACKAGE_OVERRIDES.get((spec["family"], registry), package)
    required = [o["name"] for o in request_options(spec) if o.get("required")]
    langs = registry_langs(spec)
    if registry == "separate_all":
        langs = langs or ["any"]  # separation is language-independent
    unsupported = [name for name in required if name not in SUPPORTED_REQUIRED[registry]]
    if covered:
        record.update(route="covered", reason="an existing registry entry already references this family's packages")
    elif package is None:
        record.update(route=PENDING, reason="spec ships no packages")
    elif (spec["family"], registry) in MANUAL_PENDING:
        record.update(route=PENDING, reason=MANUAL_PENDING[(spec["family"], registry)])
    elif unsupported:
        record.update(route=PENDING, reason=f"requires request option(s) the generic harness cannot supply: {', '.join(unsupported)}")
    elif not langs:
        record.update(route=PENDING, reason="no en/pl support; the registries only benchmark en and pl text/audio")
    else:
        params = {"tts_all": lambda: tts_params(spec, tasks, required), "asr_all": lambda: asr_params(spec), "aud_all": lambda: aud_params(spec, tasks),
                  "separate_all": lambda: {"task": "sep"}}[registry]()
        record.update(route=registry, package=package, langs=langs, params=params)
    return record


def family_routes(spec: dict, coverage: dict[str, list[str]], package: str | None) -> list[dict]:
    tasks = list(spec.get("tasks") or [])
    by_registry: dict[str, set[str]] = {}
    routes = []
    for task in tasks:
        if task in REGISTRY_FOR_TASK:
            by_registry.setdefault(REGISTRY_FOR_TASK[task], set()).add(task)
    for registry, group in by_registry.items():
        routes.append(route_record(spec, registry, group, coverage.get(registry, []), package))
    for task in tasks:
        if task in REGISTRY_FOR_TASK:
            continue
        covered = [ref for reg, refs in coverage.items() if reg not in REGISTRY_FOR_TASK.values() for ref in refs]
        reason = PENDING_REASONS.get(task, f"no adapter for task {task}")
        record = {"registry": None, "tasks": [task], "covered_by": covered, "status": STATUS,
                  "route": "covered" if covered else PENDING, "reason": "referenced by an existing registry" if covered else reason}
        if not covered and task in ANALYSIS_TASKS:
            record.update(analysis_route(spec, task, package))
        elif not covered and task in GENERATION_TASKS:
            record.update(generation_route(spec, task, package))
        routes.append(record)
    return routes


def registry_texts(tools: Path) -> dict[str, str]:
    texts = {}
    for registry, patterns in REGISTRY_FILES.items():
        texts[registry] = "\n".join(p.read_text() for pattern in patterns for p in sorted(tools.glob(pattern)))
    return texts


def coverage_for(spec: dict, texts: dict[str, str]) -> dict[str, list[str]]:
    """Registry -> package ids of this family already quoted in that registry's sources."""
    out = {}
    for registry, text in texts.items():
        hits = [p["id"] for p in spec.get("packages") or [] if re.search(rf"(?<![\w-]){re.escape(p['id'])}(?![\w-])", text)]
        if hits:
            out[registry] = hits
    return out


def overlay_families(tools: Path) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for registry, rel in OVERLAY_DIRS.items():
        for path in sorted((tools / rel).rglob("*.json")):
            family = json.loads(path.read_text()).get("family")
            if family:
                out.setdefault(family, []).append(f"{registry}:{path.relative_to(tools).as_posix()}")
    return out


def family_record(spec: dict, texts: dict[str, str], overlays: dict[str, list[str]]) -> dict:
    defaults = (spec.get("package_defaults") or {}).get("download") or {}
    options = request_options(spec)
    package = pick_package(spec)
    coverage = coverage_for(spec, texts)
    return {
        "family": spec["family"],
        "display_name": spec.get("display_name"),
        "category": spec.get("category"),
        "upstream_status": spec.get("status"),
        "status": STATUS,
        "tasks": spec.get("tasks") or [],
        "languages": spec.get("languages") or [],
        "required_options": [o["name"] for o in options if o.get("required")],
        "notable_options": [o["name"] for o in options if o["name"] in NOTABLE_OPTIONS and not o.get("required")],
        "default_voice": (spec.get("ui") or {}).get("default_voice"),
        "recommended_package": (spec.get("ui") or {}).get("recommended_package"),
        "selected_package": package,
        "packages": [package_record(p, defaults) for p in spec.get("packages") or []],
        "covered_by": [f"{reg}:{pid}" for reg, ids in coverage.items() for pid in ids],
        "overlay_specs": overlays.get(spec["family"], []),
        "routes": family_routes(spec, coverage, package),
    }


def default_specs() -> Path:
    paths = runpy.run_path(str(TOOLS / "tts/tools/_paths.py"))
    return paths["runtime_path"]("audio.cpp") / "model_specs"


def engine_revision(specs: Path) -> str:
    return subprocess.run(["git", "-C", str(specs.parent), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()


VERIFIED = "verified"


def apply_verification(families: list[dict], revision: str, tools: Path = TOOLS) -> None:
    """Mark routes that passed a real run on *revision* (``catalogs/audiocpp-verified.json``).

    Records for another engine revision are ignored, so a rebuilt engine starts
    every row back at ``pending runtime`` until it is re-run.
    """
    path = tools / "catalogs" / "audiocpp-verified.json"
    if not path.exists():
        return
    ledger = json.loads(path.read_text())
    if ledger.get("source_revision") != revision:
        return
    runs = ledger.get("runs", {})
    for family in families:
        for route in family["routes"]:
            key = f"{route.get('route')}:{route.get('package')}"
            tasks = route.get("tasks") or [""]
            run = runs.get(f"{key}:{tasks[0]}") or runs.get(key)  # task suffix for shared packages
            if run:
                route.update(status=VERIFIED, verification=run)


def build(specs: Path, revision: str, tools: Path = TOOLS) -> dict:
    texts = registry_texts(tools)
    overlays = overlay_families(tools)
    families = [family_record(json.loads(p.read_text()), texts, overlays) for p in sorted(specs.glob("*.json"))]
    apply_verification(families, revision, tools)
    return {"schema_version": 1, "source_revision": revision, "status": STATUS, "families": families}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--specs", type=Path, default=None, help="audio.cpp model_specs directory")
    parser.add_argument("--revision", default="", help="engine git SHA (default: git rev-parse HEAD)")
    parser.add_argument("--out", type=Path, default=None, help="output JSON (default: catalogs/audiocpp-<sha7>.json)")
    args = parser.parse_args()
    specs = (args.specs or default_specs()).expanduser().resolve()
    revision = args.revision or engine_revision(specs)
    out = args.out or TOOLS / "catalogs" / f"audiocpp-{revision[:7]}.json"
    catalog = build(specs, revision)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(catalog, indent=2, ensure_ascii=False) + "\n")
    print(f"{out}: {len(catalog['families'])} families at {revision}")


if __name__ == "__main__":
    main()
