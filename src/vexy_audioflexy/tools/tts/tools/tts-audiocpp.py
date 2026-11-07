#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = ["fire", "rich", "huggingface-hub"]
# ///
# this_file: src/vexy_audioflexy/tools/tts/tools/tts-audiocpp.py
"""Generic driver for every audio.cpp GGUF TTS family (audio-cpp/audio.cpp-gguf).

audio.cpp is a single pure-C++ ggml engine: one `audiocpp_cli` binary dispatches to
~66 model loaders via `--family`, so all of its TTS families share this one script
instead of one wrapper per model. Backend: metal (Apple Silicon GPU).

Model specifications select exact Hugging Face snapshot files and sidecars.
Downloads stay in the provider cache; native layouts contain links and markers.
"""

from __future__ import annotations

import json
import os
import runpy
import subprocess
from functools import cache
from pathlib import Path
from _paths import runtime_path

_install_package = runpy.run_path(str(Path(__file__).resolve().parents[2] / "_audiocpp_provider.py"))["install_package"]

VERSION = "1.0.0"

# audio.cpp checkout: cloned by `prep`/`setup` if missing.
REPO_URL = "https://github.com/0xShug0/audio.cpp.git"
ENGINE_ROOT = Path(
    os.environ.get(
        "AUDIOCPP_ROOT", str(runtime_path('audio.cpp'))
    )
).expanduser().resolve()
BIN_PATH = Path(os.environ.get("AUDIOCPP_BIN", str(ENGINE_ROOT / "build/macos-metal-release/bin/audiocpp_cli"))).expanduser().resolve()
# The GGUF packages run to well over a hundred gigabytes across the full family
# set, so the store is relocatable: point AUDIOCPP_MODELS_ROOT at a volume with
# room. It defaults to the configured consolidated models directory.
MODELS_ROOT = Path(
    os.environ.get("AUDIOCPP_MODELS_ROOT", str(runtime_path('models')))
).expanduser().resolve()
SPECS_DIR = ENGINE_ROOT / "model_specs"
EXTRA_SPECS_DIR = Path(__file__).resolve().parent.parent / "model_specs"
BACKEND = os.environ.get("AUDIOCPP_BACKEND", "metal")


def _require_engine() -> Path:
    """Return the audiocpp_cli path, or exit with build instructions."""
    if BIN_PATH.exists():
        return BIN_PATH
    raise SystemExit(
        f"audiocpp_cli not built at {BIN_PATH}\n"
        f"Build it with:\n"
        f"  git clone --depth 1 {REPO_URL} {ENGINE_ROOT}\n"
        f"  cd {ENGINE_ROOT} && ./scripts/build_metal.sh --target audiocpp_cli"
    )


@cache
def _specs() -> dict[str, dict]:
    """Map package id -> {family, target_directory, precision, spec}."""
    out: dict[str, dict] = {}
    for path in [*sorted(SPECS_DIR.glob("*.json")), *sorted(EXTRA_SPECS_DIR.glob("*.json"))]:
        spec = json.loads(path.read_text(encoding="utf-8"))
        for pkg in spec.get("packages", []):
            download = {**spec.get("package_defaults", {}).get("download", {}), **pkg.get("download", {})}
            out[pkg["id"]] = {
                "family": spec["family"],
                "target_directory": pkg.get("target_directory", ""),
                "files": pkg.get("files", []),
                "strip_prefix": pkg.get("strip_prefix", ""),
                "precision": pkg.get("precision", ""),
                "format": pkg.get("format", ""),
                "tasks": spec.get("tasks", []),
                "languages": spec.get("languages", []),
                "repo": download.get("repo", ""),
                "revision": download.get("revision", "main"),
                "download_kind": download.get("kind", ""),
                "specs_dir": str(path.parent),
            }
    return out


def _resolve(package: str) -> dict:
    entry = _specs().get(package)
    if entry is None:
        raise SystemExit(
            f"unknown audio.cpp package '{package}'. "
            f"List them with: python3 {SPECS_DIR.parent}/tools/model_manager_v2.py list"
        )
    return entry


def _model_path(package: str) -> Path:
    """Where the engine should be pointed for *package*.

    Usually the package directory. Some specs carry no `strip_prefix`, so their
    weights land in a subdirectory of it — GLM-TTS ships as
    `GLM-TTS-Q8/Text to audio (TTS)/GLM-TTS_Q8.gguf`. Handing the engine the
    directory then fails, because it looks for a package config beside the GGUF
    and finds neither. Pointing it at the file itself works, so resolve to the
    lone nested GGUF when the top level has none.
    """
    entry = _resolve(package)
    root = MODELS_ROOT / entry["target_directory"]
    if not root.is_dir():
        return root

    if entry["family"] == "lfm2_audio":
        return root
    if entry["family"] == "minimax_h3":
        dit = next((Path(f).name for f in entry.get("files", []) if Path(f).name in {"dit.gguf", "dit_int8.gguf"}), None)
        if dit:
            return root / dit

    top = sorted(root.glob("*.gguf"))
    declared = [Path(f).name for f in entry.get("files", []) if f.endswith(".gguf")]
    if len(declared) == 1:
        matches = sorted(root.rglob(declared[0]))
        if not matches:
            return root / declared[0]
        if top and top != [matches[0]]:
            return matches[0]
    if len(top) == 1:
        return root  # unambiguous; the directory form is what most families want

    # Two packages of one family can share a directory — PersonaPlex's q4_k and
    # q8_0, IndexTTS2's q8 and f16 — and the engine refuses a directory holding
    # more than one GGUF. The spec says which file belongs to this package.
    for declared in entry.get("files", []):
        name = Path(declared).name
        if not name.endswith(".gguf"):
            continue
        for candidate in (root / name, *root.rglob(name)):
            if candidate.is_file():
                return candidate

    if not top:
        nested = sorted(root.rglob("*.gguf"))
        if len(nested) == 1:
            return nested[0]
    return root


def _installed(package: str) -> bool:
    entry = _resolve(package)
    root = MODELS_ROOT / entry["target_directory"]
    files = entry.get("files", [])
    if files:
        prefix = entry.get("strip_prefix", "").rstrip("/") + "/"
        paths = [root / (name[len(prefix):] if prefix != "/" and name.startswith(prefix) else name) for name in files]
        return all(path.is_file() and path.stat().st_size > 0 for path in paths)
    return root.is_dir() and any(root.rglob("*.gguf"))


def prep(
    package: str = "pocket_tts_english_q8_0", deps: str = "", force: bool = False
) -> None:
    """Download and cache one audio.cpp GGUF package. Idempotent.

    Args:
        package: audio.cpp package id (e.g. `supertonic_3_orig`).
        deps: Comma-separated companion package ids a family also needs, such as
            the MioCodec decoder MioTTS loads through a session option.
        force: Reinstall even if the target directory already holds the files.
    """
    _require_engine()
    for dep in (d.strip() for d in deps.split(",") if d.strip()):
        prep(dep, force=force)
    target = _model_path(package)
    if _installed(package) and not force:
        print(f"[prep] ready: {package} -> {target}")
        return
    # These are multi-gigabyte downloads and a batch run has several in flight at
    # once; a single transport hiccup should not fail the model. The manager is
    # idempotent, so a retry cannot replace an already installed package.
    attempts = 3
    for attempt in range(1, attempts + 1):
        try:
            _install_package(_resolve(package), MODELS_ROOT, package, force)
            break
        except Exception as exc:
            if isinstance(exc, (FileExistsError, ValueError)) or attempt == attempts:
                raise
            print(f"[prep] {package}: attempt {attempt} failed, retrying")
    if not _installed(package):
        raise SystemExit(f"[prep] {package}: installation remains incomplete; inspect existing files or use --force to replace them")
    print(f"[prep] ready: {package} -> {target}")


def say(
    text: str,
    package: str = "pocket_tts_english_q8_0",
    output: str = "output.wav",
    family: str = "",
    task: str = "tts",
    deps: str = "",
    ref: str = "",
    ref_mode: str = "voice_ref",
    ref_text: str = "",
    lang: str = "",
    voice_id: str = "",
    speaker: str = "",
    instruct: str = "",
    emotion: str = "",
    request_options: str = "",
    session_options: str = "",
    seed: int = 0,
    extra: str = "",
    backend: str = "",
    play: bool = False,
    force: bool = False,
) -> None:
    """Synthesize speech with one audio.cpp GGUF package.

    Args:
        text: Text to synthesize.
        package: audio.cpp package id; picks both the family and the GGUF files.
        output: Output WAV path.
        family: Override the family from the package spec (rarely needed).
        task: audio.cpp task route — `tts`, `clon`, `vdes` or `gen`.
        deps: Comma-separated companion package ids to install alongside *package*.
        ref: Reference speaker WAV for voice cloning.
        ref_mode: How *ref* reaches the engine — `voice_ref` (the usual
            `--voice-ref`) or `voice_samples` (VibeVoice's speaker-list request
            option).
        ref_text: Transcript of *ref* for families that condition on it.
        backend: Override the engine backend for one family (e.g. "cpu" when a
            family crashes under Metal); defaults to $AUDIOCPP_BACKEND.
        lang: Language code passed as `--language` when the family takes one.
        voice_id: Built-in preset voice id (Supertonic `M1`, PocketTTS `alba`, ...).
        speaker: Packaged speaker name (`--speaker`), which is a different
            selector from `--voice-id`: Qwen3 CustomVoice takes `Vivian`/`Ryan`
            here, and no reference WAV.
        instruct: Voice-design or style instruction text (`--instruct`).
        emotion: Emotion conditioning (`--emotion`). IndexTTS-2 takes a sentence
            whose emotion is transferred; other families take a name.
        request_options: Per-request `key=value` settings, separated by `;`
            (e.g. `emotion_alpha=0.6;duration_factor=1.1`).
        session_options: Per-session `key=value` settings, separated by `;`.
        seed: Fixed sampling seed; 0 leaves the model default.
        extra: Additional raw CLI arguments, space-separated.
        play: Play the result with afplay instead of reporting the path.
        force: Overwrite an existing output file.
    """
    from rich.console import Console
    from rich.panel import Panel

    console = Console()
    binary = _require_engine()
    out_path = Path(output).resolve()

    if not force and out_path.exists() and out_path.stat().st_size > 0:
        console.print(
            f"[yellow]Skipped:[/yellow] {output} already exists (use --force to overwrite)"
        )
        return

    entry = _resolve(package)
    fam = family or entry["family"]
    # shortcut: this build's Metal lacks TQ2_1 kernels, remove after upstream fixes them.
    selected_backend = backend or ("cpu" if package == "kitten_tts2_20261021_tq2_1" and BACKEND == "metal" else BACKEND)
    model_path = _model_path(package)
    if deps or not _installed(package):
        prep(package, deps=deps)
        model_path = _model_path(package)

    cmd = [
        str(binary),
        "--task", task,
        "--family", fam,
        "--model", str(model_path),
        "--backend", selected_backend,
        "--text", text,
        "--out", str(out_path),
        "--metrics",
    ]
    if fam == "minimax_h3":
        cmd += ["--model-spec-override", str(SPECS_DIR / "minimax_h3.json")]
    if ref:
        if ref_mode == "voice_samples":
            samples = ",".join(str(Path(item.strip()).resolve()) for item in ref.split(","))
            cmd += ["--request-option", f"voice_samples={samples}"]
        else:
            cmd += ["--voice-ref", str(Path(ref).resolve())]
    if ref_text:
        cmd += ["--reference-text", ref_text]
    if lang:
        cmd += ["--language", lang]
    if voice_id:
        cmd += ["--voice-id", voice_id]
    if speaker:
        cmd += ["--speaker", speaker]
    if instruct:
        cmd += ["--instruct", instruct]
    if emotion:
        cmd += ["--emotion", emotion]
    if fam == "lfm2_audio" and "lfm2_audio.model_gguf=" not in session_options:
        backbone = next((Path(f).name for f in entry.get("files", []) if Path(f).name.startswith("LFM2.5-Audio-") and f.endswith(".gguf")), "")
        if backbone:
            cmd += ["--session-option", f"lfm2_audio.model_gguf={backbone}"]
    for flag, blob in (("--request-option", request_options),
                       ("--session-option", session_options)):
        for setting in (part.strip() for part in blob.split(";") if part.strip()):
            cmd += [flag, setting]
    if seed:
        cmd += ["--seed", str(seed)]
    if extra:
        cmd += extra.split()

    console.print(
        Panel.fit(
            f"[bold cyan]audio.cpp[/bold cyan] v{VERSION}  family=[yellow]{fam}[/yellow]\n"
            f"package=[yellow]{package}[/yellow]  task=[yellow]{task}[/yellow]  "
            f"backend=[yellow]{selected_backend}[/yellow]",
            border_style="cyan",
        )
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ENGINE_ROOT), check=False)
    if result.returncode != 0 or not out_path.is_file() or out_path.stat().st_size == 0:
        # ggml logs every Metal pipeline it compiles; keep only the decisive lines.
        noise = ("ggml_metal", "ggml_backend", "load_backend")
        lines = [
            ln
            for ln in (result.stdout + "\n" + result.stderr).splitlines()
            if ln.strip() and not ln.startswith(noise)
        ]
        console.print(f"[red]audiocpp_cli failed (exit {result.returncode})[/red]")
        console.print("\n".join(lines[-12:]))
        raise SystemExit(1)

    metrics = {
        k: v
        for k, _, v in (
            ln.partition("=")
            for ln in result.stdout.splitlines()
            if ln.startswith("metrics.")
        )
    }
    if play:
        subprocess.run(["afplay", str(out_path)], check=True)
    else:
        dur = float(metrics.get("metrics.audio_duration_ms", 0)) / 1000.0
        rtf = metrics.get("metrics.rtf", "?")
        console.print(
            f"[green]Saved[/green] [bold]{out_path.resolve()}[/bold] ({dur:.1f}s, rtf={rtf})"
        )


def info(package: str = "pocket_tts_english_q8_0") -> None:
    """Print one-line JSON with package metadata."""
    entry = _resolve(package)
    print(
        json.dumps(
            {
                "name": package,
                "family": entry["family"],
                "repo": entry.get("repo", "audio-cpp/audio.cpp-gguf"),
                "backend": BACKEND,
                "stack": "audiocpp",
                "quant": entry["precision"],
                "tasks": entry["tasks"],
                "languages": entry["languages"],
                "model_path": str(_model_path(package)),
            },
            separators=(",", ":"),
        )
    )


def packages(tts_only: bool = True) -> None:
    """List installable audio.cpp packages (id, family, precision)."""
    for pkg, entry in sorted(_specs().items()):
        if tts_only and not (set(entry["tasks"]) & {"tts", "clone", "design", "vdes"}):
            continue
        print(f"{pkg:44} {entry['family']:22} {entry['precision']}")


if __name__ == "__main__":
    import fire

    fire.Fire({"prep": prep, "say": say, "info": info, "packages": packages})
