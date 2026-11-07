#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = ["platformdirs>=4", "fire", "rich", "huggingface-hub"]
# ///
# this_file: src/vexy_audioflexy/tools/fx/tools/aud-audiocpp.py
"""Generic driver for every audio.cpp audio-in/audio-out family.

audio.cpp is a single pure-C++ ggml engine: one `audiocpp_cli` binary dispatches to
~66 model loaders via `--family`, so all of its enhancement and voice-conversion
families share this one script instead of one wrapper per model. Backend: metal.

Families covered here (task routes confirmed against `audiocpp_cli --list-loaders`):

| family      | tasks                | what it does                                  |
|-------------|----------------------|-----------------------------------------------|
| audiosr     | s2s                  | audio super-resolution (bandwidth extension)  |
| meanvc2     | vc                   | zero-shot voice conversion                    |
| personaplex | s2s                  | speech-to-speech assistant reply              |
| seed_vc     | vc, svc              | voice conversion + singing voice conversion   |
| vevo2       | tts, vc, s2s, svc    | route-selected synthesis/conversion/editing   |

Model specifications select exact Hugging Face snapshot files. Downloads stay in
the provider cache; native model layouts contain links and package markers.

The engine itself is never cloned or built by this project: `AUDIOCPP_ROOT` points at
the existing sibling checkout in tts.
"""

from __future__ import annotations


import os
from pathlib import Path
import runpy
model_path = runpy.run_path(str(Path(__file__).with_name("_audioflexy_paths.py")))["model_path"]
_install_package = runpy.run_path(str(Path(__file__).resolve().parents[2] / "_audiocpp_provider.py"))["install_package"]
from platformdirs import user_cache_path, user_data_path
RUNTIME_ROOT = Path(os.environ.get("VAFX_AUDIO_RUNTIME_ROOT", user_data_path("vexy-audioflexy") / "audio" )).expanduser()
CACHE_ROOT = Path(os.environ.get("VAFX_AUDIO_CACHE_ROOT", user_cache_path("vexy-audioflexy") / "audio")).expanduser()
import json
import os
import subprocess
from functools import cache
from pathlib import Path

VERSION = "1.0.0"

REPO_URL = "https://github.com/0xShug0/audio.cpp.git"
# The engine lives in the sibling tts project; fx never builds its own copy.
ENGINE_ROOT = Path(
    os.environ.get("AUDIOCPP_ROOT", str(RUNTIME_ROOT / "engines/audio.cpp"))
).expanduser().resolve()
BIN_PATH = Path(os.environ.get("AUDIOCPP_BIN", str(ENGINE_ROOT / "build/macos-metal-release/bin/audiocpp_cli"))).expanduser().resolve()
MODELS_ROOT = Path((os.environ.get("AUDIOCPP_MODELS_ROOT") or str(model_path("audio.cpp")))).expanduser().resolve()
SPECS_DIR = ENGINE_ROOT / "model_specs"
BACKEND = os.environ.get("AUDIOCPP_BACKEND", "metal")

# Families whose loader wants the single GGUF file rather than its directory,
# per the `--model` arguments in their docs. MeanVC2's file form is confirmed by a
# successful run; see WORK.md for what has and has not been exercised.
FILE_MODEL_FAMILIES = frozenset({"audiosr", "meanvc2"})

# Package precision -> substrings that appear in the matching GGUF filename.
_PRECISION_ALIASES = {
    "f32": ("f32", "fp32"),
    "f16": ("f16", "fp16"),
    "q8_0": ("q8_0", "q8"),
    "q4_k": ("q4_k", "q4"),
}

# Default task per family when the caller does not name one.
DEFAULT_TASK = {
    "audiosr": "s2s",
    "meanvc2": "vc",
    "personaplex": "s2s",
    "seed_vc": "vc",
    "vevo2": "vc",
    "reuse": "s2s",
    "sidon": "s2s",
    "universr": "s2s",
    "apollo": "s2s",
    "tone_color_vc": "vc",
    "sam_audio": "s2s",
    "sheetsage2": "midi",
    "minimax_music3": "gen",
}


def _require_engine() -> Path:
    """Return the audiocpp_cli path, or exit with build instructions."""
    if BIN_PATH.exists():
        return BIN_PATH
    raise SystemExit(
        f"audiocpp_cli not built at {BIN_PATH}\n"
        f"Point AUDIOCPP_ROOT at an existing checkout, or build one with:\n"
        f"  git clone --depth 1 {REPO_URL} <root>\n"
        f"  cd <root> && ./scripts/build_metal.sh --target audiocpp_cli"
    )


@cache
def _specs() -> dict[str, dict]:
    """Map package id -> {family, target_directory, precision, tasks, ...}."""
    out: dict[str, dict] = {}
    for path in sorted(SPECS_DIR.glob("*.json")):
        spec = json.loads(path.read_text(encoding="utf-8"))
        for pkg in spec.get("packages", []):
            download = {**spec.get("package_defaults", {}).get("download", {}), **pkg.get("download", {})}
            out[pkg["id"]] = {
                "family": spec["family"],
                "target_directory": pkg.get("target_directory", ""),
                "precision": pkg.get("precision", ""),
                "format": pkg.get("format", ""),
                "tasks": spec.get("tasks", []),
                "languages": spec.get("languages", []),
                "files": pkg.get("files", []),
                "strip_prefix": pkg.get("strip_prefix", ""),
                "repo": download.get("repo", ""),
                "revision": download.get("revision", "main"),
                "download_kind": download.get("kind", ""),
            }
    return out


def _resolve(package: str) -> dict:
    entry = _specs().get(package)
    if entry is None:
        raise SystemExit(
            f"unknown audio.cpp package '{package}'. "
            f"List them with: {Path(__file__).name} packages"
        )
    return entry


def _model_path(package: str, family: str = "") -> Path:
    """Path passed as `--model`.

    Most loaders take the package directory. AudioSR and MeanVC2 take the single
    GGUF file inside it (their docs' `--model` arguments are file paths), so for
    those families resolve to the lone `.gguf` when there is exactly one.
    """
    entry = _resolve(package)
    directory = MODELS_ROOT / entry["target_directory"]
    files = entry["files"]
    if len(files) == 1 and files[0].endswith(".gguf"):
        relative = files[0]
        prefix = entry["strip_prefix"]
        if prefix and relative.startswith(prefix + "/"):
            relative = relative[len(prefix) + 1:]
        return directory / relative
    fam = family or entry["family"]
    ggufs = sorted(directory.glob("*.gguf"))

    def _by_precision() -> Path | None:
        """The GGUF whose filename carries this package's precision, if unique."""
        aliases = _PRECISION_ALIASES.get(entry["precision"], (entry["precision"],))
        matches = [g for g in ggufs if any(a in g.name.lower() for a in aliases)]
        return matches[0] if len(matches) == 1 else None

    if fam in FILE_MODEL_FAMILIES:
        if len(ggufs) == 1:
            return ggufs[0]
        # Several precisions can share one target directory (MeanVC2 f32 + q4_k);
        # pick the file whose name carries this package's precision.
        picked = _by_precision()
        if picked is not None:
            return picked
    elif len(ggufs) > 1:
        # The engine refuses a directory holding more than one GGUF whatever the
        # family — "pass one of them directly with --model" — and installing two
        # precisions of one model is enough to trigger it. PersonaPlex q4_k and
        # q8_0 share PersonaPlex-GGUF and hit exactly this.
        picked = _by_precision()
        if picked is not None:
            return picked
    return directory


def _installed(package: str) -> bool:
    """Require the package marker and every declared component."""
    entry = _resolve(package)
    directory = MODELS_ROOT / entry["target_directory"]
    marker = directory / f".audiocpp-package-{package}.json"
    prefix = entry.get("strip_prefix", "").rstrip("/") + "/"
    paths = [directory / (name[len(prefix):] if prefix != "/" and name.startswith(prefix) else name) for name in entry.get("files", [])]
    return marker.is_file() and bool(paths) and all(path.is_file() and path.stat().st_size > 0 for path in paths)


def prep(package: str = "meanvc2_120ms_40ms_f32", force: bool = False) -> None:
    """Download and cache one audio.cpp package. Idempotent.

    Args:
        package: audio.cpp package id (e.g. `audiosr_basic_f32`).
        force: Reinstall even if the target directory already holds the files.
    """
    _require_engine()
    target = MODELS_ROOT / _resolve(package)["target_directory"]
    # Several packages of one family share a target directory (MeanVC2 f32 and
    # q4_k), so "the directory holds a .gguf" is not proof that THIS package is
    # installed. model_manager_v2 drops a per-package marker; that is the test.
    if _installed(package) and not force:
        print(f"[prep] ready: {package} -> {target}")
        return
    _install_package(_resolve(package), MODELS_ROOT, package, force)
    if not _installed(package):
        raise SystemExit(f"[prep] {package}: installation remains incomplete; inspect existing files or use --force to replace them")
    print(f"[prep] ready: {package} -> {target}")


def run(
    audio: str = "",
    package: str = "meanvc2_120ms_40ms_f32",
    output: str = "output.wav",
    family: str = "",
    task: str = "",
    task_route: str = "",
    voice_ref: str = "",
    target_voice: str = "",
    prosody_ref: str = "",
    style_ref: str = "",
    text: str = "",
    target_text: str = "",
    style_ref_text: str = "",
    voice_id: str = "",
    num_inference_steps: int = 0,
    request_options: str = "",
    session_options: str = "",
    seed: int = 0,
    extra: str = "",
    play: bool = False,
    force: bool = False,
) -> None:
    """Process audio with one audio.cpp package: audio in, audio out.

    Args:
        audio: Source WAV — the speech to convert, enhance or answer. Required by
            every route except VeVo2's pure text-to-speech/singing routes.
        package: audio.cpp package id; picks both the family and the weights.
        output: Output path; SAM Audio writes target.wav and residual.wav in this directory.
        family: Override the family from the package spec (rarely needed).
        task: audio.cpp task — `vc`, `svc`, `s2s` or `tts`. Defaults per family.
        task_route: Route within the task (`v2_vc`, `style_preserved_vc`,
            `editing`, `zero_shot_tts`, ...). Left unset the engine picks its
            documented default for the task.
        voice_ref: Target-timbre WAV (`--voice-ref`).
        target_voice: Same thing under VeVo2's own flag (`--target-voice`);
            only one of the two is sent.
        prosody_ref: Prosody/melody reference WAV (VeVo2 singing routes).
        style_ref: Style reference WAV (VeVo2).
        text: Target text, or PersonaPlex's system/persona prompt.
        target_text: Target text/lyrics when separate from *text* (VeVo2).
        style_ref_text: Transcript of the style/voice reference (VeVo2).
        voice_id: Packaged voice prompt id (PersonaPlex `NATF2`, `NATM1`, ...),
            sent as a request option.
        num_inference_steps: Sampler steps for families that take the flag
            (Seed-VC, VeVo2); 0 leaves the model default.
        request_options: Per-request `key=value` settings separated by `;`
            (e.g. `guidance_scale=3.5;seed=42`).
        session_options: Per-session `key=value` settings separated by `;`.
        seed: Fixed seed; 0 leaves the model default.
        extra: Additional raw CLI arguments, space-separated.
        play: Play the result with afplay instead of reporting the path.
        force: Overwrite an existing output file.
    """
    from rich.console import Console
    from rich.panel import Panel

    console = Console()
    binary = _require_engine()
    # The engine runs with cwd=ENGINE_ROOT so it can read model_specs/, so every
    # path handed to it must be absolute or the output lands in the wrong tree.
    out_path = Path(output).resolve()

    if not force and out_path.exists() and (out_path.is_dir() or out_path.stat().st_size > 0):
        console.print(
            f"[yellow]Skipped:[/yellow] {output} already exists (use --force to overwrite)"
        )
        return

    entry = _resolve(package)
    fam = family or entry["family"]
    the_task = task or DEFAULT_TASK.get(fam, "vc")

    if not _installed(package):
        prep(package)
    model_arg = _model_path(package, fam)

    cmd = [
        str(binary),
        "--task", the_task,
        "--family", fam,
        "--model", str(model_arg),
        "--backend", BACKEND,
        "--out-dir" if fam == "sam_audio" else "--out", str(out_path),
        "--metrics",
    ]
    if audio:
        cmd += ["--audio", str(Path(audio).resolve())]
    if task_route:
        cmd += ["--task-route", task_route]
    if voice_ref:
        cmd += ["--voice-ref", str(Path(voice_ref).resolve())]
    elif target_voice:
        cmd += ["--target-voice", str(Path(target_voice).resolve())]
    if prosody_ref:
        cmd += ["--prosody-ref", str(Path(prosody_ref).resolve())]
    if style_ref:
        cmd += ["--style-ref", str(Path(style_ref).resolve())]
    if text:
        cmd += ["--text", text]
    if target_text:
        cmd += ["--target-text", target_text]
    if style_ref_text:
        cmd += ["--style-ref-text", style_ref_text]
    if voice_id:
        cmd += ["--request-option", f"voice_id={voice_id}"]
    if num_inference_steps:
        cmd += ["--num-inference-steps", str(num_inference_steps)]
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
            f"package=[yellow]{package}[/yellow]  task=[yellow]{the_task}[/yellow]"
            f"{'  route=[yellow]' + task_route + '[/yellow]' if task_route else ''}  "
            f"backend=[yellow]{BACKEND}[/yellow]",
            border_style="cyan",
        )
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ENGINE_ROOT), check=False)
    outputs = [out_path / "target.wav", out_path / "residual.wav"] if fam == "sam_audio" else [out_path]
    if result.returncode != 0 or not all(p.is_file() and p.stat().st_size > 0 for p in outputs):
        # ggml logs every Metal pipeline it compiles; keep only the decisive lines.
        noise = ("ggml_metal", "ggml_backend", "load_backend")
        lines = [
            ln
            for ln in (result.stdout + "\n" + result.stderr).splitlines()
            if ln.strip() and not ln.startswith(noise)
        ]
        console.print("[red]audiocpp_cli failed[/red]")
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
    if play and fam == "sam_audio":
        subprocess.run(["afplay", str(out_path / "target.wav")], check=True)
    elif play:
        subprocess.run(["afplay", str(out_path)], check=True)
    else:
        dur = float(metrics.get("metrics.audio_duration_ms", 0)) / 1000.0
        rtf = metrics.get("metrics.rtf", "?")
        console.print(
            f"[green]Saved[/green] [bold]{out_path.resolve()}[/bold] ({dur:.1f}s, rtf={rtf})"
        )


def info(package: str = "meanvc2_120ms_40ms_f32") -> None:
    """Print one-line JSON with package metadata."""
    entry = _resolve(package)
    print(
        json.dumps(
            {
                "name": package,
                "family": entry["family"],
                "repo": entry["repo"],
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


def packages(audio_only: bool = True) -> None:
    """List installable audio.cpp packages (id, family, precision).

    Args:
        audio_only: Show audio processing, music generation, and score packages.
    """
    for pkg, entry in sorted(_specs().items()):
        if audio_only and not (set(entry["tasks"]) & {"vc", "svc", "s2s", "midi", "gen", "music"}):
            continue
        print(f"{pkg:44} {entry['family']:22} {entry['precision']}")


def loaders(family: str = "") -> None:
    """Print the tasks the BUILT binary exposes — the authority over the docs."""
    binary = _require_engine()
    # Must run inside the engine checkout: the binary reads model_specs/ relative
    # to the working directory and aborts on the first family it cannot resolve.
    result = subprocess.run(
        [str(binary), "--list-loaders", "--json"],
        capture_output=True,
        text=True,
        cwd=str(ENGINE_ROOT),
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(result.stderr.strip() or "audiocpp_cli --list-loaders failed")
    data = json.loads(result.stdout)["loaders"]
    for name, spec in sorted(data.items()):
        if family and name != family:
            continue
        print(f"{name:24} {json.dumps(spec.get('tasks', {}))}")


if __name__ == "__main__":
    import fire

    fire.Fire({"prep": prep, "run": run, "info": info,
               "packages": packages, "loaders": loaders})
