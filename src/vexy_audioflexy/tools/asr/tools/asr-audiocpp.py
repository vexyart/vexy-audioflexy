#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = ["fire", "rich", "platformdirs", "huggingface-hub"]
# ///
# this_file: src/vexy_audioflexy/tools/asr/tools/asr-audiocpp.py
"""Generic driver for every audio.cpp ASR and forced-alignment family.

audio.cpp is a single pure-C++ ggml engine: one `audiocpp_cli` binary dispatches to
~66 model loaders via `--family`, so all of its ASR families share this one script
instead of one wrapper per model. Backend: metal (Apple Silicon GPU).

Model specifications select exact GGUF/safetensors files from Hugging Face
snapshots. Downloads stay in the provider cache; native model layouts link to
those files without copying model weights.

Audio is resampled to 16 kHz mono before inference: audio.cpp's offline VAD chunker
is a Silero 16k model and hard-fails on anything else, so a 24 kHz clip from the
speaker dataset is rejected outright without this step.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
import wave
from functools import cache
from pathlib import Path

from _audioflexy_paths import runtime_root, models_root

VERSION = "1.0.0"

# audio.cpp checkout: shared read-only with the sibling tts project.
REPO_URL = "https://github.com/0xShug0/audio.cpp.git"
ENGINE_ROOT = Path(
    os.environ.get(
        "AUDIOCPP_ROOT",
        str(runtime_root("audio.cpp")),
    )
).expanduser().resolve()
BIN_PATH = Path(os.environ.get("AUDIOCPP_BIN", str(ENGINE_ROOT / "build/macos-metal-release/bin/audiocpp_cli"))).expanduser().resolve()
MODELS_ROOT = Path(os.environ.get("AUDIOCPP_MODELS_ROOT", str(models_root() / "audio.cpp"))).expanduser().resolve()
SPECS_DIR = Path(os.environ.get("AUDIOCPP_SPECS_DIR", str(ENGINE_ROOT / "model_specs"))).expanduser().resolve()
EXTRA_SPECS_DIR = Path(__file__).resolve().parent.parent / "model_specs"
BACKEND = os.environ.get("AUDIOCPP_BACKEND", "metal")

# audio.cpp's Silero VAD chunker accepts exactly this rate; every ASR family here
# expects 16 kHz mono PCM anyway.
TARGET_RATE = 16000


def _require_engine() -> Path:
    """Return the audiocpp_cli path, or exit with build instructions."""
    if BIN_PATH.exists():
        return BIN_PATH
    raise SystemExit(
        f"audiocpp_cli not built at {BIN_PATH}\n"
        f"Set AUDIOCPP_ROOT to an existing checkout, or build one:\n"
        f"  git clone --depth 1 {REPO_URL} {ENGINE_ROOT}\n"
        f"  cd {ENGINE_ROOT} && ./scripts/build_metal.sh --target audiocpp_cli"
    )


@cache
def _specs() -> dict[str, dict]:
    """Map package id -> {family, target_directory, precision, ...}."""
    out: dict[str, dict] = {}
    for path in [*sorted(SPECS_DIR.glob("*.json")), *sorted(EXTRA_SPECS_DIR.glob("*.json"))]:
        spec = json.loads(path.read_text(encoding="utf-8"))
        default_dl = spec.get("package_defaults", {}).get("download", {})
        for pkg in spec.get("packages", []):
            download = {**default_dl, **pkg.get("download", {})}
            out[pkg["id"]] = {
                "family": spec["family"],
                "target_directory": pkg.get("target_directory", ""),
                "precision": pkg.get("precision", ""),
                "format": pkg.get("format", ""),
                "tasks": spec.get("tasks", []),
                "languages": spec.get("languages", []),
                "download_kind": download.get("kind", ""),
                "repo": download.get("repo", ""),
                "revision": download.get("revision", "main"),
                "files": pkg.get("files", []),
                "strip_prefix": pkg.get("strip_prefix", ""),
                "specs_dir": str(path.parent),
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


def _package_root(package: str) -> Path:
    return MODELS_ROOT / _resolve(package)["target_directory"]


def _model_path(package: str) -> Path:
    """Select the declared GGUF when a package contains a single model file."""
    entry = _resolve(package)
    root = _package_root(package)
    files = entry["files"]
    if len(files) == 1 and files[0].endswith(".gguf"):
        relative = files[0]
        prefix = entry["strip_prefix"]
        if prefix and relative.startswith(prefix + "/"):
            relative = relative[len(prefix) + 1:]
        return root / relative
    return root


def _installed(package: str) -> bool:
    """Require every declared component, not just a sibling's weights."""
    entry = _resolve(package)
    root = MODELS_ROOT / entry["target_directory"]
    files = entry.get("files", [])
    if files:
        prefix = entry.get("strip_prefix", "").rstrip("/") + "/"
        paths = [root / (name[len(prefix):] if prefix != "/" and name.startswith(prefix) else name) for name in files]
        return all(path.is_file() and path.stat().st_size > 0 for path in paths)
    target = _model_path(package)
    if target.is_file():
        return target.stat().st_size > 0
    if not target.is_dir():
        return False
    return any(
        p.suffix in {".gguf", ".safetensors"} for p in target.rglob("*") if p.is_file()
    )


def _safe_relative(value: str) -> Path:
    """Reject model specifications that could write outside the requested layout."""
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value or value == ".":
        raise ValueError(f"Unsafe relative model path: {value}")
    return path


def _link_snapshot(entry: dict, snapshot: Path, target: Path, force: bool) -> None:
    """Expose exact provider files in native layout without duplicating model weights."""
    plan = []
    prefix = entry.get("strip_prefix", "").rstrip("/")
    for remote in entry["files"]:
        if prefix and not remote.startswith(prefix + "/"):
            raise ValueError(f"File does not match strip_prefix: {remote}")
        relative = _safe_relative(remote[len(prefix) + 1:] if prefix else remote)
        source, output = snapshot / _safe_relative(remote), target / relative
        if not source.is_file() or source.stat().st_size == 0:
            raise SystemExit(f"Provider snapshot is missing declared file: {remote}")
        if not output.parent.resolve().is_relative_to(target.resolve()):
            raise ValueError(f"Native layout escapes its target: {output}")
        exists = output.exists() or output.is_symlink()
        same = output.is_file() and output.samefile(source)
        if exists and not same and (not force or output.is_dir()):
            raise FileExistsError(f"Model path differs from pinned provider file: {output}; use --force")
        plan.append((source, output, exists, same))
    for source, output, exists, same in plan:
        if same:
            continue
        output.parent.mkdir(parents=True, exist_ok=True)
        if exists:
            output.unlink()
        output.symlink_to(source.resolve())


def prep(package: str = "parakeet_tdt_q8_0", force: bool = False) -> None:
    """Download and cache one audio.cpp package. Idempotent.

    Args:
        package: audio.cpp package id (e.g. `sensevoice_small_q8`).
        force: Reinstall even if the target directory already holds the files.
    """
    _require_engine()
    entry = _resolve(package)
    if entry["download_kind"] != "huggingface_snapshot":
        raise SystemExit(
            f"package '{package}' has download kind '{entry['download_kind']}' — "
            f"prep only installs huggingface_snapshot packages. "
            f"Pick another package for family '{entry['family']}'."
        )
    target = _package_root(package)
    if _installed(package) and not force:
        print(f"[prep] ready: {package} -> {target}")
        return
    from huggingface_hub import snapshot_download

    files = entry.get("files", [])
    if not entry.get("repo") or not files:
        raise SystemExit("HF packages must declare a repository and exact files")
    _safe_relative(entry["target_directory"])
    for name in files:
        _safe_relative(name)
    snapshot = Path(snapshot_download(entry["repo"], revision=entry.get("revision", "main"),
                                     allow_patterns=files))
    _link_snapshot(entry, snapshot, target, force)
    if not _installed(package):
        raise SystemExit(f"[prep] {package}: installation remains incomplete; inspect existing files or use --force to replace them")
    print(f"[prep] ready: {package} -> {target}")


def _wav_info(path: Path) -> tuple[int, int, float]:
    """(sample_rate, channels, duration_s) of a PCM WAV, or (0, 0, 0.0)."""
    try:
        with wave.open(str(path), "rb") as wf:
            return wf.getframerate(), wf.getnchannels(), wf.getnframes() / wf.getframerate()
    except (OSError, EOFError, wave.Error, ZeroDivisionError):
        return 0, 0, 0.0


def _ensure_16k(audio: Path, tmpdir: str) -> Path:
    """Return *audio* itself when already 16 kHz mono, else an ffmpeg-converted copy."""
    rate, channels, _ = _wav_info(audio)
    if rate == TARGET_RATE and channels == 1:
        return audio
    if shutil.which("ffmpeg") is None:
        raise SystemExit(
            f"{audio} is {rate} Hz / {channels}ch; audio.cpp needs {TARGET_RATE} Hz mono "
            f"and ffmpeg is not installed to convert it."
        )
    converted = Path(tmpdir) / "input16k.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(audio),
         "-ar", str(TARGET_RATE), "-ac", "1", str(converted)],
        check=True,
    )
    return converted


def _parse_metrics(stdout: str) -> dict[str, float]:
    """Pull `metrics.key=value` lines emitted by `--metrics` into a float dict."""
    out: dict[str, float] = {}
    for line in stdout.splitlines():
        if not line.startswith("metrics."):
            continue
        key, _, value = line.partition("=")
        try:
            out[key[len("metrics."):]] = float(value)
        except ValueError:
            pass
    return out


def _clean_engine_log(stdout: str, stderr: str) -> list[str]:
    """Drop the ggml Metal pipeline noise, keep the decisive lines."""
    noise = ("ggml_metal", "ggml_backend", "load_backend")
    return [
        ln
        for ln in (stdout + "\n" + stderr).splitlines()
        if ln.strip() and not ln.startswith(noise)
    ]


def _run_engine(cmd: list[str], console) -> tuple[str, str, int]:
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ENGINE_ROOT), check=False)
    if result.returncode != 0:
        console.print("[red]audiocpp_cli failed[/red]")
        console.print("\n".join(_clean_engine_log(result.stdout, result.stderr)[-12:]))
    return result.stdout, result.stderr, result.returncode


def transcribe(
    audio: str,
    package: str = "parakeet_tdt_q8_0",
    text_out: str = "",
    family: str = "",
    lang: str = "",
    text: str = "",
    segments_out: str = "",
    words_out: str = "",
    mode: str = "offline",
    request_options: str = "",
    session_options: str = "",
    resample: bool = True,
    extra: str = "",
    force: bool = False,
) -> None:
    """Transcribe one audio file with an audio.cpp ASR family.

    Args:
        audio: Input audio file (any rate; resampled to 16 kHz mono unless disabled).
        package: audio.cpp package id; picks both the family and the weight files.
        text_out: Transcript output path (default: *audio* with a `.txt` suffix).
        family: Override the family from the package spec (rarely needed).
        lang: Language code passed as `--language`.
        text: Instruction prompt for families that take one, e.g. higgs_audio_stt
            wants `--text "Transcribe the speech."`.
        segments_out: Write utterance segments as JSON (families that emit them).
        words_out: Write word timings as JSON. Not every family accepts this —
            parakeet_tdt rejects the underlying `return_timestamps` option.
        mode: `offline` or `streaming`. Streaming adds a default 5 s chunk size
            unless `audio_chunk_seconds` is already in *request_options*.
        request_options: Per-request `key=value` settings separated by `;`.
        session_options: Per-session `key=value` settings separated by `;`.
        resample: Convert the input to 16 kHz mono first (required by the VAD chunker).
        extra: Additional raw CLI arguments, space-separated.
        force: Overwrite an existing transcript file.
    """
    from rich.console import Console
    from rich.panel import Panel

    console = Console()
    binary = _require_engine()
    src = Path(audio).resolve()
    if not src.exists():
        raise SystemExit(f"audio not found: {src}")

    out_path = Path(text_out).resolve() if text_out else src.with_suffix(".txt")
    if not force and out_path.exists() and out_path.stat().st_size > 0:
        console.print(
            f"[yellow]Skipped:[/yellow] {out_path} already exists (use --force to overwrite)"
        )
        return

    entry = _resolve(package)
    fam = family or entry["family"]
    if not _installed(package):
        prep(package)

    console.print(
        Panel.fit(
            f"[bold cyan]audio.cpp ASR[/bold cyan] v{VERSION}  family=[yellow]{fam}[/yellow]\n"
            f"package=[yellow]{package}[/yellow]  mode=[yellow]{mode}[/yellow]  "
            f"backend=[yellow]{BACKEND}[/yellow]",
            border_style="cyan",
        )
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        wav = _ensure_16k(src, tmpdir) if resample else src
        cmd = [
            str(binary),
            "--task", "asr",
            "--family", fam,
            "--model", str(_model_path(package)),
            "--model-spec-override", str(SPECS_DIR),
            "--backend", BACKEND,
            "--audio", str(wav),
            "--text-out", str(out_path),
        ]
        streaming = bool(mode) and mode != "offline"
        if not streaming:
            # `--metrics currently supports offline mode only`; in streaming mode the
            # wrapper's own wall clock is the only timing available.
            cmd.append("--metrics")
        if streaming:
            cmd += ["--mode", mode]
            if "audio_chunk_seconds" not in request_options:
                request_options = ";".join(
                    p for p in (request_options, "audio_chunk_seconds=5") if p
                )
        if lang and fam != "parakeet_tdt":
            cmd += ["--language", lang]
        if text:
            cmd += ["--text", text]
        if segments_out:
            cmd += ["--segments-out", segments_out]
        if words_out:
            cmd += ["--words-out", words_out]
        for flag, blob in (("--request-option", request_options),
                           ("--session-option", session_options)):
            for setting in (part.strip() for part in blob.split(";") if part.strip()):
                cmd += [flag, setting]
        if extra:
            cmd += extra.split()

        t0 = time.monotonic()
        stdout, _stderr, code = _run_engine(cmd, console)
        wall = time.monotonic() - t0

    if code != 0:
        raise SystemExit(1)
    if not out_path.exists():
        console.print(f"[red]no transcript written to {out_path}[/red]")
        raise SystemExit(1)

    transcript = out_path.read_text(encoding="utf-8").strip()
    if not transcript:
        # tts hit exactly this: a run that exits 0 and produces nothing.
        console.print(f"[red]empty transcript[/red] — {package} returned no text")
        raise SystemExit(1)

    metrics = _parse_metrics(stdout)
    audio_s = metrics.get("audio_duration_ms", 0.0) / 1000.0 or _wav_info(src)[2]
    rtf = metrics.get("rtf") or (wall / audio_s if audio_s else 0.0)
    console.print(f"[green]Transcript[/green] [bold]{out_path.resolve()}[/bold]")
    console.print(f"  {transcript}")
    # Machine-readable tail line consumed by asr_all.py.
    print("asr.result=" + json.dumps(
        {
            "text": transcript,
            "text_out": str(out_path),
            "wall_s": round(wall, 4),
            "audio_duration_s": round(audio_s, 4),
            "rtf": round(rtf, 6),
            "package": package,
            "family": fam,
        },
        ensure_ascii=False, separators=(",", ":"),
    ))


def align(
    audio: str,
    text: str,
    words_out: str = "",
    package: str = "mms_forced_aligner_300m_safetensors",
    family: str = "",
    lang: str = "",
    resample: bool = True,
    request_options: str = "",
    session_options: str = "",
    extra: str = "",
    force: bool = False,
) -> None:
    """Forced-align a known transcript to audio, emitting word timings as JSON.

    Args:
        audio: Input audio file.
        text: The exact reference transcript to align.
        words_out: Word-timing JSON output path (default: *audio* with `.words.json`).
        package: Aligner package id. The MMS GGUF packages carry download kind
            `unsupported`, so the safetensors package is the working default.
        family: Override the family from the package spec.
        lang: ISO-639-3 language code, e.g. `eng`, `pol`, `nld`.
        resample: Convert the input to 16 kHz mono first.
        request_options: Per-request `key=value` settings separated by `;`.
        session_options: Per-session `key=value` settings separated by `;`.
        extra: Additional raw CLI arguments, space-separated.
        force: Overwrite an existing output file.
    """
    from rich.console import Console

    console = Console()
    binary = _require_engine()
    src = Path(audio).resolve()
    if not src.exists():
        raise SystemExit(f"audio not found: {src}")

    out_path = Path(words_out).resolve() if words_out else src.with_suffix(".words.json")
    if not force and out_path.exists() and out_path.stat().st_size > 0:
        console.print(f"[yellow]Skipped:[/yellow] {out_path} already exists")
        return

    entry = _resolve(package)
    fam = family or entry["family"]
    if not _installed(package):
        prep(package)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        wav = _ensure_16k(src, tmpdir) if resample else src
        cmd = [
            str(binary),
            "--task", "align",
            "--family", fam,
            "--model", str(_model_path(package)),
            "--backend", BACKEND,
            "--audio", str(wav),
            "--text", text,
            "--words-out", str(out_path),
            "--metrics",
        ]
        if lang:
            cmd += ["--language", lang]
        for flag, blob in (("--request-option", request_options),
                           ("--session-option", session_options)):
            for setting in (part.strip() for part in blob.split(";") if part.strip()):
                cmd += [flag, setting]
        if extra:
            cmd += extra.split()
        _stdout, _stderr, code = _run_engine(cmd, console)

    if code != 0 or not out_path.exists():
        raise SystemExit(1)
    words = json.loads(out_path.read_text(encoding="utf-8"))
    if not isinstance(words, list):
        words = words.get("words", [])
    if not words:
        console.print("[red]alignment produced no words[/red]")
        raise SystemExit(1)
    # The engine reports sample offsets against the 16 kHz input; seconds are what
    # every downstream consumer actually wants, so add them rather than make each
    # caller rediscover the rate.
    for w in words:
        if "start_sample" in w:
            w["start_s"] = round(w["start_sample"] / TARGET_RATE, 4)
            w["end_s"] = round(w["end_sample"] / TARGET_RATE, 4)
    out_path.write_text(
        json.dumps(words, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    console.print(
        f"[green]Aligned[/green] {len(words)} word(s) "
        f"({words[0].get('start_s', 0):.2f}s–{words[-1].get('end_s', 0):.2f}s) "
        f"→ [bold]{out_path.resolve()}[/bold]"
    )


def info(package: str = "parakeet_tdt_q8_0") -> None:
    """Print one-line JSON with package metadata."""
    entry = _resolve(package)
    print(json.dumps(
        {
            "name": package,
            "family": entry["family"],
            "repo": entry["repo"] or "audio-cpp/audio.cpp-gguf",
            "backend": BACKEND,
            "stack": "audiocpp",
            "quant": entry["precision"],
            "tasks": entry["tasks"],
            "languages": entry["languages"],
            "installable": entry["download_kind"] == "huggingface_snapshot",
            "installed": _installed(package),
            "model_path": str(_model_path(package)),
        },
        separators=(",", ":"),
    ))


def packages(asr_only: bool = True) -> None:
    """List audio.cpp packages (id, family, precision, installable)."""
    for pkg, entry in sorted(_specs().items()):
        if asr_only and not (set(entry["tasks"]) & {"asr", "align"}):
            continue
        flag = "" if entry["download_kind"] == "huggingface_snapshot" else "  [UNSUPPORTED DOWNLOAD]"
        print(f"{pkg:40} {entry['family']:22} {entry['precision']:10}{flag}")


if __name__ == "__main__":
    import fire

    fire.Fire({
        "prep": prep,
        "transcribe": transcribe,
        "align": align,
        "info": info,
        "packages": packages,
    })
