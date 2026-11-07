#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = ["fire", "huggingface-hub", "platformdirs", "gguf==0.19.0"]
# ///
# this_file: src/vexy_audioflexy/tools/asr/tools/asr-transcribecpp.py
"""transcribe.cpp GGUF ASR, using its pinned upstream model catalog.

Set TRANSCRIBECPP_ROOT to an existing native checkout, or place it under
VAFX_RUNTIME_DIR/transcribe.cpp. Quantization is selected explicitly with --quant.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import os
import subprocess
import tempfile
import time
import wave
from pathlib import Path

from platformdirs import user_cache_path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ROOT = Path(os.environ.get("VAFX_RUNTIME_DIR", user_cache_path("vexy-audioflexy", appauthor=False) / "runtimes")).expanduser()
ENGINE_ROOT = Path(os.environ.get("TRANSCRIBECPP_ROOT", RUNTIME_ROOT / "transcribe.cpp")).expanduser()
CATALOG = ROOT / "catalogs/transcribecpp.json"
DEFAULT_PACKAGE = "moonshine-streaming-tiny"
SORTFORMER_FRAME_S = 1280 / 16000  # Published Sortformer frame_hop and sample rate.

LOCAL_ULTRA = {
    "package": "parakeet-ultra", "quant": "Q8_0", "size_bytes": 740363168,
    "sha256": "007a59761e9258779f189df396b50d12b83bb5b79e66df4b955c230d2f2a0a59",
    "repo": "tryambientai/parakeet-ultra-gguf",
    "metadata": {"general.architecture": "parakeet", "general.name": "Parakeet Ultra",
                 "stt.variant": "tdt-0.6b-ultra", "stt.parakeet.head_kind": "tdt",
                 "general.file_type": 7, "stt.frontend.sample_rate": 16000},
}

LOCAL_PHONON = {
    "package": "phonon-2", "quant": "F32", "size_bytes": 2508435648,
    "sha256": "f2b1a5563f0a4a764ca40f4fb72da6d3a4d4ce5653e0f8fbcc874a6a81c5c10e",
    "repo": "FermionResearch/Phonon-2", "source_revision": "7e153e4054c0a10db6d47d6fff5c062d4c77c154",
    "runtime_commit": "5700f127e38dd53c4735ea23ef743b4b538eacd4",
    "metadata": {"general.architecture": "parakeet", "general.name": "Phonon-2",
                 "stt.variant": "tdt-0.6b-phonon-2", "stt.parakeet.head_kind": "tdt",
                 "general.file_type": 0, "stt.frontend.sample_rate": 16000},
}


def local_model(path: str, package: str, quant: str) -> tuple[str, dict]:
    """Accept exact qualified local identities, never another runtime's same-suffix weights."""
    profile = {"parakeet-ultra": LOCAL_ULTRA, "phonon-2": LOCAL_PHONON}.get(package)
    if profile is None or quant.upper() != profile["quant"]:
        raise SystemExit("local_model_path requires --package parakeet-ultra --quant Q8_0 or --package phonon-2 --quant F32")
    model = Path(path).expanduser().resolve()
    if not model.is_file() or model.stat().st_size != profile["size_bytes"]:
        raise SystemExit("Qualified local model file is missing or has the wrong byte size")
    digest = hashlib.sha256()
    with model.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != profile["sha256"]:
        raise SystemExit("Qualified local model SHA256 differs from the qualified artifact")
    from gguf import GGUFReader

    reader = GGUFReader(model)
    for key, expected in profile["metadata"].items():
        field = reader.get_field(key)
        if field is None or field.contents() != expected:
            raise SystemExit(f"Qualified local model metadata mismatch: {key}")
    languages = reader.get_field("general.languages")
    if languages is None or not isinstance(languages.contents(), list):
        raise SystemExit("Qualified local model must declare supported languages")
    return str(model), {"languages": languages.contents(), "local_profile": profile}


def packages() -> dict[str, dict]:
    """Read supported transcription variants; diarization-only models are excluded."""
    return {row["variant"]: row for row in json.loads(CATALOG.read_text())["models"]}


def resolve(package: str, quant: str = "Q8_0") -> tuple[dict, dict]:
    """Resolve a variant and an exact published quantization before downloading."""
    spec = packages().get(package)
    if spec is None:
        raise SystemExit(f"Unknown transcribe.cpp package: {package}")
    download = next((row for row in spec["downloads"] if row["quant"] == quant.upper()), None)
    if download is None:
        raise SystemExit(f"{package} has no {quant} weights; available: {[r['quant'] for r in spec['downloads']]}")
    return spec, download


def language_hint(languages: list[str], lang: str) -> str:
    """Map a fixture's base language to a declared locale without inventing support."""
    if not lang or lang in languages:
        return lang
    match = next((code for code in languages if code.split("-")[0] == lang), None)
    if match is None:
        raise SystemExit(f"Model does not declare language {lang}")
    return match


def _verify_published(path: str | Path, artifact: dict) -> str:
    """Read cached provider bytes without modifying them; readiness requires exact identity."""
    model = Path(path)
    if not model.is_file() or model.stat().st_size != artifact["size_bytes"]:
        raise SystemExit("Weight size disagrees with the pinned catalog")
    expected = artifact.get("published_sha256")
    if not expected:
        raise SystemExit("Published artifact must declare its SHA256 identity")
    digest = hashlib.sha256()
    with model.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise SystemExit("Weight SHA256 disagrees with the pinned provider")
    return str(model)


def prep(package: str = DEFAULT_PACKAGE, quant: str = "Q8_0", local_model_path: str = "") -> str:
    """Verify an explicit qualified local model, or cache one exact provider artifact."""
    if local_model_path:
        path, _ = local_model(local_model_path, package, quant)
        print(f"[prep] ready: {package}/{quant} -> {path}")
        return path
    from huggingface_hub import hf_hub_download

    spec, download = resolve(package, quant)
    if spec.get("availability") == "repository-unavailable":
        raise SystemExit(spec["unavailable_reason"])
    path = hf_hub_download(repo_id=spec["published_repo"], filename=download["filename"],
                           revision=spec.get("published_revision"))
    path = _verify_published(path, download)
    print(f"[prep] ready: {package}/{quant} -> {path}")
    return path


def command(model: str, audio: str, output: str, lang: str = "", backend: str = "metal", threads: int = 0) -> list[str]:
    """Use the CLI's file output contract; stdout also contains timing logs."""
    argv = [str(ENGINE_ROOT / "build/bin/transcribe-cli"), "-m", model,
            "--backend", backend, "--timestamps", "none", "-o", output]
    if threads:
        argv += ["--threads", str(threads)]
    if lang:
        argv += ["--language", lang]
    return argv + [audio]


def transcribe(
    audio: str,
    package: str = DEFAULT_PACKAGE,
    text_out: str = "",
    quant: str = "Q8_0",
    lang: str = "",
    backend: str = "metal",
    timeout: int = 900,
    force: bool = False,
    verbose: bool = False,
    local_model_path: str = "",
    threads: int = 0,
) -> None:
    """Resample audio, run inference, and emit the ASR harness result contract.

    local_model_path accepts exact Ultra Q8_0 or a qualified Phonon-2 F32 conversion.
    Phonon uses the official converter at catalog_commit with its frozen
    scripts/envs/parakeet-phonon environment; select --package phonon-2 --quant F32.
    """
    src = Path(audio).resolve()
    if not src.is_file():
        raise SystemExit(f"Audio not found: {src}")
    if threads < 0:
        raise SystemExit("threads must be nonnegative")
    if local_model_path:
        model, spec = local_model(local_model_path, package, quant)
        profile = spec.get("local_profile", LOCAL_ULTRA)
    else:
        spec, _ = resolve(package, quant)
    hint = language_hint(spec["languages"], lang)
    binary = ENGINE_ROOT / "build/bin/transcribe-cli"
    if not binary.is_file():
        raise SystemExit(f"Build transcribe.cpp first: cmake -S {ENGINE_ROOT} -B {ENGINE_ROOT}/build && cmake --build {ENGINE_ROOT}/build -j 6")
    if timeout <= 0:
        raise SystemExit("timeout must be positive")
    if not local_model_path and spec.get("runtime_commit"):
        _require_revision(spec["runtime_commit"], timeout)
    out = Path(text_out).resolve() if text_out else src.with_suffix(".txt")
    if out == src:
        raise SystemExit("Transcript output must differ from audio input")
    if out.exists() and not force:
        raise SystemExit(f"Transcript exists: {out}; use --force")
    if local_model_path:
        actual = subprocess.check_output(["git", "-C", str(ENGINE_ROOT), "rev-parse", "HEAD"],
                                         text=True, timeout=min(timeout, 30)).strip()
        catalog = json.loads(CATALOG.read_text())
        allowed = {profile["runtime_commit"]} if "runtime_commit" in profile else {catalog["commit"], catalog.get("catalog_commit", catalog["commit"])}
        if actual not in allowed:
            raise SystemExit("Local model requires the pinned transcribe.cpp runtime commit")
    else:
        model = prep(package, quant)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        wav, transcript = Path(tmp) / "input.wav", Path(tmp) / "transcript.txt"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
                        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(wav)],
                       check=True, timeout=timeout)
        with wave.open(str(wav), "rb") as wf:
            duration = wf.getnframes() / wf.getframerate()
        if duration <= 0:
            raise SystemExit("Audio is empty")
        start = time.monotonic()
        result = subprocess.run(command(model, str(wav), str(transcript), hint, backend, threads),
                                capture_output=True, text=True, timeout=timeout, cwd=ENGINE_ROOT, check=False)
        wall = time.monotonic() - start
        if verbose or result.returncode:
            print(result.stdout + result.stderr)
        if result.returncode:
            raise SystemExit(result.returncode)
        text = transcript.read_text(encoding="utf-8").strip()
        if not text:
            raise SystemExit("transcribe.cpp returned an empty transcript")
        out.write_text(text + "\n", encoding="utf-8")
    print("asr.result=" + json.dumps({"text": text, "wall_s": wall,
          "audio_duration_s": duration, "rtf": wall / duration,
          "package": package, "quant": quant, "backend": backend,
          **({"model_sha256": profile["sha256"], "model_repo": profile["repo"]} if local_model_path else {})}, ensure_ascii=False))


def roles() -> dict[str, dict]:
    """List non-ASR model roles separately from transcription variants."""
    return {row["variant"]: row for row in json.loads(CATALOG.read_text())["roles"]}


def _require_revision(expected: str, timeout: int) -> None:
    actual = subprocess.check_output(["git", "-C", str(ENGINE_ROOT), "rev-parse", "HEAD"],
                                     text=True, timeout=min(timeout, 30)).strip()
    if actual != expected:
        raise SystemExit(f"This published model/role requires transcribe.cpp commit {expected}")


def _role_spec(package: str, quant: str) -> tuple[dict, dict]:
    spec = roles().get(package)
    if spec is None:
        raise SystemExit(f"Unknown non-ASR package: {package}")
    quant = quant.upper() or ("Q8_0" if any(r["quant"] == "Q8_0" for r in spec["downloads"]) else "F32")
    artifact = next((r for r in spec["downloads"] if r["quant"] == quant), None)
    if artifact is None:
        raise SystemExit(f"{package} has no {quant} weights")
    return spec, artifact


def prep_role(package: str, quant: str = "") -> str:
    """Cache one exact non-ASR artifact through its immutable provider revision."""
    from huggingface_hub import hf_hub_download

    spec, artifact = _role_spec(package, quant)
    path = Path(hf_hub_download(spec["published_repo"], artifact["filename"], revision=spec["published_revision"]))
    return _verify_published(path, artifact)


def role_result(role: str, text: str, duration: float) -> dict:
    """Parse only the stable native role records; empty VAD/diarization is valid."""
    if role not in {"langid", "vad", "diarize"}:
        raise ValueError("Unsupported non-ASR role")
    records = []
    for line in text.splitlines():
        if role == "langid":
            match = re.fullmatch(r"candidate: (\d+) (\S+) index=(\d+) p=(\S+) logit=(\S+)", line)
            if not match:
                raise ValueError("Malformed native language candidate")
            rank, code, index, probability, logit = match.groups()
            p, z = float(probability), float(logit)
            if int(rank) != len(records) + 1 or not math.isfinite(p) or not 0 <= p <= 1 or not math.isfinite(z):
                raise ValueError("Invalid native language probabilities/rank")
            records.append({"rank": int(rank), "language": code, "index": int(index), "probability": p, "logit": z})
        elif role == "vad":
            match = re.fullmatch(r"segment: (\d+) start=(\d+) end=(\d+) t0=(\S+) t1=(\S+)", line)
            if not match:
                raise ValueError("Malformed native speech segment")
            index, start, end, t0, t1 = match.groups()
            start, end = int(start), int(end)
            if int(index) != len(records) or not 0 <= start < end <= round(duration * 16000):
                raise ValueError("Invalid native speech sample boundaries")
            if not all(math.isfinite(float(v)) and abs(float(v) - n / 16000) <= .001 for v, n in [(t0, start), (t1, end)]):
                raise ValueError("Native speech seconds disagree with sample boundaries")
            records.append({"start_sample": start, "end_sample": end, "start": start / 16000, "end": end / 16000})
        elif role == "diarize":
            match = re.fullmatch(r"\[\s*(\S+) ->\s*(\S+)\] S(\d+)", line)
            if not match:
                raise ValueError("Malformed native speaker turn")
            start, end, speaker = match.groups()
            start, end = float(start), float(end)
            if not math.isfinite(start) or not math.isfinite(end) or not 0 <= start < end or start >= duration or end > math.ceil(duration / SORTFORMER_FRAME_S) * SORTFORMER_FRAME_S + .005:
                raise ValueError("Invalid native speaker turn boundaries")
            records.append({"start": start, "end": min(end, duration), "speaker": speaker, "native_end": end})
        else:
            raise ValueError("Unsupported non-ASR role")
    if role == "langid" and not records:
        raise ValueError("Language ID returned no candidates")
    return {{"langid": "candidates", "vad": "segments", "diarize": "turns"}[role]: records}


def role(audio: str, package: str, output: str, quant: str = "", backend: str = "cpu",
         timeout: int = 900, threads: int = 4, force: bool = False, verbose: bool = False) -> None:
    """Run a model's explicit non-ASR role and save structured JSON, never ASR text."""
    source, destination = Path(audio).expanduser().resolve(), Path(output).expanduser().resolve()
    spec, artifact = _role_spec(package, quant)
    binary = ENGINE_ROOT / "build/bin/transcribe-cli"
    if not source.is_file() or source == destination or timeout <= 0 or threads <= 0:
        raise ValueError("Require audio, distinct output, positive timeout and threads")
    if destination.exists() and not force:
        raise FileExistsError("Role output exists; use --force")
    if not binary.is_file():
        raise SystemExit(f"Build transcribe.cpp at {spec['runtime_commit']} in {ENGINE_ROOT}")
    _require_revision(spec["runtime_commit"], timeout)
    model = prep_role(package, artifact["quant"])
    with tempfile.TemporaryDirectory() as temporary:
        wav, native = Path(temporary) / "input.wav", Path(temporary) / "native.txt"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(source), "-ar", "16000",
                        "-ac", "1", "-c:a", "pcm_s16le", str(wav)], check=True, timeout=timeout)
        with wave.open(str(wav), "rb") as stream:
            duration = stream.getnframes() / stream.getframerate()
        if duration <= 0:
            raise ValueError("Input audio is empty")
        argv = [str(binary), "-m", model, "--backend", backend, "--threads", str(threads), "-o", str(native), str(wav)]
        start = time.monotonic()
        result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
        wall = time.monotonic() - start
        if verbose or result.returncode:
            print(result.stdout + result.stderr)
        if result.returncode:
            raise SystemExit(result.returncode)
        raw = native.read_text(encoding="utf-8")
        data = {"role": spec["role"], "package": package, "quant": artifact["quant"], "backend": backend,
                "duration_s": duration, "wall_s": wall, "native_output": raw,
                **role_result(spec["role"], raw, duration)}
        if spec["role"] == "langid":
            data["scored_audio_duration_s"] = min(duration, 30.0)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print({"diarize": "diar", "langid": "langid", "vad": "vad"}[spec["role"]] + ".result=" + json.dumps(data))


def info() -> None:
    """Print the catalog provenance and engine location."""
    data = json.loads(CATALOG.read_text())
    print(json.dumps({"source": data["source"], "commit": data["commit"],
                      "catalog_commit": data.get("catalog_commit", data["commit"]),
                      "local_profiles": {"parakeet-ultra": LOCAL_ULTRA, "phonon-2": LOCAL_PHONON},
                      "engine": str(ENGINE_ROOT), "variants": len(data["models"]), "roles": len(data.get("roles", []))}, indent=2))


if __name__ == "__main__":
    import fire

    fire.Fire({"prep": prep, "transcribe": transcribe, "packages": packages, "info": info, "roles": roles, "prep_role": prep_role, "role": role})
