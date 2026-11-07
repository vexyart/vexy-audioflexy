#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = ["fire", "rich", "huggingface-hub"]
# ///
# this_file: src/vexy_audioflexy/tools/tts/tools/tts-crispasr-gguf.py
"""GGUF TTS models driven by the external CrispASR binary.

Backend: gguf (llama.cpp-style ggml runtime via the CrispASR CLI wrapper).

  dia               cstr/dia-1.6b-GGUF — dialogue TTS, [S1]/[S2] tags, ~44 kHz,
                    auto mode, no voice cloning
  pocket            cstr/pocket-tts-GGUF — lightweight English TTS, ~24 kHz,
                    ivc mode, cloning through --ref
  chatterbox-nano   cstr/chatterbox-nano-GGUF — GPT-2 small T3 (12L/768D) paired
                    with the Turbo S3Gen decoder, ~24 kHz, ivc mode

All English. prep() downloads weights via huggingface_hub; say() delegates
synthesis to the CrispASR binary, which must be built separately from
github.com/CrispStrobe/CrispASR.

Voice cloning through CrispASR is gated behind two operator attestations,
`--i-have-rights` (consent of the cloned speaker) and, when the audible
AI-disclosure prefix is suppressed, `--accept-marking-responsibility`. The
prefix is suppressed by default here because it would otherwise be baked into
every benchmark clip; pass `--disclaimer` to keep it.
"""

from pathlib import Path
import os
import shutil
import subprocess
from _paths import runtime_path

VERSION = "1.0.0"
DIA_REPO = "cstr/dia-1.6b-GGUF"
POCKET_REPO = "cstr/pocket-tts-GGUF"
NANO_REPO = "cstr/chatterbox-nano-GGUF"

# Repo root is two levels up from tools/; the CrispASR binary is built under
# .external/CrispASR/build/bin/ (see github.com/CrispStrobe/CrispASR).
ROOT = Path(__file__).resolve().parent.parent
_EXTERNAL_CRISPASR = runtime_path('CrispASR/build/bin')

MODELS = {
    "dia": {
        "repo": DIA_REPO,
        "backend": "dia",
        "auto_key": "dia",
        "default_file": "dia-1.6b-f16.gguf",
        "codec_file": "dac-44khz.gguf",
    },
    "pocket": {
        "repo": POCKET_REPO,
        "backend": "pocket-tts",
        "auto_key": "pocket-tts",
        "default_file": "pocket-tts-english-novc-f16.gguf",
        "codec_file": None,
    },
    "chatterbox-nano": {
        "repo": NANO_REPO,
        "backend": "chatterbox-nano",
        "auto_key": "chatterbox-nano",
        "default_file": "chatterbox-nano-t3-q8_0.gguf",
        # The S3Gen decoder is byte-identical to Chatterbox-Turbo's; CrispASR's
        # registry fetches it as the companion, so no explicit codec file here.
        "codec_file": None,
    },
}

# Backends that take the reference WAV through CrispASR's --voice flag.
_VOICE_CLONE_BACKENDS = {"pocket-tts", "chatterbox-nano", "chatterbox", "chatterbox-turbo"}


def _find_crispasr(binary: str | None) -> str:
    candidates = [
        binary,
        os.environ.get("CRISPASR_BIN"),
        shutil.which("crispasr"),
        shutil.which("crispasr-cli"),
        str(_EXTERNAL_CRISPASR / "crispasr"),
        str(_EXTERNAL_CRISPASR / "crispasr-cli"),
        "CrispASR/build/bin/crispasr",
        "CrispASR/build/bin/crispasr-cli",
        "build/bin/crispasr",
        "build/bin/crispasr-cli",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if path.exists() and os.access(path, os.X_OK):
            return str(path)
        found = shutil.which(str(candidate))
        if found:
            return found
    raise SystemExit(
        "CrispASR binary not found. Build it first, then pass --binary or set CRISPASR_BIN.\n"
        "Official route:\n"
        "  git clone https://github.com/CrispStrobe/CrispASR\n"
        "  cd CrispASR\n"
        "  cmake -B build -DCMAKE_BUILD_TYPE=Release\n"
        "  cmake --build build -j --target crispasr-cli"
    )


def _existing_file(path: str, label: str) -> str:
    resolved = Path(path).expanduser()
    if not resolved.exists() or not resolved.is_file():
        raise SystemExit(f"{label} not found: {path}")
    return str(resolved)


def _build_crispasr_command(
    crispasr: str,
    info: dict,
    text: str,
    output: str,
    model_arg: str,
    auto_download: bool,
    codec_model: str | None,
    seed: int | None,
    temperature: float | None,
    ref: str | None,
    extra_args: str | None,
    disclaimer: bool = False,
) -> list[str]:
    cmd = [
        crispasr,
        "--backend",
        str(info["backend"]),
        "-m",
        model_arg,
        "--tts",
        text,
        "--tts-output",
        output,
    ]
    if auto_download:
        cmd.append("--auto-download")
    if codec_model:
        cmd.extend(["--codec-model", _existing_file(codec_model, "Codec model")])
    if ref and info["backend"] in _VOICE_CLONE_BACKENDS:
        cmd.extend(["--voice", _existing_file(ref, "Reference audio")])
        cmd.append("--i-have-rights")
        if not disclaimer:
            cmd.extend(["--no-spoken-disclaimer", "--accept-marking-responsibility"])
    if seed is not None:
        cmd.extend(["--seed", str(seed)])
    if temperature is not None:
        cmd.extend(["-tp", str(temperature)])
    if extra_args:
        import shlex

        cmd.extend(shlex.split(extra_args))
    return cmd


def prep(force: bool = False) -> None:
    """Download and cache all model weights/assets without synthesizing. Idempotent."""
    from huggingface_hub import snapshot_download

    for key, model_info in MODELS.items():
        repo = model_info["repo"]
        print(f"[prep] downloading: {repo}")
        snapshot_download(repo_id=repo)
        print(f"[prep] ready: {repo}")


def say(
    text: str,
    output: str = "output.wav",
    model: str = "dia",
    ref: str | None = None,
    model_path: str | None = None,
    codec_model: str | None = None,
    binary: str | None = None,
    auto_download: bool = True,
    seed: int | None = 42,
    temperature: float | None = None,
    force: bool = False,
    extra_args: str | None = None,
    disclaimer: bool = False,
) -> None:
    """Generate speech through CrispASR using cstr Dia or Pocket GGUF repos.

    Args:
        text: Text to synthesize. Dia works best with [S1]/[S2] tags and longer text.
        output: Output WAV path.
        model: "dia", "pocket"/"pocket-tts", or "chatterbox-nano".
        ref: Optional reference WAV. Accepted for tts_all.py compatibility; only
            passed to CrispASR if the caller also supplies matching --extra-args.
        model_path: Local GGUF path. If omitted with --auto-download, use CrispASR registry key.
        codec_model: Dia DAC codec path. Optional if beside the Dia model or auto-download is used.
        binary: Path to crispasr/crispasr-cli. Also read from CRISPASR_BIN.
        auto_download: Let CrispASR download registered model files.
        seed: Optional --seed value.
        temperature: Optional -tp value for backends that support it.
        force: Overwrite output when it exists.
        extra_args: Additional raw CrispASR args split with shell-like syntax.
        disclaimer: Keep CrispASR's audible AI-disclosure prefix on cloned output.
            Off by default so benchmark clips contain only the synthesized text.
    """
    from rich.console import Console

    console = Console()
    out_path = Path(output)
    if not force and out_path.exists() and out_path.stat().st_size > 0:
        console.print(
            f"[yellow]Skipped:[/yellow] {output} already exists (use --force to overwrite)"
        )
        return

    key = "pocket" if model == "pocket-tts" else model
    if key not in MODELS:
        raise SystemExit(
            f"Unknown model {model!r}. Use one of: {', '.join(MODELS)}"
        )

    model_info = MODELS[key]
    crispasr = _find_crispasr(binary)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if model_path:
        model_arg = _existing_file(model_path, "GGUF model")
    elif auto_download:
        model_arg = str(model_info["auto_key"])
    else:
        raise SystemExit(
            f"Pass --model-path for {model_info['repo']} or use --auto-download.\n"
            f"Expected default file: {model_info['default_file']}"
        )

    console.print(f"[dim]Repo: {model_info['repo']}[/dim]")
    if ref and model_info["backend"] not in _VOICE_CLONE_BACKENDS:
        ref_path = Path(ref).expanduser()
        if not ref_path.exists() or not ref_path.is_file():
            raise SystemExit(f"Reference audio not found: {ref}")
        console.print(
            "[yellow]Note:[/yellow] CrispASR model card quickstarts do not document "
            "a reference-audio flag for this backend; reference validation passed, "
            "but the audio is not forwarded unless supplied through --extra-args."
        )
    cmd = _build_crispasr_command(
        crispasr=crispasr,
        info=model_info,
        text=text,
        output=str(out_path),
        model_arg=model_arg,
        auto_download=auto_download,
        codec_model=codec_model,
        seed=seed,
        temperature=temperature,
        ref=ref,
        extra_args=extra_args,
        disclaimer=disclaimer,
    )
    console.print(f"[dim]Command: {' '.join(cmd)}[/dim]")
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as exc:
        raise SystemExit(
            f"CrispASR binary vanished before execution: {crispasr}"
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"CrispASR failed with exit code {exc.returncode}.") from exc

    if not out_path.exists() or out_path.stat().st_size == 0:
        raise SystemExit(
            f"CrispASR completed but did not create a non-empty output: {out_path}"
        )
    console.print(f"[green]Saved:[/green] {out_path.resolve()}")


def info() -> None:
    """Print one-line JSON with model metadata."""
    import json

    print(
        json.dumps(
            {
                "name": "CrispASR GGUF (Dia, Pocket TTS, Chatterbox Nano)",
                "repo": [DIA_REPO, POCKET_REPO, NANO_REPO],
                "models": sorted(MODELS),
                "backend": "gguf",
                "languages": ["en"],
                "sample_rate_hz": {
                    "dia": 44100,
                    "pocket": 24000,
                    "chatterbox-nano": 24000,
                },
                "modes": ["auto", "ivc"],
                "capabilities": ["dialogue_tags", "voice_cloning"],
            }
        )
    )


if __name__ == "__main__":
    import fire

    fire.Fire({"prep": prep, "say": say, "info": info})
