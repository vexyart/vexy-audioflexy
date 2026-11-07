#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = ["fire", "rich", "huggingface-hub"]
# ///
# this_file: src/vexy_audioflexy/tools/tts/tools/tts-confucius4.py
"""Confucius4-TTS (netease-youdao/Confucius4-TTS) via local checkout or Python API. Backend: Python/PyTorch (CUDA/CPU); instanced voice cloning (IVC) from a prompt WAV; languages: en, zh, ja, ko, de, fr, es; sample rate: model-native (typically 22050 Hz); no emotion/instruct tags."""

from pathlib import Path
import os
import subprocess
import sys
from _paths import runtime_path

VERSION = "1.0.0"
MODEL_REPO = "netease-youdao/Confucius4-TTS"
ENV_PATH = "CONFUCIUS4_TTS_PATH"
ENV_PYTHON_BIN = "CONFUCIUS4_PYTHON_BIN"

# Repo root is two levels up from tools/; the Confucius4-TTS checkout and its
# dedicated venv live under .external/ (see setup in DEPENDENCIES.md).
ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CHECKOUT = runtime_path('Confucius4-TTS')
_DEFAULT_PYTHON = _DEFAULT_CHECKOUT / ".venv/bin/python"


def _validate_file(path: str, label: str) -> str:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists() or not resolved.is_file():
        raise SystemExit(f"{label} not found: {path}")
    return str(resolved)


def _output_path(path: str) -> Path:
    return Path(path).expanduser().resolve()


def _checkout(path: str | None) -> Path:
    raw = (
        path
        or os.environ.get(ENV_PATH)
        or (str(_DEFAULT_CHECKOUT) if _DEFAULT_CHECKOUT.is_dir() else "")
    )
    if not raw:
        raise SystemExit(_setup_message("No checkout path provided."))
    checkout = Path(raw).expanduser().resolve()
    if not checkout.exists() or not checkout.is_dir():
        raise SystemExit(_setup_message(f"Checkout directory not found: {checkout}"))
    return checkout


def _python_bin(path: str | None) -> str:
    if path:
        return path
    env = os.environ.get(ENV_PYTHON_BIN)
    if env:
        return env
    if _DEFAULT_PYTHON.exists():
        return str(_DEFAULT_PYTHON)
    return sys.executable


def _setup_message(reason: str) -> str:
    return (
        f"{reason}\n"
        f"This wrapper expects a local {MODEL_REPO} checkout with its own dependencies.\n"
        "Official route:\n"
        "  git clone https://github.com/netease-youdao/Confucius4-TTS.git\n"
        "  cd Confucius4-TTS\n"
        "  conda create -n confuciustts python=3.10 -y\n"
        "  conda activate confuciustts\n"
        "  pip install -r requirements.txt\n"
        f"Then set {ENV_PATH}=/path/to/Confucius4-TTS or pass --checkout."
    )


def _run_example(
    checkout: Path,
    text: str,
    ref: str,
    output: str,
    lang: str,
    config: str | None,
    python_bin: str,
) -> None:
    checkout = checkout.expanduser().resolve()
    example = checkout / "example.py"
    if not example.exists():
        raise SystemExit(
            _setup_message(f"example.py not found in checkout: {checkout}")
        )
    config_path = (
        Path(config).expanduser().resolve()
        if config
        else checkout / "config/inference_config.yaml"
    )
    if not config_path.exists():
        raise SystemExit(f"Confucius4-TTS config not found: {config_path}")
    out_path = _output_path(output)

    cmd = [
        python_bin,
        str(example),
        "--prompt_wav",
        ref,
        "--text",
        text,
        "--lang",
        lang,
        "--out",
        str(out_path),
        "--config",
        str(config_path),
    ]
    subprocess.run(cmd, cwd=str(checkout), check=True)


def _run_api(text: str, ref: str, output: str, lang: str, config: str | None) -> None:
    import torch
    import torchaudio
    from confuciustts.cli.inference import ConfuciusTTS

    config_path = config or "config/inference_config.yaml"
    model = ConfuciusTTS(
        config_path=config_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    audio = model.generate(text=text, lang=lang, prompt_wav=ref, verbose=True)
    torchaudio.save(output, audio.cpu(), model.sample_rate)


def prep(force: bool = False) -> None:
    """Download + cache all model weights/assets WITHOUT synthesizing. Idempotent."""
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=MODEL_REPO)
    print(f"[prep] ready: {MODEL_REPO}")


def say(
    text: str,
    ref: str,
    output: str = "output.wav",
    lang: str = "en",
    checkout: str | None = None,
    config: str | None = None,
    python_bin: str | None = None,
    use_api: bool = False,
    force: bool = False,
) -> None:
    """Generate speech with Confucius4-TTS via local checkout or Python API.

    Args:
        text: Text to synthesize.
        ref: Prompt/reference WAV. Confucius4-TTS does not require a transcript.
        output: Output WAV path.
        lang: Target language code, e.g. en, zh, ja, ko, de, fr, es.
        checkout: Local Confucius4-TTS checkout path. Also read from CONFUCIUS4_TTS_PATH.
        config: Inference config YAML path. Defaults to checkout/config/inference_config.yaml.
        python_bin: Python executable for example.py, usually the Confucius conda env Python.
        use_api: Use confuciustts.cli.inference. Otherwise run upstream example.py.
        force: Overwrite output when it exists.
    """
    from rich.console import Console

    console = Console()
    out_path = _output_path(output)
    if not force and out_path.exists() and out_path.stat().st_size > 0:
        console.print(
            f"[yellow]Skipped:[/yellow] {output} already exists (use --force to overwrite)"
        )
        return

    ref_path = _validate_file(ref, "Reference audio")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    console.print(f"[dim]Model repo: {MODEL_REPO}[/dim]")

    try:
        if use_api:
            _run_api(text, ref_path, str(out_path), lang, config)
        else:
            local_checkout = _checkout(checkout)
            _run_example(
                local_checkout,
                text,
                ref_path,
                str(out_path),
                lang,
                config,
                _python_bin(python_bin),
            )
    except ImportError as exc:
        raise SystemExit(
            _setup_message(f"Confucius4-TTS Python API import failed: {exc}")
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise SystemExit(
            f"Confucius4-TTS example.py failed with exit code {exc.returncode}."
        ) from exc

    if not out_path.exists() or out_path.stat().st_size == 0:
        raise SystemExit(
            f"Confucius4-TTS did not create a non-empty output: {out_path}"
        )
    console.print(f"[green]Saved:[/green] {out_path.resolve()}")


def info() -> None:
    """Print one-line JSON: name, repo, backend, languages, sample_rate_hz, modes, capabilities."""
    import json

    print(
        json.dumps(
            {
                "name": "Confucius4 TTS",
                "repo": MODEL_REPO,
                "backend": "python",
                "languages": ["en", "zh", "ja", "ko", "de", "fr", "es"],
                "sample_rate_hz": 22050,
                "modes": ["ivc"],
                "capabilities": ["voice_cloning"],
            }
        )
    )


if __name__ == "__main__":
    import fire

    fire.Fire({"prep": prep, "say": say, "info": info})
