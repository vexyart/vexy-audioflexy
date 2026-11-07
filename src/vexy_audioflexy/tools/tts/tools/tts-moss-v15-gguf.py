#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = ["fire", "requests", "rich"]
# ///
# this_file: src/vexy_audioflexy/tools/tts/tools/tts-moss-v15-gguf.py
"""MOSS-TTS v1.5 GGUF — external server wrapper (moss-tts-server + GGUF backend).

Routes synthesis requests to a locally-running moss-tts-server loaded with
smcleod/MOSS-TTS-v1.5-GGUF (q8_0 or similar quant). The server must be started
separately before calling `say`. Supports instant voice cloning (ivc) by passing a
reference WAV; without a reference the server uses its built-in default voice (auto
mode). Language can be hinted via --language. Sample rate: 24 000 Hz mono WAV.
No local model download is performed by this script — the GGUF weights are loaded
by the external server process.
"""

from pathlib import Path
import base64
import os

REPO = "smcleod/MOSS-TTS-v1.5-GGUF"
MODEL = REPO  # kept for error messages
DEFAULT_ENDPOINT = "http://127.0.0.1:8080"
TTS_PATH = "/tts"


def _skip_existing(output: str, force: bool, play: bool) -> bool:
    out_path = Path(output)
    return not play and not force and out_path.exists() and out_path.stat().st_size > 0


def _reference_wav_b64(ref: str | None) -> str | None:
    if not ref:
        return None
    ref_path = Path(ref)
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference audio not found: {ref}")
    return base64.b64encode(ref_path.read_bytes()).decode("ascii")


def _build_tts_payload(
    text: str,
    ref: str | None,
    language: str | None,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "text": text,
        "max_new_tokens": max_new_tokens,
        "sampling": {
            "text_temperature": temperature,
            "text_top_p": top_p,
            "text_top_k": top_k,
        },
    }
    audio_data = _reference_wav_b64(ref)
    if audio_data:
        payload["reference_wav_b64"] = audio_data
    if language:
        payload["language"] = language
    return payload


def prep(force: bool = False):
    """Download + cache all model weights/assets WITHOUT synthesizing. Idempotent.

    MOSS-TTS-v1.5-GGUF weights are served by an external moss-tts-server process;
    this script cannot download them directly. `prep` therefore only verifies that
    the server is reachable at the configured endpoint and prints a ready message.
    Set MOSS_TTS_SERVER_URL to override the default http://127.0.0.1:8080.
    """
    import requests

    endpoint = (os.environ.get("MOSS_TTS_SERVER_URL") or DEFAULT_ENDPOINT).rstrip("/")
    try:
        # A lightweight GET to the root is enough to confirm the server is up.
        requests.get(endpoint, timeout=5)
    except requests.RequestException as exc:
        print(
            f"[prep] WARNING: moss-tts-server not reachable at {endpoint}: {exc}\n"
            f"Start it with: moss-tts-server --model moss-tts-v1.5-q8_0.gguf "
            f"--host 127.0.0.1 --port 8080 --aux-cpu --no-webui"
        )
    print(f"[prep] ready: {REPO}")


def say(
    text: str,
    output: str = "output.wav",
    ref: str | None = None,
    language: str | None = None,
    endpoint: str | None = None,
    max_new_tokens: int = 512,
    temperature: float = 1.7,
    top_p: float = 0.8,
    top_k: int = 25,
    play: bool = False,
    force: bool = False,
):
    """Call moss-tts-server for smcleod/MOSS-TTS-v1.5-GGUF.

    Args:
        text: Text to synthesize.
        output: Output WAV path.
        ref: Optional reference audio path/URL for cloning.
        language: Optional language hint, e.g. English or Polish.
        endpoint: Server base URL. Defaults to MOSS_TTS_SERVER_URL or
            http://127.0.0.1:8080.
        max_new_tokens: Server sampling max_new_tokens.
        temperature: Server sampling temperature.
        top_p: Server top_p.
        top_k: Server top_k.
        play: Play with afplay after saving.
        force: Overwrite output if it already exists.
    """
    from rich.console import Console
    import requests

    console = Console()
    if _skip_existing(output, force, play):
        console.print(f"[yellow]Skipped:[/yellow] {output} already exists (use --force)")
        return

    endpoint = (endpoint or os.environ.get("MOSS_TTS_SERVER_URL") or DEFAULT_ENDPOINT).rstrip("/")

    try:
        payload = _build_tts_payload(
            text=text,
            ref=ref,
            language=language,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise SystemExit(1) from exc

    try:
        response = requests.post(f"{endpoint}{TTS_PATH}", json=payload, timeout=600)
        response.raise_for_status()
    except requests.RequestException as exc:
        console.print(
            f"[red]Could not reach moss-tts-server at {endpoint}{TTS_PATH}:[/red] {exc}\n"
            "Verify the external server is running and loaded with "
            f"{MODEL}. Start it with:\n"
            "  moss-tts-server --model moss-tts-v1.5-q8_0.gguf "
            "--host 127.0.0.1 --port 8080 --aux-cpu --no-webui"
        )
        raise SystemExit(1) from exc

    audio = response.content
    if not audio:
        console.print("[red]Server returned an empty WAV response.[/red]")
        raise SystemExit(1)

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(audio)

    if play:
        import subprocess

        subprocess.run(["afplay", str(out_path)], check=True)

    console.print(f"[green]Saved:[/green] {out_path.resolve()}")


def info():
    """Print one-line JSON with model metadata."""
    import json

    print(json.dumps({
        "name": "moss-v15-gguf",
        "repo": REPO,
        "backend": "gguf",
        "languages": ["en", "zh"],
        "sample_rate_hz": 24000,
        "modes": ["auto", "ivc"],
        "capabilities": ["voice_cloning", "multilingual"],
    }))


if __name__ == "__main__":
    import fire
    fire.Fire({"prep": prep, "say": say, "info": info})
