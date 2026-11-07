#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = ["fire", "requests", "rich"]
# ///
# this_file: src/vexy_audioflexy/tools/tts/tools/tts-higgs-v3.py
"""Higgs Audio v3 TTS client for an SGLang-Omni /v1/audio/speech server.

Backend: sglang-omni (external server process). Languages: en, pl.
Sample rate: 24000 Hz (WAV). Modes: ivc (instruct voice cloning via ref audio).
Special capabilities: voice_cloning. Model repo: bosonai/higgs-audio-v3-tts-4b.
NOTE: This script is a client — the model runs inside a separate sgl-omni server process.
Start the server with: sgl-omni serve --model-path bosonai/higgs-audio-v3-tts-4b --port 8000
"""

from pathlib import Path
from typing import Any

VERSION = "1.0.0"
MODEL_REPO = "bosonai/higgs-audio-v3-tts-4b"
DEFAULT_URL = "http://localhost:8000/v1/audio/speech"


def _output_path(path: str) -> Path:
    return Path(path).expanduser().resolve()


def _should_skip(output: str, force: bool, play: bool) -> bool:
    out_path = _output_path(output)
    return (
        not play
        and not force
        and out_path.exists()
        and out_path.stat().st_size > 0
    )


def _validate_ref(ref: str | None) -> str | None:
    if not ref:
        return None
    ref_path = Path(ref).expanduser().resolve()
    if not ref_path.exists() or not ref_path.is_file():
        raise SystemExit(f"Reference audio not found: {ref}")
    return str(ref_path)


def _build_speech_payload(
    text: str,
    ref: str | None,
    ref_text: str | None,
    language: str | None,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
    max_new_tokens: int | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"input": text}
    if language:
        payload["language"] = language
    if ref:
        reference: dict[str, str] = {"audio_path": ref}
        if ref_text:
            reference["text"] = ref_text
        payload["references"] = [reference]
    if temperature is not None:
        payload["temperature"] = temperature
    if top_p is not None:
        payload["top_p"] = top_p
    if top_k is not None:
        payload["top_k"] = top_k
    if max_new_tokens is not None:
        payload["max_new_tokens"] = max_new_tokens
    return payload


def _response_audio(response: Any) -> bytes:
    audio = response.content
    if not audio:
        raise SystemExit("Higgs server returned an empty response body.")

    content_type = response.headers.get("content-type", "").split(";", 1)[0].lower()
    if content_type and not content_type.startswith("audio/") and not audio.startswith(b"RIFF"):
        body = response.text[:1000]
        raise SystemExit(
            f"Higgs server returned non-audio content-type {content_type}: {body}"
        )
    return audio


def prep(force: bool = False):
    """Download + cache all model weights/assets WITHOUT synthesizing. Idempotent.

    For higgs-v3, the model runs inside an external sgl-omni server process.
    This function checks whether the server is reachable and prints instructions
    if it is not running. No local weight download is performed by this script.
    """
    import requests
    from rich.console import Console

    console = Console()
    console.print(f"[dim]Model repo: {MODEL_REPO}[/dim]")
    console.print(
        "[yellow]higgs-v3 runs via an external sgl-omni server.[/yellow]\n"
        f"Start it with:\n  sgl-omni serve --model-path {MODEL_REPO} --port 8000\n"
        "Then re-run prep to verify connectivity."
    )
    try:
        # Probe the server with a minimal HEAD/GET to see if it is up.
        response = requests.get(DEFAULT_URL.rsplit("/", 2)[0] + "/health", timeout=5)
        if response.status_code < 500:
            console.print(f"[green][prep] ready: {MODEL_REPO}[/green]")
            return
    except Exception:
        pass
    # Server not reachable — not a hard error for prep; weights may still be cached
    # on the server host. Print the repo and return without error.
    console.print(
        "[yellow]Server not reachable yet — start it before running say.[/yellow]"
    )
    print(f"[prep] ready: {MODEL_REPO}")


def say(
    text: str,
    output: str = "output.wav",
    server_url: str = DEFAULT_URL,
    ref: str | None = None,
    ref_text: str | None = None,
    language: str | None = None,
    temperature: float | None = 0.8,
    top_p: float | None = None,
    top_k: int | None = 50,
    max_new_tokens: int | None = 1024,
    timeout: float = 300.0,
    play: bool = False,
    force: bool = False,
) -> None:
    """Generate speech through an already-running SGLang-Omni Higgs server.

    Start the external server separately:
        sgl-omni serve --model-path bosonai/higgs-audio-v3-tts-4b --port 8000

    Args:
        text: Text to synthesize. Higgs inline control tags are passed through.
        output: Output WAV path.
        server_url: SGLang-Omni OpenAI-compatible speech endpoint.
        ref: Optional reference WAV for voice cloning.
        ref_text: Optional transcript for ref; recommended by the model card.
        language: Optional language code, e.g. en or pl.
        temperature: Sampling temperature sent to the server.
        top_p: Top-p sampling value sent to the server.
        top_k: Top-k sampling value sent to the server.
        max_new_tokens: Maximum generated audio-token count.
        timeout: HTTP timeout in seconds.
        play: Play with afplay after writing a temporary WAV.
        force: Overwrite output when it exists.
    """
    from rich.console import Console

    console = Console()
    if _should_skip(output, force, play):
        console.print(
            f"[yellow]Skipped:[/yellow] {output} already exists (use --force to overwrite)"
        )
        return
    ref_path = _validate_ref(ref)

    import requests

    payload = _build_speech_payload(
        text=text,
        ref=ref_path,
        ref_text=ref_text,
        language=language,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_new_tokens=max_new_tokens,
    )

    console.print(f"[dim]Model repo: {MODEL_REPO}[/dim]")
    console.print(f"[dim]Endpoint: {server_url}[/dim]")

    try:
        response = requests.post(server_url, json=payload, timeout=timeout)
        response.raise_for_status()
    except requests.exceptions.ConnectionError as exc:
        raise SystemExit(
            "Could not connect to the SGLang-Omni Higgs server.\n"
            "Start it with:\n"
            f"  sgl-omni serve --model-path {MODEL_REPO} --port 8000\n"
            f"Then retry with --server-url {DEFAULT_URL}"
        ) from exc
    except requests.exceptions.HTTPError as exc:
        body = response.text[:1000] if "response" in locals() else ""
        raise SystemExit(f"Higgs server returned HTTP {response.status_code}: {body}") from exc
    except requests.exceptions.Timeout as exc:
        raise SystemExit(f"Higgs server timed out after {timeout}s: {server_url}") from exc

    audio = _response_audio(response)

    if play:
        import subprocess
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio)
            tmp_path = Path(tmp.name)
        subprocess.run(["afplay", str(tmp_path)], check=True)
        tmp_path.unlink(missing_ok=True)
        console.print("[green]Done.[/green]")
        return

    out_path = _output_path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(audio)
    console.print(f"[green]Saved:[/green] {out_path.resolve()}")


def info():
    """Print one-line JSON: name, repo, backend, languages, sample_rate_hz, modes, capabilities."""
    import json

    print(json.dumps({
        "name": "higgs-v3",
        "repo": MODEL_REPO,
        "backend": "sglang-omni",
        "languages": ["en", "pl"],
        "sample_rate_hz": 24000,
        "modes": ["ivc"],
        "capabilities": ["voice_cloning"],
    }))


if __name__ == "__main__":
    import fire
    fire.Fire({"prep": prep, "say": say, "info": info})
