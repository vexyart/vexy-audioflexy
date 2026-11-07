#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = ["platformdirs>=4", 
#   "fire", "rich", "torch", "torchaudio", "librosa", "soundfile", "numpy",
#   "huggingface-hub", "einops", "pyworld", "torchcrepe", "setuptools<81",
# ]
# ///
# this_file: src/vexy_audioflexy/tools/fx/tools/aud-facodec.py
"""FACodec (Amphion / NaturalSpeech 3): attribute-factorised speech codec.

Amphion is a research toolkit with no PyPI package, so this wrapper clones it into
`local/Amphion` and imports `models/codec/ns3_codec` from there — the same shape as
`aud-openvoice.py` and `aud-mossformersr.py`. Weights come from the
`amphion/naturalspeech3_facodec` HuggingFace repo.

Two verbs, both audio in / audio out at 16 kHz:

- `run` — **codec resynthesis**: encode, quantize, decode. The output should be the
  input; what is measured is how much the codec costs. This is the honest way to
  benchmark a codec in a harness built for enhancers, so it is registered as op
  `codec`, not as denoising, and it is fed the clean source rather than the noisy
  fixture.
- `convert` — **zero-shot voice conversion** through the redecoder: content and
  prosody codes from the source, speaker embedding from the target. Registered as
  op `vc`, beside OpenVoice and MeanVC2.

The codec runs at 16 kHz; anything else is resampled in.

`pyworld` and `torchcrepe` are imported by the package's mel/prosody front-end at
module load, whether or not the pitch features are used, so they are declared even
though nothing here calls them.
"""

from __future__ import annotations


import os
from pathlib import Path
from platformdirs import user_cache_path, user_data_path
RUNTIME_ROOT = Path(os.environ.get("VAFX_AUDIO_RUNTIME_ROOT", user_data_path("vexy-audioflexy") / "audio" )).expanduser()
CACHE_ROOT = Path(os.environ.get("VAFX_AUDIO_CACHE_ROOT", user_cache_path("vexy-audioflexy") / "audio")).expanduser()
import json
import os
import subprocess
import sys
import time
from pathlib import Path

VERSION = "1.0.0"

REPO_URL = "https://github.com/open-mmlab/Amphion"
CKPT_REPO = "amphion/naturalspeech3_facodec"
RATE = 16000

PACKAGES: dict[str, dict] = {
    "facodec": {
        "op": "codec",
        "quant": "fp32",
        "repo": f"open-mmlab/Amphion ({CKPT_REPO})",
    },
    "facodec-vc": {
        "op": "vc",
        "quant": "fp32",
        "repo": f"open-mmlab/Amphion ({CKPT_REPO}, redecoder)",
    },
}


def _root() -> Path:
    """Checkout root. `AMPHION_ROOT` overrides it."""
    env = os.environ.get("AMPHION_ROOT")
    if env:
        return Path(env).expanduser()
    return RUNTIME_ROOT / "engines" / "Amphion"


def _spec(package: str) -> dict:
    try:
        return PACKAGES[package]
    except KeyError:
        raise SystemExit(
            f"unknown package '{package}'. Known: {', '.join(PACKAGES)}"
        ) from None


def _ensure_checkout() -> Path:
    root = _root()
    if not (root / "models" / "codec" / "ns3_codec" / "facodec.py").exists():
        root.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(root)], check=True)
    return root


def _codec(redecoder: bool = False):
    """Load the encoder, decoder and (optionally) the redecoder onto the CPU."""
    import torch
    from huggingface_hub import hf_hub_download

    root = _ensure_checkout()
    sys.path.insert(0, str(root))
    from models.codec.ns3_codec import FACodecDecoder, FACodecEncoder, FACodecRedecoder

    encoder = FACodecEncoder(ngf=32, up_ratios=[2, 4, 5, 5], out_channels=256)
    decoder = FACodecDecoder(
        in_channels=256, upsample_initial_channel=1024, ngf=32,
        up_ratios=[5, 5, 4, 2], vq_num_q_c=2, vq_num_q_p=1, vq_num_q_r=3,
        vq_dim=256, codebook_dim=8, codebook_size_prosody=10,
        codebook_size_content=10, codebook_size_residual=10,
        use_gr_x_timbre=True, use_gr_residual_f0=True, use_gr_residual_phone=True,
    )
    encoder.load_state_dict(torch.load(
        hf_hub_download(CKPT_REPO, "ns3_facodec_encoder.bin"), map_location="cpu"))
    decoder.load_state_dict(torch.load(
        hf_hub_download(CKPT_REPO, "ns3_facodec_decoder.bin"), map_location="cpu"))
    encoder.eval()
    decoder.eval()

    redec = None
    if redecoder:
        redec = FACodecRedecoder()
        redec.load_state_dict(torch.load(
            hf_hub_download(CKPT_REPO, "ns3_facodec_redecoder.bin"), map_location="cpu"))
        redec.eval()
    return encoder, decoder, redec


def _load(path: Path):
    import librosa
    import torch

    wav = librosa.load(str(path), sr=RATE)[0]
    return torch.from_numpy(wav).float().unsqueeze(0).unsqueeze(0)


def _write(samples, out_path: Path, console) -> float:
    import numpy as np
    import soundfile as sf

    data = np.asarray(samples, dtype="float32").reshape(-1)
    if data.size == 0 or not np.any(data):
        console.print("[red]FACodec produced silence[/red]")
        raise SystemExit(1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), data, RATE, subtype="PCM_16")
    return data.size / RATE


def prep(package: str = "facodec", force: bool = False) -> None:
    """Clone Amphion and download the FACodec weights. Idempotent.

    Args:
        package: `facodec` (codec) or `facodec-vc` (voice conversion).
        force: Unused; an existing checkout and cached weights are reused.
    """
    _codec(redecoder=_spec(package)["op"] == "vc")
    print(f"[prep] ready: {package} -> {_root()}")


def run(
    audio: str = "",
    package: str = "facodec",
    output: str = "output.wav",
    task: str = "",
    voice_ref: str = "",
    force: bool = False,
    play: bool = False,
    **_ignored: object,
) -> None:
    """Encode and resynthesize one clip through the codec.

    The batch runner calls every wrapper's `run`, so asking `run` for the `-vc`
    package hands off to `convert` rather than refusing — the package already says
    which operation it is.

    Args:
        audio: Input WAV; resampled to 16 kHz.
        package: `facodec` for resynthesis, `facodec-vc` to convert.
        output: Output WAV path.
        task: `codec`, optional. A mismatch is an error.
        voice_ref: Target speaker, for `facodec-vc` only.
        force: Overwrite an existing output file.
        play: Play the result with afplay instead of reporting the path.
    """
    if _spec(package)["op"] == "vc":
        return convert(audio=audio, voice_ref=voice_ref, package=package,
                       output=output, task=task, force=force, play=play, **_ignored)
    import torch
    from rich.console import Console
    from rich.panel import Panel

    if _ignored:
        raise SystemExit(
            f"{Path(__file__).name} does not accept: {', '.join(sorted(_ignored))}"
        )

    console = Console()
    out_path = Path(output).resolve()
    if not force and out_path.exists() and out_path.stat().st_size > 0:
        console.print(f"[yellow]Skipped:[/yellow] {output} already exists (use --force)")
        return
    if not audio:
        raise SystemExit("--audio is required")
    spec = _spec(package)
    if task and task != "codec":
        raise SystemExit(f"engine '{package}' does codec, not '{task}'")

    console.print(
        Panel.fit(
            f"[bold cyan]FACodec[/bold cyan] v{VERSION}  task=[yellow]codec[/yellow]  "
            f"16 kHz resynthesis",
            border_style="cyan",
        )
    )

    encoder, decoder, _ = _codec()
    wav = _load(Path(audio).resolve())
    t0 = time.monotonic()
    with torch.no_grad():
        enc_out = encoder(wav)
        vq_post_emb, _vq_id, _, _quantized, spk_embs = decoder(enc_out, eval_vq=False, vq=True)
        recon = decoder.inference(vq_post_emb, spk_embs)
    elapsed = time.monotonic() - t0

    duration = _write(recon[0][0].cpu().numpy(), out_path, console)
    console.print(
        f"[green]Saved[/green] [bold]{out_path}[/bold] "
        f"({duration:.1f}s @ {RATE} Hz, rtf={elapsed / duration:.2f})"
    )
    if play:
        subprocess.run(["afplay", str(out_path)], check=True)


def convert(
    audio: str = "",
    voice_ref: str = "",
    package: str = "facodec-vc",
    output: str = "output.wav",
    task: str = "",
    force: bool = False,
    play: bool = False,
    **_ignored: object,
) -> None:
    """Re-render `audio` in the timbre of `voice_ref` through the redecoder.

    Args:
        audio: Source speech — content and prosody are taken from here.
        voice_ref: Target speaker clip — the timbre.
        package: `facodec-vc`.
        output: Output WAV path.
        task: `vc`, optional.
        force: Overwrite an existing output file.
        play: Play the result with afplay instead of reporting the path.
    """
    import torch
    from rich.console import Console
    from rich.panel import Panel

    if _ignored:
        raise SystemExit(
            f"{Path(__file__).name} does not accept: {', '.join(sorted(_ignored))}"
        )

    console = Console()
    out_path = Path(output).resolve()
    if not force and out_path.exists() and out_path.stat().st_size > 0:
        console.print(f"[yellow]Skipped:[/yellow] {output} already exists (use --force)")
        return
    if not audio or not voice_ref:
        raise SystemExit("--audio and --voice_ref are both required for conversion")
    spec = _spec(package)
    if spec["op"] != "vc":
        raise SystemExit(f"'{package}' does {spec['op']} — use `run`")
    if task and task != "vc":
        raise SystemExit(f"engine '{package}' does vc, not '{task}'")

    console.print(
        Panel.fit(
            f"[bold cyan]FACodec[/bold cyan] v{VERSION}  task=[yellow]vc[/yellow]\n"
            f"source={Path(audio).name}  target={Path(voice_ref).name}",
            border_style="cyan",
        )
    )

    encoder, decoder, redecoder = _codec(redecoder=True)
    wav_a = _load(Path(audio).resolve())
    wav_b = _load(Path(voice_ref).resolve())
    t0 = time.monotonic()
    with torch.no_grad():
        enc_a = encoder(wav_a)
        enc_b = encoder(wav_b)
        _emb_a, vq_id_a, _, _q_a, _spk_a = decoder(enc_a, eval_vq=False, vq=True)
        _emb_b, _vq_id_b, _, _q_b, spk_b = decoder(enc_b, eval_vq=False, vq=True)
        # Residual codes carry the source's acoustic detail — including its timbre —
        # so they are dropped for conversion; upstream does the same.
        emb_a_to_b = redecoder.vq2emb(vq_id_a, spk_b, use_residual=False)
        recon = redecoder.inference(emb_a_to_b, spk_b)
    elapsed = time.monotonic() - t0

    duration = _write(recon[0][0].cpu().numpy(), out_path, console)
    console.print(
        f"[green]Saved[/green] [bold]{out_path}[/bold] "
        f"({duration:.1f}s @ {RATE} Hz, rtf={elapsed / duration:.2f})"
    )
    if play:
        subprocess.run(["afplay", str(out_path)], check=True)


def info(package: str = "facodec") -> None:
    """Print one-line JSON with engine metadata."""
    spec = _spec(package)
    print(json.dumps(
        {
            "name": package,
            "family": "facodec",
            "repo": spec["repo"],
            "stack": "torch",
            "runtime": "Amphion ns3_codec (PyTorch, local checkout)",
            "quant": spec["quant"],
            "op": spec["op"],
            "sample_rate": RATE,
        },
        separators=(",", ":"),
    ))


def packages() -> None:
    """List the engines."""
    for name, spec in PACKAGES.items():
        print(f"{name:14} {spec['op']:6} {spec['quant']:6} {RATE} Hz")


if __name__ == "__main__":
    import fire

    fire.Fire({"prep": prep, "run": run, "convert": convert,
               "info": info, "packages": packages})
