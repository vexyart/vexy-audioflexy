#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = ["fire", "rich", "sherpa-onnx", "soundfile", "numpy", "httpx"]
# ///
# this_file: src/vexy_audioflexy/tools/tts/tools/tts-sherpaonnx.py
"""sherpa-onnx TTS. Backend: onnx/sherpa-onnx, languages: en/pl/zh, 22-24 kHz mono, modes: auto, capabilities: none.

One wrapper, many models — sherpa-onnx is a single onnxruntime engine that
dispatches on which model-config slot is filled (`vits`, `kokoro`, `matcha`,
`kitten`), exactly like audio.cpp dispatches on `--family`. Add a model by adding
a dict to `PACKAGES`, not a script.

Weights are GitHub release tarballs, not HuggingFace repos: `prep` downloads and
extracts the *whole* directory, because a model is its `.onnx` plus `tokens.txt`
plus, depending on family, `espeak-ng-data/`, `lexicon*.txt`, `dict/` and
`voices.bin`. Pulling only the `.onnx` gives an engine that loads and emits noise.

None of these families is a voice cloner. They are fixed-speaker models with built-in voices (mode `auto`);
multi-speaker checkpoints select a speaker with `--sid`, and `--ref` is rejected
rather than silently ignored.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from _paths import runtime_path

VERSION = "1.0.0"

_REL = "https://github.com/k2-fsa/sherpa-onnx/releases/download"
_TTS = f"{_REL}/tts-models"
_VOC = f"{_REL}/vocoder-models"

# name -> family, tarball URL, languages, speaker count, notes.
# `sid` is the default speaker id; `speakers` is what the checkpoint carries.
PACKAGES: dict[str, dict] = {
    "vits-piper-en": {
        "family": "vits",
        "url": f"{_TTS}/vits-piper-en_US-amy-medium.tar.bz2",
        "langs": ["en"],
        "speakers": 1,
        "sample_rate": 22050,
        "quant": "f32",
        "repo": "rhasspy/piper-voices (en_US-amy-medium)",
    },
    "vits-piper-pl": {
        "family": "vits",
        "url": f"{_TTS}/vits-piper-pl_PL-gosia-medium.tar.bz2",
        "langs": ["pl"],
        "speakers": 1,
        "sample_rate": 22050,
        "quant": "f32",
        "repo": "rhasspy/piper-voices (pl_PL-gosia-medium)",
    },
    "vits-piper-pl-darkman": {
        "family": "vits",
        "url": f"{_TTS}/vits-piper-pl_PL-darkman-medium.tar.bz2",
        "langs": ["pl"],
        "speakers": 1,
        "sample_rate": 22050,
        "quant": "f32",
        "repo": "rhasspy/piper-voices (pl_PL-darkman-medium)",
    },
    "kokoro-multi-v1_1": {
        "family": "kokoro",
        "url": f"{_TTS}/kokoro-multi-lang-v1_1.tar.bz2",
        "langs": ["en"],
        "speakers": 103,
        "sample_rate": 24000,
        "quant": "f32",
        "repo": "hexgrad/Kokoro-82M (sherpa-onnx export)",
    },
    "kitten-nano-en": {
        "family": "kitten",
        "url": f"{_TTS}/kitten-nano-en-v0_1-fp16.tar.bz2",
        "langs": ["en"],
        "speakers": 8,
        "sample_rate": 24000,
        "quant": "fp16",
        "repo": "KittenML/kitten-tts-nano-0.1",
    },
    "matcha-ljspeech-en": {
        "family": "matcha",
        "url": f"{_TTS}/matcha-icefall-en_US-ljspeech.tar.bz2",
        # Matcha ships an acoustic model only; the vocoder is a separate asset.
        "vocoder": f"{_VOC}/vocos-22khz-univ.onnx",
        "langs": ["en"],
        "speakers": 1,
        "sample_rate": 22050,
        "quant": "f32",
        "repo": "icefall matcha en_US-ljspeech + vocos-22khz-univ",
    },
}


def _root() -> Path:
    """Weight cache root. `SHERPA_ONNX_MODELS_ROOT` overrides it."""
    env = os.environ.get("SHERPA_ONNX_MODELS_ROOT")
    if env:
        return Path(env).expanduser()
    return runtime_path('models/sherpa-onnx')


def _spec(package: str) -> dict:
    try:
        return PACKAGES[package]
    except KeyError:
        raise SystemExit(
            f"unknown package '{package}'. Known: {', '.join(sorted(PACKAGES))}"
        ) from None


def _download(url: str, dest: Path) -> Path:
    """Fetch `url` to `dest` unless it is already there. Atomic via .part."""
    import httpx
    from rich.console import Console

    console = Console()
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    with console.status(f"[cyan]downloading {url.rsplit('/', 1)[-1]}…"):
        with httpx.stream("GET", url, follow_redirects=True, timeout=120.0) as r:
            r.raise_for_status()
            with part.open("wb") as fh:
                for chunk in r.iter_bytes(1 << 20):
                    fh.write(chunk)
    part.replace(dest)
    return dest


def _extract(tarball: Path, into: Path) -> Path:
    """Extract a release tarball and return the single directory it contains."""
    import tarfile

    into.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tarball, "r:*") as tf:
        top = {Path(m.name).parts[0] for m in tf.getmembers() if m.name.strip("./")}
        if len(top) != 1:
            raise SystemExit(f"{tarball.name} has {len(top)} top-level entries, expected 1")
        target = into / top.pop()
        if not target.is_dir():
            tf.extractall(into, filter="data")
    return target


def _ensure(package: str) -> tuple[Path, dict]:
    """Download + extract `package`; return its model directory and spec."""
    spec = _spec(package)
    root = _root()
    tarball = _download(spec["url"], root / "_dl" / spec["url"].rsplit("/", 1)[-1])
    model_dir = _extract(tarball, root)
    if "vocoder" in spec:
        _download(spec["vocoder"], root / "_vocoder" / spec["vocoder"].rsplit("/", 1)[-1])
    return model_dir, spec


def _one(d: Path, *names: str) -> str:
    """First existing path among `names` (globs allowed), or '' if none."""
    for name in names:
        hits = sorted(d.glob(name))
        if hits:
            return str(hits[0])
    return ""


def _tts_config(package: str):
    """Build an OfflineTtsConfig by reading what the extracted package holds.

    Filenames differ per family and per export date, so every field is resolved
    from the directory rather than hard-coded: the tarball is the authority.
    """
    import sherpa_onnx

    model_dir, spec = _ensure(package)
    family = spec["family"]
    tokens = _one(model_dir, "tokens.txt")
    if not tokens:
        raise SystemExit(f"{model_dir} has no tokens.txt — layout changed")
    data_dir = _one(model_dir, "espeak-ng-data")
    dict_dir = _one(model_dir, "dict")
    lexicons = ",".join(str(p) for p in sorted(model_dir.glob("lexicon*.txt")))

    model_cfg = sherpa_onnx.OfflineTtsModelConfig(num_threads=2, provider="cpu", debug=False)
    if family == "vits":
        model_cfg.vits = sherpa_onnx.OfflineTtsVitsModelConfig(
            model=_one(model_dir, "*.onnx"),
            tokens=tokens,
            lexicon=lexicons,
            data_dir=data_dir,
            dict_dir=dict_dir,
        )
    elif family == "kokoro":
        model_cfg.kokoro = sherpa_onnx.OfflineTtsKokoroModelConfig(
            model=_one(model_dir, "model.onnx", "model.int8.onnx", "*.onnx"),
            voices=_one(model_dir, "voices.bin"),
            tokens=tokens,
            lexicon=lexicons,
            data_dir=data_dir,
            dict_dir=dict_dir,
        )
    elif family == "kitten":
        model_cfg.kitten = sherpa_onnx.OfflineTtsKittenModelConfig(
            model=_one(model_dir, "model.fp16.onnx", "model.onnx", "*.onnx"),
            voices=_one(model_dir, "voices.bin"),
            tokens=tokens,
            data_dir=data_dir,
        )
    elif family == "matcha":
        vocoder = _one(_root() / "_vocoder", "*.onnx")
        if not vocoder:
            raise SystemExit("matcha needs a vocoder; run prep first")
        model_cfg.matcha = sherpa_onnx.OfflineTtsMatchaModelConfig(
            acoustic_model=_one(model_dir, "model-steps-3.onnx", "model-steps-*.onnx", "*.onnx"),
            vocoder=vocoder,
            tokens=tokens,
            lexicon=lexicons,
            data_dir=data_dir,
            dict_dir=dict_dir,
        )
    else:
        raise SystemExit(f"unsupported family '{family}'")

    # Rule FSTs (number/date/phone normalisation) ship with the Chinese models.
    rule_fsts = ",".join(str(p) for p in sorted(model_dir.glob("*.fst")))
    config = sherpa_onnx.OfflineTtsConfig(
        model=model_cfg, rule_fsts=rule_fsts, max_num_sentences=1
    )
    if not config.validate():
        raise SystemExit(f"invalid sherpa-onnx config for {package}:\n{config}")
    return config


def prep(package: str = "vits-piper-en", force: bool = False) -> None:
    """Download + extract one package's weights WITHOUT synthesizing. Idempotent.

    Args:
        package: Registered package name. `packages` lists them.
        force: Unused; downloads are already skipped when the file is present.
    """
    model_dir, spec = _ensure(package)
    files = sorted(p.name for p in model_dir.iterdir())
    print(f"[prep] ready: {package} ({spec['family']}) -> {model_dir} [{len(files)} entries]")


def say(
    text: str,
    output: str = "output.wav",
    package: str = "vits-piper-en",
    sid: int = 0,
    speed: float = 1.0,
    ref: str = "",
    lang: str = "",
    play: bool = False,
    force: bool = False,
) -> None:
    """Synthesize `text` with one sherpa-onnx package.

    Args:
        text: Text to synthesize.
        output: Output WAV path (mono, at the model's native rate).
        package: Registered package name.
        sid: Speaker id for multi-speaker checkpoints (kokoro, kitten).
        speed: Speaking speed multiplier; larger is faster.
        ref: Rejected. None of these families does voice cloning, and accepting
            a reference clip we cannot use would report a clone that never
            happened.
        lang: Checked against the package's languages, not passed to the engine —
            these are per-language checkpoints, not multilingual ones.
        play: Play the result with afplay instead of just saving it.
        force: Overwrite an existing output file.
    """
    import time

    import soundfile as sf
    from rich.console import Console
    from rich.panel import Panel

    console = Console()
    out_path = Path(output).resolve()
    if not force and out_path.exists() and out_path.stat().st_size > 0:
        console.print(f"[yellow]Skipped:[/yellow] {output} already exists (use --force)")
        return
    if ref:
        raise SystemExit(
            f"package '{package}' is a fixed-speaker model — it cannot clone "
            f"{ref}. Use --sid to pick one of its built-in speakers."
        )

    spec = _spec(package)
    if lang and lang not in spec["langs"]:
        raise SystemExit(
            f"package '{package}' speaks {'/'.join(spec['langs'])}, not '{lang}'"
        )
    if sid >= spec["speakers"]:
        raise SystemExit(f"--sid {sid} out of range: {package} has {spec['speakers']} speakers")

    console.print(
        Panel.fit(
            f"[bold cyan]sherpa-onnx[/bold cyan] v{VERSION}  "
            f"package=[yellow]{package}[/yellow]  family=[yellow]{spec['family']}[/yellow]  "
            f"sid=[yellow]{sid}[/yellow] speed=[yellow]{speed}[/yellow]",
            border_style="cyan",
        )
    )

    import sherpa_onnx

    config = _tts_config(package)
    tts = sherpa_onnx.OfflineTts(config)
    t0 = time.monotonic()
    audio = tts.generate(text, sid=sid, speed=speed)
    wall = time.monotonic() - t0

    if len(audio.samples) == 0:
        console.print(f"[red]{package} generated no samples[/red]")
        raise SystemExit(1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), audio.samples, audio.sample_rate, subtype="PCM_16")
    duration = len(audio.samples) / audio.sample_rate
    console.print(
        f"[green]Saved[/green] [bold]{out_path}[/bold] "
        f"({duration:.1f}s @ {audio.sample_rate} Hz, rtf={wall / duration:.2f})"
    )
    if play:
        import subprocess

        subprocess.run(["afplay", str(out_path)], check=True)


def info(package: str = "vits-piper-en") -> None:
    """Print one-line JSON with package metadata."""
    spec = _spec(package)
    print(
        json.dumps(
            {
                "name": package,
                "family": spec["family"],
                "repo": spec["repo"],
                "stack": "onnx",
                "runtime": "sherpa-onnx",
                "quant": spec["quant"],
                "mode": "auto",
                "voice_clone": False,
                "langs": spec["langs"],
                "speakers": spec["speakers"],
                "sample_rate": spec["sample_rate"],
                "url": spec["url"],
            },
            separators=(",", ":"),
        )
    )


def packages() -> None:
    """List every registered package: name, family, languages, speakers."""
    for name, spec in sorted(PACKAGES.items()):
        print(
            f"{name:24} {spec['family']:8} {'/'.join(spec['langs']):6} "
            f"{spec['speakers']:>4} spk  {spec['sample_rate']} Hz  {spec['quant']}"
        )


if __name__ == "__main__":
    import fire

    fire.Fire({"prep": prep, "say": say, "info": info, "packages": packages})
