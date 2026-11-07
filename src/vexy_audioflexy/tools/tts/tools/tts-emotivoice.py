#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "fire", "rich", "torch", "torchaudio", "transformers", "numpy", "soundfile",
#   "yacs", "g2p-en", "jieba", "pypinyin", "pypinyin-dict", "huggingface-hub",
#   "tqdm", "inflect", "numba", "scipy", "librosa",
#   "setuptools<81",
# ]
# ///
# this_file: src/vexy_audioflexy/tools/tts/tools/tts-emotivoice.py
"""EmotiVoice. Backend: torch/local checkout, languages: en/zh, 16 kHz mono, modes: desi, capabilities: prompt_control.

Voice **design** by natural-language prompt: the emotion is a text string ("happy",
"angry", "悲伤") encoded by a BERT style encoder and fed to the acoustic model
alongside the phonemes. 2000+ speaker ids, no cloning.

Not pip-installable in any useful sense — the `EmotiVoice` package on PyPI contains
only the `text/` helper module, not the engine. This wrapper clones the repo into
`local/EmotiVoice` and assembles the checkpoint layout the code expects:

    outputs/prompt_tts_open_source_joint/ckpt/g_00140000   (acoustic + vocoder)
    outputs/style_encoder/ckpt/checkpoint_163431           (BERT style encoder)

The original weights were published on Google Drive; `FPVG/EmotiVoice-pretrained-models`
on HuggingFace mirrors the same three files, ungated.

`config/joint/config.py` computes its root as `os.path.dirname(os.path.abspath("__file__"))`
— that is the literal string `"__file__"`, so the root is simply the working
directory. Every path in the config is therefore relative to wherever the process
started, and the wrapper chdirs into the checkout before importing anything from it.
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
from pathlib import Path
from _paths import runtime_path

VERSION = "1.0.0"

REPO_URL = "https://github.com/netease-youdao/EmotiVoice"
CKPT_REPO = "FPVG/EmotiVoice-pretrained-models"
RATE = 16000
MAX_WAV_VALUE = 32768.0
LANGS = ["en", "zh"]


def _root() -> Path:
    """Checkout root. `EMOTIVOICE_ROOT` overrides it."""
    env = os.environ.get("EMOTIVOICE_ROOT")
    if env:
        return Path(env).expanduser()
    return runtime_path('EmotiVoice')


def _ensure_checkout() -> Path:
    root = _root()
    if not (root / "demo_page.py").exists():
        root.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(root)], check=True)
    return root


def _ensure_checkpoints() -> Path:
    """Place the three mirrored files where the upstream config looks for them."""
    from huggingface_hub import hf_hub_download

    root = _ensure_checkout()
    layout = {
        "g_00140000": root / "outputs" / "prompt_tts_open_source_joint" / "ckpt",
        "do_00140000": root / "outputs" / "prompt_tts_open_source_joint" / "ckpt",
        "checkpoint_163431": root / "outputs" / "style_encoder" / "ckpt",
    }
    for name, target in layout.items():
        target.mkdir(parents=True, exist_ok=True)
        dest = target / name
        if not dest.exists():
            dest.write_bytes(Path(hf_hub_download(CKPT_REPO, name)).read_bytes())
    return root


@contextlib.contextmanager
def _in_checkout():
    """Run inside the checkout: its config resolves every path against the cwd."""
    root = _ensure_checkpoints()
    previous = Path.cwd()
    os.chdir(root)
    sys.path.insert(0, str(root))
    try:
        yield root
    finally:
        os.chdir(previous)


def _models(device: str):
    import torch
    from yacs import config as CONFIG

    from config.joint.config import Config
    from models.prompt_tts_modified.jets import JETSGenerator
    from models.prompt_tts_modified.simbert import StyleEncoder
    from transformers import AutoTokenizer

    config = Config()
    with open(config.model_config_path) as fin:
        conf = CONFIG.load_cfg(fin)
    conf.n_vocab = config.n_symbols
    conf.n_speaker = config.speaker_n_labels

    # WangZeJun/simbert-base-chinese ships a config.json with no `model_type`, so
    # AutoModel/AutoTokenizer refuse it on current transformers ("Unrecognized
    # model"). It is a plain BERT; naming the classes directly loads it fine.
    _patch_auto_bert()

    style_encoder = StyleEncoder(config)
    ckpt = torch.load(
        f"{config.output_directory}/style_encoder/ckpt/checkpoint_163431",
        map_location="cpu",
    )
    # The published style-encoder state dict is DataParallel-wrapped: every key
    # carries a "module." prefix the bare model does not have.
    style_encoder.load_state_dict(
        {k[7:]: v for k, v in ckpt["model"].items()}, strict=False
    )

    generator = JETSGenerator(conf).to(device)
    am = torch.load(
        f"{config.output_directory}/prompt_tts_open_source_joint/ckpt/g_00140000",
        map_location=device,
    )
    generator.load_state_dict(am["generator"])
    generator.eval()

    from transformers import BertTokenizer

    tokenizer = BertTokenizer.from_pretrained(config.bert_path)
    with open(config.token_list_path) as f:
        token2id = {t.strip(): i for i, t in enumerate(f.readlines())}
    with open(config.speaker2id_path, encoding="utf-8") as f:
        speaker2id = {t.strip(): i for i, t in enumerate(f.readlines())}
    return config, style_encoder, generator, tokenizer, token2id, speaker2id


def _patch_auto_bert() -> None:
    """Point the toolkit's `AutoModel.from_pretrained` at BertModel.

    `models/prompt_tts_modified/simbert.py` calls AutoModel on a repo whose config
    predates the `model_type` key. Rather than fork the file, the class it reaches
    for is swapped for the concrete one.
    """
    import transformers
    from transformers import BertModel

    transformers.AutoModel.from_pretrained = staticmethod(BertModel.from_pretrained)


def _style_embedding(prompt: str, tokenizer, style_encoder):
    import torch

    encoded = tokenizer([prompt], return_tensors="pt")
    with torch.no_grad():
        output = style_encoder(
            input_ids=encoded["input_ids"],
            token_type_ids=encoded["token_type_ids"],
            attention_mask=encoded["attention_mask"],
        )
    return output["pooled_output"].cpu().squeeze().numpy()


def prep(force: bool = False) -> None:
    """Clone the repo and place the checkpoints. Idempotent.

    Args:
        force: Unused; an existing checkout and cached weights are reused.
    """
    root = _ensure_checkpoints()
    print(f"[prep] ready: {REPO_URL} -> {root}")


def say(
    text: str,
    output: str = "output.wav",
    prompt: str = "",
    speaker: str = "8051",
    lang: str = "en",
    ref: str = "",
    device: str = "cpu",
    play: bool = False,
    force: bool = False,
) -> None:
    """Synthesize `text` in the emotion described by `prompt`.

    Args:
        text: Text to synthesize.
        output: Output WAV path (16 kHz mono).
        prompt: Natural-language style description — "happy", "angry", "whispering",
            "悲伤". Empty means neutral: the text itself is used as its own style
            prompt, which is what upstream does.
        speaker: Speaker id from the checkpoint's 2000+ ids.
        lang: `en` or `zh`. Checked; the front-end handles mixed text either way.
        ref: Rejected. EmotiVoice designs a voice from a prompt; it does not clone.
        device: `cpu` (default) or `mps`.
        play: Play the result with afplay instead of just saving it.
        force: Overwrite an existing output file.
    """
    import time

    import numpy as np
    import soundfile as sf
    import torch
    from rich.console import Console
    from rich.panel import Panel

    console = Console()
    out_path = Path(output).resolve()
    if not force and out_path.exists() and out_path.stat().st_size > 0:
        console.print(f"[yellow]Skipped:[/yellow] {output} already exists (use --force)")
        return
    if ref:
        raise SystemExit(
            f"EmotiVoice designs a voice from --prompt and cannot clone {ref}."
        )
    if lang not in LANGS:
        raise SystemExit(f"EmotiVoice speaks {'/'.join(LANGS)}, not '{lang}'")

    console.print(
        Panel.fit(
            f"[bold cyan]EmotiVoice[/bold cyan] v{VERSION}  "
            f"speaker=[yellow]{speaker}[/yellow]  prompt=[yellow]{prompt or '(neutral)'}[/yellow]",
            border_style="cyan",
        )
    )

    with _in_checkout() as root:
        from frontend import G2p, g2p_cn_en, read_lexicon

        config, style_encoder, generator, tokenizer, token2id, speaker2id = _models(device)
        # Fire types a bare 8051 as an int; the id table is keyed by string.
        speaker = str(speaker)
        if speaker not in speaker2id:
            raise SystemExit(f"unknown speaker id '{speaker}'")

        lexicon = read_lexicon(str(root / "lexicon" / "librispeech-lexicon.txt"))
        phonemes = g2p_cn_en(text, G2p(), lexicon)
        token_ids = [token2id[ph] for ph in phonemes.split() if ph in token2id]
        if not token_ids:
            raise SystemExit("front-end produced no phonemes for this text")

        style = _style_embedding(prompt or text, tokenizer, style_encoder)
        content = _style_embedding(text, tokenizer, style_encoder)

        sequence = torch.from_numpy(np.array(token_ids)).long().unsqueeze(0).to(device)
        lengths = torch.from_numpy(np.array([len(token_ids)])).to(device)
        t0 = time.monotonic()
        with torch.no_grad():
            out = generator(
                inputs_ling=sequence,
                inputs_style_embedding=torch.from_numpy(style).unsqueeze(0).to(device),
                input_lengths=lengths,
                inputs_content_embedding=torch.from_numpy(content).unsqueeze(0).to(device),
                inputs_speaker=torch.from_numpy(np.array([speaker2id[speaker]])).to(device),
                alpha=1.0,
            )
        wall = time.monotonic() - t0
        audio = (out["wav_predictions"].squeeze() * MAX_WAV_VALUE).cpu().numpy().astype("int16")

    if audio.size == 0 or not np.any(audio):
        console.print("[red]EmotiVoice produced silence[/red]")
        raise SystemExit(1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), audio, RATE, subtype="PCM_16")
    duration = audio.size / RATE
    console.print(
        f"[green]Saved[/green] [bold]{out_path}[/bold] "
        f"({duration:.1f}s @ {RATE} Hz, rtf={wall / duration:.2f})"
    )
    if play:
        subprocess.run(["afplay", str(out_path)], check=True)


def info() -> None:
    """Print one-line JSON with model metadata."""
    print(json.dumps(
        {
            "name": "emotivoice",
            "repo": "netease-youdao/EmotiVoice",
            "stack": "torch",
            "runtime": "EmotiVoice (PyTorch, local checkout)",
            "quant": "fp32",
            "mode": "desi",
            "voice_clone": False,
            "langs": LANGS,
            "sample_rate": RATE,
            "capabilities": ["prompt_control"],
        },
        separators=(",", ":"),
    ))


def speakers(limit: int = 20) -> None:
    """Print the first `limit` speaker ids in the checkpoint."""
    with _in_checkout():
        from config.joint.config import Config

        for name in Config().speakers[:limit]:
            print(name)


if __name__ == "__main__":
    import fire

    fire.Fire({"prep": prep, "say": say, "info": info, "speakers": speakers})
