# this_file: src/vexy_audioflexy/speech/workspace.py
"""Run isolated local speech adapters without bundling models or private workspace data."""

import itertools
import json
import os
import subprocess
import tempfile
from pathlib import Path

from .io import sha256
from .qwen import check_model
from .types import Word


def root_path(value: str = "") -> Path:
    """Require the actual adapter workspace only when its backend is selected."""
    value = (
        value
        or os.getenv("VEXY_AUDIOFLEXY_MODEL_WORKSPACE", "")
        or os.getenv("VEXY_SUBS_MODEL_WORKSPACE", "")
    )
    if not value:
        return Path(__file__).resolve().parents[1] / "tools"
    root = Path(value).expanduser().resolve()
    if not any(
        (root / name / "tools/speech-issue301.py").is_file()
        for name in ("asr", "myasr")
    ):
        raise FileNotFoundError("Model workspace lacks asr/tools/speech-issue301.py")
    return root


def adapter_root(value: str = "") -> Path:
    """Resolve installed resources or an explicitly supplied legacy workspace."""
    root = root_path(value)
    return root / ("asr" if (root / "asr").is_dir() else "myasr")


def signature(value: str) -> dict:
    """Invalidate cached inference when its local adapters/catalogue/runtime change."""
    root = adapter_root(value)
    paths = [
        p
        for p in root.rglob("*")
        if p.is_file() and p.suffix in {".py", ".json", ".toml", ".patch", ".cpp", ".h"}
    ]
    binary = (
        root.parent
        / ".external/audio.cpp-issue301/build/macos-metal-release/bin/audiocpp_cli"
    )
    if binary.is_file():
        paths.append(binary)
    signature = {str(p.relative_to(root.parent)): sha256(p) for p in sorted(paths)}
    from ..tools.asr.tools._audioflexy_paths import runtime_root

    engine = Path(
        os.getenv("AUDIOCPP_ISSUE301_ROOT", str(runtime_root("audio.cpp-issue301")))
    ).resolve()
    binary = engine / "build/macos-metal-release/bin/audiocpp_cli"
    signature[str(binary)] = sha256(binary) if binary.is_file() else None
    return signature


def transcribe(
    wav: Path, hints: str, language: str, model: str, root: str = ""
) -> tuple[list[Word], dict]:
    """Consume actual worker words; forced-aligned output retains recognition wording."""
    check_model(model)
    script = adapter_root(root) / "tools/speech-issue301.py"
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary) / "recognized.json"
        subprocess.run(
            [
                "uv",
                "run",
                "--script",
                str(script),
                "recognize",
                f"--audio={wav}",
                f"--output={output}",
                f"--model={model}",
                f"--language={language}",
                f"--prompt={hints}",
            ],
            check=True,
            timeout=3600,
            capture_output=True,
            text=True,
        )
        data = json.loads(output.read_text())
    words = [Word(w["text"], float(w["start"]), float(w["end"])) for w in data["words"]]
    if any(a.start > b.start for a, b in itertools.pairwise(words)):
        raise ValueError("Workspace words are out of order")
    return words, data["raw"]


def diarize(wav: Path, family: str, root: str = "") -> dict:
    """Keep continuous native speaker activity and explicit model channel capacity."""
    capacities = {"nemotron_3_diar": 8, "sortformer_diar": 4, "sortformer_diar_v2": 4}
    if family not in capacities:
        raise ValueError("Require a supported workspace diarization family")
    script = adapter_root(root) / "tools/speech-issue301.py"
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary) / "diarization"
        mode = "streaming" if family == "sortformer_diar_v2" else "offline"
        subprocess.run(
            [
                "uv",
                "run",
                "--script",
                str(script),
                "task",
                f"--audio={wav}",
                f"--output={output}",
                f"--family={family}",
                f"--mode={mode}",
            ],
            check=True,
            timeout=3600,
            capture_output=True,
            text=True,
        )
        report = json.loads((output / "report.json").read_text())
    return {
        "turns": report["result"]["turns"],
        "capacity": capacities[family],
        "backend": "workspace",
        "native": report,
    }
