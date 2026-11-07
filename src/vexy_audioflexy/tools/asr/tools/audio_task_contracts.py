# this_file: src/vexy_audioflexy/tools/asr/tools/audio_task_contracts.py
"""Native task request and output contracts, independent of process execution."""

import itertools
import json
import math
import runpy
from pathlib import Path


def turn_result(stdout: str) -> dict:
    """Read the actual Smart Turn schema, not a transcript or invented speaker turn."""
    lines = [
        line for line in stdout.splitlines() if line.startswith("custom_schema_output=")
    ]
    if len(lines) != 1:
        raise ValueError("Expected one native turn-completion result")
    result = json.loads(lines[0].split("=", 1)[1])
    data = result.get("data", {})
    probability = data.get("probability")
    if (
        result.get("schema") != "smart_turn.v1"
        or type(probability) not in {float, int}
        or not math.isfinite(probability)
        or not 0 <= probability <= 1
        or type(data.get("complete")) is not bool
    ):
        raise ValueError("Invalid Smart Turn schema or completion probability")
    return data


def alignment_words(raw: list, duration: float) -> list[dict]:
    """Convert inspected native sample offsets to seconds; reject missing/bad words."""
    if not isinstance(raw, list) or not raw:
        raise ValueError("Alignment returned no word array")
    words = []
    for row in raw:
        start, end, text = (
            row.get("start_sample"),
            row.get("end_sample"),
            row.get("word"),
        )
        if (
            type(start) is not int
            or type(end) is not int
            or not 0 <= start <= end <= duration * 16000 + 16
            or not isinstance(text, str)
            or not text.strip()
        ):
            raise ValueError("Alignment returned invalid words or sample offsets")
        words.append({"text": text, "start": start / 16000, "end": end / 16000})
    if any(a["start"] > b["start"] for a, b in itertools.pairwise(words)):
        raise ValueError("Alignment word order is nonmonotonic")
    return words


def result_for(task: str, raw: Path, stdout: str, duration: float) -> dict:
    """Validate task-specific outputs before publishing a receipt or replacement."""
    if task == "turn":
        return {"completion": turn_result(stdout)}
    if task == "diar":
        normalize = runpy.run_path(str(Path(__file__).with_name("diarize-native.py")))[
            "normalize"
        ]
        return {
            "turns": normalize(
                json.loads(raw.read_text()) if raw.exists() else [],
                "audiocpp",
                duration,
            )
        }
    if task == "align":
        return {"words": alignment_words(json.loads(raw.read_text()), duration)}
    import numpy as np
    import soundfile as sf

    samples, rate = sf.read(raw, always_2d=True)
    if not len(samples) or not np.isfinite(samples).all():
        raise ValueError("AudioSR returned empty or non-finite audio")
    if abs(len(samples) / rate - duration) > 0.02:
        raise ValueError("AudioSR changed source duration")
    return {
        "audio": raw.name,
        "sample_rate": rate,
        "channels": samples.shape[1],
        "frames": len(samples),
    }


TASKS = {
    "nemotron_3_diar": "diar",
    "sortformer_diar": "diar",
    "sortformer_diar_v2": "diar",
    "qwen3_forced_aligner": "align",
    "mms_forced_aligner": "align",
    "smart_turn": "turn",
    "audiosr": "s2s",
}


def command(
    binary: Path,
    family: str,
    model: Path,
    audio: Path,
    raw: Path,
    backend: str,
    text: str,
    language: str,
    mode: str,
    options: str,
) -> list[str]:
    """Use official task/output flags; model-specific options remain explicit."""
    task = TASKS[family]
    args = [
        str(binary),
        "--task",
        task,
        "--family",
        family,
        "--model",
        str(model),
        "--backend",
        backend,
        "--audio",
        str(audio),
        "--mode",
        mode,
        "--log",
    ]
    if task in {"align", "diar", "s2s"}:
        args.extend(
            [
                {"align": "--words-out", "diar": "--turns-out", "s2s": "--out"}[task],
                str(raw),
            ]
        )
    if task == "align":
        args.extend(["--text", text])
        if language:
            args.extend(["--language", language])
    for option in filter(None, (p.strip() for p in options.split(";"))):
        args.extend(["--request-option", option])
    return args
