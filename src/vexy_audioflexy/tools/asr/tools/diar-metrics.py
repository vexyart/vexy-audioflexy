#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = ["fire", "pyannote.metrics==4.1"]
# ///
# this_file: src/vexy_audioflexy/tools/asr/tools/diar-metrics.py
"""Score reference speaker turns using pyannote's optimal speaker mapping.

Contract: https://pyannote.github.io/pyannote-metrics/reference.html
References use {duration_s, turns:[{start,end,speaker}]}; optional annotation_note
is copied to prevent an automatically derived reference appearing hand labelled.
"""
from __future__ import annotations

import json
import math
from pathlib import Path


def measure(reference: list[dict], hypothesis: list[dict], duration: float,
            collar: float = .25) -> dict:
    """Score the complete file, including false alarms; keep overlapping speakers."""
    from pyannote.core import Annotation, Segment, Timeline
    from pyannote.metrics.diarization import DiarizationErrorRate

    if not math.isfinite(duration) or duration <= 0 or not math.isfinite(collar) or collar < 0:
        raise SystemExit("Require finite positive duration and nonnegative collar")
    annotations = []
    for rows in (reference, hypothesis):
        annotation = Annotation()
        for index, row in enumerate(rows):
            start, end, speaker = row["start"], row["end"], row["speaker"]
            if not math.isfinite(start) or not math.isfinite(end) or not 0 <= start < end <= duration + .001:
                raise SystemExit("Invalid metric turn boundary")
            annotation[Segment(start, end), index] = str(speaker)
        annotations.append(annotation)
    if not reference:
        raise SystemExit("DER requires a nonempty reference")
    metric = DiarizationErrorRate(collar=collar, skip_overlap=False)
    return metric(*annotations, uem=Timeline([Segment(0, duration)]), detailed=True)


def score(reference: str, hypothesis: str, output: str = "", collar: float = .25) -> dict:
    """Write DER components with fixture/reference caveats and exact inputs."""
    ref, hyp = json.loads(Path(reference).read_text()), json.loads(Path(hypothesis).read_text())
    if abs(ref["duration_s"] - hyp["duration_s"]) > .001:
        raise SystemExit("Reference and hypothesis durations differ")
    report = {"reference": str(Path(reference).resolve()), "hypothesis": str(Path(hypothesis).resolve()),
              "collar_s": collar, "skip_overlap": False, "metrics": measure(ref["turns"], hyp["turns"], ref["duration_s"], collar),
              "annotation_note": ref.get("annotation_note", "User-provided reference")}
    if output:
        destination = Path(output).resolve()
        if destination in {Path(reference).resolve(), Path(hypothesis).resolve()}:
            raise SystemExit("Metric output must differ from inputs")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    import fire

    fire.Fire(score)
