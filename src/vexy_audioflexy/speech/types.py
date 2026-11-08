# this_file: src/vexy_audioflexy/speech/types.py
"""Timed words and speaker turns shared across inference and rendering."""

import math
from dataclasses import dataclass


@dataclass
class Word:
    text: str
    start: float
    end: float
    speaker: str = "unassigned"
    ambiguous: bool = False

    def __post_init__(self) -> None:
        if not self.text.strip() or not math.isfinite(self.start + self.end):
            raise ValueError("Require nonempty text and finite word times")
        if self.start < 0 or self.end < self.start:
            raise ValueError("Invalid word timing")


@dataclass(frozen=True)
class Turn:
    start: float
    end: float
    speaker: str

    def __post_init__(self) -> None:
        if not math.isfinite(self.start + self.end) or not 0 <= self.start < self.end:
            raise ValueError("Invalid speaker turn")
