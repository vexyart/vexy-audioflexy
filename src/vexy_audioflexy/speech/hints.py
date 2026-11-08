# this_file: src/vexy_audioflexy/speech/hints.py
"""User-owned UTF-8 recognition hints; never bundled into distributions."""

# Tagged recording metadata reaches ASR even when a Names:/Terms: glossary exists.
METADATA = ("Title:", "Description:")
REFERENCE_MARGIN = 1.5


def vocabulary(prompt: str) -> list[str]:
    """Use optional Names:/Terms: lines for casing; free-form hints still reach ASR."""
    terms = []
    for line in prompt.splitlines():
        if line.startswith(("Names:", "Terms:")):
            terms.extend(
                t.strip().rstrip(".") for t in line.split(":", 1)[1].split(",")
            )
    return [t for t in terms if t]


def metadata(prompt: str) -> str:
    """Return tagged Title:/Description: lines, which describe but are not speech."""
    return "\n".join(line for line in prompt.splitlines() if line.startswith(METADATA))


def excerpt(reference: list[tuple], start: float, end: float) -> str:
    """Only the reference text overlapping one audio chunk, never the whole talk."""
    lines = []
    for first, last, text in reference:
        overlaps = first < end + REFERENCE_MARGIN and last > start - REFERENCE_MARGIN
        if overlaps and (not lines or lines[-1] != text):
            lines.append(text)
    return " ".join(lines)
