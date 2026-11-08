# this_file: src/vexy_audioflexy/speech/io.py
"""Content identity for local speech runtimes."""

import hashlib
from pathlib import Path


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()
