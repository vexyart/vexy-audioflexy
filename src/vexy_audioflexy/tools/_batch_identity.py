# this_file: src/vexy_audioflexy/tools/_batch_identity.py
"""Bind ASR/TTS batch resume to the request and successful output bytes."""
import hashlib
import json
from pathlib import Path
import tempfile


def file_identity(path: Path) -> str:
    """Stream input/output bytes so model-sized inputs do not enter memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def request_identity(model: dict, adapter: Path, arguments: list, paths: list[Path], context: dict) -> str:
    """Include selected options, conditioned text, adapter and input identities."""
    payload = {"model": model["name"], "adapter": file_identity(adapter),
               "args": [str(arg) for arg in arguments], "env": model.get("env", {}),
               "inputs": [[str(path.resolve()), file_identity(path)] for path in paths],
               "context": context}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def reusable_output(path: Path, request: str) -> bool:
    """Only a complete successful receipt for these exact bytes permits reuse."""
    try:
        receipt = json.loads(path.with_name(path.name + ".request.json").read_text())
        return path.stat().st_size > 0 and receipt == {"request": request, "output_sha256": file_identity(path)}
    except (OSError, ValueError):
        return False


def record_output(path: Path, request: str) -> None:
    """Write evidence only after the batch has checked successful output."""
    receipt = path.with_name(path.name + ".request.json")
    payload = {"request": request, "output_sha256": file_identity(path)}
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(json.dumps(payload, sort_keys=True) + "\n")
    try:
        temporary.replace(receipt)
    finally:
        temporary.unlink(missing_ok=True)
