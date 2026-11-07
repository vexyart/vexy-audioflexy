#!/usr/bin/env bash
# this_file: build.sh
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p dist
artifacts=$(mktemp -d "$PWD/dist/build.XXXXXX")
uv build --no-sources --out-dir "$artifacts" >&2
python3 - "$artifacts" "${VAFX_RELEASE_TAG:-}" <<'PYTHON'
import email
from pathlib import Path
import sys
import tarfile
import zipfile

root, tag = Path(sys.argv[1]), sys.argv[2]
wheels, sources = list(root.glob("*.whl")), list(root.glob("*.tar.gz"))
assert len(wheels) == len(sources) == 1, "Expected one wheel and one sdist"
with zipfile.ZipFile(wheels[0]) as archive:
    names = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
    assert len(names) == 1, "Expected one wheel metadata file"
    wheel = email.message_from_bytes(archive.read(names[0]))
with tarfile.open(sources[0]) as archive:
    names = [name for name in archive.getnames() if name.count("/") == 1 and name.endswith("/PKG-INFO")]
    assert len(names) == 1, "Expected one sdist metadata file"
    source = email.message_from_bytes(archive.extractfile(names[0]).read())
assert wheel["Name"] == source["Name"] == "vexy-audioflexy", "Wrong distribution"
assert wheel["Version"] == source["Version"], "Distribution versions disagree"
if tag:
    assert wheel["Version"] == tag.removeprefix("v"), "Built version differs from release tag"
PYTHON
printf '%s\n' "$artifacts"
