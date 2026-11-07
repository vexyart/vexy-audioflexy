# this_file: tests/tools/asr/conftest.py
"""Let dynamic source-style imports find their installed-resource siblings."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src/vexy_audioflexy/tools/asr/tools"))
