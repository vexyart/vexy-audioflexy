# this_file: tests/test_chatterbox_mtl_codeccpp.py
"""Text preparation for Chatterbox Multilingual T3 through codec.cpp."""

import runpy
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "src/vexy_audioflexy/tools/tts/tools/tts-chatterbox-mtl-codeccpp.py"
sys.path.insert(0, str(SCRIPT.parent))  # the script imports its sibling _paths
MOD = runpy.run_path(str(SCRIPT))


def test_prepare_text_when_polish_then_lang_prefix_lowercase_nfkd():
    out = MOD["prepare_text"]("Dzień Dobry", "pl")
    assert out.startswith("[pl]"), "T3 needs the [lang] prefix"
    assert out == "[pl]dzien\u0301 dobry", "lowercase and NFKD: n + U+0301, as MTLTokenizer.encode does"


@pytest.mark.parametrize("lang", ["zh", "ja", "xx"])
def test_prepare_text_when_unsupported_language_then_exits(lang):
    with pytest.raises(SystemExit):
        MOD["prepare_text"]("text", lang)
