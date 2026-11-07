# this_file: tests/test_ported_audio_metrics_cli.py
"""Fire may parse an unquoted comma-separated metric option as a tuple."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("aud_metrics", (Path(__file__).resolve().parents[1] / "src/vexy_audioflexy/tools/fx") / "tools/aud-metrics.py")
wrapper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wrapper)


@pytest.mark.parametrize("value", ["stoi,si_sdr", ("stoi", "si_sdr"), ["stoi", "si_sdr"]])
def test_metric_selection_accepts_fire_csv_shapes(value):
    assert wrapper.selected_metrics(value) == ["stoi", "si_sdr"]


def test_empty_selection_uses_the_documented_defaults():
    assert wrapper.selected_metrics("") == list(wrapper.NOREF + wrapper.INTRUSIVE)
