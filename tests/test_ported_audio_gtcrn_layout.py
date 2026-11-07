# this_file: tests/test_ported_audio_gtcrn_layout.py
"""Published GTCRN layouts must restore grouped kernels without changing bits."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


def layout():
    path = (Path(__file__).resolve().parents[1] / "src/vexy_audioflexy/tools/fx") / "tools/gtcrn_layout.py"
    spec = importlib.util.spec_from_file_location("gtcrn_layout", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("group", [1, 2, 4])
def test_restore_grouped_transposed_convolution_preserves_every_element(group):
    original = np.arange(8 * 3 * 2 * 5, dtype=np.float32).reshape(8, 3, 2, 5)
    converted = original.reshape(group, 8 // group, 3, 2, 5).transpose(0, 2, 3, 4, 1).reshape(group * 3, 2, 5, 8 // group)
    restored = layout().restore(converted, original.shape, "ConvTranspose", group)
    np.testing.assert_array_equal(restored.view(np.uint32), original.view(np.uint32))


def test_restore_convolution_and_scalar_preserve_shape_and_bits():
    original = np.arange(120, dtype=np.float32).reshape(4, 3, 2, 5)
    restored = layout().restore(original.transpose(0, 2, 3, 1), original.shape, "Conv")
    np.testing.assert_array_equal(restored, original)
    assert layout().restore(np.array([2], dtype=np.float32), ()).shape == ()


def test_restore_rejects_wrong_dtype_and_nonfinite_values():
    for value in [np.array([1], dtype=np.float64), np.array([np.nan], dtype=np.float32)]:
        with pytest.raises(ValueError):
            layout().restore(value, (1,))
