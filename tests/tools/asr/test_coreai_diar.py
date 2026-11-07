# this_file: tests/tools/asr/test_coreai_diar.py
"""Exact CoreAI asset selection and checksum enforcement."""
import array
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


def module():
    path = Path(__file__).resolve().parents[3] / "src/vexy_audioflexy/tools/asr" / "tools/diarize-coreai.py"
    spec = importlib.util.spec_from_file_location("coreai_diar", path)
    wrapper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wrapper)
    return wrapper


def test_profiles_choose_macos_bundles_and_reject_unknown_packages():
    wrapper = module()
    assert wrapper.bundle_name("offline") == "n3d_offline_float16.aimodel"
    assert wrapper.bundle_name("streaming") == "n3d_streaming_float16.aimodel"
    with pytest.raises(SystemExit):
        wrapper.bundle_name("h18p")
    assert wrapper.host_mode("streaming", "very_low") == "vll"
    assert wrapper.host_mode("streaming", "ultra_low") == "ull"
    with pytest.raises(SystemExit):
        wrapper.host_mode("streaming", "unknown")


def test_asset_hash_is_checked_before_loading(tmp_path):
    wrapper = module()
    asset = tmp_path / "example.f32le"
    asset.write_bytes(b"finite test bytes")
    checksum = hashlib.sha256(asset.read_bytes()).hexdigest()
    (tmp_path / "metadata.json").write_text(json.dumps({"assets": {asset.name: {"sha256": checksum}}}))
    wrapper.verify_assets(tmp_path)
    asset.write_bytes(b"broken")
    with pytest.raises(SystemExit, match="hash"):
        wrapper.verify_assets(tmp_path)


def test_nonfinite_out_of_range_and_truncated_probabilities_are_rejected(tmp_path):
    wrapper = module()
    path = tmp_path / "probabilities.f32le"
    path.write_bytes(array.array("f", [.5] * 8).tobytes())
    wrapper.verify_probabilities(path, 1)
    for value in [float("nan"), 1.1, -1.]:
        path.write_bytes(array.array("f", [value] * 8).tobytes())
        with pytest.raises(SystemExit, match="invalid"):
            wrapper.verify_probabilities(path, 1)
    with pytest.raises(SystemExit, match="geometry"):
        wrapper.verify_probabilities(path, 2)
