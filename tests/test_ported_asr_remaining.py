# this_file: tests/test_ported_asr_remaining.py
"""Verify relocated ASR resources without loading or downloading models."""

import ast
import importlib.util
import json
import runpy
import socket
import subprocess
import sys
import tomllib
from importlib.resources import files
from pathlib import Path

import pytest

ROOT = Path(str(files("vexy_audioflexy").joinpath("tools/asr")))
COMMANDS = sorted(
    [str(p.relative_to(ROOT)) for p in (ROOT / "tools").glob("*.py")
     if 'if __name__ == "__main__":' in p.read_text() and p.name != "asr-transcribecpp.py"]
    + ["asr_all.py", "diarize_all.py", "translate_all.py"]
)


def paths_module():
    spec = importlib.util.spec_from_file_location("asr_paths", ROOT / "tools/_audioflexy_paths.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("relative", COMMANDS)
def test_every_relocated_command_help_when_network_and_engines_blocked_then_exits_zero(relative, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Help must not download or invoke native engines")

    monkeypatch.syspath_prepend(str(ROOT))
    monkeypatch.syspath_prepend(str(ROOT / "tools"))
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "check_output", forbidden)
    monkeypatch.setattr(sys, "argv", [str(ROOT / relative), "--help"])
    with pytest.raises(SystemExit) as caught:
        runpy.run_path(str(ROOT / relative), run_name="__main__")
    assert caught.value.code in (0, None)


def test_models_root_when_configured_then_matches_cli_toml_and_override(tmp_path, monkeypatch):
    module = paths_module()
    monkeypatch.delenv("VAFX_MODELS_DIR", raising=False)
    monkeypatch.setattr(module, "user_config_path", lambda *a, **k: tmp_path)
    monkeypatch.setattr(module, "data_root", lambda: tmp_path / "appdata")
    assert module.models_root() == tmp_path / "appdata/models"
    (tmp_path / "config.toml").write_text(f'models_dir = "{tmp_path}/saved"\n')
    assert module.models_root() == tmp_path / "saved"
    monkeypatch.setenv("VAFX_MODELS_DIR", str(tmp_path / "override"))
    assert module.models_root() == tmp_path / "override"
    monkeypatch.setenv("VAFX_MODELS_DIR", "relative")
    with pytest.raises(ValueError, match="absolute"):
        module.models_root()


def test_runtime_root_and_host_staging_when_installed_then_write_only_user_runtime(tmp_path, monkeypatch):
    module = paths_module()
    monkeypatch.setenv("VAFX_RUNTIME_DIR", str(tmp_path / "runtimes"))
    destination = module.runtime_root("coreai-diar")
    monkeypatch.setenv("VAFX_RUNTIME_DIR", "relative")
    with pytest.raises(ValueError, match="absolute"):
        module.runtime_root("coreai-diar")
    monkeypatch.setenv("VAFX_RUNTIME_DIR", str(tmp_path / "runtimes"))
    source = ROOT / "native/coreai-diar"
    original = (source / "Package.swift").read_bytes()
    module.stage_native_host(source, destination)
    assert (destination / "Sources/coder-diar/main.swift").is_file()
    assert (destination / "THIRD_PARTY_LICENSE.txt").is_file()
    (destination / "Package.swift").write_text("local change")
    assert (source / "Package.swift").read_bytes() == original
    with pytest.raises(FileExistsError, match="fresh runtime directory"):
        module.stage_native_host(source, destination)
    assert (destination / "Package.swift").read_text() == "local change"


def test_consumers_when_no_private_inputs_then_require_explicit_paths(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "tools"))
    batch = runpy.run_path(str(ROOT / "asr_all.py"))
    with pytest.raises(TypeError):
        batch["main"]()
    with pytest.raises(ValueError, match="eval_dir"):
        batch["eval_clips"]("en")
    diarization = runpy.run_path(str(ROOT / "diarize_all.py"))
    with pytest.raises(TypeError):
        diarization["main"]("input.wav")


def test_resource_closure_when_batch_registry_loaded_then_every_adapter_exists(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "tools"))
    batch = runpy.run_path(str(ROOT / "asr_all.py"))
    for model in batch["MODELS"]:
        assert (ROOT / model["script"]).is_file(), model["name"]
    catalog = json.loads((ROOT / "catalogs/issue301-audiocpp.json").read_text())
    for row in catalog["models"]:
        assert (ROOT / "model_specs/issue301" / (row["family"] + ".json")).is_file()


def test_all_script_metadata_when_ported_then_is_valid_pep723():
    for path in ROOT.rglob("*.py"):
        text = path.read_text()
        ast.parse(text)
        if "# /// script" not in text:
            continue
        block = text.split("# /// script", 1)[1].split("# ///", 1)[0]
        metadata = tomllib.loads("\n".join(line.removeprefix("# ") for line in block.splitlines()))
        assert metadata["dependencies"], path.name
