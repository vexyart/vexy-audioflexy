# this_file: tests/test_dispatch.py
"""Synthetic dispatcher checks; no model imports or downloads."""

import json
import shutil
import subprocess

import pytest

from vexy_audioflexy import dispatch

_HEADER = '# /// script\n# dependencies = []\n# ///\n'


@pytest.fixture
def resources(tmp_path, monkeypatch):
    root = tmp_path / "installed tools"
    for name in dispatch._NAMESPACES:
        (root / name / "tools").mkdir(parents=True)
    monkeypatch.setattr(dispatch, "_ROOT", root)
    return root


def test_list_classifies_resources_without_running_or_importing(resources):
    adapter = resources / "asr" / "tools"
    (adapter / "synthetic.py").write_text(_HEADER + 'raise RuntimeError("do not import")\n')
    (adapter / "helper.py").write_text("def helper(): pass\n")
    (adapter / "legacy.py").write_text('if __name__ == "__main__": pass\n')
    (adapter / "native.cpp").write_text("// native source\n")
    (adapter / "verify.sh").write_text("exit 0\n")
    (adapter / "__init__.py").write_text("")
    kinds = {item["tool"]: item["kind"] for item in dispatch.Tools().list("asr")}
    assert kinds == {"synthetic.py": "runnable", "helper.py": "helper",
                     "legacy.py": "needs-metadata", "native.cpp": "native-source",
                     "verify.sh": "shell-script"}
    assert dispatch.Tools().list() == dispatch.Tools().list("asr")


@pytest.mark.parametrize("namespace,tool", [("../asr", "example"), ("asr", "../example"),
                                          ("asr", "tools/example"), ("asr", "a\\b")])
def test_run_rejects_unlisted_paths(resources, namespace, tool):
    with pytest.raises(ValueError):
        dispatch.Tools().run(namespace, tool)


def test_run_rejects_missing_helper_and_symlink_escape(resources, tmp_path):
    with pytest.raises(ValueError, match="not installed"):
        dispatch.Tools().run("asr", "absent")
    (resources / "asr" / "tools" / "helper.py").write_text("def helper(): pass\n")
    with pytest.raises(ValueError, match="helper"):
        dispatch.Tools().run("asr", "helper")
    outside = tmp_path / "outside.py"
    outside.write_text(_HEADER)
    (resources / "asr" / "tools" / "escape.py").symlink_to(outside)
    with pytest.raises(ValueError, match="not installed"):
        dispatch.Tools().run("asr", "escape")


def test_run_preserves_argv_output_and_nonzero_status(resources, monkeypatch):
    script = resources / "asr" / "tools" / "synthetic.py"
    script.write_text(_HEADER)
    monkeypatch.setattr(dispatch.shutil, "which", lambda name: "/some path/uv")
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 17)

    monkeypatch.setattr(dispatch.subprocess, "run", run)
    with pytest.raises(SystemExit) as failure:
        dispatch.Tools().run("asr", "synthetic", "--output", "a b.wav", "--name=$(false)")
    assert failure.value.code == 17
    assert calls == [(["/some path/uv", "run", "--script", str(script), "--output",
                       "a b.wav", "--name=$(false)"], {"check": False})]


def test_run_reports_missing_uv(resources, monkeypatch):
    (resources / "asr" / "tools" / "synthetic.py").write_text(_HEADER)
    monkeypatch.setattr(dispatch.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="install it"):
        dispatch.Tools().run("asr", "synthetic")


@pytest.mark.parametrize("method,target", list(dispatch._BATCHES.items()))
def test_batch_methods_route_to_consumer(monkeypatch, method, target):
    calls = []
    monkeypatch.setattr(dispatch.Tools, "run", lambda self, *args: calls.append(args))
    getattr(dispatch.Tools(), method)("--output", "a b")
    assert calls == [(*target, "--output", "a b")]


def test_real_uv_runs_synthetic_resource_with_spaces(resources, tmp_path, monkeypatch):
    assert shutil.which("uv"), "uv is required by this project's test runner"
    monkeypatch.setenv("UV_OFFLINE", "1")
    monkeypatch.setenv("UV_PYTHON_DOWNLOADS", "never")
    output = tmp_path / "argv result.json"
    script = resources / "fx" / "tools" / "synthetic.py"
    script.write_text(_HEADER + 'import json, sys\nfrom pathlib import Path\n'
                      'Path(sys.argv[1]).write_text(json.dumps(sys.argv[2:]))\nraise SystemExit(23)\n')
    with pytest.raises(SystemExit) as failure:
        dispatch.Tools().run("fx", "synthetic.py", str(output), "a b", "--flag=value")
    assert failure.value.code == 23
    assert json.loads(output.read_text()) == ["a b", "--flag=value"]


def test_run_reports_disappearing_executable(resources, monkeypatch):
    (resources / "asr" / "tools" / "synthetic.py").write_text(_HEADER)
    monkeypatch.setattr(dispatch.shutil, "which", lambda name: "/missing/uv")

    def disappeared(*args, **kwargs):
        raise FileNotFoundError("synthetic disappearance")

    monkeypatch.setattr(dispatch.subprocess, "run", disappeared)
    with pytest.raises(RuntimeError, match="reinstall"):
        dispatch.Tools().run("asr", "synthetic")


def test_run_rejects_missing_metadata_and_ambiguous_resources(resources):
    script = resources / "asr" / "tools" / "synthetic.py"
    script.write_text('if __name__ == "__main__": pass\n')
    with pytest.raises(ValueError, match="needs-metadata"):
        dispatch.Tools().run("asr", "synthetic")
    script.write_text(_HEADER)
    (resources / "asr" / "synthetic.py").write_text(_HEADER)
    with pytest.raises(ValueError, match="Ambiguous"):
        dispatch.Tools().run("asr", "synthetic")


@pytest.mark.parametrize("args", [
    ["--backend=cpu", "--lang=pl", "--output", "a b.wav", "001", "--lang=en", "--json"],
    ["--", "--backend=cpu", "--lang=pl", "--output", "a b.wav", "001", "--lang=en", "--json"],
])
@pytest.mark.parametrize("route", [["tools", "run", "asr", "synthetic"], ["tools", "asr"],
                                   ["--json", "tools", "run", "asr", "synthetic"],
                                   ["--json", "tools", "asr"]])
def test_real_fire_cli_preserves_child_options(resources, tmp_path, route, args):
    """The actual composition root must route argv before Fire parses child flags."""
    import os
    import sys

    output = tmp_path / "received.json"
    script = _HEADER + 'import json, sys\nfrom pathlib import Path\n' + (
        f'Path({str(output)!r}).write_text(json.dumps(sys.argv[1:]))\nraise SystemExit(19)\n'
    )
    (resources / "asr" / "tools" / "synthetic.py").write_text(script)
    (resources / "asr" / "asr_all.py").write_text(script)
    source = (
        "import sys\nfrom pathlib import Path\n"
        "from vexy_audioflexy import dispatch, cli\n"
        f"dispatch._ROOT = Path({str(resources)!r})\n"
        f"sys.argv = {['vafx', *route, *args]!r}\n"
        "cli.main()\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", source], capture_output=True, text=True,
        env={**os.environ, "UV_OFFLINE": "1", "UV_PYTHON_DOWNLOADS": "never"},
    )
    assert result.returncode == 19, result.stderr
    expected = args[1:] if args[:1] == ["--"] else args
    assert json.loads(output.read_text()) == expected


@pytest.mark.parametrize("argv", [[], ["version"], ["tools"], ["tools", "list"],
                                  ["tools", "run", "--help"], ["models", "list"]])
def test_dispatch_arguments_leaves_discovery_and_other_commands_to_fire(argv):
    assert dispatch.dispatch_arguments(argv) is False


def test_public_namespace_and_batch_contract():
    assert dispatch._NAMESPACES == ("asr", "fx", "tts")
    assert dispatch._BATCHES["fx"] == dispatch._BATCHES["aud"] == ("fx", "aud_all")
    assert {item["namespace"] for item in dispatch.Tools().list()} == {"asr", "fx", "tts"}
    for namespace in ("asr", "fx", "tts"):
        assert dispatch.Tools().list(namespace)
    for namespace in ("myasr", "myaud", "mypvc"):
        with pytest.raises(ValueError, match="Unknown namespace"):
            dispatch.Tools().list(namespace)
        with pytest.raises(ValueError, match="Unknown namespace"):
            dispatch.Tools().run(namespace, "example")
