# this_file: tests/test_release_scripts.py
"""Exercise release chaining and failure guards without remote mutations."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).parents[1]
FAKE = r'''import io, json, os
from pathlib import Path
import shutil, sys, tarfile, zipfile
name = Path(sys.argv[0]).name
args = sys.argv[1:]
with open(os.environ["COMMAND_LOG"], "a") as log:
    log.write(json.dumps([name, *args]) + "\n")
store = Path(os.environ["REMOTE_ASSETS"])
if name == "git":
    if args[0] == "describe":
        if os.environ.get("FAKE_UNTAGGED"): sys.exit(128)
        print(os.environ.get("FAKE_TAG", "v1.2.3"))
    elif args[0] == "rev-parse": print("abc123")
    elif args[0] == "status":
        tagged = (store.parent / "tagged").exists()
        print(os.environ.get("FAKE_AFTER_DIRTY" if tagged else "FAKE_DIRTY", ""), end="")
elif name == "uvx":
    assert args in (["gitnextver", "--pull=False"], ["gitnextver", "--pull=False", "--commit=False"])
    if os.environ.get("TAG_EXIT"): sys.exit(int(os.environ["TAG_EXIT"]))
    if not os.environ.get("FAKE_DIRTY") and "--commit=False" not in args:
        sys.exit(0)  # gitnextver's default commit mode stops when nothing changed.
    (store.parent / "tagged").touch()
    print(os.environ.get("FAKE_NEXT_TAG", "v1.2.3"))
elif name == "uv":
    if args[0] == "build":
        root = Path(args[args.index("--out-dir") + 1])
        version = os.environ.get("FAKE_VERSION", "1.2.3")
        metadata = f"Name: vexy-audioflexy\nVersion: {version}\n".encode()
        with zipfile.ZipFile(root / f"vexy_audioflexy-{version}-py3-none-any.whl", "w") as archive:
            archive.writestr(f"vexy_audioflexy-{version}.dist-info/METADATA", metadata)
        with tarfile.open(root / f"vexy_audioflexy-{version}.tar.gz", "w:gz") as archive:
            info = tarfile.TarInfo(f"vexy_audioflexy-{version}/PKG-INFO")
            info.size = len(metadata)
            info.mtime = 0
            archive.addfile(info, io.BytesIO(metadata))
    elif args[0] == "publish":
        assert os.environ.get("UV_HTTP_RETRIES") == "0"
        sys.exit(int(os.environ.get("PUBLISH_EXIT", "0")))
elif name == "gh":
    command = args[1]
    if command == "view":
        if not store.exists(): sys.exit(1)
        if "--jq" in args:
            print("\n".join(p.name for p in store.iterdir()))
    elif command in ("create", "upload"):
        store.mkdir(exist_ok=True)
        for item in args[3:]:
            if item.endswith((".whl", ".tar.gz")): shutil.copyfile(item, store / Path(item).name)
    elif command == "download":
        asset = store / args[args.index("--pattern") + 1]
        if not asset.exists(): sys.exit(1)
        shutil.copyfile(asset, Path(args[args.index("--dir") + 1]) / asset.name)
'''


@pytest.fixture
def release_tree(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    binary = tmp_path / "bin"
    binary.mkdir()
    for script in ("build", "install", "release", "publish"):
        shutil.copyfile(ROOT / f"{script}.sh", project / f"{script}.sh")
        (project / f"{script}.sh").chmod(0o755)
    for name in ("uv", "uvx", "git", "gh"):
        path = binary / name
        path.write_text(f"#!{sys.executable}\n" + FAKE)
        path.chmod(0o755)
    env = dict(os.environ, PATH=f"{binary}:{os.environ['PATH']}",
               COMMAND_LOG=str(tmp_path / "commands.jsonl"),
               REMOTE_ASSETS=str(tmp_path / "remote"))
    env.pop("VAFX_RELEASE_TAG", None)
    return project, env


def run_script(tree, script="publish", *args, **overrides):
    project, env = tree
    result = subprocess.run([str(project / f"{script}.sh"), *args], cwd=project.parent,
                            env=dict(env, **overrides), capture_output=True, text=True)
    log = Path(env["COMMAND_LOG"])
    commands = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    return result, commands


def test_publish_when_clean_tag_then_build_install_release_publish_exact_artifacts(release_tree):
    result, commands = run_script(release_tree, "publish", "v1.2.3")
    assert result.returncode == 0, result.stderr
    uv_commands = [row for row in commands if row[0] == "uv"]
    assert [row[1] for row in uv_commands] == ["build", "tool", "publish"]
    assert uv_commands[1][2:5] == ["install", "--force", "--reinstall"]
    release = next(row for row in commands if row[:3] == ["gh", "release", "create"])
    assert "--verify-tag" in release
    selected = Path(result.stdout.strip())
    assert selected.is_dir() and selected.name.startswith("released.")
    assert set(uv_commands[-1][-2:]) == {str(p) for p in selected.iterdir()}


@pytest.mark.parametrize("tag", ["v01.2.3", "1.2.3", "v1.2.3rc1", "v1.2.3.4"])
def test_release_when_tag_not_stable_semver_then_stops_before_build(release_tree, tag):
    result, commands = run_script(release_tree, "release", tag)
    assert result.returncode != 0
    assert not any(row[0] == "uv" for row in commands)


def test_release_when_dirty_then_stops_before_build(release_tree):
    result, commands = run_script(release_tree, "release", "v1.2.3", FAKE_DIRTY=" M source.py")
    assert result.returncode != 0
    assert "clean checkout" in result.stderr
    assert not any(row[0] == "uv" for row in commands)


@pytest.mark.parametrize("version", ["1.2.4", "1.2.3.dev1+gdeadbeef", "1.2.3+dirty"])
def test_release_when_built_version_differs_then_stops_before_install(release_tree, version):
    result, commands = run_script(release_tree, "release", "v1.2.3", FAKE_VERSION=version)
    assert result.returncode != 0
    assert "Built version differs" in result.stderr
    assert [row[1] for row in commands if row[0] == "uv"] == ["build"]


def test_publish_when_existing_asset_differs_then_preserves_remote_and_stops(release_tree):
    project, env = release_tree
    store = Path(env["REMOTE_ASSETS"])
    store.mkdir()
    wheel = store / "vexy_audioflexy-1.2.3-py3-none-any.whl"
    wheel.write_bytes(b"existing different release")
    result, commands = run_script(release_tree, "publish", "v1.2.3")
    assert result.returncode != 0
    assert "refusing replacement" in result.stderr
    assert wheel.read_bytes() == b"existing different release"
    assert not any(row[:2] == ["uv", "publish"] for row in commands)
    assert all("--clobber" not in row for row in commands)


def test_publish_when_upload_fails_then_does_not_retry(release_tree):
    result, commands = run_script(release_tree, "publish", "v1.2.3", PUBLISH_EXIT="43")
    assert result.returncode == 43
    assert sum(row[:2] == ["uv", "publish"] for row in commands) == 1


def test_build_when_no_release_tag_then_allows_development_version(release_tree):
    result, commands = run_script(release_tree, "build", FAKE_VERSION="1.2.4.dev1")
    assert result.returncode == 0, result.stderr
    assert Path(result.stdout.strip()).is_dir()
    assert len(commands) == 1 and commands[0][:2] == ["uv", "build"]


@pytest.mark.parametrize("script", ["release", "publish"])
@pytest.mark.parametrize("state", [{"FAKE_UNTAGGED": "1"}, {"FAKE_DIRTY": " M source.py"}])
def test_release_when_no_tag_argument_then_gitnextver_before_build(release_tree, script, state):
    result, commands = run_script(release_tree, script, **state)
    assert result.returncode == 0, result.stderr
    expected = ["uvx", "gitnextver", "--pull=False"]
    if "FAKE_UNTAGGED" in state:
        expected.append("--commit=False")
    tagging = commands.index(expected)
    build = next(i for i, row in enumerate(commands) if row[:2] == ["uv", "build"])
    assert tagging < build
    assert sum(row[0] == "uvx" for row in commands) == 1


def test_release_when_no_argument_and_clean_stable_tag_then_reuses_tag(release_tree):
    result, commands = run_script(release_tree, "release")
    assert result.returncode == 0, result.stderr
    assert not any(row[0] == "uvx" for row in commands)


def test_release_when_gitnextver_fails_then_no_build_or_upload(release_tree):
    result, commands = run_script(release_tree, "publish", FAKE_UNTAGGED="1", TAG_EXIT="42")
    assert result.returncode == 42
    assert not any(row[0] in ("uv", "gh") for row in commands)


def test_release_when_gitnextver_leaves_dirty_checkout_then_no_build(release_tree):
    result, commands = run_script(release_tree, "release", FAKE_UNTAGGED="1", FAKE_AFTER_DIRTY=" M source.py")
    assert result.returncode != 0
    assert "clean checkout" in result.stderr
    assert not any(row[0] == "uv" for row in commands)


def test_release_when_gitnextver_returns_development_tag_then_no_build(release_tree):
    result, commands = run_script(release_tree, "release", FAKE_UNTAGGED="1", FAKE_NEXT_TAG="v1.2.4.dev1")
    assert result.returncode != 0
    assert "stable semantic tag" in result.stderr
    assert not any(row[0] == "uv" for row in commands)
