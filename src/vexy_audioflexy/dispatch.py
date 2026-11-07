# this_file: src/vexy_audioflexy/dispatch.py
"""Run packaged tools in their declared uv environments without importing models."""

import ast
from pathlib import Path
import shutil
import subprocess

_NAMESPACES = ("asr", "fx", "tts")
_ROOT = Path(__file__).parent / "tools"
_BATCHES = {
    "asr": ("asr", "asr_all"),
    "diarize": ("asr", "diarize_all"),
    "translate": ("asr", "translate_all"),
    "fx": ("fx", "aud_all"),
    "aud": ("fx", "aud_all"),
    "detect": ("fx", "detect_all"),
    "separate": ("fx", "separate_all"),
    "tts": ("tts", "tts_all"),
}


def _namespaces(namespace: str | None) -> tuple[str, ...]:
    if namespace is None:
        return _NAMESPACES
    if namespace not in _NAMESPACES:
        raise ValueError(f"Unknown namespace {namespace!r}; choose {', '.join(_NAMESPACES)}")
    return (namespace,)


def _paths(namespace: str) -> list[Path]:
    root = (_ROOT / namespace).resolve()
    if not root.is_relative_to(_ROOT.resolve()):
        return []
    return sorted(
        path for directory in (root, root / "tools")
        for path in directory.glob("*")
        if path.is_file() and path.suffix in {".py", ".sh", ".cpp"}
        and path.resolve().is_relative_to(root)
        and path.name != "__init__.py"
    )


def _kind(path: Path) -> str:
    if path.suffix != ".py":
        return "native-source" if path.suffix == ".cpp" else "shell-script"
    source = path.read_text(encoding="utf-8")
    if "# /// script" in source:
        return "runnable"
    tree = ast.parse(source)
    has_main = any(
        isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "__name__"
        and any(isinstance(value, ast.Constant) and value.value == "__main__"
                for value in node.test.comparators)
        for node in tree.body
    )
    return "needs-metadata" if has_main else "helper"


class Tools:
    """List packaged namespaces and forward positional arguments to isolated tools."""

    def list(self, namespace: str | None = None) -> list[dict[str, str]]:
        """List packaged basenames and distinguish runnable scripts from helpers."""
        return [
            {"namespace": name, "tool": path.name, "kind": _kind(path)}
            for name in _namespaces(namespace) for path in _paths(name)
        ]

    def run(self, namespace: str, tool: str, *args: str) -> None:
        """Run a basename (with optional .py suffix); preserve argv and exit status."""
        _namespaces(namespace)
        if not isinstance(tool, str) or not tool or any(c in tool for c in "/\\"):
            raise ValueError("Tool must be a listed source basename, without directories")
        names = {tool, f"{tool}.py"}
        paths = [path for path in _paths(namespace) if path.name in names]
        if not paths:
            raise ValueError(f"Tool {tool!r} is not installed in {namespace}; use tools list")
        if len(paths) != 1:
            raise ValueError(f"Ambiguous tool {tool!r} in {namespace}")
        path = paths[0]
        kind = _kind(path)
        if kind != "runnable":
            raise ValueError(
                f"{namespace}/{path.name} is {kind}, not a uv script; "
                "use a runnable adapter or add its declared PEP 723 dependencies"
            )
        uv = shutil.which("uv")
        if uv is None:
            raise RuntimeError("uv is required to run tools; install it from https://docs.astral.sh/uv/")
        try:
            result = subprocess.run([uv, "run", "--script", str(path), *map(str, args)], check=False)
        except FileNotFoundError as error:
            raise RuntimeError("uv or the packaged tool disappeared; reinstall uv and vexy-audioflexy") from error
        raise SystemExit(result.returncode)

    def asr(self, *args: str) -> None:
        """Run the ASR batch consumer."""
        self.run(*_BATCHES["asr"], *args)

    def diarize(self, *args: str) -> None:
        """Run the diarization batch consumer."""
        self.run(*_BATCHES["diarize"], *args)

    def translate(self, *args: str) -> None:
        """Run the translation batch consumer."""
        self.run(*_BATCHES["translate"], *args)

    def fx(self, *args: str) -> None:
        """Run the audio-processing batch consumer."""
        self.run(*_BATCHES["fx"], *args)

    def aud(self, *args: str) -> None:
        """Run the audio-processing batch consumer."""
        self.run(*_BATCHES["aud"], *args)

    def detect(self, *args: str) -> None:
        """Run the detection batch consumer."""
        self.run(*_BATCHES["detect"], *args)

    def separate(self, *args: str) -> None:
        """Run the separation batch consumer."""
        self.run(*_BATCHES["separate"], *args)

    def tts(self, *args: str) -> None:
        """Run the synthesis batch consumer."""
        self.run(*_BATCHES["tts"], *args)


def dispatch_arguments(argv: list[str]) -> bool:
    """Route tool execution before Fire can consume the child script's options."""
    if len(argv) < 2 or argv[0] != "tools":
        return False
    command = argv[1]
    if command == "run" and len(argv) >= 4 and not argv[2].startswith("-"):
        namespace, tool = argv[2:4]
        args = argv[4:]
    elif command in _BATCHES:
        namespace, tool = _BATCHES[command]
        args = argv[2:]
    else:
        return False
    if args[:1] == ["--"]:
        args = args[1:]
    Tools().run(namespace, tool, *args)
    return True
