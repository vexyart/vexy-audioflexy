# this_file: src/vexy_audioflexy/cli.py
"""Compose the public CLI from explicitly selected commands."""

import json
import sys

import fire
import yaml

from vexy_audioflexy import __version__
from vexy_audioflexy.dispatch import Tools, dispatch_arguments
from vexy_audioflexy.models import Models


def version() -> str:
    """Return the installed package version."""
    return __version__


def _serialize(value: object, as_json: bool) -> object:
    """Format command data while leaving Fire's command groups available for help."""
    if isinstance(value, dict) and any(callable(item) for item in value.values()):
        return value
    if isinstance(value, (dict, list, tuple)):
        if as_json:
            return json.dumps(value, ensure_ascii=False, indent=2)
        return yaml.safe_dump(value, allow_unicode=True, sort_keys=False).rstrip("\n")
    if as_json and isinstance(value, (str, int, float, bool, type(None))):
        return json.dumps(value, ensure_ascii=False)
    return value


def main() -> None:
    """Run the vafx command line interface."""
    args = sys.argv[1:]
    as_json = False
    while args[:1] == ["--json"]:
        as_json, args = True, args[1:]
    if dispatch_arguments(args):
        return
    end = args.index("--") if "--" in args else len(args)
    as_json = as_json or "--json" in args[:end]
    args = [arg for arg in args[:end] if arg != "--json"] + args[end:]
    fire.Fire({"version": version, "models": Models, "tools": Tools}, command=args,
              name="vafx", serialize=lambda value: _serialize(value, as_json))


if __name__ == "__main__":
    main()
