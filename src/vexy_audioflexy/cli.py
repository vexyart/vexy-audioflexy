# this_file: src/vexy_audioflexy/cli.py
"""Compose the public CLI from explicitly selected commands."""

import sys

import fire

from vexy_audioflexy import __version__
from vexy_audioflexy.models import Models
from vexy_audioflexy.dispatch import Tools, dispatch_arguments


def version() -> str:
    """Return the installed package version."""
    return __version__


def main() -> None:
    """Run the vafx command line interface."""
    dispatch_arguments(sys.argv[1:])
    fire.Fire({"version": version, "models": Models, "tools": Tools}, name="vafx")


if __name__ == "__main__":
    main()
