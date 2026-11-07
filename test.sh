#!/usr/bin/env bash
# this_file: test.sh
set -euo pipefail
cd "$(dirname "$0")"
uv run pytest "$@"
uv run python examples/basic.py
