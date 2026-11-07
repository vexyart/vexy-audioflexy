#!/usr/bin/env bash
# this_file: install.sh
set -euo pipefail
cd "$(dirname "$0")"
artifacts=$(./build.sh)
uv tool install --force --reinstall "$artifacts"/*.whl >&2
printf '%s\n' "$artifacts"
