#!/usr/bin/env bash
# this_file: publish.sh
set -euo pipefail
cd "$(dirname "$0")"
artifacts=$(./release.sh "$@")
UV_HTTP_RETRIES=0 uv publish --check-url https://pypi.org/simple/ "$artifacts"/*.whl "$artifacts"/*.tar.gz >&2
printf '%s\n' "$artifacts"
