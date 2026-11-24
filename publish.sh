#!/usr/bin/env bash
# this_file: publish.sh
set -euo pipefail
cd "$(dirname "$0")"
artifacts=$(./release.sh "$@")
tag=${1:-$(git describe --tags --exact-match HEAD)}
run_url=$(gh workflow run pypi-upload.yml -f "tag=$tag")
if [[ ! "$run_url" =~ ^https://github\.com/[^/]+/[^/]+/actions/runs/([0-9]+)$ ]]; then
    echo "Check the PyPI upload workflow in GitHub Actions for $tag before dispatching again." >&2
    exit 1
fi
printf '%s\n' "$run_url" >&2
gh run watch "${BASH_REMATCH[1]}" --exit-status >&2
printf '%s\n' "$artifacts"
