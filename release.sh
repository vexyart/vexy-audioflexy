#!/usr/bin/env bash
# this_file: release.sh
set -euo pipefail
cd "$(dirname "$0")"
tag=${1:-}
if [[ -z "$tag" ]]; then
    if [[ -n $(git status --porcelain) ]]; then
        tag=$(uvx gitnextver --pull=False)
    elif ! tag=$(git describe --tags --exact-match HEAD 2>/dev/null); then
        tag=$(uvx gitnextver --pull=False --commit=False)
    fi
fi
if [[ ! "$tag" =~ ^v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$ ]]; then
    echo "Expected stable semantic tag vMAJOR.MINOR.PATCH" >&2
    exit 1
fi
if [[ $(git rev-parse "$tag^{commit}") != $(git rev-parse HEAD) ]] || [[ -n $(git status --porcelain) ]]; then
    echo "Release requires a clean checkout at the selected tag" >&2
    exit 1
fi
export VAFX_RELEASE_TAG="$tag"
artifacts=$(./install.sh)
if gh release view "$tag" --json tagName >/dev/null 2>&1; then
    names=$(gh release view "$tag" --json assets --jq '.assets[].name')
    for asset in "$artifacts"/*.whl "$artifacts"/*.tar.gz; do
        name=$(basename "$asset")
        if ! printf '%s\n' "$names" | grep -Fxq "$name"; then
            gh release upload "$tag" "$asset" >&2
        fi
    done
else
    gh release create "$tag" "$artifacts"/*.whl "$artifacts"/*.tar.gz --verify-tag --title "$tag" --generate-notes >&2
fi
released=$(mktemp -d "$PWD/dist/released.XXXXXX")
for asset in "$artifacts"/*.whl "$artifacts"/*.tar.gz; do
    name=$(basename "$asset")
    gh release download "$tag" --pattern "$name" --dir "$released" >&2
    if ! cmp -s "$asset" "$released/$name"; then
        echo "GitHub asset differs from this tag's build: $name; refusing replacement" >&2
        exit 1
    fi
done
printf '%s\n' "$released"
