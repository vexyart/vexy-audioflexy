---
this_file: src_docs/md/development.md
---
# Development and releases

## Build documentation

From the repository root:

```sh
uv run --no-project --with-requirements requirements-docs.txt properdocs build --strict -f src_docs/mkdocs.yml
```

The pinned stack uses ProperDocs as the builder and MaterialX as the theme.
`docs_dir: md` and `site_dir: ../docs` resolve relative to
`src_docs/mkdocs.yml`, producing the repository's `docs/` directory.

## Build, install, release and publish

Versions come from stable `vMAJOR.MINOR.PATCH` Git tags through hatch-vcs; there is
no manually edited package version. Its build hook writes an ignored
`src/vexy_audioflexy/__version__.py`; builds never track this generated file.
Development builds include the Git revision and may include a dirty suffix.
Run these executable scripts from any working directory:

```sh
./build.sh
./install.sh
./release.sh
./publish.sh
# Resume an existing release from its clean tagged checkout:
./publish.sh v1.0.0
```

The chain is `publish → release → install → build`. Build creates a fresh ignored
`dist/build.*` directory and checks wheel/sdist metadata. Install replaces the
installed `vafx` uv tool with that exact wheel. Each script prints its selected
artifact directory on stdout; command progress goes to stderr.

Without a tag argument, release reuses a clean stable tag at HEAD. Otherwise it
runs `uvx gitnextver --pull=False`, which commits all pending changes, creates the
next stable semantic tag and atomically pushes the branch and tag. A clean untagged
checkout adds `--commit=False` to tag the existing commit. Review pending
changes first and synchronize the branch before running release; automatic pulls
and stash restoration are disabled. A tagging or push failure stops the chain.
An explicit tag selects an existing release without creating a commit or tag.

Before building, release requires a clean checkout at the selected stable tag.
The wheel and sdist must both have exactly the selected stable version; development
or dirty versions stop the chain before installation or publication.
It creates the GitHub release when missing, uploads missing package assets, and
compares existing assets byte for byte without replacing them. Publish uploads
those verified GitHub assets with `uv publish --check-url`; it stops on any error,
including rate limits. Existing matching files can be skipped on a later resume.
Retain selected artifacts when investigating an uncertain upload. An explicit-tag
script resume still rebuilds and reinstalls before checking GitHub asset bytes;
build differences stop the chain. To resume an upload using already verified
GitHub-downloaded archives, keep the exact files and run:

```sh
UV_HTTP_RETRIES=0 uv publish --check-url https://pypi.org/simple/ dist/released.EXAMPLE/*.whl dist/released.EXAMPLE/*.tar.gz
```

Replace `dist/released.EXAMPLE` with the retained artifact directory. Stop on
rate limits and retain those same files for the next attempt; do not change the
version to work around an uncertain upload.

Authenticate GitHub CLI with `gh auth login` or `GH_TOKEN`. Configure PyPI upload
credentials through `UV_PUBLISH_TOKEN` or uv's supported credential store. Keep
credentials outside the repository and out of shell command arguments and logs.
For CI-first releases, run `uvx gitnextver --pull=False` after validation (add
`--commit=False` when there are no pending changes), wait for the tag workflow,
then run `./publish.sh` from the clean tagged checkout. Direct
release/publish invocations can race CI to create GitHub assets; immutable asset
comparisons stop on a mismatch and reruns resume the same clean tagged version.

## Tag workflow

GitHub Actions runs on pushed tags matching `v*.*.*`, then validates the exact
stable semantic form. It fetches full history for hatch-vcs, runs locked tests,
builds the packages with `build.sh` and strict ProperDocs documentation, and
creates the GitHub release with package assets. Documentation and packages are
also retained as workflow artifacts. Existing release assets must match on rerun.

PyPI upload is performed by `publish.sh`, separately from CI, so the tag workflow
and local publication cannot race to upload the same distribution. No PyPI token
or trusted-publisher configuration is required by the build/release workflow.

Before releasing, inspect source files, both distribution contents and generated
documentation for private data, credentials, identifying metadata, local paths
and unintended model assets. Local validation does not prove remote publication;
verify the Actions run, release assets and downloaded PyPI distributions afterward.
