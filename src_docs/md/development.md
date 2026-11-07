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
no manually edited package version. Development builds include the Git revision.
Run these executable scripts from any working directory:

```sh
./build.sh
./install.sh
./release.sh v1.0.0
./publish.sh v1.0.0
```

The chain is `publish → release → install → build`. Build creates a fresh ignored
`dist/build.*` directory and checks wheel/sdist metadata. Install replaces the
installed `vafx` uv tool with that exact wheel. Each script prints its selected
artifact directory on stdout; command progress goes to stderr.

Release requires a clean checkout at the selected stable tag and a pushed tag.
It creates the GitHub release when missing, uploads missing package assets, and
compares existing assets byte for byte without replacing them. Publish uploads
those verified GitHub assets with `uv publish --check-url`; it stops on any error,
including rate limits. Existing matching files can be skipped on a later resume.
Retain selected artifacts when investigating an uncertain upload.

Authenticate GitHub CLI with `gh auth login` or `GH_TOKEN`. Configure PyPI upload
credentials through `UV_PUBLISH_TOKEN` or uv's supported credential store. Keep
credentials outside the repository and out of shell command arguments and logs.
Scripts do not create commits, tags or pushes. After validation, the release owner
commits, creates the stable tag and pushes it, then waits for the tag workflow
before running publish. This prevents concurrent release creation.

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
