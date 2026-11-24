---
this_file: src_docs/md/development.md
---
# Development and releases

## Build documentation

From the repository root:

```sh
uv run --no-project --python 3.13 --with-requirements requirements-docs.txt properdocs build --strict -f src_docs/mkdocs.yml
```

ProperDocs builds the site with fly-january 0.4.0 or later (remote theme assets). The `chrome: none`
setting selects the native MaterialX header and footer.
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
# Build and verify a specific release from its clean tagged checkout
./publish.sh v1.0.0
```

The chain is `publish → release → install → build`. Build creates a fresh ignored
`dist/build.*` directory and checks wheel/sdist metadata. Install replaces the
installed `vafx` uv tool with that exact wheel. Each script prints its selected
artifact directory on stdout. Command progress goes to stderr.

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
compares existing assets byte for byte. `./publish.sh` dispatches the PyPI upload
workflow for that release and waits for the result. The workflow downloads the
GitHub release assets and publishes with PyPI Trusted Publishing (OIDC).

Use GitHub CLI 2.105.0 or a release that returns the workflow run URL from
`gh workflow run`. Authenticate with `gh auth login`. Set the GitHub `pypi`
environment's deployment branch policy to allow only the protected default
branch. Configure the PyPI project
`vexy-audioflexy` with GitHub repository `vexyart/vexy-audioflexy`,
workflow `pypi-upload.yml`, and environment `pypi`. Use a pending publisher
for the first upload if the project does not exist. [PyPI setup](https://docs.pypi.org/trusted-publishers/)

For a CI-built release or to retry an upload of the same GitHub assets, dispatch
the workflow from the default branch with the existing release tag:

```sh
gh workflow run pypi-upload.yml -f tag=v1.0.3
```

Wait for the Actions run, then compare the PyPI file hashes with the GitHub
release assets. If PyPI rate-limits the upload, keep the same tag and assets for
the next attempt.

## Tag workflow

GitHub Actions runs on pushed tags matching `v*.*.*`, then validates the exact
stable semantic form. It fetches full history for hatch-vcs, runs locked tests,
builds the packages with `build.sh` and strict ProperDocs documentation, and
creates the GitHub release with package assets. Documentation and packages are
also retained as workflow artifacts. Existing release assets must match on rerun.

Before releasing, inspect source files, both distribution contents and generated
documentation for private data, credentials, identifying metadata, local paths
and unintended model assets. Local validation does not prove remote publication;
verify the Actions run, release assets and downloaded PyPI distributions afterward.
