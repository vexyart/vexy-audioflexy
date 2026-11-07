---
this_file: README.md
---

# vexy-audioflexy

Audioflexy brings ASR, audio processing and voice tools into one installed package
while keeping model runtimes in isolated uv environments. The `vafx` Python Fire
CLI lists adapters, forwards their arguments, and manages local model locations.
No model weights or private recordings are bundled.

Requires Python 3.12 or newer and [uv](https://docs.astral.sh/uv/).

```sh
uv sync
uv run vafx version
uv run vafx tools list
uv run vafx models config
uv run vafx -- --help
./test.sh
uv build --no-sources
```

Install from source with `./install.sh`; it builds and installs the exact wheel
using `uv tool install`. Build alone with `./build.sh`.

## Tools

The `asr`, `fx` and `tts` namespaces retain their source adapter names.
List a namespace before selecting a tool:

```sh
vafx tools list asr
vafx tools run asr asr-transcribecpp --help
vafx tools asr --help
```

Tool execution uses `uv run --script` with each script's declared dependencies.
Child flags, ordering and exit status are preserved. Running a tool may install
its dependencies; preparation or inference may require separate model downloads
and native builds. Helpers and native source files are listed with their kind
and cannot be run as Python adapters.

Batch aliases are `asr`, `diarize`, `translate`, `fx` (`aud` alias), `detect`, `separate` and
`tts`. Supply your own inputs, text and output paths according to each command's
help. Availability of an adapter does not establish every model's compatibility.

## Model locations

Use the platform configuration/data directories or override the models folder:

```sh
vafx models --models_dir=/absolute/path/to/models config
vafx models configure /absolute/path/to/models
vafx models discover
vafx models list
```

`configure` writes `config.toml`; registration uses `models.toml` in the app data
directory. Discovery reads LM Studio, Hugging Face and configured local stores
without downloading or loading models. Discovery persists records only when
requested with `--save=True`. Provider download destinations prefer their own
stores. Model and output paths remain user supplied.

The `vexy_audioflexy.migration` Python API prepares reviewed, content-bound local
model relocations and receipts. It preserves upstream paths with symlinks; provider
deduplication has a separate reviewed-file path. It does not classify mixed trees
or authorize moving private data automatically.

## Acceptance and development

The package includes adapters for transcribe.cpp, exact Parakeet Ultra GGUF,
Bark-named Parakeet MLX and Nemotron SALM. Short local acceptance runs are distinct from
coverage of every catalog variant, language, quantization or long-form quality.
Native runtimes and provider model stores remain external.

Versions derive from semantic-version Git tags through `hatch-vcs`; development
builds include the Git revision. Public archives select code, synthetic tests,
examples and documentation source. Keep private inputs, outputs, machine paths
and model weights outside this repository.

See `PLAN.md` and `TODO.md` for remaining acceptance work. Documentation source
is in `src_docs/`. Pushing a stable `vMAJOR.MINOR.PATCH` tag runs tests, builds
packages and strict docs, and creates a GitHub release. `./release.sh` and
`./publish.sh` use `uvx gitnextver` to commit pending changes, create the next
stable tag and push when HEAD is untagged or dirty. A clean tagged HEAD is reused.
Publish chains release → install → build, verifies GitHub assets, then runs
`uv publish`. Pass an existing tag explicitly to resume a clean tagged release.
See the release instructions in `src_docs/md/development.md` for authentication.
