---
this_file: DEPENDENCIES.md
---

# Dependencies

- `fire`: explicit Python command composition for `vafx`.
- `PyYAML`: readable YAML output for structured CLI command results.
- `platformdirs`: portable configuration and data directories for model settings.
- `tomli-w`: write portable model settings and registry TOML.
- `hatchling`: wheel and source distribution builds.
- `hatch-vcs`: derive versions from Git tags without a hand-maintained version.
- `pytest` (development): synthetic functional tests.
- `uv` (development tool): dependency resolution, execution, builds, installation.

Model runtimes remain isolated rather than becoming mandatory package dependencies.
The audio.cpp audio/TTS adapters declare `huggingface-hub` in their own inline
dependencies for provider downloads; native layout links use the standard library.

## Migrated regression tests (development only)

The dev group also installs `numpy` and `soundfile` for synthetic audio arrays
and WAV fixtures, `rich` for migrated console imports, `huggingface-hub` for
provider interfaces without downloads, `jiwer` and `jieba==0.42.1` for transcript
scoring, and `pyannote-metrics` for synthetic diarization scores. These packages
are excluded from mandatory runtime dependencies; no model inference runtime
or weights are installed by the test dependency group.

- `gguf==0.19.0` (development): generate tiny synthetic GGUF metadata fixtures
  for exact model-identity rejection tests; no weights are installed.
- `sacrebleu` (development): exercise relocated translation scoring/help imports.
