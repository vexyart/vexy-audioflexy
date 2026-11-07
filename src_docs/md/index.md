---
this_file: src_docs/md/index.md
---
# Audioflexy

Audioflexy is being consolidated into the `vexy-audioflexy` Python package and
the `vafx` command. The CLI provides version reporting, model configuration and
discovery, and dispatch to packaged tools in isolated uv environments.

ASR, diarization, translation, audio processing, detection, separation, and TTS
engine consolidation and acceptance remain in progress. A packaged tool, package
build, or model registry entry
does not demonstrate that an engine can run inference.

## Install

From a source checkout with Python 3.12 or newer and [uv](https://docs.astral.sh/uv/):

```sh
uv sync
uv run vafx version
uv run vafx --help
```

Published package availability must be checked separately from local builds.

## Privacy

Keep audio, transcripts, prompts, credentials, identifying metadata, model weights,
and local configuration outside this public repository. Public examples use
synthetic data. Private reproduction wrappers should call the public package or
CLI and keep inputs and results in a separate private workspace.

Rankings require a separate review that removes private content, identifying
metadata, and local paths. Wheel, source archive, and generated documentation
contents must also be inspected before publication.
