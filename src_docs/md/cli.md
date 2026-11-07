---
this_file: src_docs/md/cli.md
---
# CLI

The Python Fire entry point is `vafx`. Use `uv run vafx` from a source checkout,
or `vafx` after installation.

```sh
vafx version
vafx --help
vafx models config
vafx tools list
vafx tools list asr
```

`version` prints the installed distribution version, derived by Hatch/hatch-vcs
from Git history and semantic version tags. Editable installs may need rebuilding
after a tag changes. Model commands are described in [Models](models.md).

## Command output

Structured results use YAML by default. Request a complete JSON document with
`--json` before or after the command, before Fire's standalone `--`:

```sh
vafx models list
vafx models list --json
vafx tools list asr --json
```

Version and path results remain plain text unless `--json` is selected. Tool
execution preserves child arguments and output, including a child's `--json`.

## Run packaged tools

Namespaces are `asr`, `fx`, and `tts`. Listings distinguish `runnable`
Python scripts from `helper`, `needs-metadata`, `shell-script`, and `native-source`
entries. Only scripts marked `runnable` can be dispatched with `tools run`.
Use the listed basename, with or without its `.py` suffix:

```sh
vafx tools run asr asr-transcribecpp --help
vafx tools run asr asr-transcribecpp -- --help
```

The dispatcher forwards child arguments before Fire parses them. An optional
leading `--` after the tool name is removed; the remaining arguments retain their
order and spelling, including options and filenames containing spaces. The child
runs through `uv run --script` using its declared dependencies, and its exit status
becomes the dispatcher's exit status. The runtime requires `uv` on `PATH`.

Batch aliases route to the existing packaged consumers:

| Alias | Namespace and tool |
| --- | --- |
| `vafx tools asr` | `asr/asr_all.py` |
| `vafx tools diarize` | `asr/diarize_all.py` |
| `vafx tools translate` | `asr/translate_all.py` |
| `vafx tools fx` (`aud` alias) | `fx/aud_all.py` |
| `vafx tools detect` | `fx/detect_all.py` |
| `vafx tools separate` | `fx/separate_all.py` |
| `vafx tools tts` | `tts/tts_all.py` |

For example, `vafx tools asr --help` forwards help to the ASR batch consumer.
A runnable listing establishes the dispatch contract; it does not establish that
every dependency, model, native runtime, or batch workflow has passed acceptance.

ASR and TTS batches save a `.request.json` receipt beside each successful output.
Reuse requires the same input bytes, text, reference transcript, selected options,
adapter and output bytes. Changed, tampered or legacy outputs are preserved and
reported as failures until `--force` explicitly permits regeneration. TTS also
checks leftover raw WAVs before invoking adapters; relative reference paths resolve
from the caller's directory before children enter the output directory.

## Isolated engines and private reproduction

Tools retain separate uv-managed dependencies when model runtimes conflict.
The shared package does not install every engine into one environment. Native
runtime roots are separate from model stores; see [Models](models.md).

The private reproduction wrapper has the interface
`python reproduce.py {asr,audio,voice} [--show-command] [child arguments...]`.
It invokes `vafx tools run`, supplies private input/output locations, forwards
additional child arguments, and returns the child exit status. `--show-command`
prints the invocation without running it; its output contains private paths and
must remain private. The wrapper and its inputs are not part of this package.
