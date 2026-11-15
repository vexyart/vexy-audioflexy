---
this_file: CHANGELOG.md
---

# Changelog

## Unreleased

- Run audio.cpp packages on upstream 83b1354 (2026-11-15). The new engine resolves model paths and picks the tensor format from the file extension, so the provider now hardlinks cached files under their package names (falling back to a symlink across volumes) and upgrades older blob symlinks in place.
- Follow the upstream package rename `voxcpm1_0.5b_q8_0` → `voxcpm1_0_5b_q8_0`. Point entries that defaulted to the missing `audio.cpp-20261030` / `macos-metal-full` runtime at the standard `audio.cpp` release build.
- Pick the newest installed speech.cpp runtime instead of the fixed 0.8.11 path; the adapter's CLI is verified against speech.cpp 0.11.0.
- Update `mlx-audio` from 0.5.7 to 0.6.5 in the `apple` extra and the pinned ASR/diarization scripts.

## 1.0.3

- Reuse existing vexy-subs NLTK cache weights for offline English tagging; keep new downloads in the Audioflexy cache.

- Add reusable lazy speech APIs for Qwen/Whisper, native audio.cpp, alignment, diarization, punctuation, langid and NLTK, with shared validated Word/Turn types and packaged workspace execution.

- Print structured `vafx` results as readable YAML by default and support `--json` for complete JSON documents, while preserving forwarded tool arguments and output.
- Create and push stable release tags with `uvx gitnextver` when release/publish is run without a tag on untagged or dirty work; reuse clean tagged releases for resumable publication.
- Generate an ignored `__version__.py` through hatch-vcs and retain exact stable wheel/sdist version checks before release or publication.

## 1.0.0

- Add chained build, uv tool installation, GitHub release and uv publication scripts.
- Build tagged releases and strict documentation in GitHub Actions, with version checks and immutable asset verification.

- Rename public tool namespaces to `asr`, `fx`, `tts`; update CLI/resource consumers and retain `aud` as a batch alias.

- Bind ASR/TTS batch resume to request and output bytes; preserve stale and legacy outputs until explicit force, and resolve caller-relative TTS references before dispatch.
- Prepare audio.cpp audio/TTS packages through provider snapshots and native links; preflight IndexTTS checkpoint/companion bindings outside immutable provider stores.
- Correct the ASR audio.cpp provider-layout description and document bounded installed acceptance.
- Add the Python package foundation with a Fire `vafx` command and Git-derived version.
- Add uv development dependencies, synthetic CLI tests, example, and build file selection.
- Add portable model configuration, provider discovery and a local model registry.
- Mount namespaced audio/voice/ASR tools with isolated uv execution and raw argument forwarding.
- Include migrated helper, model-specification, catalog, patch and native resources.
- Add reviewed model relocation and file-deduplication APIs with private receipts.
- Include documentation sources and build requirements in source distributions.
- Extend synthetic regression dependencies without adding mandatory model runtimes.
- Refresh public implementation and acceptance pointers while retaining open storage and whole-scope verification tasks.
- Preserve exact OS symlink target strings and retain owner-replaced aliases during rollback.
- Reconcile independently accepted finite storage/private/caller closure and 482-test evidence; retain whole-scope review as the remaining implementation gate.
