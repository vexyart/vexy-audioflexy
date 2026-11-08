---
this_file: PLAN.md
---

# Consolidation plan

The public package consolidates the ASR, audio-processing and voice tool
namespaces. Public implementation and bounded acceptance are established;
finite mapped storage, private extraction and caller closure are independently
verified. Independent whole-scope acceptance is complete within the recorded finite bounds.

## Implemented and reviewed

- Hatch/hatch-vcs packaging, Git-derived versions, uv development environment,
  Python Fire `vafx`, installed version/help and a synthetic example.
- Portable model configuration, provider discovery and registry commands. Explicit
  CLI settings override saved TOML; downloads prefer provider-managed stores.
- All 168 original mapped tools and batch consumers, supporting resources and
  isolated uv invocation. Raw child arguments and nonzero exit status are retained.
- Reviewed model relocation, provider compatibility and runtime-path helper APIs.
  Finite mapped storage moves, provider placement and caller compatibility have
  separate independent acceptance; historical source and recovery evidence remain.
- Documentation source built with pinned ProperDocs/MaterialX dependencies and a
  semantic-tag release workflow that builds docs and publishes through `uv publish`.
  Workflow configuration is distinct from an executed release.

## Bounded acceptance

The latest public snapshot passed 510 source tests and 510 tests against an
isolated Python 3.12 wheel installation. All mapped resources were present; 163
Python tool entries had valid inline dependency metadata, with helpers, shell and
native resources explicitly classified. Wheel/sdist member and content audits
found no private artifacts or identifying machine paths in the inspected patterns.

The transcribe.cpp catalog represents 77 declared variants and 398 declared
artifacts: 74 ASR variants and three non-ASR roles. Published metadata covers 392
artifacts; the six declared Phonon mirror artifacts remain unavailable. A qualified
local Phonon conversion is a separate accepted route. Installed short-sample
checks cover the exact requested local models, VAD, language ID, diarization and
local Phonon. They do not establish every variant, language, quantization, backend
or long-form quality. The seven installed model/role profiles were also checked
after consolidation; their finite acceptance does not broaden those limits.

## Implementation acceptance complete

Independent whole-scope review and the final correction-wave rereview passed
with no residual findings. The current finite implementation is accepted with
510 source and isolated installed regression tests and retained strict-docs,
resource/privacy and bounded installed model/role evidence. Later changes require
fresh affected checks; this acceptance does not extend native compatibility or
resolve source redistribution rights.

## Future release

The initial implementation excluded publication. The subsequently authorized
1.0.0 release status and remaining PyPI verification are recorded below.

## Tool namespace rename

Rename installed public tool namespaces from `myasr`, `myaud`, `mypvc` to
`asr`, `fx`, `tts`. Preserve original `_incoming` paths and historical receipts.
Update dispatch, tests, resource headers, CLI examples, private wrapper routes and
current migration destination/hash pins. Use canonical `fx` batch dispatch while
retaining the existing `aud` batch alias. Old namespace imports/routes are replaced.

Issue 304 remains regression scope: model configuration/discovery, provider
preference, registry and upstream links retain existing behavior and state.
No storage migration, model download, native rebuild or publication is needed.

- [x] Move all three public resource trees and update their callers and headers.
- [x] Update current docs, private wrappers and authoritative migration maps.
- [x] Verify source/installed suites, CLI/resource closure, archives and strict docs.
- [x] Obtain independent namespace review and resolve findings.

## Version 1.0.0 release

- [x] Add and verify chained build/install/GitHub release/PyPI scripts.
- [x] Verify stable tag configuration and semver-tag GitHub release workflow; actual tagged version is checked before publication.
- [x] Complete privacy audit, commit, tag v1.0.0, push and verify GitHub release.
- [ ] Publish the preserved 1.0.0 assets to PyPI and verify hashes and installation.

## Release version repair

Use `uvx gitnextver --pull=False` to create a committed, pushed stable tag before
no-argument release/publish builds; add `--commit=False` for clean untagged HEAD.
Reuse clean tagged HEAD for resumable uploads;
explicit tags still require the selected clean checkout. Generate an ignored
`__version__.py` using the hatch-vcs build hook. Keep development build versions
honest and reject them at the existing release wheel/sdist metadata gate.

- [x] Implement version generation, tagging and clean stable release guards.
- [x] Cover automatic tagging, tagged reuse, tagging failures and dirty/development rejection with focused script tests.
- [x] Verify actual isolated tagged wheel/sdist and installed version parity, full regression suite and independent review.

- [x] Provide YAML structured CLI output and --json with source and installed-wheel verification.

## Reusable speech integration

- [x] Port reusable speech engines, neutral types and language/tagging APIs with lazy optional dependencies.
- [x] Verify speech regressions, packaged workspace route, full suite and installable artifacts.

Subtitle consumer integration and independent whole-goal/correction review passed.
Final provider586 and consumer149 tests, installed export/resume and legacy-only
offline POS checks pass. Native-build limits remain explicit.

## Version 1.0.3 integration release

The reviewed speech API is ready for the authorized provider/consumer release.
Publication and installation from the registry require fresh stable artifacts;
local development-wheel acceptance alone does not establish either outcome.

- [x] Commit and tag the reviewed provider integration after registry/version checks.
- [x] Verify tagged CI, strict docs and exact public wheel/sdist contents.
- [ ] Publish the verified stable assets and verify registry hashes and installation.

PyPI 1.0.3 upload attempted; HTTP 429 new-project quota blocks publication.
Retain exact GitHub assets for retry; do not rebuild or bump to evade the limit.
