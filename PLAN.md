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

An actual release is outside this implementation goal and requires separate
publication authorization and published-artifact verification. No commit, push or
package publication is claimed by local validation.

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
- [ ] Complete privacy audit, commit, tag v1.0.0, push and verify publication.
