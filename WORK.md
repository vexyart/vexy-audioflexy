---
this_file: WORK.md
---

# Work

## Package foundation

Baseline: destination contained README, LICENSE and ignore rules; no package
configuration or existing test runner. Added Hatch/hatch-vcs configuration,
explicit Fire composition root, version/help, synthetic CLI checks and example.
Validation pending below. Models and tool commands require separate integration
review before mounting in the CLI.

Validation: `uv sync` passed using Python 3.14.10. `./test.sh` passed all four CLI
tests and the synthetic example. `uv build --no-sources` built a wheel through
the sdist. Installing that wheel into a separate Python 3.12.12 environment
passed installed `vafx version`, help, all four CLI tests and the example;
version was `0.1.dev1+gd44e9a0f2.d20261107`. Archive member inspection found only
selected code, tests, examples, metadata and docs; no source inputs or private
artifacts. Concurrent model module was included but remains separately reviewed.
Existing LICENSE was checked and is Apache-2.0. `git diff --check` passed.

Source suites have separate baseline failures recorded in the workspace source
audit; this foundation result does not imply a clean source migration baseline.
No model inference, final documentation build, release/tag validation or
publication was performed here. Final integrated artifacts need another audit.

## Integrated package update

Reviewed model configuration/discovery/registry and namespaced tools are mounted
in vafx. Raw child arguments are routed before Fire parsing. All168 mapped source
resources are included with helper/specification/patch/native support files.
Migration APIs provide reviewed local relocation and separate file deduplication.
README now describes these interfaces and portable examples instead of the old
foundation-only status.

The reproducible synthetic suite reached376 passing tests plus the example on
Python3.12.12 after adding gguf and sacrebleu as dev-only dependencies. The active
artifact audit covers installed entrypoints, resource closure and public archive
contents; its final snapshot follows remaining model-adapter review. Model
acceptance is separate: short successful local runs do not establish broad
variant coverage or long-form quality. No package publication has occurred.

## Public acceptance pointers refreshed

PLAN and TODO now reflect reviewed package/model/tool/helper/documentation work
and the bounded 471-test source/installed snapshot. Private storage/caller closure
and independent whole-scope acceptance remain open. Catalog declarations remain
separate from available published artifacts and exact local inference evidence.
A final public artifact/docs refresh follows these documentation edits; no commit,
push or publication is part of this pass.

## Finite implementation closure and literal-link correction

The current source and isolated installed-wheel snapshot passes 482 regression
tests. Reviewed migration, overlay and runtime-link helpers now read exact OS
symlink strings, preserving dot components and trailing slashes. Rollback removes
only its own exact generated alias; an owner replacement is retained with recovery
evidence. Historical completed-operation pins remain unchanged.

Finite mapped storage placement, private extraction and caller/reproduction
closure have independent acceptance. Installed model discovery and all seven
post-consolidation model/role profiles passed; these are exact short-profile
results, not universal variant, backend or long-form quality coverage.

Independent whole-scope review is the remaining implementation completion gate.
The final documentation reconciliation changes no executable code; retain the
482-test, strict-docs and offline CLI evidence and refresh affected archive hashes.
Actual release/publication is future work outside this implementation goal. No
commit, push or package publication has occurred.

## Final review correction wave

Reproduced the final review's four Important findings with synthetic files and
intercepted providers/processes. ASR/TTS batches now bind successful reuse to
request and output bytes, preserve stale/legacy results until explicit force,
and resolve TTS references before the child's writable output cwd. Audio.cpp
audio/TTS preparation uses provider snapshots, native links and package markers.
IndexTTS checks its complete checkpoint/companion binding set before writes.
Corrected the ASR audio.cpp module description; its executable code is unchanged.

The final executable/test snapshot passes 510 source tests plus the synthetic
example and 510 tests against an isolated Python 3.12 wheel installation. The
installed suite reports three upstream jieba SyntaxWarnings. Strict pinned
ProperDocs/MaterialX documentation builds successfully. Archive/resource/metadata
and offline installed CLI checks are retained with the final artifact manifest.
All 51 resources used by the seven accepted installed model profiles remain
byte-identical; no new inference claim is made. Provider/model/configuration,
registry and accepted relocation state were not changed. Scoped independent
rereview remains pending. No model jobs, downloads, migrations or publication
were performed in this correction wave.

## Finite implementation acceptance complete

Independent whole-goal scoped rereview passed with no residual findings. All
current implementation checkboxes are complete. The final executable snapshot
passes 510 source and isolated installed tests; all accepted inference resources
and frozen configuration/storage identities remain unchanged. Earlier pending
review statements above describe historical stages, now superseded by this PASS.

Acceptance remains bounded to the reviewed finite scope, exact short native
profiles and tested platforms. It does not establish every backend, language,
quantization or long-form quality; source redistribution rights and future remote
release prerequisites remain separate. No commit, push or publication occurred.

## Tool namespace rename in progress

Baseline: 510 source tests and the synthetic example passed before edits.
Before-file snapshot is retained privately under `.superpowers/namespace-rename`.
Scope includes public resources, dispatch, tests, current docs/wrappers and
authoritative migration pins; historical source/evidence and issue304 storage
state are preserved. Verification and independent review follow the rename.

Namespace verification: 512 source tests plus example and 512 tests against a
fresh isolated Python3.12 installed wheel passed (three upstream jieba warnings).
ProperDocs strict build passed. All255 source resources match wheel, sdist and
installed files; all168 original tool mappings and259 current destination/hash
pins resolve. Twelve installed CLI checks pass, including canonical namespace
lists and all eight batch names (`fx` plus retained `aud`). Seven private wrapper
routes use the new namespaces. Existing configuration/registry/runtime identity
and native-helper hashes are unchanged; accepted engine resources differ only
by namespace text, with the dispatch change separately tested. Independent
review is pending. No model jobs, migrations, downloads or publication occurred.

## Tool namespace rename accepted

Independent namespace review passed with no findings. The final snapshot passes
512 source and 512 isolated installed tests, examples and strict documentation.
All 255 packaged resources and 259 current pins reconcile. Issue 304 remains in
scope; configuration, registry and storage identities are unchanged. Evidence:
.superpowers/namespace-rename/review.md and implementation-report.md in the
workspace root. No inference rerun or publication is claimed.

## Version 1.0.0 release preparation

Baseline 512 tests and the synthetic example pass. Release scripts stage exact
artifacts in unique ignored directories and chain publish/release/install/build.
Tag CI builds/tests/docs and GitHub assets; local uv publication remains separate.
Privacy audit and independent review proceed before authorized external actions.

Release preparation checks: 522 tests and synthetic example pass; the ten new
release tests cover chaining, exact artifacts, stable-tag/dirty/version guards,
immutable existing assets and a single publish attempt with retries disabled.
Shell syntax, shellcheck, actionlint, lock check and strict docs pass. Two fresh
builds from the same selected tree produce identical wheel/sdist bytes. Actual
tag metadata, installation, GitHub run and PyPI publication remain the release
owner's next steps; development-build checks do not prove remote acceptance.
