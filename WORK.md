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

## Version 1.0.0 release status

Commit fa700ce and immutable tag v1.0.0 are pushed. The public tree and stable
wheel/sdist privacy audit passed; three private-derived comments/docstrings were
redacted with originals preserved privately. All 255 installed resources match.
The Linux release run passed 522 tests and strict docs and created GitHub assets
byte-identical to local tagged builds. The workflow now also supports rebuilding
an existing tag and explicitly installs FFmpeg for the full acceptance suite.

GitHub release: https://github.com/vexyart/vexy-audioflexy/releases/tag/v1.0.0
CI: https://github.com/vexyart/vexy-audioflexy/actions/runs/42414796783

The complete publish -> release -> install -> build chain ran. PyPI rejected
the first wheel with HTTP 429, "Too many new projects created". No retry was
made; project and version endpoints still returned 404. Exact release assets
and hashes are preserved for resumption after the external limit is resolved.
PyPI publication and installation from PyPI remain incomplete. No model jobs
or model/storage changes were performed for this release.

## Release version repair

Baseline: 522 tests and the synthetic example passed before edits. The release
script formerly required an exact tag even when no tag argument was supplied.
No-argument release/publish now uses `uvx gitnextver --pull=False` when HEAD is
untagged or dirty, committing pending changes and pushing the next stable tag.
Clean untagged HEAD uses `--commit=False` because gitnextver otherwise stops
without tagging when there is nothing to commit. Clean tagged HEAD is reused,
preserving resumable publication. Explicit tags
retain the clean checkout guard. Existing wheel/sdist metadata equality and exact
stable tag checks remain mandatory before installation or publication.

Added the hatch-vcs generated `src/vexy_audioflexy/__version__.py` build hook and
ignored its filename so generation cannot dirty tagged checkouts. Runtime version
reporting still uses installed package metadata. No mandatory dependency changed.
Current official hatch-vcs build-hook docs and `uvx gitnextver --help` confirm the
configuration and CLI flags. Focused tests cover automatic tagging, clean tagged
reuse, failed tagging/push, residual dirty state and invalid returned tags.
Final source suite: 532 tests and the synthetic example pass. Shellcheck and
`git diff --check` pass. A first in-progress suite overlapped the clean-tag flag
correction and failed against an older fixture snapshot; the settled-tree rerun
above resolves those failures. Isolated artifact and independent review results
follow verification.

Independent real-command verification in disposable repositories found that
gitnextver's default commit mode exits successfully without a tag on unchanged
HEAD. The clean branch now supplies `--commit=False`; the fake CLI reproduces
that no-change behavior. Real gitnextver, local bare-remote pushes, uv build and
isolated install then passed for both clean and dirty untagged states. Selected
versions 8.0.1 and 8.0.2 matched wheel/sdist metadata, generated modules and
installed CLI output; generated version files remained ignored and checkout
status stayed clean. Explicit dirty publication was refused before building.
GitHub asset operations and PyPI upload were intercepted for this verification.

Final strict ProperDocs build, uv lock check, shellcheck, actionlint and diff
whitespace checks pass. Regenerated tracked development documentation and search
index reflect the new release contract. Final independent diff review passed
with no blocking findings; no placeholders, skipped tests or private data were
introduced. No live release or PyPI publication is claimed by this repair pass.

## Readable CLI output

Baseline: 532 tests and the synthetic example passed. A new process-level test
reproduced the previous output problem: a one-record model list parsed as a
mapping rather than the complete list. The shared Fire serializer now emits
structured results as YAML, retaining insertion order and Unicode; `--json`
selects a complete JSON document. Scalar version/path output remains plain text
unless JSON is requested. Python API results are unchanged.

Leading output flags are handled before raw tool dispatch; discovery-command
flags are handled before Fire parsing. Child options, including `--json`, and
Fire's standalone `--` control arguments retain their existing meaning. Root and
model help remain available. PyYAML is now an explicit runtime dependency rather
than an accidental development dependency. Its safe dumper and Fire's serializer
callback were checked against official documentation and in an isolated proof.

Full regression: 551 tests and the synthetic example pass, including nested
metadata, Unicode, YAML-sensitive strings, empty registries, JSON scalar output,
help and real child argument forwarding. CLI Ruff check, lock consistency and
diff whitespace checks pass. Installed-wheel and separate review results follow.

## Reusable speech model API port

Ported current recognition/alignment/native/punctuation behavior into
`vexy_audioflexy.speech`, preserving Qwen bare retry and dropped-chunk receipts,
reference excerpts, vocabulary/metadata safeguards, lexical alignment checks,
Whisper timings, native identities and diarization channel capacity. Shared
validated Word/Turn types exclude subtitle Cue/rendering concerns. The language
and syntax APIs keep langid sampling/probabilities/override checks and lazy NLTK
English tagging; their dependencies are optional editorial extras.

Workspace result APIs now default to installed ASR resources and retain explicit
legacy workspace/environment compatibility. Their signatures bind resource bytes;
subprocess exceptions propagate. Native environment settings retain legacy names
and accept Audioflexy equivalents. No incoming tools, model storage, release
behavior or model weights changed.

Verification: established baseline 551 tests; 32 focused speech tests and full
583-test suite pass, along with the synthetic example, Ruff, lock consistency and
diff whitespace checks. Built wheel and sdist successfully. A fresh Python 3.12
wheel installation imports the speech types and locates/hashes installed adapter
resources without optional model dependencies. Speech wheel members contain no
machine paths or weight/media artifacts in the inspected patterns. Real packaged
`uv run --script speech-issue301.py --help` succeeds; invalid native family exits
nonzero with its expected ValueError. Model inference remains unverified here;
engine tests use synthetic isolated responses. Independent review is external
to this implementation pass.

### Speech workspace fingerprint review correction

The independent review reproduced stale fingerprints when the packaged adapter's
actual runtime binary changed. The signature now uses the existing packaged
`_audioflexy_paths.runtime_root` resolver and the adapter's
`AUDIOCPP_ISSUE301_ROOT` override, including VAFX_RUNTIME_DIR/platform-cache
selection. It hashes that selected executable under its resolved path (or records
its absence); foreign paths do not enter relative-to-workspace assumptions.
Existing explicit legacy workspace resources and `.external` binary fingerprints
remain present.

Regression reproduced before the fix: 1 failed, 3 passed in the workspace suite.
After the fix: all 34 speech tests pass, including platform-default byte changes,
configured location changes with identical bytes, adapter override precedence,
byte replacement/missing binary, and legacy workspace binary changes. Ruff and
diff whitespace checks pass. Full-suite/artifact refresh remains part of final
integration verification; no inference or native storage changes were made.

## Subtitle consumer verification

The final speech port passes 585 provider tests plus the example. The subtitle
consumer passes 149 tests and uses the provider modules and neutral types.
Exact wheels pass archive/private-path scans, portable metadata checks, and a
fresh base-only installed synthetic API/export/resume check. Installed real MLX
Qwen/forced-alignment/Nemotron inference produced two verified subtitle cues; a
separate CLI resume retained an identical receipt. No source path overrides were
used in the installed check; Python bytes match the checked wheels. Whole-scope
independent review remains pending. Native Nemotron's existing local binary
rejected its family name; no successful native execution is claimed.

## Offline English POS compatibility correction

The final review found that the speech port ignored existing vexy-subs NLTK
weights. The tagger now searches the canonical Audioflexy cache first and the
legacy vexy-subs cache second before attempting a download. New downloads keep
the canonical destination; existing weights are neither moved nor rewritten.

A synthetic legacy-only offline regression failed before the fix and passes
after it. All 11 focused model tests and 586 provider tests plus the runnable
example pass. Ruff check/format and diff whitespace checks pass. Rebuilt wheel
and sdist pass Twine and archive privacy checks. Fresh installed base integration
passes with two speakers, subtitle export and verified resume. Actual offline
NLTK verification and independent final approval are recorded in the private
integration evidence rather than inferred from the synthetic test.

## Integration completion

Independent whole-goal review and the scoped correction review passed with no
material open findings. The final provider suite passes 586 tests and its example;
the unchanged consumer suite passes 149 tests with 91.68% coverage and Ruff clean.
The provider now searches existing subtitle NLTK caches before downloading. Real
installed NLTK succeeds with legacy-only weights and downloads forbidden, without
changing owner weights. Final installed Python bytes match the checked wheels.
An editorial-only provider update reused unchanged real recognition/alignment/
diarization caches; refreshed captions verify and a separate CLI resume skips
inference. Exact archives, clean base-installed synthetic integration and strict
docs passed. Recovery evidence and exact wheels are retained privately.

This completes the local integration. The reviewed API is ready for the authorized
stable provider/consumer release; registry publication and installation remain
release acceptance gates. The local
native Nemotron binary remains incompatible with that model family. Short real
MLX smoke checks establish integration, not universal long-form model quality.
No commit, push, publication or model-storage migration was performed.

## Version 1.0.3 release preparation

Publication of the reviewed provider and subtitle consumer is authorized. The
next provider version is expected to be 1.0.3, subject to registry/version checks.
Reconciled the current speech API status and release checklist without changing
historical publication records or claiming remote publication. Stable tagged
artifacts, CI, registry hashes and installed behavior remain controller-owned
release checks. No model/storage changes are required.

The release scripts accept an explicit clean HEAD tag, but still rebuild and
replace the installed uv tool before comparing immutable GitHub assets. Preserve
verified GitHub-downloaded archives for upload retries; direct uv publication of
those exact files avoids rebuilding on a rate-limit resume.

Preparation verification: strict ProperDocs/MaterialX build passed. Generated
tracked HTML was restored after the check because it introduced theme whitespace;
source-document diff whitespace checks pass. Executable suites were not rerun
for these documentation-only changes.

## Version 1.0.3 publication attempt

Committed, tagged and pushed v1.0.3. Tagged GitHub CI passed package tests,
distribution build and strict docs, and published wheel/sdist assets. Downloaded
GitHub assets match the independently built stable archives byte-for-byte.
Twine, metadata/privacy checks and a clean base wheel installation passed.

PyPI rejected the first wheel upload with HTTP 429, "Too many new projects
created". No retry/version bump was performed. Registry reconciliation still
returns 404 for 1.0.3. Exact verified assets and the sanitized failure log are
retained privately for a later uv publish retry without rebuilding. The consumer
upload is held because this required provider is unavailable from PyPI.
