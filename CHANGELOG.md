---
this_file: CHANGELOG.md
---

# Changelog

## Unreleased

- Generate `tools/catalogs/audiocpp-83b1354.json` from the pinned audio.cpp `model_specs` with `tools/gen_audiocpp_catalog.py`. It records all 146 families, their packages (repo, revision, gated, files), existing registry coverage and a route per task. `tts_all`, `asr_all` and `aud_all` append one `ac83-*` row per uncovered routed family (26 TTS, 9 ASR, 10 audio-to-audio; VieNeu, AuK, MiniMax-H3 and SamsOne were later held as `pending adapter`), each marked `pending runtime` until a real run. `verified` means one real run produced usable output, not that quality is good.
- Record real runs in `tools/catalogs/audiocpp-verified.json`, keyed `registry:package` and tied to engine SHA 83b1354. The generator marks matching routes `verified` and keeps rows from any other engine revision `pending runtime`. 15 routes are verified on Metal: 6 ASR (lfm2_audio, lfm3_audio, owsm, phonon2_asr, underdog_bark3, whistle_asr) and 9 audio-to-audio (apollo, chatterbox vc, lfm2/lfm3 spoken dialogue, miocodec, mossformer2 SE, realtime_reuse, rvc, tone_color_vc).
- Pass OWSM's language as an ISO 639-3 code (`eng`, `pol`) through a catalog `lang_map`. Before this, `--lang en` failed with "OWSM v4 tokenizer is missing <en>".
- Route Apollo to a 44.1 kHz manifest source (`noisy44k`); the engine rejects other rates.
- Mark SamsOne `pending adapter` for ASR, because it is an audio captioner: given speech, it returned captions such as "the audio clip is a clear and distinct recording of a woman speaking".
- Run audio.cpp packages on upstream 83b1354 (2026-11-15). The new engine resolves model paths and picks the tensor format from the file extension, so the provider now hardlinks cached files under their package names (falling back to a symlink across volumes) and upgrades older blob symlinks in place.
- Follow the upstream package rename `voxcpm1_0.5b_q8_0` → `voxcpm1_0_5b_q8_0`. Point entries that defaulted to the missing `audio.cpp-20261030` / `macos-metal-full` runtime at the standard `audio.cpp` release build.
- Pick the newest installed speech.cpp runtime instead of the fixed 0.8.11 path; the adapter's CLI is verified against speech.cpp 0.11.0.
- Update `mlx-audio` from 0.5.7 to 0.6.5 in the `apple` extra and the pinned ASR/diarization scripts.
- Verify 21 audio.cpp TTS families on 83b1354 by ASR round-trip: each spoke one English sentence and `parakeet_tdt_q8_0` transcribed it (WER 0% for 20, 6.7% for ZipVoice).
- Route M5-TTS to its CustomVoice package: `voice_id=1` is a CustomVoice preset and the VoiceClone checkpoint rejects it. Pass `--voice-id`/`--speaker` to the engine as strings; fire had turned `1` into an int.
- Mark VieNeu v3 Turbo (needs SEA-G2P phoneme input and a speaker embedding) and AuK (needs its Qwen2.5-Omni-3B and VAE component packages) `pending adapter`.
- Verify MOSS-TTS v1.5, MOSS-TTSD (WER 0%), Ming3 Audio Tiny ASR and Nemotron Transcribe IQ4_NL (WER 15.4% each). The ASR adapter strips Ming3's leading language tag and passes Nemotron's experts GGUF as `--model`, since its package also holds the mmproj encoder. MiniMax-H3 is marked `pending adapter`: it is text-to-audio/video generation, not TTS.
- `prep` accepts a cross-volume copy of a cached blob (same size and mtime) as installed, so re-running it no longer asks for `--force`; a failed link removes its `.partial` file.
- Add audio.cpp source separation: `aud-audiocpp.py run --task sep` writes the engine's named stems to `--out-dir`, and `separate_all.py` accepts the generated `ac83-*` separation rows. Verified TF-GridNet and MossFormer2 (a two-speaker mix split into both sentences) plus HTDemucs, BS-RoFormer and Mel-Band RoFormer (speech recovered from a synthetic music bed). 45 audio.cpp routes are now verified.
- Fix the TTS dashboard step: `build_dashboard.py` is loaded by path, so it now adds its own directory to `sys.path` before importing `_paths`. Every `tts_all.py` run had crashed there after synthesis.

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
