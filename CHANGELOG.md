---
this_file: CHANGELOG.md
---

# Changelog

## Unreleased

- Add `aud-audiocpp.py stage` to install a locally converted file as a package (linked under the spec name, marker records size and SHA-256), and route Sortformer Diar v2.1 through it. Verified on a two-voice clip; 71 of 166 routes are verified.
- Route ACE-Step (text2music with lyrics and duration; `aud-audiocpp.py run` gained `lyrics` and `duration_seconds`, passed as `--lyrics` and `--duration-seconds`), MiniMax Music 3 (lyrics request option) and sherpa_kws (embedded keywords) from the catalog. ACE-Step and sherpa_kws verified; MiniMax Music 3 q4_0 produced a gong-like clip, so it stays `pending runtime`. Wake-word analysis with no detections now writes an empty list instead of failing. 70 of 166 routes are verified.
- Route VeVo2 (`zero_shot_tts` with `--voice-ref`, no reference transcript) and AuK Flash (instruction TTS with its Qwen2.5-Omni-3B and VAE packages, English only) into `tts_all`. Both verified on 83b1354 by an ASR round trip. AuK also speaks the first words of the voice description; its reference-conditioned TTS and `--task gen` editing abort in the Metal flow sampler, and AuK does not support the CPU backend. 68 of 166 routes are now verified.
- `tts_all` catalog rows pass `deps`, `request_options`, `session_options` and `extra` as `--key=value`, so a value starting with `--` reaches the adapter intact. `tts-audiocpp.py prep` accepts the tuple Fire makes from a comma-separated `--deps`; before this, every multi-package call failed with "'tuple' object has no attribute 'split'".
- VieNeu v3 Turbo runs from phoneme text with its packaged `minh_quan_pro` voice (OWSM transcribes it exactly). It stays out of the registries, which only feed en/pl text. SAM Audio stays pending: three different text prompts gave the same split.
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

- Route audio.cpp analysis families to `aud-audiocpp.py analyze` (catalog route `aud_analyze`). Verified on 83b1354: MMS, Qwen3 and CrisperWhisper word alignment; FireRed VAD; Sortformer 4spk and Nemotron-3 diarization on a two-speaker clip; openWakeWord "Alexa" (0.9997) and microWakeWord "Okay Nabu" (0.973); AST and CED classification; LFM-D1 audio questions (speech 0.89 on speech, music 0.94 on music). LFM-D1 gets its required `questions=` JSON from the catalog. Sherpa KWS (BPE keyword rows) and Sortformer v2 (local conversion only) stay `pending adapter`.
- Route audio.cpp music, sound-effect and MIDI families to `aud-audiocpp.py run --task gen|midi` (route `aud_generate`); ledger keys may carry a `:task` suffix where one package has several tasks. Verified, each checked with the AST classifier: MOSS SoundEffect (rain), ControlFoley (bird), Stable Audio 3 (piano music; dog-bark effect), Magenta RealTime 2 (funk), HeartMuLa (sung verse with tags), MiDashengLM gen (dog bark; a rain prompt came out as train noise), MuScriptor (MIDI notes) and SheetSage 2 (an ABC lead sheet, not MIDI). Families that need lyrics, melody, style, video or image inputs stay `pending adapter`.
- Chatterbox Multilingual via codec.cpp: prefix the text with `[lang]`, lowercased and NFKD-normalised, as upstream's MTLTokenizer does; codec.cpp leaves that to the caller. Polish had come out as Thai script and now round-trips word-perfect, so `tts_all` runs it for en and pl. `say --language` rejects zh and ja, whose script conversions are not implemented. codec.cpp d5e3852 is current upstream.
- transcribe.cpp v0.4.2 accepts the local Phonon-2 F32 conversion: transcripts are byte-identical to 5700f12 on both English clips. `LOCAL_PHONON` records it under `verified_runtimes`; the conversion's own `runtime_commit` is unchanged.

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
