---
this_file: src_docs/md/models.md
---
# Model configuration and discovery

The `models` CLI provides configuration, registry, and discovery operations:

```sh
vafx models config
vafx models configure /absolute/model-store
vafx models discover
vafx models discover --save=True
vafx models list
vafx models download_destination huggingface
vafx models download_destination lmstudio
```

Replace `/absolute/model-store` with your own absolute directory. `configure`
saves the setting; `discover` only reads locations unless `--save=True` is given.
`list` includes registered locations whose files are now missing. Discovery and
registration do not download weights or establish model compatibility.

An invocation-only override precedes the subcommand:

```sh
vafx models --models_dir=/absolute/other-store config
```

`config.toml` belongs in the platformdirs configuration directory for
`vexy-audioflexy`. It accepts a user-supplied `models_dir` string. The registry,
`models.toml`, belongs in the platformdirs data directory. Registry records contain
`identity`, `provider`, and `path`. These files contain local paths and belong
outside public artifacts.

Model-folder precedence is an explicit override, then saved configuration, then
the application's default data-directory models folder.

## Provider stores

Hugging Face models should remain in their provider cache. Discovery respects
`HF_HUB_CACHE`, its legacy equivalent `HUGGINGFACE_HUB_CACHE`, `HF_HOME`, and
the platform cache environment. LMStudio discovery reads the home-directory
`.lmstudio-home-pointer` file, with `.lmstudio` as its fallback home.

Other model sources use the configured Audioflexy models folder. Selecting a
download destination performs no download. Discovery records locations without
loading weights; a discovered file or directory is not proof of runtime
compatibility. Missing files, unsupported models, and untested engines must remain
explicit in acceptance records.

Audio.cpp audio and TTS preparation downloads each specification's repository,
revision and file list into the Hugging Face cache, then creates native layout
links and package markers. It preserves conflicting files unless `--force` is
explicit. IndexTTS's checkpoint and CAMPPlus companion need a separate writable
model view; preparation rejects provider snapshots, their aliases and conflicting
views before creating any destination directories or links.

## Model stores and runtime roots

The model CLI uses its explicit `--models_dir` override, then saved configuration,
then the application data directory. Independently executed ASR helpers also
support `VAFX_MODELS_DIR` before the saved setting. These are model-weight roots.
Runtime-root conventions vary by namespace:

| Namespace | Runtime override | Default runtime location |
| --- | --- | --- |
| ASR helpers using `_audioflexy_paths` and transcribe.cpp | `VAFX_RUNTIME_DIR` | Application cache directory, then `runtimes` and engine name |
| Audio tools | `VAFX_AUDIO_RUNTIME_ROOT` | Application data directory, then `audio`; native checkouts usually live under `engines/` |
| Voice tools using `_paths` | `VAFX_VOICE_ROOT` | Application data directory, then `voice` and engine name |

Audio tools separately support `VAFX_AUDIO_CACHE_ROOT`, defaulting to the
application cache directory's `audio` child. Voice helper model requests use
registered identities or `VAFX_MODELS_DIR`/saved model configuration rather than
the voice runtime root. Setting a model store does not install or relocate a
native runtime.

Specific adapters may override the runtime with `TRANSCRIBECPP_ROOT`,
`AUDIOCPP_ROOT`, or `AUDIOCPP_BIN`. The audio.cpp adapter separately accepts
`AUDIOCPP_MODELS_ROOT` and `AUDIOCPP_SPECS_DIR`. Inspect each tool's help and
requirements before running it; provider store discovery does not configure all
engine-specific model formats.

## Bounded native evidence

Synthetic nine-word native smoke checks passed for the exact requested Underdog
Bark MLX 8-bit, Parakeet Ultra Q8_0 GGUF, and Nemotron 3.5 30B AD-Q5_0_XL decoder.
Bark required explicit `model_type=parakeet`. Nemotron required a pinned
SALM-capable native source build with `NEMO_SPEECH_BUILD_SPEECHLM=ON`, the exact
decoder, and its pinned companion encoder; an earlier runtime failed recognizer
dispatch. Each successful output matched all nine normalized reference words.

These are private native-command acceptance results. They do not establish
installed public-adapter acceptance for those three exact named targets,
multilingual or long-recording quality, complete tool consolidation, or complete
storage migration. Native commands, transcripts, logs, hashes, and machine paths
remain in private evidence. A public adapter needs its own reproducible invocation
and output checks against the installed package before its route is accepted.

Later acceptance covered seven installed short profiles after consolidation:
the three exact named targets, Silero VAD, VoxLingua language ID, Sortformer and
the qualified local Phonon route. Each completed successfully with saved output
and terminal process evidence. This separate tier does not broaden the earlier
native results to other variants, languages, backends or long recordings.


## transcribe.cpp catalog and dedicated roles

The checked catalog at upstream commit
`5700f127e38dd53c4735ea23ef743b4b538eacd4` covers all 77 upstream variants:
73 ASR variants with published accessible artifacts, one qualified official-source
local Phonon-2 F32 conversion, and three dedicated non-ASR roles. The catalog
records 398 declared quantization artifacts. Published metadata is accessible for
392 artifacts; the six declared artifacts in the official Phonon mirror remain
unavailable. The converted F32 file is a locally produced derivative, not an
official downloadable mirror artifact.

Every accessible published artifact has an immutable provider revision, byte size,
and published SHA256. Preparation checks both size and SHA256 before treating a
cached file as ready; incorrect cached bytes are rejected without rewriting them.
This exact catalog coverage does not imply inference acceptance for every model,
quantization, backend, or upstream-declared capability. The ASR adapter exposes
its offline transcript contract; catalog declarations of streaming, translation,
or timestamps do not add those methods to that contract.

Inventory and preparation commands:

```sh
vafx tools run asr asr-transcribecpp packages
vafx tools run asr asr-transcribecpp roles
vafx tools run asr asr-transcribecpp prep_role silero-vad-v6.2 --quant=F32
```

Set `TRANSCRIBECPP_ROOT` to a checkout built at the pinned commit before inference.
The `role` method resamples audio to mono 16 kHz, runs the native role, and writes
structured JSON rather than inventing an ASR transcript:

```sh
vafx tools run asr asr-transcribecpp role /absolute/synthetic.wav --package=silero-vad-v6.2 --quant=F32 --output=/absolute/vad.json --backend=cpu
vafx tools run asr asr-transcribecpp role /absolute/synthetic.wav --package=lang-id-voxlingua107-ecapa --quant=Q8_0 --output=/absolute/langid.json --backend=cpu
vafx tools run asr asr-transcribecpp role /absolute/synthetic.wav --package=diar_streaming_sortformer_4spk-v2.1 --quant=Q8_0 --output=/absolute/diarize.json --backend=cpu
```

Use your own absolute input and output paths. VAD retains exact sample intervals;
language ID retains ranked language codes, indices, probabilities and logits;
Sortformer retains speaker turns and original native output. Language ID scores
at most the first 30 seconds, recorded in the result. Sortformer uses 80 ms frames;
normalized end times are clipped to the real audio duration while retaining native
frame padding separately. Empty VAD and diarization results are valid. Malformed,
nonfinite, or out-of-range native values are rejected before replacing output.

Real public source-adapter invocations on one synthetic 2.67075-second fixture
passed CPU checks for exact local Ultra Q8_0 and qualified Phonon F32 (nine
normalized words each), Silero VAD F32 (one segment), VoxLingua107 Q8_0 (107 ranked
candidates), and Sortformer Q8_0 (one turn). These are bounded route checks, not DER,
all-language, streaming, all-artifact, or installed-wheel acceptance.

### Reproduce the qualified Phonon conversion

The official `handy-computer/phonon-2-gguf` mirror remains unavailable. The
supported local route uses the official `FermionResearch/Phonon-2` packed source
at revision `7e153e4054c0a10db6d47d6fff5c062d4c77c154`, converted by the pinned
transcribe.cpp converter and its committed frozen environment. With that checkout
as the working directory, replace the output placeholder with your chosen path:

```sh
uv run --frozen --project scripts/envs/parakeet-phonon scripts/convert-parakeet.py FermionResearch/Phonon-2 /absolute/model-store/conversions/transcribecpp/phonon-2/phonon-2-F32.gguf --revision 7e153e4054c0a10db6d47d6fff5c062d4c77c154
```

The source archive `phonon-2.bps.tar.zst` has SHA256
`98125795b6dda72f5c6eee9ba33d19815df65dcb18b50a357bf9f73c9935309e`.
The qualified F32 output is 2,508,435,648 bytes with SHA256
`f2b1a5563f0a4a764ca40f4fb72da6d3a4d4ce5653e0f8fbcc874a6a81c5c10e`.
The converter verifies the packed source and uses pinned tokenizer/config assets
without downloading teacher weights. The public adapter additionally checks the
converted file's exact hash, size, architecture and variant metadata:

```sh
vafx tools run asr asr-transcribecpp prep phonon-2 --quant=F32 --local_model_path=/absolute/model-store/conversions/transcribecpp/phonon-2/phonon-2-F32.gguf
vafx tools run asr asr-transcribecpp transcribe /absolute/synthetic.wav --package=phonon-2 --quant=F32 --local_model_path=/absolute/model-store/conversions/transcribecpp/phonon-2/phonon-2-F32.gguf --text_out=/absolute/phonon.txt --backend=cpu --threads=4
```

Keep the derivative in your configured external model store. Other GGUF exports
with the same model name are not interchangeable with this route. Other Phonon
quantizations remain unavailable through the mirror and untested locally.
