---
this_file: TODO.md
---

# Acceptance tasks

- [ ] Publish the preserved v1.0.0 GitHub release assets through Trusted Publishing and verify PyPI hashes and a clean install.
- [ ] Publish the preserved v1.0.3 GitHub release assets through Trusted Publishing and verify PyPI hashes and a clean install.
- [ ] Smoke-test remaining routable audio.cpp 83b1354 entries marked `pending runtime`, record usable runs in `catalogs/audiocpp-verified.json` and regenerate the catalog.
- [ ] Recheck AuK reference TTS and audio editing after the Metal flow-sampler abort is fixed. Verify a reference-conditioned clip and an instruction edit.
- [ ] Recheck SAM Audio on the same mix with distinct prompts and verify that target and residual tracks change with the prompt.
- [ ] Verify MiniMax-H3 audio generation through `--task gen` and check where SamsOne captions belong in the audio analysis routes.
- [ ] Add input-aware routes for melody and reference music (MuLaCover and VeVo2) and audio editing. Recheck MuLaCover with a reference clip once its allocation failure is fixed.
- [ ] Compare MiniMax Music 3 against upstream example audio before retesting lyric intelligibility.
- [ ] Decide whether the spoken-dialogue outputs from LFM2, LFM3 and Ming3 belong in the `aud_all` report.
- [ ] Switch the default transcribe.cpp runtime link to v0.4.2 and confirm the default CLI route yields the verified Phonon-2 transcripts.
