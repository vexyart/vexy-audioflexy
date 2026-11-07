#!/usr/bin/env bash
# this_file: src/vexy_audioflexy/tools/tts/tools/verify-samples.sh
#
# Transcribe a 10-second slice of every clip in out/ and print it beside the
# filename. An engine can run, report success and still emit noise, so this is
# how a sample is confirmed to be speech.
#
# Whisper's whole-file pass often returns nothing on a 40-second clip that is
# perfectly fine, hence the slice. Start later in a clip (SEEK=10) when a model
# opens with a long quiet lead-in.
#
# Results are also written to out/transcripts.json, which the dashboard reads so
# each clip shows what it actually says beside its player.
#
#   ./tools/verify-samples.sh            # every clip
#   ./tools/verify-samples.sh 'en__*'    # only the English ones
#   SEEK=10 ./tools/verify-samples.sh
set -uo pipefail

OUTDIR="${VAFX_VERIFY_OUTDIR:?Set VAFX_VERIFY_OUTDIR to the clips directory}"
CRISPASR="${CRISPASR_BIN:?Set CRISPASR_BIN to the built executable}"
PATTERN="${1:-*}"
SEEK="${SEEK:-0}"
LEN="${LEN:-10}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

if [[ ! -x "$CRISPASR" ]]; then
  echo "CrispASR binary not found at $CRISPASR (set CRISPASR_BIN)." >&2
  exit 1
fi

JSON="$OUTDIR/transcripts.json"
[[ -f "$JSON" ]] || echo '{}' > "$JSON"

record() {  # record <filename> <verdict-text>
  CLIP="$1" TEXT="$2" JSON="$JSON" python3 -c 'import json, os
path = os.environ["JSON"]
try:
    data = json.load(open(path))
except Exception:
    data = {}
data[os.environ["CLIP"]] = os.environ["TEXT"]
json.dump(data, open(path, "w"), indent=2, ensure_ascii=False, sort_keys=True)'
}

shopt -s nullglob
for clip in "$OUTDIR"/${PATTERN}.mp3; do
  ffmpeg -y -loglevel error -ss "$SEEK" -t "$LEN" -i "$clip" "$TMP/slice.wav" || continue
  printf '%-62s ' "$(basename "$clip")"
  "$CRISPASR" --backend whisper -m auto --auto-download \
    -f "$TMP/slice.wav" --no-timestamps >"$TMP/hyp.txt" 2>"$TMP/err.txt"
  text="$(tr '\n' ' ' < "$TMP/hyp.txt" | sed 's/  */ /g; s/^ //; s/ $//')"
  if [[ -n "$text" ]]; then
    printf '%s\n' "${text:0:80}"
    verdict="$text"
  elif grep -q 'no text produced for' "$TMP/err.txt"; then
    # Whisper's own diagnosis: there IS audio here, it just is not speech.
    verdict="[NOT SPEECH - audio present but nothing transcribable]"
    echo "$verdict"
  else
    verdict="[no transcript - clip may be silent, or starts later]"
    echo "$verdict"
  fi
  record "$(basename "$clip")" "$verdict"
done
echo "transcripts recorded in $JSON"
