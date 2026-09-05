#!/usr/bin/env bash
#
# Records the submission demo: processing, appearance counts, and the finished collage for
# each of the three sample clips, cut to 60 seconds.
#
# The phone must be UNLOCKED and stay awake — the script drives the UI by tapping.
#
#   bash tools/record-demo.sh
#
# Output: demo.mp4 in the repo root.

set -euo pipefail

ADB="${ADB:-/d/Android/sdk/platform-tools/adb.exe}"
FFMPEG="${FFMPEG:-ffmpeg}"
PKG=com.iykyk.collage

# Each clip takes roughly 55 s to process on a 2018 device; allow headroom.
PROCESS_WAIT="${PROCESS_WAIT:-75}"
# How long the finished collage is held on screen before moving on.
HOLD="${HOLD:-12}"

say() { printf '\n== %s\n' "$1"; }

"$ADB" shell input keyevent KEYCODE_WAKEUP >/dev/null 2>&1 || true
if "$ADB" shell dumpsys window 2>/dev/null | grep -q "mCurrentFocus=.*Bouncer"; then
  echo "Phone is locked. Unlock it and re-run." >&2
  exit 1
fi

# Keep the screen on for the whole recording.
"$ADB" shell svc power stayon true >/dev/null 2>&1 || true

# Tap targets, resolved from the running UI rather than hardcoded.
coords_for() {
  "$ADB" shell uiautomator dump /sdcard/ui.xml >/dev/null 2>&1
  "$ADB" pull /sdcard/ui.xml /tmp/ui.xml >/dev/null 2>&1
  python - "$1" <<'PY'
import re, sys
label = sys.argv[1]
xml = open('/tmp/ui.xml', encoding='utf-8', errors='ignore').read()
for m in re.finditer(r'text="([^"]*)"[^>]*?bounds="\[(\d+),(\d+)\]\[(\d+),(\d+)\]"', xml):
    if m.group(1).strip().lower() == label.lower():
        x1, y1, x2, y2 = map(int, m.groups()[1:])
        print((x1 + x2) // 2, (y1 + y2) // 2)
        break
PY
}

record_one() {
  local index="$1" out="$2"
  say "Sample $index"

  "$ADB" shell am force-stop "$PKG"
  "$ADB" shell am start -n "$PKG/.MainActivity" >/dev/null
  sleep 3

  "$ADB" shell screenrecord --time-limit 180 --bit-rate 8000000 /sdcard/seg.mp4 &
  local recpid=$!
  sleep 2

  read -r tx ty < <(coords_for "Sample $index")
  if [ -z "${tx:-}" ]; then
    echo "Could not find the 'Sample $index' button on screen." >&2
    kill "$recpid" 2>/dev/null || true
    exit 1
  fi
  "$ADB" shell input tap "$tx" "$ty"

  sleep "$PROCESS_WAIT"
  sleep "$HOLD"

  "$ADB" shell pkill -SIGINT screenrecord 2>/dev/null || true
  wait "$recpid" 2>/dev/null || true
  sleep 2
  "$ADB" pull /sdcard/seg.mp4 "$out" >/dev/null
}

record_one 1 /tmp/seg1.mp4
record_one 2 /tmp/seg2.mp4
record_one 3 /tmp/seg3.mp4

"$ADB" shell svc power stayon false >/dev/null 2>&1 || true

# Trim each segment to: a few seconds of processing, then the collage held on screen.
# 3 x 20 s = 60 s exactly.
say "Editing to 60 s"
for i in 1 2 3; do
  dur=$("$FFMPEG" -i "/tmp/seg$i.mp4" 2>&1 | sed -n 's/.*Duration: \([0-9:.]*\).*/\1/p' \
        | awk -F: '{print ($1*3600)+($2*60)+$3}')
  tail_start=$(python -c "print(max(0, $dur - $HOLD))")

  "$FFMPEG" -y -loglevel error -ss 3 -t 7 -i "/tmp/seg$i.mp4" \
    -vf scale=540:-2 -r 30 -an "/tmp/proc$i.mp4"
  "$FFMPEG" -y -loglevel error -ss "$tail_start" -t 13 -i "/tmp/seg$i.mp4" \
    -vf scale=540:-2 -r 30 -an "/tmp/coll$i.mp4"
done

printf "file '%s'\n" /tmp/proc1.mp4 /tmp/coll1.mp4 \
                     /tmp/proc2.mp4 /tmp/coll2.mp4 \
                     /tmp/proc3.mp4 /tmp/coll3.mp4 > /tmp/concat.txt

"$FFMPEG" -y -loglevel error -f concat -safe 0 -i /tmp/concat.txt -c:v libx264 -crf 23 demo.mp4

say "Done"
"$FFMPEG" -i demo.mp4 2>&1 | grep -E "Duration|Stream #0:0"
echo "Wrote demo.mp4 - confirm it is under 60 s and all three collages are legible."
