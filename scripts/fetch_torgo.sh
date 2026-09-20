#!/usr/bin/env bash
# Robust TORGO fetch: resumable download -> verify size -> extract audio -> delete archive.
#
# A naive `curl | tar -xj` cannot be retried: if the connection drops mid-stream
# curl restarts from byte 0 and the tar stream is corrupt, silently leaving out
# whole speakers. So the archive is staged on disk, checked against the server's
# Content-Length, and only then extracted.
#
# Only acoustic + prompt data is extracted; the articulatory (.pos) and video
# bulk is discarded, which keeps ~18GB of corpus down to a few hundred MB.
#
# Licence: free for academic (non-profit) use; cite Rudzicz et al. (2012).
set -uo pipefail
BASE="http://www.cs.toronto.edu/~complingweb/data/TORGO"
DEST="${1:?usage: fetch_torgo.sh <dest_dir> <set> [staging_dir]}"
SET="${2:?}"
STAGE="${3:-$DEST/_archives}"
mkdir -p "$DEST" "$STAGE"
AR="$STAGE/$SET.tar.bz2"

want=$(curl -sI "$BASE/$SET.tar.bz2" | tr -d '\r' \
        | awk 'tolower($1)=="content-length:"{print $2}' | tail -1)
echo "[$(date +%T)] $SET: expecting $want bytes"

for attempt in 1 2 3 4 5 6 7 8; do
  have=$(stat -c%s "$AR" 2>/dev/null || echo 0)
  if [ -n "$want" ] && [ "$have" = "$want" ]; then break; fi
  echo "[$(date +%T)] $SET: attempt $attempt (have $have)"
  curl -L -C - --connect-timeout 30 --max-time 5400 \
       --speed-limit 2048 --speed-time 120 \
       -o "$AR" "$BASE/$SET.tar.bz2"
  sleep 5
done

have=$(stat -c%s "$AR" 2>/dev/null || echo 0)
if [ -z "$want" ] || [ "$have" != "$want" ]; then
  echo "[$(date +%T)] $SET: FAILED size check ($have != $want)"; exit 1
fi

echo "[$(date +%T)] $SET: extracting"
# --force-local: without it GNU tar reads a Windows path like C:/... as a
# remote "host:path" spec and fails with "Cannot connect to C".
tar --force-local -xjf "$AR" -C "$DEST" --wildcards \
    '*/wav_headMic/*' '*/wav_arrayMic/*' '*/prompts/*'
rc=$?
if [ $rc -ne 0 ]; then echo "[$(date +%T)] $SET: tar rc=$rc"; exit $rc; fi
rm -f "$AR"
echo "[$(date +%T)] $SET: OK"
