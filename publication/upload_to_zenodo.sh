#!/bin/bash
# Upload the bundle archives to a reserved Zenodo draft. Files only -- metadata is left
# alone, so a title set on the website is not clobbered.
#
#   DEPOSITION=<id> BUNDLE_DIST=~/bundle_dist ./upload_to_zenodo.sh
#
# Reads the API token from ~/.zenodo_token and sends it as an Authorization header, so it
# never lands in a URL, a process listing, or shell history. Does NOT publish: that is
# irreversible and belongs to a human.
#
# Retries each file: Zenodo returns 504s under load, and a multi-GB PUT is worth retrying
# rather than restarting the whole run. Zenodo echoes a checksum for every file it
# accepts, which is compared against the local one.
set -uo pipefail

DEPOSITION=${DEPOSITION:?set DEPOSITION to the draft record id}
HOST=${ZENODO_HOST:-zenodo.org}
DIST=${BUNDLE_DIST:-$HOME/bundle_dist}
AUTH="Authorization: Bearer $(tr -d '\r\n' < "${ZENODO_TOKEN_FILE:-$HOME/.zenodo_token}")"
ATTEMPTS=6
# Retried like everything else: Zenodo returns 504s under load, and losing this one
# call means the whole run never starts.
bucket=""
for attempt in $(seq $ATTEMPTS); do
  code=$(curl -sS -o /tmp/zdep.json -w '%{http_code}' -H "$AUTH" --max-time 120 \
         "https://$HOST/api/deposit/depositions/$DEPOSITION")
  if [[ $code == 200 ]]; then
    bucket=$(python3 -c 'import json; print(json.load(open("/tmp/zdep.json"))["links"]["bucket"])')
    [[ -n $bucket ]] && break
  fi
  echo "resolving bucket, attempt $attempt: http=$code"
  sleep $((attempt * 15))
done
[[ -n $bucket ]] || { echo "could not resolve the bucket after $ATTEMPTS attempts" >&2; exit 1; }
echo "bucket resolved"

# What the draft already holds, so a resumed run does not re-send gigabytes. Zenodo
# reports each file's md5, so "already there" means "there and intact".
python3 - <<'PYEOF' > /tmp/zhave.txt
import json
d = json.load(open("/tmp/zdep.json"))
for f in d.get("files", []):
    print(f["filename"], f.get("checksum", "").replace("md5:", ""))
PYEOF
echo "draft already holds $(wc -l < /tmp/zhave.txt) file(s)"

failed=()
for n in 2 3 4 5 6; do
  for name in figure${n}_data.tar.gz figure${n}_data.tar.gz.sha256; do
    path="$DIST/$name"
    local_md5=$(md5sum "$path" | cut -d' ' -f1)
    if grep -qx "$name $local_md5" /tmp/zhave.txt; then
      echo "== $name  already uploaded, md5 matches, skipping"
      continue
    fi
    echo "== $name ($(du -h "$path" | cut -f1))"
    ok=0
    for attempt in $(seq $ATTEMPTS); do
      code=$(curl -sS -o /tmp/zup.json -w '%{http_code}' -H "$AUTH" \
             -X PUT "$bucket/$name" --upload-file "$path")
      if [[ $code == 20* ]]; then
        remote_md5=$(python3 -c '
import json
print(json.load(open("/tmp/zup.json")).get("checksum","").replace("md5:",""))')
        if [[ $remote_md5 == "$local_md5" ]]; then
          echo "   ok  md5 $remote_md5 matches"
          ok=1
          break
        fi
        echo "   CHECKSUM MISMATCH local=$local_md5 remote=$remote_md5"
      else
        echo "   attempt $attempt: http=$code $(head -c 120 /tmp/zup.json | tr -d '\n')"
      fi
      sleep $((attempt * 10))
    done
    [[ $ok == 1 ]] || failed+=("$name")
  done
done

echo
if [[ ${#failed[@]} -eq 0 ]]; then
  echo "ALL FILES UPLOADED"
else
  echo "FAILED: ${failed[*]}"
fi
curl -sS -H "$AUTH" --max-time 60 "https://$HOST/api/deposit/depositions/$DEPOSITION" |
  python3 -c '
import json, sys
d = json.load(sys.stdin)
print(f"draft now holds {len(d.get(\"files\", []))} files:")
for f in d.get("files", []):
    print(f"   {f[\"filename\"]:34} {f[\"filesize\"]:>13,} bytes")'
