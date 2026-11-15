#!/usr/bin/env bash
# Copy local files to /workspace/fb/ on the pod over the SSH proxy (no scp there): base64 through a heredoc.
# usage: tools/push.sh exp/a.py exp/b.py
cmd="mkdir -p /workspace/fb/exp"
for f in "$@"; do
  cmd+=$'\n'"base64 -d > /workspace/fb/$f <<'B64END'"$'\n'"$(base64 -w 76 "$f")"$'\n'"B64END"
done
cmd+=$'\n'"cd /workspace/fb && md5sum $*"
"$(dirname "$0")/pod.sh" "$cmd"
