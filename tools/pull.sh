#!/usr/bin/env bash
# Copy files from /workspace/fb/ on the pod into ./out/ (base64 over the SSH proxy).
# usage: tools/pull.sh exp/a.png exp/b.npz
mkdir -p out
for f in "$@"; do
  "$(dirname "$0")/pod.sh" "echo B64START; base64 -w 0 /workspace/fb/$f; echo; echo B64END" \
    | sed -n '/^B64START$/,/^B64END$/p' | sed '1d;$d' | tr -d '\n' | base64 -d > "out/$(basename "$f")"
  echo "out/$(basename "$f") $(wc -c < "out/$(basename "$f")") bytes"
done
