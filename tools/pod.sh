#!/usr/bin/env bash
# Run a command on the RunPod pod through the SSH proxy (the proxy has no exec channel, so the command goes in on stdin).
# usage: tools/pod.sh 'command ...'
POD="${POD:-e0h3ra0rnvlkno-64411330@ssh.runpod.io}"
printf '%s\nexit\n' "stty -echo; $1" | ssh -tt -q -i ~/.ssh/runpod_ed25519 -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 "$POD" 2>&1 \
  | sed -e 's/\x1b\[[0-9;?]*[a-zA-Z]//g' -e 's/\x1b\][^\a]*\a//g' | tr -d '\r' | sed -n '/^root@.*# stty -echo/,$p' | sed '1d'
