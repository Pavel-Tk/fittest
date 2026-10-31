#!/bin/bash
# usage (from Git Bash): colab/evoc/jc.sh <jobctl args...>  e.g. status setup --tail 40
J=/mnt/c/Users/ptkac/.claude/plugins/cache/colab-cli-ops/colab-cli-ops/0.1.0/skills/colab-cli-ops/scripts/jobctl.py
ARGS=$(printf '%q ' "$@")
T=/tmp/jc_$$_$RANDOM.sh
echo "cd /home/pavel && python3 $J send -s evoc $ARGS --colab-bin /home/pavel/.local/bin/colab" > $T
WP=$(cygpath -am $T | sed -E 's#^([A-Za-z]):#/mnt/\L\1#')
MSYS_NO_PATHCONV=1 wsl -d Ubuntu --cd /home/pavel -- bash "$WP" 2>&1
rm -f $T
