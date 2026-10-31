#!/bin/bash
# usage: jc.sh <jobctl args...>
J=/mnt/c/Users/ptkac/.claude/plugins/cache/colab-cli-ops/colab-cli-ops/0.1.0/skills/colab-cli-ops/scripts/jobctl.py
cd /home/pavel && python3 $J "$@" --colab-bin /home/pavel/.local/bin/colab
