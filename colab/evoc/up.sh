#!/bin/bash
# usage: colab/evoc/up.sh <local file> <remote path>
W=$(cygpath -aw "$1")
MSYS_NO_PATHCONV=1 powershell -File "C:/Users/ptkac/.claude/plugins/cache/colab-cli-ops/colab-cli-ops/0.1.0/skills/colab-cli-ops/scripts/colab.ps1" upload -s evoc "$W" "$2" 2>&1 | tail -3
