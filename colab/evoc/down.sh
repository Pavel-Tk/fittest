#!/bin/bash
# usage: colab/evoc/down.sh <remote path> <local file>
W=$(cygpath -aw "$2")
MSYS_NO_PATHCONV=1 powershell -File "C:/Users/ptkac/.claude/plugins/cache/colab-cli-ops/colab-cli-ops/0.1.0/skills/colab-cli-ops/scripts/colab.ps1" download -s evoc "$1" "$W" 2>&1 | tail -3
