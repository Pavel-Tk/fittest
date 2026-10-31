#!/bin/bash
# usage: colab/evoc/cx.sh <local .py file> [timeout]   (session evoc)
F=$(cygpath -aw "$1"); T=${2:-300}
MSYS_NO_PATHCONV=1 powershell -File "C:/Users/ptkac/.claude/plugins/cache/colab-cli-ops/colab-cli-ops/0.1.0/skills/colab-cli-ops/scripts/colab.ps1" exec -s evoc --timeout $T -f "$F" 2>&1 | grep -v -E "^\[colab\] (A new|Run 'colab|To silence)|NativeCommandError|CategoryInfo|^At C:|^\+ |^powershell : \[colab.ps1\] exit=0|^ *$"
