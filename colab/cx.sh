#!/bin/bash
# usage: colab/cx.sh <local .py file> [timeout]
F=$(cygpath -aw "$1"); T=${2:-300}
powershell -File "C:/Software_projects/Workshops/science_claw/trustswarm/colab/cx.ps1" -File "$F" -Timeout $T 2>&1 | grep -v -E "^\[colab\] (A new|Run 'colab|To silence)|NativeCommandError|CategoryInfo|^At C:|^\+ |^powershell : \[colab.ps1\] exit=0|^ *$"
