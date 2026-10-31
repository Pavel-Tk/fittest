#!/bin/bash
# usage: colab/fetch.sh "<remote paths relative to /content/fittest>" -> same relative local paths
set -e
REL="$1"; ID=$$_$RANDOM; ROOT=/c/Software_projects/Workshops/science_claw/fittest
echo "import subprocess; print(subprocess.run('cd /content/fittest && tar czf /content/f_$ID.tgz $REL && echo PACK_OK', shell=True, capture_output=True, text=True))" > /tmp/pack_$ID.py
$ROOT/colab/cx.sh /tmp/pack_$ID.py 200 | grep -q PACK_OK || { echo PACK_FAIL; exit 1; }
MSYS_NO_PATHCONV=1 powershell -File "C:/Users/ptkac/.claude/plugins/cache/colab-cli-ops/colab-cli-ops/0.1.0/skills/colab-cli-ops/scripts/colab.ps1" download -s tswarm /content/f_$ID.tgz "/mnt/c/Software_projects/Workshops/science_claw/fittest/f_$ID.tgz" 2>&1 | grep -i -E "error|fail" || true
cd $ROOT && tar xzf f_$ID.tgz && rm -f f_$ID.tgz /tmp/pack_$ID.py && echo FETCH_OK $REL
