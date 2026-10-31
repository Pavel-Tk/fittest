#!/bin/bash
# Tar the package + scripts and push to /content/trustswarm on the VM. Safe to run concurrently.
set -e
cd /c/Software_projects/Workshops/science_claw/trustswarm
ID=$$_$RANDOM
tar --exclude=__pycache__ -czf /tmp/ts_$ID.tgz trustswarm scripts tests pyproject.toml 2>/dev/null
W=$(cygpath -aw /tmp/ts_$ID.tgz)
MSYS_NO_PATHCONV=1 powershell -File "C:/Users/ptkac/.claude/plugins/cache/colab-cli-ops/colab-cli-ops/0.1.0/skills/colab-cli-ops/scripts/colab.ps1" upload -s tswarm "$W" /content/ts_$ID.tgz 2>&1 | grep -i -E "error|fail" | head -3 || true
echo "import subprocess; r=subprocess.run('mkdir -p /content/trustswarm && cd /content/trustswarm && tar xzf /content/ts_$ID.tgz && rm /content/ts_$ID.tgz && echo SYNC_OK', shell=True, capture_output=True, text=True); print(r.stdout, r.stderr)" > /tmp/untar_$ID.py
/c/Software_projects/Workshops/science_claw/trustswarm/colab/cx.sh /tmp/untar_$ID.py 120 | grep -E "SYNC_OK|rror" 
rm -f /tmp/ts_$ID.tgz /tmp/untar_$ID.py
