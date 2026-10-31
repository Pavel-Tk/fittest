#!/usr/bin/env bash
# Deploy / update the hosted Fittest MCP server. Idempotent; run from the laptop:
#     bash deploy/deploy.sh                 # pull master on the VM, (re)install, restart MCP server
#     RESTART_MODELS=1 bash deploy/deploy.sh  # also restart the warm model agents
# Needs: the code pushed to GitHub (the VM checks out origin/master into its own dir).
set -euo pipefail
VM=${VM:-azureuser@20.125.63.101}
KEY=${KEY:-$HOME/.ssh/id_rsa}
BRANCH=${BRANCH:-master}
HERE="$(cd "$(dirname "$0")" && pwd)"
SSH=(ssh -i "$KEY" -o StrictHostKeyChecking=accept-new "$VM")

"${SSH[@]}" "mkdir -p ~/fittest-mcp-deploy"
scp -q -i "$KEY" "$HERE/Caddyfile" "$HERE/fittest-mcp.service" "$HERE/fittest-model@.service" \
    "$HERE/fittest-mcp.target" "$HERE/remote_setup.sh" "$VM:fittest-mcp-deploy/"
"${SSH[@]}" "BRANCH=$BRANCH RESTART_MODELS=${RESTART_MODELS:-0} bash ~/fittest-mcp-deploy/remote_setup.sh"
