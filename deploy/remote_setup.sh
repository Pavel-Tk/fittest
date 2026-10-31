#!/usr/bin/env bash
# Runs ON the VM (called by deploy.sh). Idempotent.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
BRANCH=${BRANCH:-master}
SRC=$HOME/fittest-mcp-src          # server's own checkout; never touches ~/fittest (experiments)
STATE=/var/lib/fittest-mcp         # usage.sqlite + salt (never in git), owned by the fittest-mcp user
OLD_DATA=$HOME/fittest-mcp-data    # pre-hardening location, migrated once
DEP=$HOME/fittest-mcp-deploy
CORE_PY=${CORE_PY:-$HOME/envs/core/bin/python}
ORB_PY=${ORB_PY:-$HOME/envs/orb/bin/python}

echo "== source ($BRANCH)"
if [ ! -d "$SRC/.git" ]; then git clone -q -b "$BRANCH" https://github.com/Pavel-Tk/fittest.git "$SRC"; fi
git -C "$SRC" fetch -q origin "$BRANCH" && git -C "$SRC" reset -q --hard "origin/$BRANCH"
git -C "$SRC" log --oneline -1

echo "== MCP env (no torch: the server only talks to the model agents)"
[ -x "$HOME/envs/mcp/bin/python" ] || uv venv -q "$HOME/envs/mcp" --python 3.12
uv pip install -q --python "$HOME/envs/mcp/bin/python" -e "$SRC"
"$HOME/envs/mcp/bin/python" -c "import importlib.metadata as m; print('mcp', m.version('mcp'))"

echo "== service user + state dir + secrets"
id fittest-mcp >/dev/null 2>&1 || sudo useradd --system --no-create-home --home-dir "$STATE" --shell /usr/sbin/nologin fittest-mcp
sudo mkdir -p "$STATE"
if [ -d "$OLD_DATA" ] && ! sudo test -s "$STATE/salt"; then
  echo "migrating $OLD_DATA -> $STATE"
  sudo systemctl stop fittest-mcp 2>/dev/null || true
  sudo cp -a "$OLD_DATA"/. "$STATE"/
fi
sudo test -s "$STATE/salt" || head -c 32 /dev/urandom | od -An -tx1 | tr -d ' \n' | sudo tee "$STATE/salt" >/dev/null
sudo chown -R fittest-mcp:fittest-mcp "$STATE"; sudo chmod 700 "$STATE"; sudo chmod 600 "$STATE/salt"
if [ -d "$OLD_DATA" ] && sudo test -s "$STATE/salt"; then rm -rf "$OLD_DATA"; fi
sudo mkdir -p /etc/fittest-mcp
if ! sudo test -s /etc/fittest-mcp/common.env; then
  AK=$(head -c 24 /dev/urandom | od -An -tx1 | tr -d ' \n')
  printf 'FITTEST_AUTHKEY=%s\nFITTEST_SNAPSHOT=%s\n' "$AK" "$SRC/results/trust_snapshot.json" \
    | sudo tee /etc/fittest-mcp/common.env >/dev/null
fi
sudo chmod 640 /etc/fittest-mcp/common.env; sudo chown root:azureuser /etc/fittest-mcp/common.env
# evolution family trees are read (read-only) from the experiments' results dir
sudo sed -i '/^FITTEST_FAMILY_TREE=/d;/^FITTEST_RESULTS_DIR=/d' /etc/fittest-mcp/common.env
echo "FITTEST_RESULTS_DIR=$HOME/fittest/results" | sudo tee -a /etc/fittest-mcp/common.env >/dev/null
port=7300
for m in mace_mpa0 aimnet2 orb; do
  py=$CORE_PY; [ "$m" = orb ] && py=$ORB_PY
  printf 'MODEL_PYTHON=%s\nMODEL_PORT=%s\n' "$py" "$port" | sudo tee /etc/fittest-mcp/model-$m.env >/dev/null
  port=$((port+1))
done

echo "== systemd"
sudo cp "$DEP/fittest-mcp.service" "$DEP/fittest-model@.service" "$DEP/fittest-mcp.target" /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable -q fittest-mcp.target fittest-mcp.service \
  fittest-model@mace_mpa0.service fittest-model@aimnet2.service fittest-model@orb.service
if [ "${RESTART_MODELS:-0}" = 1 ]; then
  sudo systemctl restart fittest-model@mace_mpa0 fittest-model@aimnet2 fittest-model@orb
else
  sudo systemctl start fittest-model@mace_mpa0 fittest-model@aimnet2 fittest-model@orb
fi
sudo systemctl restart fittest-mcp

echo "== caddy"
if ! command -v caddy >/dev/null; then
  sudo apt-get install -y -q debian-keyring debian-archive-keyring apt-transport-https curl gnupg >/dev/null
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor --yes -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list >/dev/null
  sudo apt-get update -q >/dev/null && sudo apt-get install -y -q caddy >/dev/null
fi
caddy version
sudo mkdir -p /var/log/caddy && sudo chown caddy:caddy /var/log/caddy || true
sudo -u caddy caddy validate --adapter caddyfile --config "$DEP/Caddyfile" >/dev/null 2>&1 || { sudo -u caddy caddy validate --adapter caddyfile --config "$DEP/Caddyfile"; exit 1; }
sudo cp "$DEP/Caddyfile" /etc/caddy/Caddyfile
sudo systemctl enable -q caddy && sudo systemctl reload-or-restart caddy

echo "== health"
for i in $(seq 1 60); do
  if out=$(curl -fsS http://127.0.0.1:8765/health 2>/dev/null); then echo "$out"; break; fi; sleep 2
done
systemctl --no-pager --plain list-units 'fittest-*' | head -8
