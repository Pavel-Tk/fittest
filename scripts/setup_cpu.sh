#!/usr/bin/env bash
# Reproduce the Fittest CPU environment (what runs on the Azure VM: Ubuntu 24.04, 8 vCPU, no GPU).
#
#   bash scripts/setup_cpu.sh [WORKDIR]        # WORKDIR default: $HOME (holds toolkit, envs, logs)
#
# Idempotent: re-running reuses existing clones/venvs and only re-resolves packages.
# Creates
#   core env : $WORKDIR/envs/core (Python 3.12 venv): CPU torch (download.pytorch.org/whl/cpu),
#              nvalchemi-toolkit 0.3.0-rc editable with extras mace,aimnet,ase (NOT cu13)
#              -> torch 2.15+cpu, mace-torch 0.3.15, aimnet 0.2.5; plus h5py scikit-learn matplotlib
#              fastapi uvicorn httpx pytest mcp, and this repo (editable).
#   orb env  : $WORKDIR/envs/orb (Python 3.12 venv): CPU torch + orb-models (0.5.0 resolves) + dm-tree>=0.1.9.
# Data: no Transition1x download needed. Copy the repo's data/ dir (frames.pkl + preds/ + ckpt/, 23 MB,
# gitignored) from a machine that has it, e.g.   rsync -a data/ azureuser@<vm>:$WORKDIR/fittest/data/
set -euo pipefail

WORKDIR=$(cd "${1:-$HOME}" && pwd)
REPO=${REPO:-$WORKDIR/fittest}
TOOLKIT_REF=${TOOLKIT_REF:-0.3.0-rc}
CPU_IDX=https://download.pytorch.org/whl/cpu
IDX=(--extra-index-url "$CPU_IDX" --index-strategy unsafe-best-match)

export PATH="$HOME/.local/bin:$PATH"
# g++ + Python headers: the toolkit's NeighborListHook is torch.compile'd; on CPU inductor builds
# C++ against Python.h (the venvs use the system python3.12)
if ! command -v g++ >/dev/null || [ ! -f /usr/include/python3.12/Python.h ]; then
  sudo apt-get update -qq
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq g++ python3.12-dev
fi
command -v uv >/dev/null || { echo "installing uv"; curl -LsSf https://astral.sh/uv/install.sh | sh; }
mkdir -p "$WORKDIR"/{envs,logs}
cd "$WORKDIR"

[ -d "$REPO/.git" ] || git clone -q https://github.com/Pavel-Tk/fittest.git "$REPO"
[ -d nvalchemi-toolkit ] || git clone -q -b "$TOOLKIT_REF" https://github.com/NVIDIA/nvalchemi-toolkit.git

# --- orb env in the background (independent resolver run)
(
  [ -x envs/orb/bin/python ] || uv venv envs/orb --python 3.12
  uv pip install --python envs/orb/bin/python torch --index-url "$CPU_IDX"
  uv pip install --python envs/orb/bin/python "dm-tree>=0.1.9" orb-models ase h5py numpy "${IDX[@]}"
  envs/orb/bin/python -c "import orb_models, torch; print('ORB OK', torch.__version__)"
) > logs/orb_install.log 2>&1 &
ORB_PID=$!

# --- core env: install CPU torch first so the toolkit resolve keeps it (no CUDA wheels pulled in)
[ -x envs/core/bin/python ] || uv venv envs/core --python 3.12
PY=envs/core/bin/python
uv pip install --python "$PY" torch --index-url "$CPU_IDX"
uv pip install --python "$PY" -e "./nvalchemi-toolkit[mace,aimnet,ase]" "${IDX[@]}"
uv pip install --python "$PY" h5py scikit-learn matplotlib fastapi uvicorn httpx pytest mcp "${IDX[@]}"
uv pip install --python "$PY" -e "$REPO" "${IDX[@]}"
"$PY" -c "import nvalchemi, torch, mace, aimnet, fittest; print('CORE OK', torch.__version__, 'cuda', torch.cuda.is_available())"

wait "$ORB_PID" || { echo "orb env failed, see $WORKDIR/logs/orb_install.log"; exit 1; }
tail -1 logs/orb_install.log

[ -f "$REPO/data/frames.pkl" ] || echo "WARNING: $REPO/data/frames.pkl missing; copy the data/ dir (see header)"

cat <<EOF

Done. Add to your shell:
  export FITTEST_CORE_PYTHON=$WORKDIR/envs/core/bin/python
  export FITTEST_ORB_PYTHON=$WORKDIR/envs/orb/bin/python
  export FITTEST_NEB_DIR=$WORKDIR/nvalchemi-toolkit/examples/advanced/batched_neb_endpoints
  export FITTEST_LOG_DIR=$WORKDIR/logs
CPU notes: pass --device cpu to model_server / finetune.py / service (their default is cuda);
cuEquivariance kernels are skipped automatically on CPU (e3nn path). Threads: OMP_NUM_THREADS=N.
EOF
