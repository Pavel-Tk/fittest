#!/usr/bin/env bash
# Reproduce the Fittest GPU environment (what we ran on a Colab L4, CUDA 13 driver).
#
#   bash scripts/setup_gpu.sh [WORKDIR]        # WORKDIR default: $PWD/.. (holds toolkit, envs, data)
#
# Creates
#   core env  : the interpreter in $CORE_PYTHON (default: python3 on PATH; on Colab /usr/bin/python3,
#               Python 3.13) gets nvalchemi-toolkit 0.3.0-rc (editable, extras cu13,mace,aimnet,ase)
#               -> torch 2.11+cu130, mace-torch 0.3.15, aimnet 0.2.5, cuequivariance; plus h5py,
#               scikit-learn, matplotlib, fastapi, uvicorn, httpx, pytest.
#               Set CORE_VENV=1 to install into a fresh uv venv ($WORKDIR/envs/core) instead.
#   orb env   : $WORKDIR/envs/orb (Python 3.12 venv): orb-models + torch from the cu130 index.
#               Separate because orb-models pins different deps than the toolkit.
#   data      : $WORKDIR/data/Transition1x.h5 (6.6 GB, md5 faab0413cf6b042b6d50e6272dea9a8c)
# and prints the FITTEST_* exports to use afterwards (see fittest/paths.py).
set -euo pipefail

WORKDIR=$(cd "${1:-$PWD/..}" && pwd)
CORE_PYTHON=${CORE_PYTHON:-$(command -v python3)}
TOOLKIT_REF=${TOOLKIT_REF:-0.3.0-rc}
T1X_URL=https://ndownloader.figshare.com/files/36035789
T1X_MD5=faab0413cf6b042b6d50e6272dea9a8c
IDX=(--extra-index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://pypi.nvidia.com
     --index-strategy unsafe-best-match)

command -v uv >/dev/null || { echo "installing uv"; curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
mkdir -p "$WORKDIR"/{data,envs,logs}
cd "$WORKDIR"

# --- Transition1x download in the background (figshare rejects requests without a browser UA)
if [ ! -f data/Transition1x.h5 ] || [ "$(md5sum data/Transition1x.h5 | cut -d' ' -f1)" != "$T1X_MD5" ]; then
  ( curl -sL -A 'Mozilla/5.0 (X11; Linux x86_64)' -o data/Transition1x.h5 "$T1X_URL" ) > logs/t1x_dl.log 2>&1 &
  DL_PID=$!
fi

# --- core env: ALCHEMI toolkit 0.3.0-rc (editable, so examples/ incl. batched_neb_endpoints are on disk)
[ -d nvalchemi-toolkit ] || git clone -q -b "$TOOLKIT_REF" https://github.com/NVIDIA/nvalchemi-toolkit.git
if [ "${CORE_VENV:-0}" = "1" ]; then
  uv venv envs/core --python 3.12
  CORE_PYTHON="$WORKDIR/envs/core/bin/python"
  PIPTGT=(--python "$CORE_PYTHON")
else
  PIPTGT=(--python "$CORE_PYTHON" --system --break-system-packages)
fi
uv pip install "${PIPTGT[@]}" -e "./nvalchemi-toolkit[cu13,mace,aimnet,ase]" "${IDX[@]}"
uv pip install "${PIPTGT[@]}" h5py scikit-learn matplotlib fastapi uvicorn httpx pytest
"$CORE_PYTHON" -c "import nvalchemi, torch, mace, aimnet; print('CORE OK', nvalchemi.__version__, torch.__version__, torch.cuda.is_available())"

# --- orb env (Python 3.12): dm-tree>=0.1.9 has 3.12 wheels; older pins fail to build
rm -rf envs/orb
uv venv envs/orb --python 3.12
UV_SYSTEM_PYTHON=0 uv pip install --python envs/orb/bin/python "dm-tree>=0.1.9" orb-models ase h5py numpy torch \
  --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match
envs/orb/bin/python -c "import orb_models, torch; print('ORB OK', torch.__version__, torch.cuda.is_available())"

# --- data check
if [ -n "${DL_PID:-}" ]; then wait "$DL_PID"; fi
echo "$T1X_MD5  data/Transition1x.h5" | md5sum -c -

cat <<EOF

Done. Add to your shell (or colab/env.sh):
  export FITTEST_T1X=$WORKDIR/data/Transition1x.h5
  export FITTEST_CORE_PYTHON=$CORE_PYTHON
  export FITTEST_ORB_PYTHON=$WORKDIR/envs/orb/bin/python
  export FITTEST_NEB_DIR=$WORKDIR/nvalchemi-toolkit/examples/advanced/batched_neb_endpoints
  export FITTEST_LOG_DIR=$WORKDIR/logs
EOF
