# Run C setup on a fresh Colab L4: scripts/setup_gpu.sh minus the Transition1x download
# (run C only needs data/frames.pkl + data/preds/*.npz, uploaded with the code).
set -x
cd /content
mkdir -p data envs logs
command -v uv >/dev/null || pip install -q uv
IDX="--extra-index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match"
[ -d nvalchemi-toolkit ] || git clone -q -b 0.3.0-rc https://github.com/NVIDIA/nvalchemi-toolkit.git
uv pip install --python /usr/bin/python3 --system --break-system-packages -e "./nvalchemi-toolkit[cu13,mace,aimnet,ase]" $IDX
uv pip install --python /usr/bin/python3 --system --break-system-packages h5py scikit-learn matplotlib pytest
/usr/bin/python3 -c "import nvalchemi, torch, mace, aimnet; print('CORE OK', torch.__version__, torch.cuda.is_available())"
rm -rf envs/orb
uv venv envs/orb --python 3.12
UV_SYSTEM_PYTHON=0 uv pip install --python envs/orb/bin/python "dm-tree>=0.1.9" orb-models ase h5py numpy torch \
  --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match
envs/orb/bin/python -c "import orb_models, torch; print('ORB OK', torch.__version__, torch.cuda.is_available())"
echo SETUP_DONE
