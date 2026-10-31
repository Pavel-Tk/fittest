set -x
cd /content
mkdir -p data envs logs
( wget -q -O data/Transition1x.h5 https://figshare.com/ndownloader/files/36035789 && ls -la data && echo T1X_OK ) > logs/t1x_dl.log 2>&1 &
[ -d nvalchemi-toolkit ] || git clone -q -b 0.3.0-rc https://github.com/NVIDIA/nvalchemi-toolkit.git
cd nvalchemi-toolkit
uv venv /content/envs/core --python 3.12
. /content/envs/core/bin/activate
uv pip install -e ".[cu13,mace,aimnet,ase]" --extra-index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match
uv pip install h5py pytest scikit-learn matplotlib fastapi uvicorn httpx
python -c "import nvalchemi, torch; print('CORE', torch.__version__, torch.cuda.is_available())"
deactivate
uv venv /content/envs/orb --python 3.12
. /content/envs/orb/bin/activate
uv pip install orb-models ase h5py numpy --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match
python -c "import orb_models, torch; print('ORB', torch.__version__, torch.cuda.is_available())"
wait
cat /content/logs/t1x_dl.log
echo INSTALL_DONE
