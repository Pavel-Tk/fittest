set -x
cd /content
( curl -sL -A 'Mozilla/5.0 (X11; Linux x86_64)' -o data/Transition1x.h5 https://ndownloader.figshare.com/files/36035789; ls -la data; md5sum data/Transition1x.h5 ) > logs/t1x_dl.log 2>&1 &
rm -rf /content/envs/orb
uv venv /content/envs/orb --python 3.12
export UV_SYSTEM_PYTHON=0
uv pip install --python /content/envs/orb/bin/python "dm-tree>=0.1.9" orb-models ase h5py numpy torch --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match
/content/envs/orb/bin/python -c "import orb_models, torch; print('ORB', torch.__version__, torch.cuda.is_available())"
/content/envs/orb/bin/python -m pip --version || true
wait
cat logs/t1x_dl.log
echo INSTALL2_DONE
