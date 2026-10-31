# Run C: four budget processes in parallel (GPU slots shared via flock), logs in /content/logs/evc_*.log
cd /content/fittest && tar xzf /content/code.tgz
export PYTHONPATH=/content/fittest
O="--out results/evolution_c --core-python /usr/bin/python3 --orb-python /content/envs/orb/bin/python --gpu-slots 6 --parallel 3 --threads 2"
python3 scripts/evolve_c.py run --budget 200 --arms static,evolved --seeds 0,1 $O > /content/logs/evc_b200.log 2>&1 &
python3 scripts/evolve_c.py run --budget 400 --arms static,evolved --seeds 0,1 $O > /content/logs/evc_b400.log 2>&1 &
python3 scripts/evolve_c.py run --budget 800 --arms static,evolved,noselect --seeds 0 $O > /content/logs/evc_b800.log 2>&1 &
python3 scripts/evolve_c.py run --budget 800 --tag _s1 --arms static,evolved,noselect --seeds 1 $O > /content/logs/evc_b800_s1.log 2>&1 &
wait
grep -h RUN_DONE /content/logs/evc_*.log
echo RUNC_ALL_DONE
