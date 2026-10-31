cd /content/fittest
export PYTHONPATH=/content/fittest
mkdir -p /content/smoke
python3 - <<'PY'
import json, sys
sys.path.insert(0, "/content/fittest")
from fittest.cache import load_frames
from fittest.data import assign_leakfree_splits
fr, _ = assign_leakfree_splits(load_frames("data/frames.pkl"), 0)
pool = [f.uid for f in fr if f.split == "pool"][:40]
allu = [f.uid for f in fr if f.split in ("pool", "eval")]
for fam in ("aimnet2", "orb", "mace"):
    s = {"id": f"smoke_{fam}", "family": fam, "parent_ckpt": None, "seed": 0,
         "recipe": {"lr": 1e-4, "energy_weight": 1.0, "force_weight": 1.0, "steps": 30, "reactions_per_batch": 4},
         "train_uids": pool, "predict_uids": allu, "out": f"/content/smoke/{fam}.npz",
         "ckpt": f"/content/smoke/{fam}.pt", "result": f"/content/smoke/{fam}.result.json"}
    json.dump(s, open(f"/content/smoke/{fam}.json", "w"))
PY
for fam in aimnet2 mace; do echo "=== $fam"; python3 scripts/evolve_c_offspring.py --spec /content/smoke/$fam.json --device cuda 2>&1 | grep -v Warning | tail -12; done
echo "=== orb"; /content/envs/orb/bin/python scripts/evolve_c_offspring.py --spec /content/smoke/orb.json --device cuda 2>&1 | tail -12
nvidia-smi --query-gpu=memory.used --format=csv
echo SMOKE_DONE
