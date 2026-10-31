# Run C handoff (Sat Oct 31; laptop shutdown, Colab session `evoc` STOPPED)

**Design:** `PREREG.md` (committed before the run, 957a91d). It covers niche specialists (AIMNet2/Orb/MACE
head-only fine-tunes on the breed labels of the label-free `flagged` niche, σ_F ≥ pool q70 of the base swarm)
with a reactant-relative energy loss, a niche gate, fitness under the Calibrator's routing, budgets
200/400/800 × 4 generations, and static / evolved / noselect arms.

## What ran (Colab L4, about 15 min)
`partial_run1/` holds blackboards, logs and child training results (`cache/*.result.json`, `*.log`). It has
no npz/ckpt files. **This run is invalid as a result** and must not be quoted as the outcome. Several children
died of **CUDA OOM** (too many concurrent GPU processes) and were counted as stillborn. The runs were killed
mid-way, so b400/b800 never reached generation 3.

Fixed since then (in the code committed here):
- OOM retry (up to 4 tries) in `GPUFactory`
- `expandable_segments`
- MACE predict chunk of 1000 atoms

Use `--gpu-slots 3` on a 24 GB GPU.

Interim observations (gates use select labels only, not eval), to be confirmed by a clean rerun:
- **AIMNet2 specialists**: niche hold-out force RMSE falls from about 0.55 to 0.38 eV/Å, but rel-E MAE rises
  from about 0.12 to 0.18 eV, so they are stillborn in every run. This is run A's failure mode, despite the
  rel-E loss at w_E = 1.
- **Orb specialists**: pass the gate (niche force 0.84 to 0.50; rel-E better). Selection then prunes **base
  Orb**, because fitness labels are concentrated in the flagged niche. Served eval F-RMSE got worse in the
  evolved runs. Example at b200 seed 0: evolved g3 0.236 vs static g3 0.202.
- **MACE readout-only specialists**: barely move.
- **Design weakness seen**: children of a *base* parent always get the base recipe, so a stillborn lineage
  never explores w_E. Any change to this is a post-hoc amendment and must be labelled exploratory.

## What's left
1. A clean rerun of the pre-registered design: `scripts/evolve_c.py run` for each budget, then
   `scripts/evolve_c.py summarize`. Summarize writes `SUMMARY.md` inputs (`summary.json`,
   `summary_table.md`) and `fig9_budget.png/svg`.
2. Write `SUMMARY.md` with the table, the verdict per the pre-registered rule, and a mention of the aborted
   OOM run.

## How to run
GPU (Colab L4, as before; setup = `colab/evoc/setup.sh`, data = frames.pkl + data/preds):
```
O="--out results/evolution_c --core-python /usr/bin/python3 --orb-python /content/envs/orb/bin/python --gpu-slots 3 --parallel 3"
python3 scripts/evolve_c.py run --budget 200 --arms static,evolved --seeds 0,1 $O
python3 scripts/evolve_c.py run --budget 400 --arms static,evolved --seeds 0,1 $O
python3 scripts/evolve_c.py run --budget 800 --arms static,evolved,noselect --seeds 0 $O
python3 scripts/evolve_c.py run --budget 800 --tag _s1 --arms static,evolved,noselect --seeds 1 $O
python3 scripts/evolve_c.py summarize --out results/evolution_c
```
Measured on L4: AIMNet2 child about 30–50 s, MACE about 70 s, Orb about 120 s. One run is about 7 min
(3 breeding generations × the slowest child). The whole design takes about 30–40 min with 4 processes in
parallel.

**Azure CPU VM (no GPU)**, AIMNet2-only niche specialists. Pass families via a config override. The CLI has no
flag yet, so add `families=("aimnet2",)` in `budget_config(...)` in `cmd_run`, or add a `--families` arg; this
is a one-line change. Example:
```
ssh azureuser@20.125.63.101
cd /home/azureuser/fittest && git pull
PY=/home/azureuser/envs/core/bin/python
$PY scripts/evolve_c.py run --budget 200 --arms static,evolved --seeds 0 --device cpu --core-python $PY \
    --orb-python $PY --parallel 2 --gpu-slots 2 --threads 2 --out results/evolution_c
```
Expected CPU cost: run B reports about 2 min per AIMNet2 child (200 steps, 1 thread). Prediction on 3157
frames takes about 1–3 min on CPU. AIMNet2-only means 1 child per generation and 3 per run, so about
10–15 min per run, and about 1–1.5 h for 200/400/800 + noselect + seed 1. Orb on CPU is about 10× slower;
skip it. Tests: `python -m pytest tests/test_evolution_c.py` (CPU, no torch, about 1 s).
