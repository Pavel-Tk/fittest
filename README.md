# TrustSwarm

**A collective of agents that decides which machine-learned interatomic potential (MLIP) to trust,
where, and repairs it where none can be.** Built overnight at ScienceClaw 2026 (MIT Media Lab) on
NVIDIA ALCHEMI Toolkit 0.3.0-rc.

> Labs spend months deciding whether a foundation MLIP is good enough for their chemistry, and
> the models fail silently off-distribution — at transition states, exactly where reactions are
> decided. TrustSwarm turns *disagreement between independent foundation models* into a
> label-free signal, spends a fixed DFT budget where it matters, audits its own "trusted"
> verdicts, and fine-tunes the weakest model on what the oracle returns.

<!-- RESULTS:START -->
## Results

**TrustSwarm on Transition1x (leak-free split, 40×5 = 200 oracle calls per arm, 95% reaction-bootstrap CIs).**

* **Better served answers:** TrustSwarm's served force RMSE on held-out eval is **0.206 eV/Å** vs 0.286–0.317 for every baseline; wrong on 40% of eval frames vs 54%–68%. It learned to serve the subset mean of the models that is most often right.
* **Calibrated trust:** ECE 0.080, Brier 0.185; significantly better than A_single, A_random, B_seed, B_mace_seeds, C_collective, tied with C_audit — the gain comes from the Skeptic's random audits of trusted answers.
* **Negative result — error ranking:** TrustSwarm's AUROC (eval 0.730) is significantly worse than A_single, B_seed, B_mace_seeds, C_collective, C_audit. Best ranker on eval: B_mace_seeds (0.853).
* **Heterogeneous disagreement detects AIMNet2's failures best:** AUROC vs 'AIMNet2 wrong' 0.917 (C_collective).
* **Recall@K:** TrustSwarm 0.186; significantly better than A_random.
* **Ranking ≠ calibration:** on eval the median seed spread is MACE LoRA seeds 0.009 vs median error 0.266 eV/Å (×29.7 under-dispersed); AIMNet2 ensemble members 0.109 vs median error 0.257 eV/Å (×2.4 under-dispersed); yet B_mace_seeds ranks errors best.
* **Self-repair:** the Curator chose and LoRA-fine-tuned `mace_mpa0` on its oracle labels; not shipped — forces 0.878 → 0.500 eV/Å but relative-energy MAE 0.402 → 0.414 eV on held-out labels, so our gate refuses it.
* **Leakage controlled:** pool/eval and skeptic_ref/planted share 0 endpoint geometries (Transition1x reuses reactants: a per-reaction split had 732 near-duplicate pool/eval endpoints).
* Full tables, per-version history and failures: `results/RESULTS.md`.

![Ablation](results/fig/fig5_ablation.png)

| Science finding | Figure |
|---|---|
| Every model's force error rises toward the transition state (AIMNet2: 2.1× reactant) | `results/fig/fig1_ts_failure.png` |
| Independent-model spread tracks error (Spearman 0.73); a seed ensemble's spread is 0.4× its error | `results/fig/fig2_spread_vs_error.png` |
| In ALCHEMI batched NEB, spread follows force error along the path (median ρ 0.96, 8 reactions) | `results/fig/fig3_neb.png` |
| Averaging AIMNet2 + Orb halves wrong answers (57% → 26%); adding MACE undoes it | `results/fig/fig4_best_subset.png` |
| Curator's LoRA repair: forces 0.88 → 0.50 eV/Å, relative energies slightly worse, gate refuses | `results/fig/fig6_repair.png` |

Pitch deck: see the link in the hackathon submission. Announcement text: `docs/discord_post.md`.
<!-- RESULTS:END -->

## How it works

```
 Transition1x frames ──► ALCHEMI in-flight batch ──► DisagreementHook ──σ < τ──► trusted (Skeptic audits)
                          MACE · AIMNet2 · Orb         per-system σ_F   │
                          (one process each)                            └σ ≥ τ──► Blackboard ──bids──► Oracle (k DFT labels / round)
                                                                                     ▲                     │ true error
                         Curator ◄── worst model + labels ── Calibrator ledger ◄─────┴─────────────────────┘
                         LoRA fine-tune + acceptance report  reputation · τ · per-region trust
```

| Agent | Holds privately | Decides |
|---|---|---|
| ModelAgent ×3 (MACE-MPA-0, AIMNet2, Orb-v3) | its own weights (own OS process; Orb in its own env), its prediction, its error history | how much to bid on each flagged system |
| Oracle | Transition1x ωB97x/6-31G(d) labels (lookup only), the budget | which top-k bids get paid |
| Skeptic | held-out reference reactions, planted known failures | vetoes a "trusted" verdict; plants cases |
| Calibrator | the ledger | reputation, threshold τ, trust + recommended model per region |
| Curator | fine-tune compute | which model to repair, whether the repair ships |

Agents only communicate through an append-only blackboard (`results/blackboard_armD.jsonl`);
the round driver advances the clock and makes no decisions.

## Reuse it (collaboration artefacts)

**1. `DisagreementHook`** — drop into any ALCHEMI Toolkit run with any set of models:

```python
from trustswarm import paths
from trustswarm.hook import DisagreementHook
from trustswarm.model_server import spawn

committee = {m: spawn(m, 6100 + k, python=paths.model_python(m)).connect()   # one process per model agent
             for k, m in enumerate(["mace_mpa0", "aimnet2", "orb"])}
hook = DisagreementHook(committee, tau=0.57, every=10, stage="both",   # admission + every 10 steps
                        graduate=True,                                 # σ_F ≥ τ → leaves batch, sampler refills
                        id_map=lambda sid: ds.load_order[sid])
fused = FusedStage(sub_stages=[(0, relax)], sampler=sampler, refill_frequency=1, hooks=[hook])
fused.run(batch=None, n_steps=20000)
hook.flagged   # → your DFT / oracle queue; hook.records holds σ per system and step
```
Full working example (dataset protocol, neighbour hooks, NEB variant): `docs/REPRODUCE.md`.

**2. Trust service** — structures in, per-system trust and recommended model out:

```bash
python -m trustswarm.service serve --snapshot results/trust_snapshot.json --port 8000
curl -X POST localhost:8000/trust -H 'content-type: application/json' \
     -d '{"systems":[{"Z":[8,1,1],"R":[[0,0,0],[0.96,0,0],[-0.24,0.93,0]]}]}'
python -m trustswarm.service score my_molecules.xyz --snapshot results/trust_snapshot.json
```

## Try it on a laptop (CPU, 2 minutes)

```bash
git clone https://github.com/Pavel-Tk/trustswarm && cd trustswarm
uv venv .venv --python 3.11 && source .venv/bin/activate      # Windows: .venv\Scriptsctivate
uv pip install -e ".[dev,service]"
pytest                                                          # 82 CPU tests
python scripts/make_synthetic.py                                # synthetic molecules + models
python scripts/run_ablation.py --frames data/synthetic/frames.pkl --preds data/synthetic/preds --out results/synthetic
# trust service API with CPU stand-in models (verdicts meaningless; for integration only):
python -m trustswarm.service serve --snapshot results/trust_snapshot.json --fake-models --port 8000
```

## Reproduce the real results (GPU)

```bash
bash scripts/setup_gpu.sh                               # envs + Transition1x (see docs/REPRODUCE.md)
python3 scripts/predict_pool.py --build-frames          # T1x frames + every model's predictions (~2.5 min, L4)
python3 scripts/analyze_signal.py                       # disagreement vs DFT error
python3 scripts/screen_alchemi.py                       # ALCHEMI in-flight batch + DisagreementHook
python3 scripts/run_neb.py                              # batched NEB, σ along the path
python3 scripts/run_ablation.py --frames data/frames.pkl --preds data/preds --out results --budgets 20,40,80
python3 scripts/finetune.py --labelled-uids results/oracle_labels_armD.json ...   # Curator: LoRA repair + gate
python3 scripts/run_ablation.py ... && python3 scripts/write_results.py && python3 scripts/make_figures.py
python3 scripts/demo_service.py                         # trust service against the real model agents
```
Step-by-step with runtimes and environment variables: `docs/REPRODUCE.md`.

## Honest caveats

* Level of theory differs: MACE-MPA-0 and Orb-v3-omat are PBE materials models, AIMNet2 is
  ωB97M-D3, Transition1x is ωB97x/6-31G(d). We never compare absolute energies across models;
  errors are force RMSE and energies relative to each reaction's reactant. Part of every
  pre-fine-tune error is the level-of-theory gap, not model failure.
* Leakage: Orb models trained on OMol25 have seen Transition1x geometries (OMol25 recomputed
  T1x), so we use `orb-v3-conservative-inf-omat` instead. MACE-MPA-0 is crystals-only.
  AIMNet2's training set contains no T1x as far as we could verify.
* The official T1x split is by chemical formula. Transition1x builds many reactions from the same
  reactant, so a per-reaction split leaks geometry; our pool/eval split is by connected components of
  reactions that share an endpoint geometry (37 components, 0 shared geometries; `results/split_report.json`).
  Pool and eval still share chemical formulas.
* The oracle is a DFT label lookup, not live DFT.
* The trust service's calibration is learned on Transition1x-like C/H/N/O reaction chemistry; outside that
  domain (e.g. water, metals, charged species) its trust numbers extrapolate and should not be believed.
  `results/service_demo.json` shows it scoring 12 held-out frames with the real model agents.
* In the ablation, agents replay predictions cached by the model-agent processes on the GPU, and the hook's
  logic runs as its CPU twin (`DisagreementMonitor`); the ALCHEMI `DisagreementHook` itself is demonstrated in
  the in-flight screen (`results/screen_alchemi.json`).

See `docs/DESIGN.md` for the full contract, `docs/data_and_models.md` and
`docs/alchemi_api_notes.md` for the research notes.

MIT licence.
