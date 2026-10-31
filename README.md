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
_Results are being produced; see `results/`._
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
from trustswarm.hook import DisagreementHook
from trustswarm.model_server import spawn
clients = [spawn("mace_mpa0", 6000).connect(), spawn("aimnet2", 6001).connect(),
           spawn("orb", 6002, python="/path/to/orb-env/bin/python").connect()]
hook = DisagreementHook(clients, tau=0.35)   # graduates systems with σ_F ≥ τ out of the batch
# FusedStage(sub_stages=[(0, FIRE2(...))], sampler=SizeAwareSampler(...), hooks=[hook, ...]).run(...)
hook.flagged  # → oracle queue
```

**2. Trust service** — structures in, per-system trust and recommended model out:

```bash
python -m trustswarm.service serve --snapshot results/trust_snapshot.json --port 8000
curl -X POST localhost:8000/trust -H 'content-type: application/json' \
     -d '{"systems":[{"Z":[8,1,1],"R":[[0,0,0],[0.96,0,0],[-0.24,0.93,0]]}]}'
python -m trustswarm.service score my_molecules.xyz --snapshot results/trust_snapshot.json
```

## Reproduce

```bash
# GPU (Colab L4 used here). Core env: nvalchemi-toolkit[cu13,mace,aimnet,ase] @ 0.3.0-rc; Orb in its own venv.
python3 scripts/predict_pool.py --build-frames          # T1x frames + every model's predictions
python3 scripts/analyze_signal.py                       # disagreement vs DFT error
python3 scripts/screen_alchemi.py                       # ALCHEMI in-flight batch + DisagreementHook
python3 scripts/run_neb.py                              # batched NEB, σ along the path
python3 scripts/run_ablation.py --frames data/frames.pkl --preds data/preds --out results/
python3 scripts/finetune.py ...                         # Curator: LoRA repair + acceptance report
# CPU-only smoke test, no GPU or data needed:
python scripts/make_synthetic.py && python scripts/run_ablation.py --frames data/synthetic/frames.pkl --preds data/synthetic/preds --out results/synthetic
pytest -q
```

## Honest caveats

* Level of theory differs: MACE-MPA-0 and Orb-v3-omat are PBE materials models, AIMNet2 is
  ωB97M-D3, Transition1x is ωB97x/6-31G(d). We never compare absolute energies across models;
  errors are force RMSE and energies relative to each reaction's reactant. Part of every
  pre-fine-tune error is the level-of-theory gap, not model failure.
* Leakage: Orb models trained on OMol25 have seen Transition1x geometries (OMol25 recomputed
  T1x), so we use `orb-v3-conservative-inf-omat` instead. MACE-MPA-0 is crystals-only.
  AIMNet2's training set contains no T1x as far as we could verify.
* The official T1x split is by chemical formula; our pool/eval split is by reaction within the
  T1x test formulas, so pool and eval share formulas (but not reactions).
* The oracle is a DFT label lookup, not live DFT.

See `docs/DESIGN.md` for the full contract, `docs/data_and_models.md` and
`docs/alchemi_api_notes.md` for the research notes.

MIT licence.
