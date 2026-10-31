# TrustSwarm — design contract (v1, frozen for overnight build)

> A collective that decides which MLIP to trust, where, and repairs it where none can be.

Everything below is the contract between modules. Implementers: do not change a public
signature without updating this file.

## 0. Scientific framing (what we claim, what we test)

Claim: **cross-model disagreement between heterogeneous foundation MLIPs is a usable,
label-free signal of where a prediction is wrong**, and a collective of agents that (a) spends
a fixed oracle budget where models disagree, (b) audits "trusted" verdicts with held-out
references, and (c) repairs the worst model on what the oracle returns, beats a single model
and a seed-ensemble of one fine-tuned model **at the same oracle budget**.

Validation follows NVIDIA's own wording: *held-out reference data, planted known cases,
agreement between independent models*. We report failures explicitly.

## 1. Data

Transition1x (T1x), ωB97x/6-31G(d), C/H/N/O, neutral, non-periodic. Units: eV, Å, eV/Å.

Splits are **by reaction** (never by frame), using the official T1x train/val/test split:

| Split name | Source | Who may see labels | Purpose |
|---|---|---|---|
| `pool` | T1x test, reactions 50% | Oracle only (revealed per call) | Stream the collective triages |
| `eval` | T1x test, other 50% | Nobody during the run; final scoring only | Post-fine-tune error, held-out AUROC |
| `skeptic_ref` | T1x val, 50% of reactions | Skeptic only | Skeptic's private audit references |
| `planted` | T1x val, other 50%, filtered | Skeptic plants them; labels private | Known failures injected into the pool |
| `seed_train` | T1x train, small subset | Arm B training only | Seed-ensemble fine-tunes |

Per reaction we sample frames from the NEB path (reactant, product, TS and intermediate
images) — role ∈ {reactant, product, ts, path}. The reactant frame of every pool/eval reaction
is always included so relative energies can be formed.

`trustswarm.data.Frame` (dataclass): `uid:str, reaction:str, formula:str, split:str,
role:str, Z:np.ndarray[int] (n,), R:np.ndarray[float] (n,3), E:float|None, F:np.ndarray|None (n,3)`.
Labels `E,F` are stripped (None) before frames leave the Oracle for pool/eval splits.

## 2. Error definition ("wrong") — fixed before running anything

For a prediction (E_hat, F_hat) on a frame with DFT (E, F):

* `force_rmse = sqrt(mean over atoms and xyz (F_hat - F)^2)` [eV/Å]
* `rel_e_err = |(E_hat - E_hat_reactant) - (E - E_reactant)|` [eV] (same composition
  within a reaction, so level-of-theory offsets cancel)
* **wrong** ⇔ `force_rmse > 0.25 eV/Å` **or** `rel_e_err > 0.25 eV` (≈ 5.8 kcal/mol).
  (Both thresholds are config values `WRONG_F`, `WRONG_E`; report sensitivity at 0.15/0.4.)

## 3. Disagreement signal

For models m=1..M on one structure with per-atom forces F_m (n,3):
`F_bar = mean_m F_m`; per-atom spread `s_i = sqrt(mean_m ||F_m,i - F_bar,i||^2)`;
`sigma_F = mean_i s_i` (headline) and `sigma_F_max = max_i s_i`. Energy spread on relative
energies `sigma_E = std_m (E_m - E_m,reactant)` when the reactant is known.
Implemented once in `trustswarm.signals.force_spread(forces: np.ndarray (M,n,3)) -> (mean,max)`
and a torch twin inside the hook.

## 4. Agents (each holds private info; they talk only through the Blackboard)

| Agent | Private info | Posts | Reads |
|---|---|---|---|
| ModelAgent ×3 (MACE-MPA-0, AIMNet2, Orb) | own weights; own prediction; own oracle-revealed error history | `prediction`, `bid` | `flag`, `oracle_result`, `reputation` |
| Oracle | DFT labels; budget k per round | `oracle_result` (true error per model) | `bid` |
| Skeptic | `skeptic_ref` labels + descriptors; `planted` labels | `plant`, `veto`, `bid` | `verdict`, `oracle_result` |
| Calibrator | the ledger | `reputation`, `threshold`, `verdict` (trust + recommended model) | `oracle_result`, `prediction`, `flag` |
| Curator | fine-tune compute | `finetune_report`, `model_update` | `reputation`, `oracle_result` |
| DisagreementHook (in the ALCHEMI batch) | — | `flag` (sigma ≥ tau) / `pass` | `threshold` |

No central planner: each agent has `step(bb: Blackboard) -> None` that reads new events and
posts new events. The round driver only advances the clock (`bb.tick()`), it makes no decisions.

### Blackboard
`trustswarm.blackboard.Blackboard`: append-only event log (in-memory list + optional JSONL
file). `post(kind, author, payload:dict) -> int`, `read(kind=None, since=0, round=None) -> list[Event]`,
`tick()`. `Event(id, round, kind, author, payload, t)`. JSONL log is a deliverable (shown in demo).

### Bidding (oracle allocation)
Each round, flagged systems (and Skeptic audit targets) receive bids. A ModelAgent's bid on
system x is its private estimate that the collective's served answer for x is wrong:
`bid_m(x) = rep_m * g(d_m(x), sigma(x))`, where `d_m(x)` is the agent's own deviation from the
ensemble mean and `rep_m` its current reputation (Calibrator). Skeptic bids on vetoed / planted
systems with `bid_S(x) = rep_S * p_S(x)` (kNN wrongness probability from its references).
Oracle pays the top-k summed bids per round (k = `BUDGET_PER_ROUND`). Settlement (Calibrator):
a winning bid whose system is revealed wrong earns `+1*bid share`; a false alarm costs
`-lambda*bid share`. Reputations are a Beta(α,β) mean, clipped to [0.05, 0.95].

### Calibrator → verdict
Per (model, context bucket) Beta posterior of correctness from oracle results. Context bucket
is label-free: `(sigma_F quantile bin [4], mean |F_bar| bin [3])`. Verdict per system:
`trust = 1 - P(served answer wrong)`, `recommended_model = argmax_m P(m correct | bucket)`,
served answer = recommended model's prediction. Threshold tau per round = smallest sigma
such that ledger-estimated P(wrong | sigma ≥ tau) ≥ 0.5 (isotonic on labelled points),
floored at the 70th percentile of sigma on round 0.

### Skeptic
Descriptor (label-free): composition counts (C,H,N,O), n_atoms, min interatomic distance,
max bond-stretch ratio vs covalent radii, mean |F_bar|, sigma_F. Standardised, kNN (k=7).
On every `verdict` with trust high, Skeptic computes `p_S = fraction of k neighbours in
skeptic_ref whose consensus was wrong`; if `p_S > 0.5` it posts `veto` and bids.
Planted cases are injected into the pool stream at round 0 (Skeptic knows their labels;
nobody else does). `planted` and `skeptic_ref` are disjoint reactions — no leakage.

### Curator
After the last triage round: worst model = highest ledger mean force RMSE on oracle-labelled
systems. LoRA fine-tune it (ALCHEMI 0.3.0-rc `FineTuningStrategy` + `LoRAConfig`) on all
oracle labels collected (pool frames only); acceptance report vs the base model on a held-out
slice of oracle labels (NOT `eval`). Ship ⇔ report verdict passes; then re-predict.

### CPU implementation additions (agents + event kinds)
* `PoolStream` (`agents/stream.py`, author `stream`): passive source, no decisions. Posts
  label-free `system` events {uid, reaction, formula, role, phase, Z, R}: pool reactions in one
  chunk per triage round (`phase="triage"`), eval frames once after triage (`phase="serve"`;
  served and scored, never labelled). Skeptic injects planted structures as `plant` (same
  payload); agents treat `system` and `plant` identically.
* `DisagreementMonitor` (`agents/monitor.py`, author `hook`): CPU twin of the hook; reads
  `prediction` + `threshold`, posts `flag` per system (σ stats + per-model deviation d_m) and
  batched `pass`. Holds systems until the first `threshold`.
* `tick` (author `clock`): posted by `Blackboard.tick()`.
* Batched kinds: `prediction` (one batch per ModelAgent step), `pass`, `verdict` are
  column-wise `{uid: [...], key: [...], ...}`; verdicts reference prediction rows as
  `[event_id, row]`. Calibrator candidates = models + `consensus` (ensemble mean); verdict
  carries `trust` (bucket Beta), `risk` (isotonic P(served wrong | σ_F)), `recommended_model`.
* Auction: bids sealed per round; Oracle pays top-k when the clock leaves the round; unused
  allowance carries over (reported); `oracle_result` is the only label channel.
* JSONL audit log: arrays (`forces`, `F`, `R`) are replaced by `{"omitted","shape"}` and floats
  rounded; in-memory events keep the arrays. `bid` is batched too.
* v2 (after the first real-data run; v1 kept in `results/v1_flag_only_bidding/`):
  - Candidates = every non-empty subset mean of the models (`trustswarm.candidates`: singletons,
    `mean(a+b)`, `consensus`); `oracle_result.candidate_errors` gives each one's error.
  - Exploration audits: each triage round the Skeptic bids `{"audit": true}` on a seeded uniform
    sample of trusted systems (σ < τ, unflagged), `audit_fraction` (0.25) of k; the Oracle pays
    them first out of the same k. Audits feed the ledger but not reputations. Baseline
    `C_audit` = C + the same audits (isolates their effect).
  - skeptic_ref / planted are split by formula; planted = AIMNet2 force RMSE > 0.4 eV/Å
    (arm-independent); replayed fine-tunes ship only if `train_uids` ⊆ this run's labels and
    disjoint from held-out frames.

## 5. Ablation (same oracle budget K = rounds × k)

| Arm | Models | Uncertainty score | Skeptic | Bidding | Fine-tune |
|---|---|---|---|---|---|
| A single | AIMNet2 (strongest single model on organics) | max|F| heuristic (and random) | – | top-k score | – |
| B seed ensemble | AIMNet2 × 4 shipped ensemble members (same data, different seeds) | seed sigma_F | – | top-k sigma | – |
| C collective | MACE + AIMNet2 + Orb | sigma_F | – | top-k sigma | – |
| D TrustSwarm | MACE + AIMNet2 + Orb | sigma_F + rep bids | on | sigma + rep | 1 round |

Metrics (on `pool` for triage, `eval` for generalisation):
1. AUROC(score → served answer wrong). Also AUROC vs a common target (MACE wrong).
2. Recall@K: fraction of wrong served answers that received an oracle call.
3. Planted failures caught: fraction of planted cases flagged, vetoed, or oracle-checked.
4. Served-answer error (force RMSE mean, rel_e MAE) on `eval`, before/after fine-tune.
5. Fine-tuned model error on `eval` vs base.
Bootstrap 95% CI over reactions (1000 resamples). Seeds fixed. Failures reported.

## 6. ALCHEMI usage
* `DisagreementHook`: a Toolkit hook in the in-flight batch; every N steps it gathers forces
  from the three ModelAgent processes for every system, computes sigma per system, and
  graduates systems with sigma ≥ tau out of the batch into the oracle queue (the size-aware
  sampler refills the slot). Unflagged systems converge normally and are reported with sigma.
* Batched NEB (example 11 pattern) on T1x reactions with the hook watching: sigma along the
  path vs DFT error along the path (figure).
* LoRA fine-tune from 0.3.0-rc (`nvalchemi.training`), acceptance report.

## 7. Reusable artefacts (collab bonus)
* `trustswarm.hook.DisagreementHook` — drop-in for any Toolkit run, any set of models.
* `trustswarm.service` — FastAPI: `POST /trust` with structures (xyz or JSON Z/R) →
  per-system `{sigma_F, trust, recommended_model, flagged, per_model_energy}`; `GET /health`.
  CLI: `trustswarm score file.xyz`.

## 8. Repo layout
```
trustswarm/            package
  data.py              T1x reader, splits, Frame
  signals.py           force_spread, errors, wrong()
  blackboard.py        Blackboard, Event
  agents/              model_agent.py oracle.py skeptic.py calibrator.py curator.py
  hook.py              DisagreementHook (ALCHEMI)
  model_server.py      per-model process server + client (own process/env)
  models/              orb_wrapper.py (Toolkit BaseModelMixin) + loaders
  ablation.py          arms A-D, metrics, bootstrap
  service.py           FastAPI trust service + CLI
scripts/               colab job scripts: predict_pool.py, run_neb.py, finetune.py, run_ablation.py
tests/                 pytest, CPU-only, synthetic data (no GPU, no T1x needed)
results/               json + figures (committed)
docs/                  this file, API notes, results.md
```

## 9. Prediction cache (GPU stage → agent stage)
GPU scripts write, agent/ablation code reads. `trustswarm.cache`:
* `save_frames(path, frames: list[Frame])` / `load_frames(path) -> list[Frame]` (pickle; this file
  holds labels and is read only by Oracle/Skeptic/final scoring).
* `save_predictions(path, model: str, uids: list[str], energies: np.ndarray (N,), forces: list[np.ndarray (n_i,3)], meta: dict)`
  → npz with `uids`, `energy`, `forces_flat`, `offsets` (N+1), `meta_json`.
* `load_predictions(path) -> Predictions` with `.model`, `.energy[uid]`, `.forces[uid]`, `.meta`.
Files: `data/frames.pkl`, `data/preds/{model}.npz` where model ∈ {mace_mpa0, aimnet2, orb,
mace_seed0..3, mace_lora_curated}. GPU timings go in `meta` (atoms/s) for the cost column.

## 10. Revisions after data/model research (see docs/data_and_models.md)
* Orb checkpoint = `orb-v3-conservative-inf-omat` (OMol25-trained Orb models saw T1x geometries → excluded from headline; leakage).
* AIMNet2 checkpoint = `aimnet2` (ωB97M-D3), MACE = `medium-mpa-0`.
* Arm A single = `aimnet2` (model file `aimnet2`); Arm B = `aimnet2_m0..m3` (the four shipped ensemble members); Arms C/D = `mace_mpa0, aimnet2, orb`. Optional extra arm B' = `mace_seed0..3` if produced.
* T1x official split is by chemical formula (test 8 formulas, val 8). Our `pool`/`eval` split is by reaction within T1x test; report that pool and eval share formulas.
* "Wrong" thresholds are config values; primary reported with threshold-free Spearman(sigma, error) and AUROC at several thresholds.
