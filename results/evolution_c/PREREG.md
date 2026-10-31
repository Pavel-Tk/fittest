# Evolution run C: niche specialists × budget axis (pre-registered)

Written and committed **before** the real run (Sat Oct 31, about 14:05 EST). Code: `fittest/evolution_c.py`
(engine, numpy, subclass of `fittest.evolution.Evolution`), `scripts/evolve_c.py` (runs, controls,
summary, figure), `scripts/evolve_c_offspring.py` (one child, torch, GPU), `tests/test_evolution_c.py`.
Hardware: Colab NVIDIA L4 (24 GB), session `evoc`.

**Question.** Run A (MACE LoRA offspring, budget 200) found no improvement: the static swarm already
serves about the hindsight-optimal blend, so selecting among existing models leaves little headroom.
Run C asks two things. (1) Can *niche specialists* (strong parents fine-tuned only where the swarm is weak)
beat the static swarm when the Calibrator routes them per bucket? (2) Does evolution start to beat the
static swarm as ground truth accumulates (budgets 200, 400, 800)?

## Fixed design

**Data and splits** are those of run A (`results/evolution/METHODS.md`): leak-free split of
`data/frames.pkl` (seed 0); the pool is split by whole endpoint-geometry component into *breed* (offspring
training) and *select* (fitness, gate; never trained on) parts; eval = 1485 frames / 135 reactions, read
only by the scorer. The reactant reference energy of a labelled frame's reaction is supplied by the Oracle
contract (as in runs A and B).

**Budget axis.** Total budgets 200, 400 and 800 oracle labels. Each has 4 generations of B/4 labels, with
breeding after generations 0, 1 and 2 (3 breeding generations). Triage, oracle and audits are unchanged from
run A: 40 % of each generation's labels go to select and 60 % to breed; in each part 75 % go to the
highest-σ_F unlabelled frames and 25 % to random audits of trusted frames.

**Niches (speciation), label-free.** These are fixed once from the base swarm (MACE-MPA-0 + AIMNet2 + Orb).
`flagged` = σ_F ≥ τ₀, where τ₀ is the 70th percentile of pool σ_F, the Calibrator's flag threshold
(the high-σ "TS-like" region). `trusted` = the rest. Structure role is not used because it is not known at
query time. Only the `flagged` niche breeds specialists: that is where the swarm is weak, and triage
concentrates labels there. A niche breeds only if it has at least 10 breed labels.

**Offspring.** Each breeding generation produces one specialist per family (AIMNet2, Orb-v3-conservative-inf-omat,
MACE-MPA-0). Each is trained only on the breed labels of the `flagged` niche, accumulated so far.
- Head-only fine-tune: AIMNet2 `mlps.2` + `outputs` (run B's recipe), Orb `heads`, MACE `model.readouts`.
- Loss: `w_F · mean |F − F_DFT|² + w_E · mean ((E_i − E_r) − (E_i − E_r)_DFT)²`, with r the reactant of
  frame i's reaction (reaction-relative energy loss).
- Optimiser: Adam, 200 steps, 4 reactions per batch (each reactant plus its labelled frames), gradient clipping
  at 10, w_F = w_E = 1.
- Learning rate: AIMNet2 1e-4, Orb 1e-4, MACE 1e-3 (readout only, about 2k parameters).
- Generation-0 children use these base recipes. Later children mutate their parent's recipe (learning rate
  and w_E, each ×0.5 or ×2, seeded) and start from the parent's head weights.
- Parent = the fittest alive member of the family: a base model or a `flagged` specialist of that family.
- Each child predicts all pool and eval frames once.

**Gate.** On the select labels of the child's niche (component-disjoint from every training label; asserted),
the child must not be worse than its parent on force RMSE and on reactant-relative ΔE MAE. If the niche has
fewer than 5 select labels, all select labels are used. Then the clone check of run A applies (Pearson r of
deviations from the population-mean force > 0.95, or median per-frame force RMS difference < 0.02 eV/Å ⇒ clone,
rejected). The ALCHEMI acceptance-report check of run A is not used (it is MACE/LoRA-specific).

**Fitness and selection** are unchanged from run A: marginal contribution of each member under the
Calibrator's per-bucket routing (out-of-fold ledger, σ_F × |F̄| buckets), z-scored answerer and critic Δ with
reaction bootstrap. A specialist that wins only in its niche is served there and scores positive. Pruning is
as in run A (min population 3, max 6).

**Arms (per budget and seed).**
- `static`: same triage/oracle, no breeding, no pruning.
- `evolved`: niche specialists, gate, fitness selection.
- `noselect`, "fine-tune without selection": every child that trained is admitted (no gate, no clone
  rejection, no fitness pruning). The population is the base models plus the newest child of each family
  lineage, and the lineage continues from its newest child.

Children are content-addressed (family, parent, recipe, training uids, seed), so identical children across
arms are trained once.

**Runs.** Seed 0 runs all three arms at budget 800 and static + evolved at 200 and 400. Seed 1 runs
static + evolved at 800 (and noselect if time allows). If time allows, seed 1 is also run at 200 and 400. Cut
order if short of time (decided now): seed 1 at 200/400, then budget 400, then seed 1 at 800. 200 vs 800 and
the no-selection control are kept.

**Evaluation** (eval split only; 1000 reaction-bootstrap resamples shared by all runs of a budget, so
comparisons are paired): served force RMSE, served rel-E MAE, ECE and Brier of P(wrong), trusted share,
badly-wrong (F-RMSE > 0.4) trusted share, plus served error by niche (descriptive). Paired 95 % CIs:
evolved vs static at the same budget (final generation), noselect vs static, evolved vs noselect, and final
generation vs gen 0. A family tree JSON is written per run.

**Decision rule (same as run A).** Evolution *holds* at a budget iff the evolved swarm's served force RMSE at
the final generation (after 3 breeding generations) is significantly lower (paired CI excludes 0) than both
the static swarm at the same budget and its own generation 0, with offspring in the final population, and with
no significant regression vs static in rel-E MAE or ECE. With two seeds, it must hold in each seed. A
*crossover* is the smallest budget at which it holds. Served rel-E MAE improvements are reported as
secondary. Nothing is tuned on eval; negative results are reported as they come.

## Done before this file was written (disclosed)
- Smoke tests of the offspring script, one per family: 30 steps on 40 pool frames. Only training losses were
  read.
- A CPU dry run of the `static` arm at budget 800 to check the pipeline. It printed eval metrics of the static
  swarm (served force RMSE ≈ 0.20 eV/Å). No design choice was changed after it. The static arm has no free
  parameters.
- The AIMNet2 recipe (head = `mlps.2` + `outputs`, Adam 1e-4, force + reactant-relative energy loss) follows
  run B's interim experience as relayed by the coordinator. Run B's own code
  (`scripts/evolve_offspring_aimnet.py`) is not imported. One trainer covers all three families.
