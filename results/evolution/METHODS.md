# Evolutionary loop — methods (fixed before the real run)

Code: `fittest/evolution.py` (engine, numpy), `scripts/evolve.py` (runs + controls),
`scripts/evolve_offspring.py` (one MACE LoRA child, torch, CPU), `scripts/write_evolution_summary.py`
(tables + decision rule), `scripts/make_evolution_figures.py` (fig7, fig8). Tests: `tests/test_evolution.py`.

**Data.** Leak-free split of `data/frames.pkl` (`assign_leakfree_splits`, seed 0): pool 1672 frames /
152 reactions / 16 endpoint-geometry components; eval 1485 frames / 135 reactions (never labelled,
never used for any decision; read only by `EvalScorer`).

**Breed / select split.** The pool's 16 components are split once (seeded greedy, ≈35 % of frames)
into *select* (7 components, 572 frames, 52 reactions) and *breed* (9 components, 1100 frames,
100 reactions). Offspring train only on breed labels; fitness and the gate use labels no member
trained on (select labels always qualify), so selection is never measured on a child's own
training geometry.

**Budget.** 4 generations × 50 Transition1x look-ups = 200 (arm D's budget). Per generation 20 select
+ 30 breed; in each part 75 % to the highest σ_F unlabelled frames, 25 % random audits of trusted
frames (σ_F < τ = 70th percentile of pool σ_F). The static control runs the same procedure with the
population fixed at MACE-MPA-0 + AIMNet2 + Orb.

**Serving (Calibrator twin).** Candidates = every non-empty subset mean of the population; per
(candidate, σ_F × mean|F̄| bucket) Beta(2,1) ledger of correctness from oracle labels (a label counts
for a candidate only if no member of it trained on that label's component); served = argmax
P(correct | bucket); trust = that probability; P(wrong) = 1 − trust. Differences from arm D:
no Skeptic kNN vetoes, no bid auction (σ-ranked oracle + audits instead), no planted cases; risk
score for AUROC = population σ_F.

**Fitness** (on clean labels): answerer Δ = OOF served error of P∖m − of P (error in wrongness units
F_RMSE/0.25 + |ΔE_rel|/0.25; ledger fitted out-of-fold, 5 reaction-grouped folds); critic Δ =
AUROC(σ_F(P) → P's served answer wrong) − same for P∖m. 400 paired reaction-bootstrap resamples →
SE, 95 % CI. fitness = Δ_ans/SE_ans + Δ_crit/SE_crit; role = larger z.

**Selection.** Recompute after each death. Eligible = neither CI lower bound > 0 and fitness ≤ 0.5;
least fit eligible dies while population > 3; then least fit dies while > 6.

**Breeding.** 3 children per generation (generations 0–2) from the fittest MACE-family member(s)
(the second-fittest parents child c when there are two); the foundation MACE-MPA-0 is the germline
if no MACE-family member is alive. Recipe mutations from the parent (base E:F 10:10, LoRA rank 16,
α 2, lr 1e-3, 40 epochs, batch 16, ≥ 160 optimizer steps): child a E-weight ×3; child b E-weight ×10
and LoRA rank ×½ or ×2; child c lr ×½ or ×2, epochs −10/+20, 80 % reaction subsample of the labels.
E-weight clipped to [1, 300]. A child of an offspring starts from its parent's merged weights (new
LoRA on top, E0 inherited); a child of the foundation model refits E0 on its training labels.
Clone rejection: Pearson r of child's and a member's deviations from the population-mean force over
pool atoms > 0.95, or median per-frame force RMS difference < 0.02 eV/Å.

**Run B (pre-planned extension, run after run A's outcome was known): AIMNet2 offspring.**
`scripts/evolve_offspring_aimnet.py`. Same budget, generations, split, fitness, selection, gate
(minus the ALCHEMI report, not applicable to AIMNet2; recorded as n/a), clone check, eval and
decision rule; only the bred family changes (germline `aimnet2`, ensemble member 0). Fine-tune with
`AIMNet2Calculator(train=True)`, Adam, cosine schedule, ≥ 200 steps, batch 16; base recipe lr 1e-4,
head only (mlps.2.* + outputs.*). Loss = w_F · MSE(forces) + w_rel · MSE((Ê_i − Ê_reactant) −
(E_i − E_reactant)). The relative-energy target uses the Oracle's free reactant reference energy —
the same contract every arm and run A get (DESIGN.md: the reactant reference is not charged to the
budget); it is not an extra label. No absolute-energy term. Each job also reports how many training
labels have a *labelled* same-reaction partner (16 of 30 at generation 0, seed 0). Sibling mutations:
a: w_rel ×3; b: train the whole network at lr ÷3 (or back to head at lr ×3); c: lr ×3 or ÷3,
epochs −20/+30, 80 % reaction subsample. Level of theory: AIMNet2 is ωB97M-D3/def2-TZVPP, T1x is
ωB97x/6-31G(d); part of any gain is learning the T1x level from budgeted labels (legitimate, the
labels came out of the budget). **Newest-child control**: one child per generation with the
parent's recipe unchanged, no fitness, no pruning, no gate, no clone check; the newest child always
replaces the previous one.

**Gate** (select labels, component-disjoint from the child's training labels, asserted in both the
engine and the torch job): force RMSE ≤ parent; reactant-relative ΔE MAE ≤ parent; ALCHEMI
`build_acceptance_report` from-scratch ratio (child vs parent, same hold-out batches) ≤ 1.01.
Tolerance note: the toolkit's energy-per-atom MAE is dominated by the frozen E0 offsets, which are
extrapolated across compositions and identical for child and parent (smoke test: 2644 eV MAE for
both, ratio 1.000008), so at 1.00 the check is a float-precision coin flip; the 1 % tolerance was set
after the smoke test, before the real run, and not on eval. The toolkit's own verdict at 1.0 is
recorded too.

**Evaluation.** Per generation (after selection, ledger on all labels so far) on eval: served force
RMSE, served ΔE_rel MAE, wrong rate, ECE/Brier of P(wrong), AUROC(σ_F → served wrong), trusted
fraction, badly wrong (F-RMSE > 0.4) trusted, silent (badly wrong, trusted, σ_F < τ). 1000
reaction-bootstrap resamples shared by every run → paired CIs.

**Decision rule** (`scripts/write_evolution_summary.py`): the flagship holds iff, in every seed run,
served force RMSE at the final generation is significantly lower than both the run's own generation
0 and the static swarm at the same budget, over ≥ 3 generations, with offspring in the final
population and no significant regression vs static in rel-E MAE or ECE.
