# Evolution: does a swarm that breeds and selects its own models beat the static swarm?

**Verdict under the pre-registered rule: no.** Runs A, B and C each fail. Below, every pre-registered run
is reported side by side, including the failures. This is followed by exploratory runs (extra seeds and design
variants) that were **not** pre-registered and are labelled as such. Those runs do not change the verdict.

## The rule (fixed before any run; `METHODS.md`, `../evolution_c/PREREG.md`)
Evolution *holds* iff, **in every seed run**, the evolved swarm's served force RMSE on the eval split at
the final generation (after 3 breeding generations, ≥ 3 generations) is significantly lower (paired 95 %
reaction-bootstrap CI below 0) than **both** the static swarm at the same DFT budget **and** its own generation
0. Offspring must be in the final population, and there must be no significant regression vs static in served
rel-E MAE or ECE. "Static" uses the same triage and oracle, with no breeding and no pruning. Eval = 1485 frames / 135
reactions, never labelled and never used for any decision.

## Pre-registered results

| run | what breeds | budget | seed 0 | seed 1 | verdict |
|---|---|---|---|---|---|
| **A** | MACE-MPA-0 LoRA offspring | 200 | does not hold | does not hold | **does not hold** |
| **B** | AIMNet2 head fine-tunes (rel-E loss) | 200 | does not hold | **holds** | **does not hold** |
| **C** | niche specialists, AIMNet2 + Orb + MACE (L4 GPU, as pre-registered) | 200 | does not hold | holds | **does not hold** |
| | | 400 | does not hold | does not hold | **does not hold** |
| | | 800 | **holds** | does not hold | **does not hold** |
| C′ | same, AIMNet2 specialists only (CPU; deviation) | 200 | does not hold | holds | does not hold |
| | | 400 | does not hold | holds | does not hold |
| | | 800 | does not hold | does not hold | does not hold |

Run B is a pre-planned extension that was started after run A's outcome was known. Run C was pre-registered
before it ran (`PREREG.md`, commit 957a91d). Run C's first GPU attempt is **invalid** and is not quoted: CUDA
out-of-memory failures were counted as stillborn children (`../evolution_c/partial_run1/`). The clean GPU rerun
(two Colab L4 sessions) is the faithful run C. C′ was run first on the CPU VM while no GPU was available; it
deviates from the pre-registration by breeding AIMNet2 specialists only and is reported for completeness.

### Final generation vs static, same budget (paired 95 % CI; * = excludes 0)

| run / seed | served force RMSE Δ (eV/Å) | served rel-E MAE Δ (eV) | ECE Δ | offspring alive | failed criterion |
|---|---|---|---|---|---|
| A s0 | +0.010 [+0.008, +0.013] * (worse) | −0.007 [−0.011, −0.002] * | +0.004 n.s. | none | forces worse than static |
| A s1 | −0.011 [−0.017, −0.005] * | **+0.056 [+0.044, +0.068] *** (worse) | **+0.069 *** (worse) | g1_a (MACE replaced) | rel-E and ECE regress; not better than gen 0 |
| B s0 | −0.001 [−0.007, +0.005] | −0.027 [−0.039, −0.014] * | −0.007 n.s. | g1_a (AIMNet2 replaced) | forces not better than static |
| B s1 | −0.024 [−0.031, −0.016] * | −0.029 [−0.042, −0.017] * | +0.008 n.s. | g2_b | — (holds) |
| C b200 s0 | **+0.034 [+0.026, +0.041] *** (worse) | −0.011 [−0.022, −0.001] * | **regresses *** | g3_orb_flagged (base Orb pruned) | forces worse; ECE regresses |
| C b200 s1 | −0.027 [−0.031, −0.023] * | −0.008 [−0.013, −0.004] * | no sig. regression | g2_aimnet2_flagged | — (holds) |
| C b400 s0 | **+0.019 [+0.012, +0.026] *** (worse) | −0.025 [−0.035, −0.016] * | no sig. regression | g1_mace, g3_orb (base MACE, Orb pruned) | forces worse |
| C b400 s1 | −0.016 [−0.022, −0.009] * | −0.018 [−0.027, −0.008] * | no sig. regression | g1_aimnet2, g2_orb, g3_mace | not better than own gen 0 |
| C b800 s0 | −0.017 [−0.023, −0.012] * | −0.044 [−0.055, −0.034] * | no sig. regression | g1_orb, g3_aimnet2, g3_orb (all three bases pruned) | — (holds) |
| C b800 s1 | **+0.011 [+0.003, +0.019] *** (worse) | −0.028 [−0.040, −0.017] * | no sig. regression | g1_orb_flagged (base Orb pruned) | forces worse |
| C′ b200 / b400 s1 | −0.027 * / −0.036 * | −0.008 * / −0.006 * | no sig. regression | one AIMNet2 specialist | — (holds) |
| C′ s0 (all budgets), b800 s1 | 0 | 0 | 0 | none (every child failed the gate) | no offspring |

Sources: `summary_tables.md` and `verdict.json` here (run A), `../evolution_aimnet/` (run B), and
`../evolution_c/summary.json` and `summary_table.md` (run C, GPU), `../evolution_c/cpu_aimnet2_run/` (C′). Figures: `../fig/fig7_evolution_run{A,B}_s{0,1}.png`,
`../fig/fig8_family_tree_run{A,B}_s{0,1}.png` and `../evolution_c/fig9_budget.png`.

### Controls (pre-registered)
- **Run B, "newest" control** (always keep the newest child, no gate, no selection). Forces vs static:
  s0 −0.003 n.s., s1 −0.027 *. rel-E: s0 −0.030 *, s1 −0.017 *. **ECE regresses significantly in both seeds**
  (+0.056 *, +0.044 *); the evolved arm's ECE never regresses. So selection's measurable contribution in run B is
  calibration, not accuracy.
- **Run C, no-selection arm at budget 800** (every trained child admitted; base models kept). GPU, three families:
  forces vs static s0 −0.006 n.s., s1 −0.041 *; rel-E s0 −0.029 *, s1 −0.028 *. C′ (AIMNet2 only): forces s0
  −0.016 *, s1 −0.039 *; rel-E s0 +0.013 n.s., s1 −0.014 *.

### Secondary outcome (pre-registered as secondary): relative energies
Every evolved run of run C lowers served reaction-relative energy error vs static, significantly in **6 of 6**
(−0.008 to −0.044 eV), including the runs where forces got worse. Run B: 2 of 2 (−0.027, −0.029). Run A: 1 of 2
(s1 is +0.056, worse).

## What we learned
1. **The gate kills most children, mostly on reaction energy.** Fine-tuned children reliably lower force error
   on held-out labels but often raise the reactant-relative energy error. MACE LoRA children did this at every
   energy weight tried (run A: 15 of 18 children stillborn across both seeds). In C′ (AIMNet2 only), seed 0 kept
   no child at any budget.
2. **Selection prunes the wrong members.** Fitness is measured on labels that triage concentrates in the
   high-σ niche, so a niche specialist looks more useful than the base model it descends from, and the base
   model is pruned. Outside the niche the specialist is worse, and served force error rises: run C b200 s0,
   b400 s0 and b800 s1 are significantly *worse* than static on forces after base Orb or MACE was pruned.
3. **Surviving children can help, but it is a coin flip across seeds** (holds: B s1, C b200 s1, C b800 s0,
   C′ b200/b400 s1). Some wins are partly seeds where the static swarm happened to do worse (static force RMSE
   0.220 vs 0.202 eV/Å in run B s1 vs s0).
4. **Fine-tuning without selection is the more reliable improvement** (exploratory evidence below). This points
   at the selection machinery (a gate on a few select labels and pruning of base models) as the weak link, not at
   the idea of growing the swarm.
5. The static swarm (arm D, served force RMSE 0.206 eV/Å, `../RESULTS.md`) remains the reported system. Nothing
   in this file changes the static ablation results.

## Exploratory (NOT pre-registered; does not change the verdict)

**Extra seeds 2–5** (same code and configs; CPU; run C AIMNet2-only):
- Run B (`../evolution_aimnet_ext_seeds2-5/`): evolved holds in 2 of 4 seeds (s4, s5). In s3 it is significantly worse on
  forces than static (+0.007 *). The newest control beats static on forces in 4 of 4 seeds.
  Across all 6 seeds, evolved holds in 3 of 6.
- Run C (`../evolution_c/cpu_aimnet2_ext_seeds2-5/`): evolved holds in 1, 2 and 1 of 4 seeds at budgets 200, 400 and 800
  (2, 3 and 1 of 6 including the pre-registered seeds). At 800 in s5 it is significantly worse than static
  (+0.009 *). **No-selection at 800 beats static on forces and on rel-E in 4 of 4 seeds** (forces −0.021 to
  −0.033 *, rel-E −0.018 to −0.025 *). Including the pre-registered seeds, it beats static on forces in 6 of 6.

**GPU design variants** (2 children per family, base-model children explore the energy weight, gate tolerates
10 % worse rel-E, base models never pruned; each knob alone; budget 1600): running at the time of writing; results
will be added here.

## Failures and limits, reported
- Run C's first GPU attempt is invalid (OOM children counted as stillborn). In the clean GPU rerun, the first launch deadlocked:
  a run with more parallel children than GPU slots blocked on itself (fixed in `scripts/evolve_c.py`, commit
  0b9939f). It was killed and rerun from cached children.
- Run C's summary first reported "holds at some budget" when a single seed held; the pre-registration requires every
  seed. This was fixed before any write-up.
- The harness has no Skeptic vetoes, so the "badly-wrong trusted" share is about 40 % in every arm (the full arm D
  pipeline: 0 of 96). The comparison is fair, but the absolute number is not the product's number.
- Two seeds per pre-registered condition give little power. The extra seeds are exploratory and run on the same eval
  split. Nothing was tuned on eval.
