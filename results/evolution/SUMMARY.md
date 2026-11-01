# Evolution: does a swarm that breeds and selects its own models beat the static swarm?

**Verdict under the pre-registered rule: no.** Runs A, B and C each fail. A post-hoc fix (never prune the base
models, soften the gate) held in 5 of 6 exploratory seeds at budget 800 (CPU). Its pre-registered confirmation on new
seeds (run D) fell short: on CPU (primary) it held in 4 of 6 new seeds where 5 were required (none worse than
static); on GPU with three families (secondary) it held in 4 of 6 and was significantly worse than static in 2. **A narrower claim did confirm (run E, pre-registered, 12 new seeds): growing the swarm with fine-tuned
specialists while keeping the base models, without selection, beats the static swarm on served force error
(−0.018 eV/Å [−0.024, −0.013], 0 of 12 seeds worse) and on relative energies.** **And once the gate tests only forces (run F, pre-registered after a diagnosis, 12 new seeds), the full
evolutionary loop beats the static swarm in 12 of 12 seeds (−0.026 eV/Å [−0.030, −0.021]), passing even the original
every-seed rule. That fix was chosen after reading earlier eval results; a pre-registered check on never-read frames
is reported in the run F section: it passed (fresh, never-read Transition1x val formulas: −0.026 [−0.034, −0.019],
12 of 12 seeds better).** Below, every pre-registered run
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

Run B was designed after run A's outcome was known and its methods were committed before run B ran
(`METHODS.md`). Run C was pre-registered before it ran (`PREREG.md`, commit 957a91d, 13:16 EDT). Code commits
between the pre-registration and the faithful GPU run (751d4ca OOM retry, 57a6391/ceba4db `--families` and
opt-in exploratory knobs whose defaults equal PREREG, 0b9939f deadlock fix) do not change the design. Run C's first GPU attempt is **invalid** and is not quoted: CUDA
out-of-memory failures were counted as stillborn children (`../evolution_c/partial_run1/`). The clean GPU rerun
(two Colab L4 sessions) is the faithful run C. C′ was run first on the CPU VM while no GPU was available; it
deviates from the pre-registration by breeding AIMNet2 specialists only and is reported for completeness.

### Final generation vs static, same budget (paired 95 % CI; * = excludes 0)

| run / seed | served force RMSE Δ (eV/Å) | served rel-E MAE Δ (eV) | ECE Δ | offspring alive | failed criterion |
|---|---|---|---|---|---|
| A s0 | +0.010 [+0.008, +0.013] * (worse) | −0.007 [−0.011, −0.002] * | +0.004 n.s. | none (population = static's) | no offspring; forces worse ¹ |
| A s1 | −0.011 [−0.017, −0.005] * | **+0.056 [+0.044, +0.068] *** (worse) | **+0.069 *** (worse) | g1_a (MACE replaced) | rel-E and ECE regress; not better than gen 0 |
| B s0 | −0.001 [−0.007, +0.005] | −0.027 [−0.039, −0.014] * | −0.007 n.s. | g1_a (AIMNet2 replaced) | forces not better than static |
| B s1 | −0.024 [−0.031, −0.016] * | −0.029 [−0.042, −0.017] * | +0.008 n.s. | g2_b | — (holds) |
| C b200 s0 | **+0.034 [+0.026, +0.041] *** (worse) | −0.011 [−0.022, −0.001] * | **+0.116 [+0.075, +0.159] *** (worse) | g3_orb_flagged (base Orb pruned) | forces worse (vs static and gen 0); ECE regresses |
| C b200 s1 | −0.027 [−0.031, −0.023] * | −0.008 [−0.013, −0.004] * | no sig. regression | g2_aimnet2_flagged | — (holds) |
| C b400 s0 | **+0.019 [+0.012, +0.026] *** (worse) | −0.025 [−0.035, −0.016] * | no sig. regression | g1_mace, g3_orb (base MACE, Orb pruned) | forces worse (vs static and gen 0) |
| C b400 s1 | −0.016 [−0.022, −0.009] * | −0.018 [−0.027, −0.008] * | no sig. regression | g1_aimnet2, g2_orb, g3_mace | not better than own gen 0 |
| C b800 s0 | −0.017 [−0.023, −0.012] * | −0.044 [−0.055, −0.034] * | no sig. regression | g1_orb, g3_aimnet2, g3_orb (all three bases pruned) | — (holds) |
| C b800 s1 | **+0.011 [+0.003, +0.019] *** (worse) | −0.028 [−0.040, −0.017] * | no sig. regression | g1_orb_flagged (base Orb pruned) | forces worse (vs static and gen 0) |
| C′ b200 / b400 s1 | −0.027 * / −0.036 * | −0.008 * / −0.006 * | no sig. regression | one AIMNet2 specialist | — (holds) ² |
| C′ s0 (all budgets), b800 s1 | 0 | 0 | 0 | none (every child failed the gate) | no offspring |

¹ **The CIs cover eval resampling only, not run-to-run noise.** A s0 ends with exactly the static population, yet
differs "significantly" (+0.010 forces, −0.007 rel-E): the brief life of child g1_c changed which frames were
labelled, so the Calibrator's ledger serves AIMNet2 alone on 225 of 1485 frames instead of the AIMNet2+Orb mean.
The static swarm itself moves 0.202 → 0.220 eV/Å between seeds and 0.202–0.233 across generations. Effects below
about 0.02 eV/Å should be read as within run noise.
² C′ b200 s1 is numerically identical to C b200 s1 (the same AIMNet2 child; the Orb and MACE children died), so
it is not an independent confirmation.

Sources: `summary_tables.md` and `verdict.json` here (run A), `../evolution_aimnet/` (run B), and
`../evolution_c/summary.json` and `summary_table.md` (run C, GPU), `../evolution_c/cpu_aimnet2_run/` (C′). Figures: `../fig/fig7_evolution_run{A,B}_s{0,1}.png`,
`../fig/fig8_family_tree_run{A,B}_s{0,1}.png` and `../evolution_c/fig9_budget.png`.

### Controls (pre-registered)
- **Run B, "newest" control** (the newest child always replaces the previous one; no gate, no pruning; parents are
  still picked by fitness, so it is not selection-free). Forces vs static: s0 −0.003 n.s., s1 −0.027 *. rel-E:
  s0 −0.030 *, s1 −0.017 *. ECE regresses significantly in both pre-registered seeds (+0.056 *, +0.044 *), but not
  in any of the exploratory seeds 2–5, so this is not a reliable difference from the evolved arm.
- **Run C, no-selection arm at budget 800** (every trained child admitted; base models kept). GPU, three families:
  forces vs static s0 −0.006 n.s., s1 −0.041 *; rel-E s0 −0.029 *, s1 −0.028 *. C′ (AIMNet2 only): forces s0
  −0.016 *, s1 −0.039 *; rel-E s0 +0.013 n.s., s1 −0.014 *.
- **Run C, evolved vs no-selection (pre-registered head-to-head, GPU, budget 800).** Splits by seed: s0 evolved is
  better (forces −0.011 [−0.015, −0.008] *, rel-E −0.015 *); s1 no-selection is better (forces +0.052
  [+0.047, +0.056] * for evolved). No-selection differs from evolved in several ways at once (no gate, no clone
  check, no pruning, base models always kept), so this control cannot isolate which one matters.

### Secondary outcome: relative energies (pre-registered as secondary for run C only)
Every evolved run of run C lowers served reaction-relative energy error vs static, significantly in **6 of 6**
(−0.008 to −0.044 eV), including the runs where forces got worse. Run B: 2 of 2 (−0.027, −0.029). Run A: 1 of 2
(s1 is +0.056, worse; s0's −0.007 comes from a population identical to static, see ¹). This is not evidence for
selection: children in B and C are trained on a rel-E loss and gated on rel-E, so the gain is built in, and the
no-selection arm gains as much (−0.029 *, −0.028 *). In C′ (AIMNet2 only) evolved rel-E is significantly better in
only 2 of 6 cells.

## What we learned
1. **The gate kills most children, mostly on reaction energy.** In GPU run C, 36 of 54 children were stillborn,
   35 of them on reactant-relative energy. In run A, 15 of 18 MACE LoRA children were stillborn; several also failed
   on force and the ALCHEMI check, and only E:F 10 and 30 children passed. In C′ (AIMNet2 only), seed 0 kept no
   child at any budget. The gate is small: at budget 200 it decides on about 15 frames from 4 reactions.
2. **Why the evolved swarm sometimes gets worse is not established.** In the three GPU run C cells that are
   significantly worse than static on forces (b200 s0, b400 s0, b800 s1), a base model was pruned. But a base model
   was also pruned in three of the five pre-registered runs that pass the rule (B s1; C b800 s0, with all three
   bases pruned; C′ b400 s1), and in exploratory C′ s3. The worse runs are worse both inside the flagged niche (served force RMSE 0.308–0.310 vs
   static 0.297) and outside it (0.211–0.214 vs 0.171–0.204). "Selection prunes the wrong members" is a hypothesis,
   not a finding.
3. **Surviving children can help, but it is a coin flip across seeds** (holds: B s1, C b200 s1, C b800 s0,
   C′ b200/b400 s1). Both pre-registered seed-1 wins at budget 200 (B s1, C b200 s1) are against a weak static
   seed (0.220 eV/Å). In absolute terms, B s1's evolved swarm (0.197) is only 0.005 below seed 0's static (0.202).
4. **Fine-tuning without selection looks more consistent, but the evidence is mixed and confounded.** In the
   pre-registered head-to-head the two arms split by seed (above). In the exploratory AIMNet2-only runs it beats
   static on forces in 6 of 6 seeds at budget 800, but at budgets 200/400 it is n.s. in some seeds and makes rel-E
   significantly worse in seed 0 (below). It changes several things at once, so it does not show that selection is
   the weak link.
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

**No-selection at budgets 200 and 400** (`../evolution_c/cpu_aimnet2_noselect_lowbudget/`, AIMNet2 only, seeds 0–5):
forces better than static in 4 of 6 seeds at 200 (s2, s5 n.s.) and 5 of 6 at 400 (s5 n.s.). rel-E better in 4 of 6
at 200 and 5 of 6 at 400, but **significantly worse in s0 at both** (+0.023 *, +0.017 *).

**Design variants: protect the base models, soften the gate** (all post hoc, chosen after seeing runs A–C):
| variant | where | seeds | holds at 400 | holds at 800 | worse than static on forces |
|---|---|---|---|---|---|
| base models never pruned (`--protect-base`) | CPU, AIMNet2 | 0–5 | 2 of 6 | 2 of 6 | 0 of 12 |
| + gate tolerates 10 % worse rel-E (`--gate-tol 0.1`) | CPU, AIMNet2 | 0–5 | 3 of 6 | **5 of 6** | 0 of 12 |
| + 2 children per family, base children explore energy weight and lr | L4, 3 families | 0–1 | **2 of 2** | 1 of 2 | 0 of 4 |

Sources: `../evolution_c/expl_cpu_protectbase/`, `../evolution_c/expl_cpu_protect_gatetol/`,
`../evolution_c/expl_gpu_v2_combined/`. With base models protected, every run whose children all failed the gate
equals static (Δ = 0). Every other run except GPU b800 s0 (−0.003 n.s.)
improved forces significantly (−0.005 to −0.044 eV/Å), and rel-E in all but one cell (b400 s0
with the gentle gate: +0.003 n.s.). The pattern suggests that pruning base models causes the harm and the strict gate
causes the missed gains. Because these designs were chosen from these seeds, that is a hypothesis. Run D tests it.
The one-knob GPU runs below already cut against the first half: with only the softer gate, all three base models
were pruned in both seeds and forces still improved most.

**One knob at a time** (L4, 3 families, budget 800, seeds 0–1; `../evolution_c/expl_gpu_ablation_*`). Forces Δ vs static
(eV/Å): protect-base −0.010 * / −0.018 *; gate-tol 0.1 −0.029 * / −0.038 *; explore-base (energy weight and lr) + 2 children
+0.000 n.s. / −0.035 *; 3 children per family −0.011 * / −0.036 * (without explore-base, siblings bred from a base
model share the recipe and differ only in training seed). Every knob alone holds in seed 1, three of four also in seed 0,
and none is worse than static. Pre-registered run C at the same budget and seeds: −0.017 * / +0.011 * (worse).
Two seeds per knob cannot rank the knobs.

**Run C extension, seeds 2–5, budget 800, three families** (L4, same pre-registered design; exploratory because of the
extra seeds; `../evolution_c/expl_gpu_ext_seeds2-5/`): evolved holds in 2 of 4 (s3 −0.030 *, s5 −0.032 *) and is
**significantly worse than static in 2 of 4** (s2 +0.038 *, s4 +0.017 *). No-selection beats static on forces
(−0.020 to −0.027 *) and on rel-E (−0.022 to −0.041 *) in 4 of 4. Across all six three-family seeds at 800, evolved
holds in 3 of 6 and is worse in 3 of 6. No-selection beats static on forces in 5 of 6 (s0 n.s.) and on rel-E in 6 of 6.

A third GPU job (the run C extension at budgets 200/400 and a budget-1600 run) was lost with its Colab session before
any result was downloaded. It is not reported.

Why evolved runs sometimes get worse than static (exploratory diagnosis): see `DIAGNOSIS.md`.

## Run D: pre-registered confirmation on new seeds (`../evolution_d/PREREG.md`)
Design: run C plus `--protect-base --gate-tol 0.1`, budget 800, **new seeds 6–11**. D-cpu (primary) breeds
AIMNet2 specialists only, as in C′; D-gpu (secondary) breeds all three families. The pre-registration was committed
(3238b18) before any run D seed ran. **Confirmed iff the rule holds in ≥ 5 of 6 seeds of D-cpu and no seed is
significantly worse than static.**

| seed | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|
| D-cpu forces Δ vs static (eV/Å) | −0.028 * | −0.015 * | −0.008 * | −0.017 * | 0 (no child passed) | −0.005 * |
| D-cpu rel-E Δ (eV) | −0.025 * | −0.013 * | −0.024 * | −0.018 * | 0 | −0.014 * |
| D-cpu rule | holds | holds | no (not better than own gen 0) | holds | no (no offspring) | holds |

**D-cpu: the rule holds in 4 of 6 seeds, below the pre-registered 5, so the fix is NOT confirmed.** No seed is
worse than static, and forces are better or equal in 6 of 6. Under the original every-seed rule, both D-cpu and
D-gpu fail. Sources: `../evolution_d/cpu/summary.json` and
`summary_table.md`.

| seed | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|
| D-gpu forces Δ vs static (eV/Å) | −0.014 * | −0.022 * | −0.019 * | **+0.015 *** (worse) | **+0.005 *** (worse) | −0.010 * |
| D-gpu rel-E Δ (eV) | −0.036 * | −0.031 * | −0.038 * | −0.020 * | −0.027 * | −0.031 * |
| D-gpu rule | holds | holds | holds | no | no | holds |

**D-gpu (three families, secondary): holds in 4 of 6, and 2 seeds are significantly worse than static on forces,
although base models were protected** (`../evolution_d/gpu/`). In the three-family setting, protecting the base
models did not prevent harm, which further weakens the hypothesis above. In all 6 seeds, rel-E is
better than static.

**Run D verdict: not confirmed** (D-cpu 4 of 6 < 5; D-gpu 4 of 6 with 2 seeds worse). What recurs is narrower:
evolved swarms whose children survive usually lower reaction-relative energy error (not in run A s1 or most C′
cells), and at budget 800 no-selection beat static on forces more often than the evolved arm (exploratory; not
tested in run D). Neither was the pre-registered claim. Run E (`../evolution_e/PREREG.md`) tests the second on new
seeds.

## Run E: growing the swarm without selection (pre-registered, `../evolution_e/PREREG.md`) — CONFIRMED
Design: the run C no-selection arm (each breeding generation fine-tunes one niche specialist per family and keeps the
newest; base models are never removed; no gate, no clone check, no pruning) vs static, plus the evolved arm, budget 800,
**new seeds 12–23**. The pre-registration was committed (2f8114e, 00:07 UTC) before any run E seed ran. E-gpu (three
families, L4) is primary, E-cpu (AIMNet2 specialists only) secondary. Rule: seed-level 95 % t-interval of mean Δ
force RMSE (arm − static) entirely below 0, at most 1 of 12 seeds significantly worse, and no ECE regression at the
seed level.

| | mean Δ force RMSE (eV/Å) [95 % t-CI over seeds] | seeds sig. better / worse | mean Δ rel-E (eV) | mean Δ ECE | run-C rule per seed |
|---|---|---|---|---|---|
| **E-gpu no-selection** | **−0.018 [−0.024, −0.013]** | 10 / **0** | −0.026 [−0.033, −0.019] | −0.008 [−0.022, +0.005] | holds in 10 of 12 |
| E-gpu evolved (same seeds) | −0.001 [−0.017, +0.016] | 6 / 4 | −0.032 [−0.037, −0.026] | +0.000 [−0.017, +0.018] | holds in 6 of 12 |
| E-cpu no-selection | −0.018 [−0.027, −0.009] | 11 / 1 | −0.015 [−0.021, −0.009] | −0.013 [−0.027, −0.000] | — |
| E-cpu evolved | −0.008 [−0.015, −0.000] | 4 / 0 | −0.012 [−0.023, −0.001] | +0.002 [−0.014, +0.017] | — |

**Verdict: confirmed.** All three conditions hold in E-gpu. Adding fine-tuned niche specialists to a swarm that
keeps its base models lowers served force error vs the static swarm at the same DFT budget (−0.018 eV/Å, about 9 % of
static's ~0.21) and lowers reaction-relative energy error. The secondary CPU replication agrees. On the same seeds,
the evolved arm (gate + fitness pruning) shows no force gain on average and is worse than static in 4 of 12
seeds. Sources: `../evolution_e/gpu{1,2}/summary.json`, `../evolution_e/seed_level_gpu_{noselect,evolved}.json`,
`../evolution_e/cpu/`.

What this does and does not show: the swarm improves by **growing** (variation plus keeping parents), not by
**selection**: our gate and fitness pruning removed the gain. This was not the flagship claim, which needed the
evolved arm to win and did not. Children are trained only on labels drawn from the same 800-label budget, so both
arms use the same DFT cost.

## Run F: gate on forces only (pre-registered, `../evolution_f/PREREG.md`) — CONFIRMED on eval
Design: exactly run C (gate + fitness pruning; base models can be pruned) except that the gate's
reaction-relative-energy check is switched off (`--gate-tol 1e9`). Budget 800, **new seeds 24–35**. The
pre-registration was committed (94b4376, 00:23 UTC) before any run F seed ran. Motivation: the exploratory diagnosis
(`DIAGNOSIS.md`) found that the rel-E check on the small select set does not predict eval (r = −0.15) and mostly
rejects the AIMNet2 children with the largest force gains, and that harmed runs are those with no AIMNet2
specialist.

| | mean Δ force RMSE vs static (eV/Å) [95 % t-CI] | seeds sig. better / worse | mean Δ rel-E (eV) | mean Δ ECE | run-C rule per seed |
|---|---|---|---|---|---|
| **F-gpu evolved (3 families)** | **−0.026 [−0.030, −0.021]** | **12 / 0** | −0.039 [−0.044, −0.034] | −0.021 [−0.028, −0.014] | **holds in 12 of 12** |
| F-cpu evolved (AIMNet2 only) | −0.017 [−0.023, −0.010] | 10 / 0 | −0.026 [−0.032, −0.020] | +0.002 [−0.017, +0.020] | holds in 10 of 12 |

**Verdict: confirmed under the pre-registered rule.** With a force-only gate, the full evolutionary loop
(breeding, gate, fitness pruning) beats the static swarm in every one of 12 new seeds. It even passes the original
flagship rule, which required every seed. It also beats run E's no-selection arm on average (−0.026 vs −0.018, on
different seeds).

**Caveat: adaptivity to the eval split.** Every run is scored on the same 1485 eval frames. The run F design came from a
diagnosis that read earlier runs' eval results, so new seeds do not rule out that the fix suits this eval split. A
pre-registered check on frames no evolution run has ever read (Transition1x val formulas, 2475 frames,
`../evolution_f/PREREG_fresh_frames.md`, committed f817523 before scoring) **passed** (`../evolution_f/FRESH_FRAMES.md`). Each final swarm was replayed from its cached children,
the eval numbers reproduced exactly (max |Δ| = 0 in all 6 reruns), and the fresh frames were then served and scored:

| on fresh frames (2475 frames, 225 reactions, new formulas) | mean Δ force RMSE vs static [95 % t-CI] | seeds better / worse | rule |
|---|---|---|---|
| **F-gpu evolved (force-only gate)** | **−0.026 [−0.034, −0.019]** | **12 / 0** | **holds** |
| **E-gpu grow (no selection)** | **−0.018 [−0.026, −0.011]** | **12 / 0** | **holds** |
| F-cpu evolved (secondary) | −0.015 [−0.024, −0.005] | 9 / 0 | holds |
| E-cpu grow (secondary) | −0.019 [−0.033, −0.005] | 9 / 2 | **fails** (2 seeds worse; at most 1 allowed) |
| E-gpu evolved, original gate (reference) | −0.010 [−0.026, +0.006] | 7 / 4 | — |

The gains carry over to chemistry no run has seen, at the same size as on eval (F: −0.026 on both).

**Run E controls (exploratory).** Sham children (learning rate 0, identical to their parent; everything else
equal), CPU, seeds 12–17: forces −0.004 [−0.013, +0.005] (n.s.) vs −0.018 for real children, so run E's force gain
comes from learning. But sham rel-E is −0.011 [−0.016, −0.006]: part of the rel-E gain in runs E/F is a pipeline effect
(different labels and routing), not learning (`../evolution_e/expl_cpu_sham/`). Growing across budgets (CPU,
seeds 12–17): forces −0.017 / −0.026 / −0.017 eV/Å at 200 / 400 / 1600 labels, 6 of 6 seeds better at each, with
no trend (`../evolution_e/expl_cpu_grow_budget/`).

## Failures and limits, reported
- Run C's first GPU attempt is invalid (OOM children counted as stillborn). In the clean GPU rerun, the first launch deadlocked:
  a run with more parallel children than GPU slots blocked on itself (fixed in `scripts/evolve_c.py`, commit
  0b9939f). It was killed and rerun from cached children.
- Run C's summary first reported "holds at some budget" when a single seed held; the pre-registration requires every
  seed. This was fixed before any write-up.
- The harness has no Skeptic vetoes. Its "badly-wrong trusted" share (badly wrong among trusted frames) ranges from
  6 % to 57 % across arms and runs, and it is not comparable to arm D's 0 of 96 silent failures in `../RESULTS.md`,
  which is a different metric.
- Paired CIs cover eval resampling only (see ¹). Two seeds per pre-registered condition give little power. The extra seeds are exploratory and run on the same eval
  split. Nothing was tuned on eval.
