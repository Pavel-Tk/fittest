# Run E: does growing the swarm without selection beat the static swarm? (pre-registered)

Written and committed **before** any run E seed was run (Sat Oct 31, about 20:05 EDT = Nov 1 00:05 UTC). Run E tests a
finding that showed up repeatedly in **exploratory** data: the run C *no-selection* arm. It fine-tunes one niche
specialist per family per breeding generation, keeps the newest one, and **never removes the base models**; there
is no gate, no clone check and no pruning. Across seeds 0–5 at budget 800 with three families, its
final-vs-static served force RMSE averaged −0.024 eV/Å (seed-level 95 % t-interval [−0.036, −0.012]; 5 of 6 seeds
significantly better, none worse). The evolved arm averaged −0.002 [−0.032, +0.028]. Those seeds suggested
the hypothesis, so they cannot confirm it. Run E uses new seeds only.

## Fixed design
- Engine, data, splits, triage, oracle, niches, specialists and eval scorer: exactly run C (`../evolution_c/PREREG.md`),
  code at the commit that adds this file. Arms: **static, noselect, evolved** (pre-registered run C settings; no
  run D knobs).
- Budget **800** (4 generations × 200 labels; 3 breeding generations).
- **New seeds 12–23** (12 seeds). Seeds 0–11 are not part of run E.
- **E-gpu (primary):** AIMNet2 + Orb + MACE specialists on Colab L4, seeds 12–17 on one session and 18–23 on another.
- **E-cpu (secondary):** AIMNet2 specialists only, CPU VM, same seeds.

## Primary outcome and decision rule (fixed now)
Per seed: Δ = noselect − static, served force RMSE on eval at the final generation, with the paired 95 % reaction
bootstrap CI as in run C. Seed-level aggregation: `scripts/seed_level.py` (mean of per-seed Δ with a two-sided 95 %
Student-t interval over seeds).

**"Growing the swarm beats the static swarm" is confirmed iff, in E-gpu with all 12 seeds:**
1. the seed-level 95 % t-interval of mean Δ force RMSE lies entirely below 0, **and**
2. at most 1 of 12 seeds is significantly worse than static on force RMSE, **and**
3. the seed-level 95 % t-interval of mean Δ ECE does not lie entirely above 0 (no calibration regression).

If fewer than 12 E-gpu seeds finish by 08:00 EST Nov 1, the rule is applied to the seeds that finished, and the
report says how many finished. With fewer than 8 seeds the result is reported as inconclusive.

## Secondary (reported, not decisive)
- Seed-level Δ rel-E MAE for noselect; the per-seed run C rule applied to noselect (`verdict_noselect` in
  `summary.json`).
- Head-to-head noselect vs evolved (seed-level), and the evolved arm vs static under the original rule.
- E-cpu: the same statistics.

The flagship verdicts for runs A, B, C and D do not change whatever run E shows. Run E can only support a
different, narrower claim: fine-tuned specialists **added** to a swarm that keeps its base models beat the static
swarm at the same DFT budget.
