# Fresh-frame check of runs E and F: results

Pre-registered rule and procedure: `PREREG_fresh_frames.md` (f817523) and its Amendment 1 (5fe14cb), both committed before
any fresh-frame number was computed. Fresh frames are the leak-free `planted` + `skeptic_ref` frames (Transition1x
**val** formulas, 2475 frames, 225 reactions). No evolution run has labelled, selected on or scored them. Code: e9fd169
(`scripts/evolve_c_fresh_predict.py`, `scripts/evolve_c.py run --also-score fresh`, `scripts/seed_level.py --split fresh`).

## Procedure as run
1. **Fresh predictions from checkpoints.** For every cached child (CPU: 59 for E, 36 for F; GPU: 89, 54, 90 and 54), the
   saved head state was loaded onto the same family model and its fresh frames were predicted with the same `predict()` as
   `scripts/evolve_c_offspring.py`. Output went to `<cache entry>.fresh.npz`; cache keys, specs and the cached `.npz` were not touched.
2. **Reruns.** The original commands (`~/evo_runs/re.sh`, `rf.sh`, `colab/e1.sh`, `e2.sh`, `f1.sh`, `f2.sh`) were rerun with the
   same seeds, out dirs and caches, plus `--also-score fresh`. The fresh scorer refits the population's buckets, τ
   and ledger with the same call as `serve_eval` (pool labels only). It only buckets, serves and scores the fresh frames,
   with the same metrics and a paired reaction bootstrap over the 225 fresh reactions (1000 resamples, shared by all runs of a
   budget). Nothing it computes feeds back into the run. The unit test `tests/test_evolution_c.py::test_fresh_scorer_is_read_only`
   checks that eval states, bootstrap samples, history, labels and child requests are identical with and without it.

## Reproduction checks (gate, done before reading any fresh number)
- **Children:** 0 children were retrained in any rerun (every child was a cache hit).
- **Eval numbers:** every numeric leaf of the committed `summary.json` and of `states`, `comparisons` and `served_by_niche`
  in every `b800*/generations.json` was compared with the rerun (`results/evolution_{e,f}/{cpu,gpu1,gpu2}`
  vs `fresh_{cpu,gpu1,gpu2}`). **Max |Δ| = 0 in all 6 reruns**, with no non-numeric mismatches. `history.json` and
  `family_tree.json` are byte-identical. Details are in each `fresh_*/repro_check.json`.
- **Checkpoint re-prediction:** 3 children per family per cache had their full pool + eval prediction list recomputed from the
  checkpoint and compared with the cached `.npz` (42 children). On CPU (AIMNet2) the match is exact (0). On GPU, max |ΔE| =
  1.5e-5 eV and max |ΔF| = 1.2e-5 eV/Å, which is float32 GPU noise and about 4 orders of magnitude below the effects
  below (`fresh_*/cache/fresh_verify.json`).

All runs pass, so no fresh result is flagged unreliable.

## Seed-level results (final generation, Δ = arm − static, eV/Å for force and eV for rel-E; 95 % t-interval over 12 seeds)
| run | arm − static | split | n | Δ force RMSE mean [95% t-CI] | sig better / worse | Δ rel-E MAE mean [CI] | Δ ECE mean [CI] | rule (CI < 0, ≤1 worse) |
|---|---|---|---|---|---|---|---|---|
| E-gpu | noselect | eval | 12 | -0.0181 [-0.0237, -0.0125] | 10 / 0 | -0.0260 [-0.0327, -0.0193] | -0.0084 [-0.0220, +0.0051] | **holds** |
| E-gpu | noselect | fresh | 12 | -0.0184 [-0.0262, -0.0107] | 12 / 0 | -0.0106 [-0.0179, -0.0034] | +0.0046 [-0.0145, +0.0237] | **holds** |
| E-gpu | evolved | eval | 12 | -0.0007 [-0.0170, +0.0157] | 6 / 4 | -0.0318 [-0.0371, -0.0264] | +0.0004 [-0.0171, +0.0178] | (fails; not decisive) |
| E-gpu | evolved | fresh | 12 | -0.0099 [-0.0255, +0.0056] | 7 / 4 | -0.0176 [-0.0227, -0.0124] | -0.0301 [-0.0532, -0.0070] | (fails; not decisive) |
| F-gpu | evolved | eval | 12 | -0.0257 [-0.0302, -0.0212] | 12 / 0 | -0.0392 [-0.0443, -0.0340] | -0.0208 [-0.0278, -0.0139] | **holds** |
| F-gpu | evolved | fresh | 12 | -0.0263 [-0.0336, -0.0190] | 12 / 0 | -0.0251 [-0.0323, -0.0178] | -0.0372 [-0.0537, -0.0206] | **holds** |
| E-cpu | noselect | eval | 12 | -0.0180 [-0.0273, -0.0087] | 11 / 1 | -0.0151 [-0.0209, -0.0093] | -0.0134 [-0.0267, -0.0001] | **holds** |
| E-cpu | noselect | fresh | 12 | -0.0190 [-0.0333, -0.0047] | 9 / 2 | -0.0108 [-0.0171, -0.0044] | +0.0152 [-0.0055, +0.0358] | **fails** |
| E-cpu | evolved | eval | 12 | -0.0076 [-0.0151, -0.0001] | 4 / 0 | -0.0123 [-0.0232, -0.0013] | +0.0018 [-0.0138, +0.0174] | (holds; not decisive) |
| E-cpu | evolved | fresh | 12 | -0.0076 [-0.0181, +0.0029] | 4 / 0 | -0.0085 [-0.0167, -0.0003] | -0.0047 [-0.0185, +0.0092] | (fails; not decisive) |
| F-cpu | evolved | eval | 12 | -0.0165 [-0.0232, -0.0097] | 10 / 0 | -0.0260 [-0.0319, -0.0201] | +0.0015 [-0.0174, +0.0203] | **holds** |
| F-cpu | evolved | fresh | 12 | -0.0147 [-0.0241, -0.0053] | 9 / 0 | -0.0202 [-0.0256, -0.0148] | +0.0052 [-0.0218, +0.0321] | **holds** |

The rule (from the addendum): the CI of mean Δ force RMSE lies entirely below 0, and at most 1 of 12 seeds is significantly worse. The
eval rows reproduce the committed seed-level analyses. The E-evolved rows are reported for reference; they are not a rule arm.

## Verdicts
- **F holds on fresh frames (primary, F-gpu): YES.** Mean Δ force RMSE is −0.0263 [−0.0336, −0.0190] eV/Å, 12/12 seeds
  significantly better and 0 worse. This matches eval (−0.0257). Rel-E and ECE also improve.
- **E holds on fresh frames (primary, E-gpu noselect − static): YES.** Mean Δ force RMSE is −0.0184 [−0.0262, −0.0107]
  eV/Å, 12/12 seeds significantly better and 0 worse (eval: −0.0181, 10/0). The mean ECE change, +0.0046 [−0.0145, +0.0237], is not
  significant (2 seeds significantly better, 3 worse). The addendum's rule has no ECE condition; it is shown for completeness. The
  seed-level numbers are in `fresh_seed_level.json`.
- **CPU (secondary).** F-cpu holds: −0.0147 [−0.0241, −0.0053], 9 better / 0 worse. E-cpu noselect does **not** meet the rule:
  the CI is below 0 (−0.0190 [−0.0333, −0.0047]), but 2 of 12 seeds are significantly worse (the limit is 1). Its ECE also leans worse
  (+0.0152 [−0.0055, +0.0358]; 5 seeds significantly worse).

The force gains measured on eval carry over to chemistry no evolution run has seen. The effect size is the same
on fresh frames for F (GPU and CPU) and for E-gpu. The secondary E-cpu result is weaker on fresh frames than on eval.
