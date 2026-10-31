# STATUS — CPU core (ablation / agents)

LABELS_FINAL ad5f14229a51679267d4fd9456feb589  results/oracle_labels_armD.json

- v3 split: pool/eval by connected component of shared endpoint geometry (37 test components),
  skeptic_ref/planted by formula (components kept whole). 0 shared endpoint geometries and
  0 near-duplicate endpoints (max|ΔR| < 1e-3 Å) across pool|eval and skeptic_ref|planted
  (before: 732 and 481 near-duplicate pairs). See results/split_report.json.
- The labels file is produced by arm D (audit_fraction 0.25, K = 40 x 5, seed 0). The v3 risk
  score is reporting-only: md5 re-checked after the final run (with --budgets 20,40,80):
  unchanged, ad5f14229a51679267d4fd9456feb589.
- v4 (risk feature chosen by grouped CV on own labels: sigma_F, CV AUROC 0.756 vs 0.752 for
  sigma_served) rerun with --budgets 20,40,80: labels md5 still ad5f14229a51679267d4fd9456feb589.
  Summary: results/RESULTS.md.
