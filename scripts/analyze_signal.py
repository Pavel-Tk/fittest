"""Is cross-model disagreement a signal of error? Label-free signal vs DFT truth.

Produces ``results/signal.json`` and ``results/signal.png``:
  (a) per-frame force RMSE vs DFT by role (reactant / product / path / TS) per model,
  (b) disagreement vs actual error, heterogeneous committee vs AIMNet2 seed ensemble,
  (c) calibration: spread / error ratio (an ensemble that "collapses" sits far below 1),
  (d) AUROC(spread → wrong) at several error thresholds with bootstrap CIs over reactions.

    python3 scripts/analyze_signal.py --frames data/frames.pkl --preds data/preds
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trustswarm.cache import load_frames, load_predictions  # noqa: E402
from trustswarm.signals import force_spread  # noqa: E402

HET = ["mace_mpa0", "aimnet2", "orb"]
SEED = ["aimnet2_m0", "aimnet2_m1", "aimnet2_m2", "aimnet2_m3"]
ROLES = ["reactant", "product", "path", "ts"]
THRESHOLDS = [0.15, 0.25, 0.4]


def per_frame(frames, preds, models):
    sig = np.array([force_spread(np.stack([preds[m].forces[f.uid] for m in models]))[0] for f in frames])
    return sig


def auroc_ci(score, wrong, groups, n_boot=1000, seed=0):
    from sklearn.metrics import roc_auc_score

    if wrong.all() or not wrong.any():
        return None
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx_by = {g: np.flatnonzero(groups == g) for g in uniq}
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([idx_by[g] for g in rng.choice(uniq, size=len(uniq))])
        if wrong[idx].all() or not wrong[idx].any():
            continue
        vals.append(roc_auc_score(wrong[idx], score[idx]))
    return {"auroc": float(roc_auc_score(wrong, score)),
            "ci95": [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="data/frames.pkl")
    ap.add_argument("--preds", default="data/preds")
    ap.add_argument("--out", default="results")
    ap.add_argument("--target", default="aimnet2", help="model whose error is the target")
    a = ap.parse_args()

    from scipy.stats import spearmanr

    frames = [f for f in load_frames(a.frames) if f.split in ("pool", "eval")]  # held-out T1x test
    avail = {m: load_predictions(os.path.join(a.preds, f"{m}.npz"))
             for m in set(HET + SEED + [a.target]) if os.path.exists(os.path.join(a.preds, f"{m}.npz"))}
    het = [m for m in HET if m in avail]
    seed = [m for m in SEED if m in avail]
    err = {m: np.array([np.sqrt(((p.forces[f.uid] - f.F) ** 2).mean()) for f in frames]) for m, p in avail.items()}
    roles = np.array([f.role for f in frames])
    groups = np.array([f.reaction for f in frames])
    y = err[a.target]
    s_het, s_seed = per_frame(frames, avail, het), per_frame(frames, avail, seed)

    out = {"n_frames": len(frames), "n_reactions": int(len(np.unique(groups))), "target": a.target,
           "het_models": het, "seed_models": seed,
           "error_by_role": {m: {r: float(np.median(err[m][roles == r])) for r in ROLES} for m in err},
           "spearman": {"het": float(spearmanr(s_het, y)[0]), "seed": float(spearmanr(s_seed, y)[0])},
           "median_spread": {"het": float(np.median(s_het)), "seed": float(np.median(s_seed))},
           "median_error_target": float(np.median(y)),
           "auroc": {str(t): {"het": auroc_ci(s_het, y > t, groups), "seed": auroc_ci(s_seed, y > t, groups),
                              "frac_wrong": float((y > t).mean())} for t in THRESHOLDS}}
    # silent failures: badly wrong (> 0.4 eV/Å) but spread below the median spread of its own ensemble
    bad = y > 0.4
    out["silent_failures"] = {"het": float((bad & (s_het < np.median(s_het))).sum() / max(bad.sum(), 1)),
                              "seed": float((bad & (s_seed < np.median(s_seed))).sum() / max(bad.sum(), 1)),
                              "n_bad": int(bad.sum())}
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "signal.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    blue, orange, grey = "#1f5fa8", "#c4561b", "#8a8f98"
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    w = 0.8 / len(err)
    for k, m in enumerate(sorted(err)):
        ax[0].bar(np.arange(len(ROLES)) + k * w, [out["error_by_role"][m][r] for r in ROLES], w, label=m)
    ax[0].set_xticks(np.arange(len(ROLES)) + 0.4 - w / 2, ROLES)
    ax[0].set_ylabel("median force RMSE vs DFT (eV/Å)")
    ax[0].set_title("(a) Foundation MLIPs fail at transition states")
    ax[0].legend(fontsize=7, frameon=False)

    ax[1].scatter(y, s_het, s=4, alpha=0.35, color=orange, label=f"heterogeneous ({'+'.join(het)})")
    ax[1].scatter(y, s_seed, s=4, alpha=0.35, color=blue, label="AIMNet2 seed ensemble ×4")
    lim = float(np.quantile(np.concatenate([y, s_het]), 0.995))
    ax[1].plot([0, lim], [0, lim], color=grey, lw=1, ls="--", label="spread = error")
    ax[1].set_xlim(0, lim); ax[1].set_ylim(0, lim)
    ax[1].set_xlabel(f"{a.target} force RMSE vs DFT (eV/Å)")
    ax[1].set_ylabel("committee force spread σ_F (eV/Å)")
    ax[1].set_title(f"(b) Spearman ρ: het {out['spearman']['het']:.2f}, seed {out['spearman']['seed']:.2f}")
    ax[1].legend(fontsize=7, frameon=False, markerscale=3)

    xs = np.arange(len(THRESHOLDS))
    for k, (key, col) in enumerate((("het", orange), ("seed", blue))):
        vals = [out["auroc"][str(t)][key] for t in THRESHOLDS]
        mid = [v["auroc"] if v else np.nan for v in vals]
        lo = [v["auroc"] - v["ci95"][0] if v else 0 for v in vals]
        hi = [v["ci95"][1] - v["auroc"] if v else 0 for v in vals]
        ax[2].bar(xs + (k - 0.5) * 0.38, mid, 0.38, yerr=[lo, hi], color=col, capsize=3,
                  label="heterogeneous" if key == "het" else "seed ensemble")
    ax[2].set_xticks(xs, [f">{t} eV/Å\n({out['auroc'][str(t)]['frac_wrong']:.0%} wrong)" for t in THRESHOLDS])
    ax[2].set_ylim(0.5, 1.0); ax[2].set_ylabel("AUROC (spread → wrong), 95% CI")
    ax[2].set_title("(c) AUROC at three error thresholds")
    ax[2].legend(fontsize=7, frameon=False)
    fig.suptitle(f"Held-out Transition1x test reactions: {out['n_frames']} frames, {out['n_reactions']} reactions",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "signal.png"), dpi=160)


if __name__ == "__main__":
    main()
