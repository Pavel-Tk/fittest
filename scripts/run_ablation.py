"""Run the TrustSwarm ablation (arms A-D, same oracle budget) on cached predictions.

    python scripts/run_ablation.py --frames data/frames.pkl --preds data/preds --out results/ \
        --budget-per-round 40 --rounds 5 --seed 0 [--budgets 20,40,80]

Writes to --out:
  ablation.json, ablation.png      all arms, all metrics, failures (one figure)
  blackboard_armD.jsonl            arm D's audit log (arrays summarised; kept small)
  trust_snapshot.json              arm D's final Calibrator state (read by the trust service)
  ablation_budget.json/.png        only with --budgets: the sweep over per-round budgets
The Curator's repair request is written to ``oracle_labels_armD.json``. Fine-tuning is replayed from ``{preds}/mace_lora_curated.npz`` if present (``--no-finetune``
disables it).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path
from typing import Any

from trustswarm.ablation import (AblationConfig, plot_ablation, plot_budget, run_ablation,
                                 run_budget_sensitivity)
from trustswarm.ablation import default_finetune
from trustswarm.agents import NullFinetune
from trustswarm.cache import load_frames


class RecordingFinetune:
    """Write the Curator's first repair request (model + labelled uids) for the GPU fine-tune.

    The Curator decides *which* model to repair and *which* oracle labels to use; this
    records that decision to ``oracle_labels_armD.json`` so ``scripts/finetune.py`` trains on
    exactly those frames, then delegates (replay of a finished fine-tune, or no-op).
    """

    def __init__(self, inner, path: Path) -> None:
        self.inner, self.path, self.recorded = inner, path, False

    def __call__(self, model_name, labelled_frames):
        if not self.recorded:
            self.path.write_text(json.dumps({"model": model_name,
                                             "uids": [f.uid for f in labelled_frames]}, indent=1))
            self.recorded = True
        return self.inner(model_name, labelled_frames)

    def bind_held_out(self, uids) -> None:
        bind = getattr(self.inner, "bind_held_out", None)
        if bind is not None:
            bind(uids)

SUMMARY_KEYS = ("auroc_pool", "auroc_eval", "recall_at_k", "planted_caught", "audit_hit_rate",
                "eval_force_rmse_post", "ece_eval", "silent_failures_eval")


def _fmt(metric: dict[str, Any] | None) -> str:
    if not metric or metric.get("value") is None:
        return "n/a"
    ci = metric.get("ci")
    return f"{metric['value']:.3f} [{ci[0]:.2f},{ci[1]:.2f}]" if ci else f"{metric['value']:.3f}"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", default="data/frames.pkl")
    ap.add_argument("--preds", default="data/preds")
    ap.add_argument("--out", default="results")
    ap.add_argument("--budget-per-round", type=int, default=40)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument("--max-planted", type=int, default=40, help="cap on planted cases (<=0: all)")
    ap.add_argument("--no-finetune", action="store_true", help="use NullFinetune")
    ap.add_argument("--audit-fraction", type=float, default=0.25,
                    help="arm D: share of each round's budget for Skeptic audits of trusted systems")
    ap.add_argument("--budgets", default="",
                    help="comma-separated per-round budgets for a sensitivity sweep, e.g. 20,40,80")
    ap.add_argument("--budget-bootstrap", type=int, default=200,
                    help="bootstrap resamples per budget in the sweep")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = AblationConfig(budget_per_round=args.budget_per_round, rounds=args.rounds,
                         seed=args.seed, n_bootstrap=args.n_bootstrap,
                         max_planted=args.max_planted if args.max_planted > 0 else None,
                         audit_fraction=args.audit_fraction)
    inner = NullFinetune() if args.no_finetune else default_finetune(args.preds, cfg)
    finetune = RecordingFinetune(inner, out / "oracle_labels_armD.json")
    frames = load_frames(args.frames)
    t0 = time.perf_counter()
    results = run_ablation(frames, args.preds, cfg, finetune_fn=finetune,
                           blackboard_path=out / "blackboard_armD.jsonl",
                           snapshot_path=out / "trust_snapshot.json")
    results["runtime_s"] = round(time.perf_counter() - t0, 1)
    (out / "ablation.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    if results.get("arms"):
        plot_ablation(results, out / "ablation.png")

    print(f"{'arm':<15}" + "".join(f"{k[:20]:<22}" for k in SUMMARY_KEYS))
    for name, e in results["arms"].items():
        if e.get("status") != "ok":
            print(f"{name:<15}{e.get('status')}: {e.get('reason')}")
            continue
        print(f"{name:<15}" + "".join(f"{_fmt(e['metrics'].get(k)):<22}" for k in SUMMARY_KEYS))
    print(f"\n{len(results['failures'])} failures/caveats (see ablation.json):")
    for f in results["failures"]:
        print("  -", f)
    print(f"\nwrote {out / 'ablation.json'}, {out / 'ablation.png'}, "
          f"{out / 'blackboard_armD.jsonl'}, {out / 'trust_snapshot.json'} "
          f"in {results['runtime_s']} s")

    if args.budgets:
        budgets = [int(b) for b in args.budgets.split(",") if b.strip()]
        sweep_cfg = dataclasses.replace(cfg, n_bootstrap=args.budget_bootstrap)
        t0 = time.perf_counter()
        sweep = run_budget_sensitivity(frames, args.preds, budgets, sweep_cfg, finetune_fn=finetune.inner)
        sweep["runtime_s"] = round(time.perf_counter() - t0, 1)
        (out / "ablation_budget.json").write_text(json.dumps(sweep, indent=2), encoding="utf-8")
        plot_budget(sweep, out / "ablation_budget.png")
        print(f"budget sweep {budgets}: wrote {out / 'ablation_budget.json'}, "
              f"{out / 'ablation_budget.png'} in {sweep['runtime_s']} s")


if __name__ == "__main__":
    main()
