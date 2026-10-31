"""Run the TrustSwarm ablation (arms A-D, same oracle budget) on cached predictions.

    python scripts/run_ablation.py --frames data/frames.pkl --preds data/preds --out results/ \
        --budget-per-round 40 --rounds 5 --seed 0

Writes ``ablation.json``, ``ablation.png``, ``blackboard_armD.jsonl`` and
``trust_snapshot.json`` (arm D's final Calibrator state, read by the trust service) to --out.
Fine-tuning is replayed from ``{preds}/mace_lora_curated.npz`` if present (``--no-finetune``
disables it).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from trustswarm.ablation import AblationConfig, plot_ablation, run_ablation
from trustswarm.agents import NullFinetune
from trustswarm.cache import load_frames


def _fmt(metric: dict | None) -> str:
    if not metric or metric.get("value") is None:
        return "   n/a          "
    ci = metric.get("ci")
    return f"{metric['value']:.3f} [{ci[0]:.2f},{ci[1]:.2f}]" if ci else f"{metric['value']:.3f}      "


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
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = AblationConfig(budget_per_round=args.budget_per_round, rounds=args.rounds,
                         seed=args.seed, n_bootstrap=args.n_bootstrap,
                         max_planted=args.max_planted if args.max_planted > 0 else None)
    t0 = time.perf_counter()
    results = run_ablation(load_frames(args.frames), args.preds, cfg,
                           finetune_fn=NullFinetune() if args.no_finetune else None,
                           blackboard_path=out / "blackboard_armD.jsonl",
                           snapshot_path=out / "trust_snapshot.json")
    results["runtime_s"] = round(time.perf_counter() - t0, 1)
    (out / "ablation.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    if results.get("arms"):
        plot_ablation(results, out / "ablation.png")

    keys = ("auroc_pool", "auroc_eval", "recall_at_k", "planted_caught", "eval_force_rmse_post")
    print(f"{'arm':<15}" + "".join(f"{k:<22}" for k in keys))
    for name, e in results["arms"].items():
        if e.get("status") != "ok":
            print(f"{name:<15}{e.get('status')}: {e.get('reason')}")
            continue
        print(f"{name:<15}" + "".join(f"{_fmt(e['metrics'].get(k)):<22}" for k in keys))
    print(f"\n{len(results['failures'])} failures/caveats (see ablation.json):")
    for f in results["failures"]:
        print("  -", f)
    print(f"\nwrote {out / 'ablation.json'}, {out / 'ablation.png'}, {out / 'blackboard_armD.jsonl'}"
          f" in {results['runtime_s']} s")


if __name__ == "__main__":
    main()
