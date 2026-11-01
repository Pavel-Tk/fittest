"""Seed-level analysis across evolve_c summary.json files (run E pre-registration).

Within-seed paired CIs cover eval resampling only. This treats each seed as one observation:
for an arm (e.g. noselect) at a budget, it collects the final-vs-static difference per seed and reports
the mean with a two-sided 95 % Student-t interval over seeds, plus per-seed significance counts.

    python scripts/seed_level.py --arm noselect --budget 800 results/evolution_e/gpu/summary.json [...]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

# two-sided 95 % t critical values, df 1..30
T95 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145,
       2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045,
       2.042]


def t_interval(xs: list[float]) -> dict:
    n = len(xs)
    m = sum(xs) / n
    if n < 2:
        return {"n": n, "mean": m, "ci": [None, None]}
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    h = T95[min(n - 1, 30) - 1] * sd / math.sqrt(n)
    return {"n": n, "mean": m, "sd": sd, "ci": [m - h, m + h]}


def collect(paths: list[str], arm: str, budget: int) -> dict:
    out: dict = {"arm": arm, "budget": budget, "seeds": {}}
    for p in paths:
        S = json.loads(Path(p).read_text())
        for entry in S["budgets"].values():
            if entry["budget"] != budget:
                continue
            for name, r in entry["runs"].items():
                if not name.startswith(arm + "_s") or "vs_static" not in r:
                    continue
                out["seeds"][f"{Path(p).parent.name}/{name}"] = {
                    m: r["vs_static"][m] for m in ("force_rmse", "rel_e_mae", "ece")}
    res = {}
    for m in ("force_rmse", "rel_e_mae", "ece"):
        d = [v[m]["diff"] for v in out["seeds"].values()]
        sig_better = sum(1 for v in out["seeds"].values() if v[m]["significant"] and v[m]["diff"] < 0)
        sig_worse = sum(1 for v in out["seeds"].values() if v[m]["significant"] and v[m]["diff"] > 0)
        res[m] = {**t_interval(d), "sig_better": sig_better, "sig_worse": sig_worse} if d else None
    out["seed_level"] = res
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("summaries", nargs="+")
    ap.add_argument("--arm", default="noselect")
    ap.add_argument("--budget", type=int, default=800)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    r = collect(a.summaries, a.arm, a.budget)
    txt = json.dumps(r, indent=1)
    if a.out:
        Path(a.out).write_text(txt)
    for m, v in r["seed_level"].items():
        if v:
            print(f"{m}: n={v['n']} mean={v['mean']:+.4f} 95% t-CI [{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}] "
                  f"sig better {v['sig_better']}, sig worse {v['sig_worse']}")


if __name__ == "__main__":
    main()
