"""Evolution run C: niche specialists × budget axis (results/evolution_c/PREREG.md).

    # one budget, its arms and seeds (GPU: offspring are subprocesses of evolve_c_offspring.py)
    python scripts/evolve_c.py run --budget 800 --arms static,evolved,noselect --seeds 0 \
        --out results/evolution_c --orb-python /content/envs/orb/bin/python
    # merge every budget: SUMMARY.md, summary.json, fig9_budget.png/svg
    python scripts/evolve_c.py summarize --out results/evolution_c

Per budget B (4 generations of B/4 labels; breeding after generations 0-2) and seed it runs
``static`` (same triage/oracle, no breeding, no pruning), ``evolved`` (niche specialists, gate,
fitness selection) and optionally ``noselect`` (fine-tune without selection: every child is
admitted, the newest child of each lineage replaces the previous one). Every run is scored on
eval each generation; all runs of one budget share the eval bootstrap resamples, so their
comparisons are paired. Offspring are cached by content (family, parent, recipe, training
uids, seed), so identical children across arms train once.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fittest.blackboard import Blackboard  # noqa: E402
from fittest.cache import load_frames, load_models, load_predictions  # noqa: E402
from fittest.data import assign_leakfree_splits  # noqa: E402
from fittest.evolution import EvalScorer, OffspringResult, jsonable  # noqa: E402
from fittest.evolution_c import (NicheEvolution, budget_config, noselect_c, served_by_niche,  # noqa: E402
                                 static_c)

ARMS = ("static", "evolved", "noselect")
PRIMARY = "force_rmse"
SECONDARY = ("rel_e_mae", "ece")


# ============================================================ offspring factory (GPU)
class GPUFactory:
    """One evolve_c_offspring.py subprocess per child, ``parallel`` at a time per run, and at
    most ``gpu_slots`` across all runs on the machine (flock slots). Content-addressed cache."""

    def __init__(self, out: Path, frames: str, core_python: str, orb_python: str, device: str,
                 parallel: int, gpu_slots: int, threads: int) -> None:
        self.cache = out / "cache"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.frames, self.device = frames, device
        self.py = {"aimnet2": core_python, "mace": core_python, "orb": orb_python}
        self.parallel, self.gpu_slots, self.threads = parallel, gpu_slots, threads
        self.max_tries = 4
        self.log: list[dict[str, Any]] = []

    @staticmethod
    def key(r) -> str:
        h = hashlib.sha256(json.dumps({"family": r.family, "parent": r.parent_ckpt, "recipe": r.recipe,
                                       "train": sorted(r.train_uids), "seed": r.seed,
                                       "n_predict": len(r.predict_uids)}, sort_keys=True).encode())
        return h.hexdigest()[:16]

    def spec(self, r) -> dict[str, Any]:
        k = self.key(r)
        base = self.cache / f"{r.family}_{k}"
        return {"id": r.id, "key": k, "family": r.family, "niche": r.niche, "parent_id": r.parent.id,
                "parent_ckpt": r.parent_ckpt, "recipe": r.recipe, "seed": r.seed, "train_uids": r.train_uids,
                "predict_uids": r.predict_uids, "out": f"{base}.npz", "ckpt": f"{base}.pt",
                "result": f"{base}.result.json"}

    def _slot(self):
        """Try once to take a free GPU slot; None if all are busy. Never blocks: the caller must keep
        polling its own running children, or a run with parallel > gpu_slots deadlocks on itself."""
        import fcntl

        for i in range(self.gpu_slots):
            fh = open(self.cache / f".gpu_slot{i}", "w")
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return fh
            except OSError:
                fh.close()
        return None

    def __call__(self, requests):
        specs = [self.spec(r) for r in requests]
        pending, running, done = list(specs), [], {}
        while pending or running:
            while pending and len(running) < self.parallel:
                s = pending.pop(0)
                if Path(s["result"]).exists() and Path(s["out"]).exists():
                    done[s["id"]] = 0
                    print(f"[factory] {s['id']}: cached {s['key']}", flush=True)
                    continue
                slot = self._slot()
                if slot is None:  # all GPU slots busy: poll running children first, retry next tick
                    pending.insert(0, s)
                    break
                sp = Path(s["result"]).with_suffix("").with_suffix(".spec.json")
                sp.write_text(json.dumps({k: v for k, v in s.items() if not k.startswith("_")}))
                log = open(Path(s["result"]).with_suffix("").with_suffix(".log"), "w")
                env = {**os.environ, "PYTHONPATH": str(ROOT), "OMP_NUM_THREADS": str(self.threads),
                       "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
                cmd = [self.py[s["family"]], str(ROOT / "scripts" / "evolve_c_offspring.py"), "--frames",
                       self.frames, "--spec", str(sp), "--device", self.device]
                print(f"[factory] start {s['id']} ({s['key']}, parent {s['parent_id']}, recipe {s['recipe']}, "
                      f"n_train {len(s['train_uids'])})", flush=True)
                running.append((s, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env,
                                                    cwd=str(ROOT)), time.time(), log, slot))
            time.sleep(3)
            still = []
            for s, p, t0, log, slot in running:
                if p.poll() is None:
                    still.append((s, p, t0, log, slot))
                    continue
                log.close()
                slot.close()
                lp = Path(s["result"]).with_suffix("").with_suffix(".log")
                oom = p.returncode != 0 and "OutOfMemoryError" in lp.read_text(errors="replace")
                tries = s.setdefault("_tries", 1)
                if oom and tries < self.max_tries:  # infrastructure failure, not an outcome: retry
                    s["_tries"] = tries + 1
                    lp.rename(lp.with_suffix(f".oom{tries}.log"))
                    print(f"[factory] {s['id']} CUDA OOM, retry {tries + 1}/{self.max_tries}", flush=True)
                    pending.append(s)
                    continue
                done[s["id"]] = p.returncode
                print(f"[factory] {s['id']} exit {p.returncode} after {time.time() - t0:.0f} s", flush=True)
            running = still
        results = []
        for s in specs:
            if done.get(s["id"]) != 0 or not Path(s["result"]).exists():
                lp = Path(s["result"]).with_suffix("").with_suffix(".log")
                msg = lp.read_text()[-1500:] if lp.exists() else "no log"
                results.append(OffspringResult(None, error=f"exit {done.get(s['id'])}: {msg}"))
                continue
            res = json.loads(Path(s["result"]).read_text())
            preds = load_predictions(s["out"])
            preds.model = s["id"]
            self.log.append({"id": s["id"], "key": s["key"], "wall_seconds": res.get("wall_seconds"),
                             "curve_last": res["curve"][-1] if res.get("curve") else None})
            results.append(OffspringResult(preds, toolkit={}, ckpt=s["ckpt"], wall_seconds=res.get("wall_seconds")))
        return results


# ============================================================ one budget
def cmd_run(args) -> None:
    out = Path(args.out) / f"b{args.budget}{args.tag}"
    out.mkdir(parents=True, exist_ok=True)
    frames, split_report = assign_leakfree_splits(load_frames(args.frames), 0)
    base_cfg = budget_config(args.budget, args.generations, n_bootstrap_eval=args.n_boot,
                             families=tuple(f for f in args.families.split(",") if f),
                             children_per_niche=args.children, explore_base=args.explore_base,
                             gate_rel_e_tol=args.gate_tol, protect_base=args.protect_base)
    base = load_models(args.preds, list(base_cfg.base_models))
    scorer = EvalScorer(args.n_boot, 0)
    factory = GPUFactory(Path(args.out), args.frames, args.core_python, args.orb_python, args.device,
                         args.parallel, args.gpu_slots, args.threads)
    runs: dict[str, NicheEvolution] = {}
    niche_eval: dict[str, Any] = {}
    arms = [a for a in args.arms.split(",") if a]
    with Blackboard(out / "blackboard.jsonl") as bb:
        for seed in [int(s) for s in args.seeds.split(",")]:
            cfg = dataclasses.replace(base_cfg, seed=seed)
            for arm in arms:
                acfg = {"static": static_c(cfg), "evolved": cfg, "noselect": noselect_c(cfg)}[arm]
                name = f"{arm}_s{seed}"
                t0 = time.time()
                evo = NicheEvolution(frames, base, acfg, None if arm == "static" else factory, bb, name)
                # run with per-generation niche-resolved scoring
                for g in range(acfg.generations):
                    info: dict[str, Any] = {"generation": g, "triage": evo.triage(g)}
                    info["selection"] = evo.select(g)
                    state = evo.serve_eval()
                    scorer.record(f"{name}_g{g}", evo, state)
                    niche_eval[f"{name}_g{g}"] = served_by_niche(evo, state)
                    info["served_eval"] = state["served"]
                    info["population_after_selection"] = list(evo.population)
                    if g < acfg.generations - 1:
                        info["breeding"] = evo.breed(g)
                    info["population_next"] = list(evo.population)
                    evo.history.append(info)
                    evo.post("generation", {k: v for k, v in info.items() if k != "triage"})
                    v = scorer.values[f"{name}_g{g}"]
                    print(f"[{name}] g{g} labels {v['labels']} pop {v['population']} F-RMSE "
                          f"{v['force_rmse']['value']:.4f} relE {v['rel_e_mae']['value']:.4f} "
                          f"({time.time() - t0:.0f} s)", flush=True)
                runs[name] = evo
                write_outputs(out, runs, scorer, base_cfg, split_report, niche_eval, factory, args.budget)
    print("RUN_DONE", out, flush=True)


def write_outputs(out: Path, runs, scorer: EvalScorer, cfg, split_report, niche_eval, factory, budget) -> None:
    G = cfg.generations
    comp: dict[str, Any] = {}
    names = set(runs)
    for name in runs:
        arm, seed = name.rsplit("_s", 1)
        last = f"{name}_g{G - 1}"
        c: dict[str, Any] = {"final_vs_gen0": scorer.paired(last, f"{name}_g0")}
        if arm != "static" and f"static_s{seed}" in names:
            c["final_vs_static"] = scorer.paired(last, f"static_s{seed}_g{G - 1}")
            c["per_generation_vs_static"] = {g: scorer.paired(f"{name}_g{g}", f"static_s{seed}_g{g}")
                                             for g in range(G)}
        if arm == "evolved" and f"noselect_s{seed}" in names:
            c["final_vs_noselect"] = scorer.paired(last, f"noselect_s{seed}_g{G - 1}")
        comp[name] = c
    any_run = next(iter(runs.values()))
    payload = {"budget": budget, "config": dataclasses.asdict(cfg), "pool_split": any_run.split_report,
               "leakfree_split": split_report, "niches": any_run.niche_report(), "states": scorer.values,
               "comparisons": comp, "served_by_niche": niche_eval, "offspring_log": factory.log,
               "wrongness_units": "F_RMSE/WRONG_F + |dE_rel|/WRONG_E"}
    (out / "generations.json").write_text(json.dumps(jsonable(payload), indent=1))
    (out / "family_tree.json").write_text(json.dumps(jsonable(
        {n: e.family_tree() for n, e in runs.items() if not n.startswith("static")}), indent=1))
    (out / "history.json").write_text(json.dumps(jsonable({n: e.history for n, e in runs.items()}), indent=1))


# ============================================================ summary over budgets
def _fmt(v, nd=3):
    return "–" if v is None else f"{v:.{nd}f}"


def verdict_for(comp: dict[str, Any], final_pop: list[str]) -> dict[str, Any]:
    """Pre-registered rule: final generation significantly lower served force RMSE than both
    static (same budget) and its own gen 0, offspring in the final population, and no significant
    regression vs static in rel-E MAE or ECE."""
    vs_s, vs_0 = comp["final_vs_static"], comp["final_vs_gen0"]
    beats_static = vs_s[PRIMARY]["significant"] and vs_s[PRIMARY]["diff"] < 0
    beats_g0 = vs_0[PRIMARY]["significant"] and vs_0[PRIMARY]["diff"] < 0
    regress = [m for m in SECONDARY if vs_s[m]["significant"] and (vs_s[m]["diff"] or 0) > 0]
    has_off = any(m.startswith("g") for m in final_pop)
    holds = bool(beats_static and beats_g0 and has_off and not regress)
    return {"holds": holds, "beats_static": bool(beats_static), "beats_gen0": bool(beats_g0),
            "offspring_in_final_population": has_off, "regressions_vs_static": regress}


def cmd_summarize(args) -> None:
    root = Path(args.out)
    rows, summary = [], {"budgets": {}}
    for d in sorted(root.glob("b*/generations.json"), key=lambda p: (int(''.join(c for c in p.parent.name[1:]
                                                                                if c.isdigit()) or 0), p.parent.name)):
        J = json.loads(d.read_text())
        B, G = J["budget"], J["config"]["generations"]
        st, cp = J["states"], J["comparisons"]
        entry: dict[str, Any] = {"dir": d.parent.name, "runs": {}}
        for name in sorted({k.rsplit("_g", 1)[0] for k in st}):
            last = st[f"{name}_g{G - 1}"]
            c = cp.get(name, {})
            r = {"final": {m: last[m] for m in ("force_rmse", "rel_e_mae", "ece", "brier", "trusted_fraction",
                                                "badly_wrong_trusted", "wrong_rate")},
                 "population": last["population"], "labels": last["labels"],
                 "gen0_force_rmse": st[f"{name}_g0"]["force_rmse"]["value"],
                 "served_by_niche_final": J["served_by_niche"].get(f"{name}_g{G - 1}")}
            if "final_vs_static" in c:
                r["vs_static"] = {m: c["final_vs_static"][m] for m in ("force_rmse", "rel_e_mae", "ece", "brier")}
            r["vs_gen0"] = {m: c["final_vs_gen0"][m] for m in ("force_rmse", "rel_e_mae")}
            if "final_vs_noselect" in c:
                r["vs_noselect"] = {m: c["final_vs_noselect"][m] for m in ("force_rmse", "rel_e_mae")}
            if name.startswith("evolved") and "final_vs_static" in c:
                r["verdict"] = verdict_for(c, last["population"])
            entry["runs"][name] = r
            rows.append((B, d.parent.name, name, r))
        summary["budgets"][d.parent.name] = {"budget": B, **entry}
    ev = [r["verdict"]["holds"] for _, _, n, r in rows if "verdict" in r]
    # Pre-registered aggregation: at a budget, evolution holds iff it holds in EVERY seed run there.
    by_budget: dict[int, list[bool]] = {}
    for B, _, _, r in rows:
        if "verdict" in r:
            by_budget.setdefault(B, []).append(bool(r["verdict"]["holds"]))
    per_budget = {str(B): {"seeds_holding": sum(v), "seeds": len(v), "holds": bool(v) and all(v)}
                  for B, v in sorted(by_budget.items())}
    crossover = next((int(B) for B, v in per_budget.items() if v["holds"]), None)
    summary["verdict"] = {"rule": "holds at a budget iff it holds in every seed run (PREREG.md)",
                          "per_budget": per_budget, "evolution_holds_any_budget": crossover is not None,
                          "crossover_budget": crossover, "any_single_run_holds": any(ev), "per_run": {
        f"{dn}/{n}": r["verdict"] for _, dn, n, r in rows if "verdict" in r}}
    (root / "summary.json").write_text(json.dumps(jsonable(summary), indent=1))
    lines = ["| budget | run | served F-RMSE [95% CI] | Δ vs static F-RMSE [CI] | served rel-E MAE | "
             "Δ vs static rel-E [CI] | ECE | Brier | trusted | badly-wrong trusted | final population | verdict |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for B, dn, n, r in rows:
        f = r["final"]
        vs = r.get("vs_static")
        dF = f"{vs['force_rmse']['diff']:+.4f} [{vs['force_rmse']['ci'][0]:+.4f}, {vs['force_rmse']['ci'][1]:+.4f}]" if vs else "–"
        dE = f"{vs['rel_e_mae']['diff']:+.4f} [{vs['rel_e_mae']['ci'][0]:+.4f}, {vs['rel_e_mae']['ci'][1]:+.4f}]" if vs else "–"
        v = r.get("verdict")
        vt = "–" if v is None else ("HOLDS" if v["holds"] else "does not hold")
        lines.append(f"| {B} | {dn}/{n} | {_fmt(f['force_rmse']['value'], 4)} [{_fmt(f['force_rmse']['ci'][0], 4)}, "
                     f"{_fmt(f['force_rmse']['ci'][1], 4)}] | {dF} | {_fmt(f['rel_e_mae']['value'], 4)} | {dE} | "
                     f"{_fmt(f['ece']['value'])} | {_fmt(f['brier']['value'])} | {_fmt(f['trusted_fraction']['value'])} | "
                     f"{_fmt(f['badly_wrong_trusted']['value'])} | {', '.join(r['population'])} | {vt} |")
    (root / "summary_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    make_figure(root, rows)


def make_figure(root: Path, rows) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"static": "#6b7280", "evolved": "#2563eb", "noselect": "#d97706"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric, lab in ((axes[0], "force_rmse", "served force RMSE on eval [eV/Å]"),
                            (axes[1], "rel_e_mae", "served rel-E MAE on eval [eV]")):
        for arm, off in (("static", -0.04), ("evolved", 0.0), ("noselect", 0.04)):
            pts = [(B, r) for B, dn, n, r in rows if n.startswith(arm + "_s0") and not dn.endswith("_s1")]
            pts += [(B, r) for B, dn, n, r in rows if n.startswith(arm + "_s1")]
            if not pts:
                continue
            for i, (B, r) in enumerate(sorted(pts, key=lambda t: t[0])):
                v = r["final"][metric]
                x = B * (1 + off) * (1.015 if r is not None and "_s1" in json.dumps(r.get("labels", "")) else 1)
                ax.errorbar([x], [v["value"]], yerr=[[v["value"] - v["ci"][0]], [v["ci"][1] - v["value"]]],
                            fmt="o", color=colors[arm], capsize=3, label=arm if i == 0 else None)
            xs = sorted({B for B, _ in pts})
            ys = [min((r["final"][metric]["value"] for B2, r in pts if B2 == B)) for B in xs]
            ax.plot([b * (1 + off) for b in xs], ys, color=colors[arm], alpha=0.5)
        ax.set_xscale("log")
        ax.set_xticks([200, 400, 800])
        ax.set_xticklabels(["200", "400", "800"])
        ax.set_xlabel("total oracle labels (4 generations)")
        ax.set_ylabel(lab)
        ax.grid(alpha=0.3)
    axes[0].legend(frameon=False)
    fig.suptitle("Run C: static vs evolved (niche specialists) vs fine-tune without selection; 95% reaction-bootstrap CIs",
                 fontsize=9)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(root / f"fig9_budget.{ext}", dpi=150)
    print("wrote", root / "fig9_budget.png")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--frames", default="data/frames.pkl")
    r.add_argument("--preds", default="data/preds")
    r.add_argument("--out", default="results/evolution_c")
    r.add_argument("--budget", type=int, required=True)
    r.add_argument("--tag", default="", help="suffix of the output dir, e.g. _s1")
    r.add_argument("--generations", type=int, default=4)
    r.add_argument("--arms", default="static,evolved")
    r.add_argument("--seeds", default="0")
    r.add_argument("--core-python", default=sys.executable)
    r.add_argument("--orb-python", default=sys.executable)
    r.add_argument("--device", default="cuda")
    r.add_argument("--parallel", type=int, default=3)
    r.add_argument("--gpu-slots", type=int, default=5)
    r.add_argument("--threads", type=int, default=2)
    r.add_argument("--n-boot", type=int, default=1000)
    r.add_argument("--families", default="aimnet2,orb,mace",
                   help="specialist families to breed (CPU VM: aimnet2 only)")
    # exploratory amendments (not pre-registered; defaults = PREREG.md)
    r.add_argument("--children", type=int, default=1, help="children per family per generation")
    r.add_argument("--explore-base", action="store_true", help="base-parent children explore energy weight / lr")
    r.add_argument("--gate-tol", type=float, default=0.0, help="relative rel-E tolerance of the gate")
    r.add_argument("--protect-base", action="store_true", help="never prune base models")
    s = sub.add_parser("summarize")
    s.add_argument("--out", default="results/evolution_c")
    args = ap.parse_args(argv)
    {"run": cmd_run, "summarize": cmd_summarize}[args.cmd](args)


if __name__ == "__main__":
    main()
