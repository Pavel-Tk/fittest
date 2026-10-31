"""Run the evolutionary loop (Fittest: survival of the fittest) and its controls.

    # VM (CPU): offspring are LoRA fine-tunes run as parallel subprocesses
    python scripts/evolve.py --frames data/frames.pkl --preds data/preds_base --out results/evolution \
        --generations 4 --labels-per-generation 50 --seeds 0 --python ~/envs/core/bin/python

    # dry run without torch (no offspring: evolved == static)
    python scripts/evolve.py --no-breeding ...

Per seed it runs the **static** swarm (same triage/oracle procedure and budget, no breeding,
no pruning) and the **evolved** swarm (optionally a **clone** control: breeding without
mutation), scores every generation on eval with reaction-bootstrap CIs (paired across all
runs), and writes to --out:

  blackboard.jsonl        every triage / fitness / death / breed / birth / generation event
  family_tree.json        every member ever: parent, generation, recipe, gate, fitness, fate
  generations.json        eval metrics per run and generation (+ CIs) and paired comparisons
  numbers.json            the headline numbers used by SUMMARY.md and the figures
  jobs/<run>/<id>.{json,log,result.json}, preds/<run>/<id>.npz, ckpt/<run>/<id>/   offspring
"""

from __future__ import annotations

import argparse
import dataclasses
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
from fittest.evolution import (EvalScorer, Evolution, EvoConfig, OffspringRequest,  # noqa: E402
                               OffspringResult, jsonable, static_config)


class TorchFactory:
    """Runs one ``scripts/evolve_offspring.py`` subprocess per child, in parallel; resumes
    finished children (result file present) instead of re-training them."""

    def __init__(self, out: Path, run: str, python: str, frames: str, device: str, threads: int,
                 parallel: int) -> None:
        self.jobs, self.preds, self.ckpt = (out / "jobs" / run, out / "preds" / run, out / "ckpt" / run)
        for d in (self.jobs, self.preds, self.ckpt):
            d.mkdir(parents=True, exist_ok=True)
        self.python, self.frames, self.device = python, frames, device
        self.threads, self.parallel = threads, parallel

    def spec(self, r: OffspringRequest) -> dict[str, Any]:
        parent = r.parent.id if r.parent.kind == "base" else r.parent.ckpt
        return {"id": r.id, "family": r.parent.family, "parent": parent, "parent_id": r.parent.id, "recipe": r.recipe,
                "seed": r.seed, "train_uids": r.train_uids, "holdout_uids": r.holdout_uids,
                "out": str(self.preds / f"{r.id}.npz"), "ckpt": str(self.ckpt / r.id),
                "result": str(self.jobs / f"{r.id}.result.json")}

    def __call__(self, requests):
        specs = [self.spec(r) for r in requests]
        procs: list[tuple[dict, subprocess.Popen | None, float]] = []
        pending = list(specs)
        running: list[tuple[dict, subprocess.Popen, float, Any]] = []
        done: dict[str, int] = {}
        while pending or running:
            while pending and len(running) < self.parallel:
                s = pending.pop(0)
                if Path(s["result"]).exists() and Path(s["out"]).exists():
                    done[s["id"]] = 0
                    print(f"[factory] {s['id']}: reusing finished job", flush=True)
                    continue
                sp = self.jobs / f"{s['id']}.json"
                sp.write_text(json.dumps(s, indent=1))
                log = open(self.jobs / f"{s['id']}.log", "w")
                env = {**os.environ, "OMP_NUM_THREADS": str(self.threads), "MKL_NUM_THREADS": str(self.threads)}
                script = "evolve_offspring_aimnet.py" if s["family"] == "aimnet2" else "evolve_offspring.py"
                cmd = [self.python, str(ROOT / "scripts" / script), "--frames", self.frames,
                       "--spec", str(sp), "--device", self.device, "--threads", str(self.threads)]
                print(f"[factory] start {s['id']} (parent {s['parent_id']}, recipe {s['recipe']})", flush=True)
                running.append((s, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env,
                                                    cwd=str(ROOT)), time.time(), log))
            time.sleep(5)
            still = []
            for s, p, t0, log in running:
                if p.poll() is None:
                    still.append((s, p, t0, log))
                else:
                    log.close()
                    done[s["id"]] = p.returncode
                    print(f"[factory] {s['id']} exit {p.returncode} after {time.time() - t0:.0f} s", flush=True)
            running = still
        results = []
        for s in specs:
            if done.get(s["id"]) != 0 or not Path(s["result"]).exists():
                tail = (self.jobs / f"{s['id']}.log")
                msg = tail.read_text()[-1500:] if tail.exists() else "no log"
                results.append(OffspringResult(None, error=f"exit {done.get(s['id'])}: {msg}"))
                continue
            res = json.loads(Path(s["result"]).read_text())
            preds = load_predictions(s["out"])
            preds.model = s["id"]
            results.append(OffspringResult(preds, toolkit=res["toolkit"], ckpt=res["ckpt"],
                                           wall_seconds=res.get("wall_seconds")))
        return results


def run_one(name: str, cfg: EvoConfig, frames, base, scorer: EvalScorer, bb: Blackboard,
            factory=None) -> Evolution:
    t0 = time.time()
    evo = Evolution(frames, base, cfg, factory, blackboard=bb, label=name)
    evo.run(scorer)
    print(f"[{name}] done in {time.time() - t0:.0f} s; final population {evo.population}", flush=True)
    return evo


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="data/frames.pkl")
    ap.add_argument("--preds", default="data/preds")
    ap.add_argument("--out", default="results/evolution")
    ap.add_argument("--generations", type=int, default=4)
    ap.add_argument("--labels-per-generation", type=int, default=50)
    ap.add_argument("--offspring", type=int, default=3)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--controls", default="", help="comma list: clone,random,newest")
    ap.add_argument("--family", default="mace", choices=("mace", "aimnet2"),
                    help="which family the Curator breeds (run A: mace, run B: aimnet2)")
    ap.add_argument("--control-seeds", default="0", help="seeds that also run the controls")
    ap.add_argument("--no-breeding", action="store_true")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--parallel", type=int, default=3)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--epochs-scale", type=float, default=1.0, help="multiply recipe epochs (CPU budget)")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frames, split_report = assign_leakfree_splits(load_frames(args.frames), 0)
    base_cfg = EvoConfig(generations=args.generations, labels_per_generation=args.labels_per_generation,
                         offspring_per_generation=args.offspring, n_bootstrap_eval=args.n_boot,
                         breed_family=args.family,
                         finetunable=("aimnet2",) if args.family == "aimnet2" else ("mace_mpa0",))
    if args.epochs_scale != 1.0:
        import fittest.evolution as ev
        ev.BASE_RECIPE["epochs"] = int(round(ev.BASE_RECIPE["epochs"] * args.epochs_scale))
    base = load_models(args.preds, list(base_cfg.base_models))
    scorer = EvalScorer(args.n_boot, 0)
    runs: dict[str, Evolution] = {}
    with Blackboard(out / "blackboard.jsonl") as bb:
        for seed in [int(s) for s in args.seeds.split(",")]:
            cfg = dataclasses.replace(base_cfg, seed=seed)
            runs[f"static_s{seed}"] = run_one(f"static_s{seed}", static_config(cfg), frames, base, scorer, bb)
            write_outputs(out, runs, scorer, base_cfg, split_report)
            fac = None if args.no_breeding else TorchFactory(out, f"evolved_s{seed}", args.python, args.frames,
                                                             args.device, args.threads, args.parallel)
            runs[f"evolved_s{seed}"] = run_one(f"evolved_s{seed}", cfg, frames, base, scorer, bb, fac)
            write_outputs(out, runs, scorer, base_cfg, split_report)
            ctls = [c for c in args.controls.split(",") if c] if str(seed) in args.control_seeds.split(",") else []
            for ctl in ctls:
                ccfg = {"clone": dataclasses.replace(cfg, breeding_mode="clone"),
                        "random": dataclasses.replace(cfg, selection_mode="random"),
                        "newest": dataclasses.replace(cfg, breeding_mode="clone", selection_mode="newest",
                                                      offspring_per_generation=1)}[ctl]
                fac = None if args.no_breeding else TorchFactory(out, f"{ctl}_s{seed}", args.python, args.frames,
                                                                 args.device, args.threads, args.parallel)
                runs[f"{ctl}_s{seed}"] = run_one(f"{ctl}_s{seed}", ccfg, frames, base, scorer, bb, fac)
                write_outputs(out, runs, scorer, base_cfg, split_report)
    write_outputs(out, runs, scorer, base_cfg, split_report)


def write_outputs(out: Path, runs: dict[str, Evolution], scorer: EvalScorer, cfg: EvoConfig,
                  split_report: dict) -> None:
    G = cfg.generations
    comparisons: dict[str, Any] = {}
    for name in runs:
        if name.startswith("static"):
            continue
        seed = name.rsplit("_s", 1)[1]
        static = f"static_s{seed}_g{G - 1}"
        comparisons[name] = {
            "final_vs_gen0": scorer.paired(f"{name}_g{G - 1}", f"{name}_g0"),
            "final_vs_static_same_budget": scorer.paired(f"{name}_g{G - 1}", static),
            "per_generation_vs_gen0": {g: scorer.paired(f"{name}_g{g}", f"{name}_g0") for g in range(1, G)},
            "per_generation_vs_static": {g: scorer.paired(f"{name}_g{g}", f"static_s{seed}_g{g}") for g in range(G)},
        }
    trees = {name: evo.family_tree() for name, evo in runs.items() if not name.startswith("static")}
    histories = {name: evo.history for name, evo in runs.items()}
    any_run = next(iter(runs.values()))
    payload = {"config": dataclasses.asdict(cfg), "pool_split": any_run.split_report,
               "leakfree_split": split_report, "states": scorer.values, "comparisons": comparisons,
               "wrongness_units": "F_RMSE/WRONG_F + |dE_rel|/WRONG_E"}
    (out / "generations.json").write_text(json.dumps(jsonable(payload), indent=1))
    (out / "family_tree.json").write_text(json.dumps(jsonable(trees), indent=1))
    (out / "history.json").write_text(json.dumps(jsonable(histories), indent=1))
    nums = {"states": {k: {m: v[m] for m in EvalScorer.METRICS} | {"population": v["population"],
                                                                     "labels": v["labels"]}
                       for k, v in scorer.values.items()},
            "comparisons": {k: {"final_vs_gen0": c["final_vs_gen0"],
                                "final_vs_static_same_budget": c["final_vs_static_same_budget"]}
                            for k, c in comparisons.items()}}
    (out / "numbers.json").write_text(json.dumps(jsonable(nums), indent=1))
    print(f"wrote {out}/generations.json, family_tree.json, history.json, numbers.json", flush=True)


if __name__ == "__main__":
    main()
