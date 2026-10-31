"""In-flight ALCHEMI screening run with the DisagreementHook (DESIGN.md §6).

Transition1x frames stream through a ``FusedStage`` (Mode 2: ``SizeAwareSampler`` refills
freed slots) running FIRE2 relaxation driven by a Toolkit MACE (or AIMNet2) wrapper. The
DisagreementHook asks the three model agents (MACE-MPA-0, AIMNet2, Orb; one process each)
for forces at admission (pristine dataset geometry) and every ``--every`` steps; systems
with sigma_F >= tau graduate out of the batch into the oracle queue and the sampler
refills the slot. Unflagged systems relax for up to ``--n-steps`` steps.

    python3 scripts/screen_alchemi.py --frames data/frames.pkl --split pool --max-frames 200

Writes ``results/screen_alchemi.json`` (per-frame records + summary) and
``data/oracle_queue_alchemi.pkl`` (flagged geometries with per-model forces).
No labels are used during the run; ``--score-labels`` adds post-hoc per-model force
errors at the admission geometry for frames that carry DFT labels (validation only).
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SERVERS = {  # name -> (port, python)
    "mace_mpa0": (6100, "/usr/bin/python3"),
    "aimnet2": (6101, "/usr/bin/python3"),
    "orb": (6102, "/content/envs/orb/bin/python"),
}


def load_frames(a):
    if a.frames and Path(a.frames).exists():
        from trustswarm.cache import load_frames as _load

        frames = [f for f in _load(a.frames) if a.split in (None, "all") or f.split == a.split]
        src = f"{a.frames}:{a.split}"
    else:
        from trustswarm import t1x

        frames = t1x.sample_split(a.h5, "test", a.n_reactions, a.n_path, seed=a.seed)
        src = f"{a.h5}:test sample (n_reactions={a.n_reactions}, n_path={a.n_path}, seed={a.seed})"
    if a.max_frames and len(frames) > a.max_frames:
        rng = np.random.default_rng(a.seed)
        frames = [frames[i] for i in np.sort(rng.choice(len(frames), a.max_frames, replace=False))]
    return frames, src


def auto_tau(a, frames) -> tuple[float, str]:
    """70th percentile of sigma_F over the cached pool predictions (Calibrator's round-0 floor)."""
    try:
        from trustswarm.cache import load_models
        from trustswarm.config import HETERO_MODELS, TAU_FLOOR_QUANTILE
        from trustswarm.signals import force_spread

        preds = load_models(ROOT / "data" / "preds", HETERO_MODELS)
        uids = [f.uid for f in frames if all(f.uid in p.forces for p in preds.values())]
        if len(uids) >= 20:
            sig = [force_spread(np.stack([preds[m].forces[u] for m in HETERO_MODELS]))[0] for u in uids]
            return float(np.quantile(sig, TAU_FLOOR_QUANTILE)), \
                f"q{TAU_FLOOR_QUANTILE:.2f} of cached sigma_F over {len(uids)} frames"
    except Exception as exc:  # cache absent: fall back
        return a.tau_fallback, f"fallback ({type(exc).__name__}: {exc})"
    return a.tau_fallback, "fallback (too few cached predictions)"


class FrameDataset:
    """Toolkit sampler dataset protocol over ``trustswarm.data.Frame`` (labels never exposed)."""

    def __init__(self, frames, device):
        self.frames, self.device, self.load_order = frames, device, []

    def __len__(self):
        return len(self.frames)

    def get_metadata(self, i):
        return len(self.frames[i].Z), 0

    def __getitem__(self, i):
        from trustswarm.models.toolkit_io import atomic_data

        f = self.frames[i]
        self.load_order.append(i)   # sampler stamps system_id = len(load_order) - 1 right after
        return atomic_data(f.Z, f.R, charge=0.0, device=self.device, with_dynamics_fields=True), {}


def start_servers(names, log_dir, report):
    from trustswarm.model_server import spawn

    procs = {}
    for n in names:
        port, py = SERVERS[n]
        if not Path(py).exists():
            report[n] = f"python {py} not found"
            continue
        try:
            procs[n] = spawn(n, port, python=py, log_dir=log_dir)
        except Exception as exc:
            report[n] = f"spawn failed: {exc}"
    clients = {}
    for n, c in procs.items():
        try:
            c.connect(timeout=900)
            clients[n] = c
            report[n] = "ok"
        except Exception as exc:
            log = Path(log_dir) / f"server_{n}.log"
            tail = log.read_text()[-800:] if log.exists() else ""
            report[n] = f"connect failed: {exc} | {tail}"
            try:
                c.proc.kill()
            except Exception:
                pass
    return clients


def make_driver(name, device):
    import torch

    if name == "mace":
        from trustswarm.models.mace_batched import BatchedMACE

        return BatchedMACE(device=device).model, "MACE-MPA-0 (Toolkit MACEWrapper)"
    if name == "aimnet2":
        from nvalchemi.models.aimnet2 import AIMNet2Wrapper

        m = AIMNet2Wrapper.from_checkpoint("aimnet2", device=torch.device(device), compile_model=False)
        m.model_config.active_outputs = {"energy", "forces"}
        return m, "AIMNet2 (Toolkit AIMNet2Wrapper, no D3)"
    raise ValueError(name)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default=str(ROOT / "data" / "frames.pkl"))
    ap.add_argument("--split", default="pool")
    ap.add_argument("--h5", default="/content/data/Transition1x.h5")
    ap.add_argument("--n-reactions", type=int, default=30)
    ap.add_argument("--n-path", type=int, default=3)
    ap.add_argument("--max-frames", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--models", default="mace_mpa0,aimnet2,orb")
    ap.add_argument("--driver", default="mace", choices=["mace", "aimnet2"])
    ap.add_argument("--tau", default="auto")
    ap.add_argument("--tau-fallback", type=float, default=0.30)
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--n-steps", type=int, default=50, help="per-system FIRE2 budget")
    ap.add_argument("--fmax", type=float, default=0.05)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--max-atoms", type=int, default=1024)
    ap.add_argument("--max-batch", type=int, default=48)
    ap.add_argument("--max-engine-steps", type=int, default=20000)
    ap.add_argument("--score-labels", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--log-dir", default="/content/logs")
    ap.add_argument("--out", default=str(ROOT / "results" / "screen_alchemi.json"))
    ap.add_argument("--queue-out", default=str(ROOT / "data" / "oracle_queue_alchemi.pkl"))
    a = ap.parse_args()

    import torch
    from nvalchemi.dynamics import FIRE2, ConvergenceHook, FusedStage, SizeAwareSampler
    from nvalchemi.dynamics.base import DynamicsStage

    from trustswarm.hook import DisagreementHook

    frames, src = load_frames(a)
    tau, tau_src = (float(a.tau), "cli") if a.tau != "auto" else auto_tau(a, frames)
    out: dict = {"config": {**vars(a), "tau": tau, "tau_source": tau_src, "frames_source": src,
                            "n_frames": len(frames)}, "servers": {}, "errors": []}
    print(f"[screen] {len(frames)} frames from {src}; tau={tau:.4f} ({tau_src})", flush=True)

    clients = start_servers([m for m in a.models.split(",") if m], a.log_dir, out["servers"])
    print("[screen] servers:", out["servers"], flush=True)
    if len(clients) < 2:
        out["errors"].append("fewer than two model servers available; nothing to compare")
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=1, default=str))
        return 1

    ds = FrameDataset(frames, a.device)
    flag_queue: list[dict] = []
    hook = None
    t_wall = 0.0
    try:
        driver, driver_desc = make_driver(a.driver, a.device)
        out["config"]["driver_desc"] = driver_desc
        hook = DisagreementHook(
            clients, tau=tau, every=a.every, stage="both", graduate=True,
            id_map=lambda sid: ds.frames[ds.load_order[sid]].uid,
            on_flag=lambda rec: flag_queue.append(rec["key"]))
        relax = FIRE2(model=driver, dt=a.dt, n_steps=a.n_steps,
                      convergence_hook=ConvergenceHook.from_fmax(a.fmax))
        for h in driver.make_neighbor_hooks():
            relax.register_hook(h, stage=DynamicsStage.BEFORE_COMPUTE)
        sampler = SizeAwareSampler(ds, max_atoms=a.max_atoms, max_edges=None,
                                   max_batch_size=a.max_batch, bin_width=1, shuffle=False)
        fused = FusedStage(sub_stages=[(0, relax)], sampler=sampler, sinks=[],
                           refill_frequency=1, hooks=[hook])
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.perf_counter()
        with fused:
            result = fused.run(batch=None, n_steps=a.max_engine_steps)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t_wall = time.perf_counter() - t0
        out["engine"] = {"final_step_count": int(getattr(fused, "step_count", -1)),
                         "sampler_exhausted": result is None,
                         "left_in_batch": 0 if result is None else int(result.num_graphs)}
    except Exception as exc:
        out["errors"].append(f"{type(exc).__name__}: {exc}")
        out["traceback"] = traceback.format_exc()
        print(out["traceback"], flush=True)

    # ---------------------------------------------------------------- report
    by_uid = {f.uid: f for f in frames}
    per_frame = []
    atom_steps = 0
    if hook is not None:
        for uid, rec in hook.records.items():
            f = by_uid[uid]
            n_steps = rec.get("n_steps")
            atom_steps += len(f.Z) * (n_steps or 0)
            row = {
                "uid": uid, "reaction": f.reaction, "role": f.role, "n_atoms": int(len(f.Z)),
                "sigma_admission": rec.get("sigma_admission"),
                "flagged": rec["flagged"], "flag_phase": rec.get("flag_phase"),
                "sigma_at_flag": rec.get("sigma_at_flag"),
                "step_graduated": rec.get("graduated_step"), "admitted_step": rec.get("admitted_step"),
                "n_steps": n_steps, "reason": rec.get("reason", "unfinished"),
                "fmax_at_graduation": rec.get("fmax_at_graduation"),
                "converged": (rec.get("reason") == "dynamics" and rec.get("fmax_at_graduation") is not None
                              and rec["fmax_at_graduation"] <= a.fmax),
                "sigma_trace": [[c.get("local_step"), c["sigma"]] for c in rec["checks"]],
                "energy_admission": rec["checks"][0].get("energy") if rec["checks"] else None,
            }
            if a.score_labels and f.F is not None and "at_admission" in rec:
                row["force_rmse_admission"] = {
                    m: float(np.sqrt(np.mean((F - f.F) ** 2))) for m, F in rec["at_admission"]["forces"].items()}
            per_frame.append(row)
        with open(a.queue_out, "wb") as fh:  # oracle queue: flagged geometry + per-model forces
            pickle.dump([{"uid": k, "sigma": hook.records[k]["sigma_at_flag"],
                          "step": hook.records[k]["flag_step"], **hook.records[k].get("at_flag", {})}
                         for k in hook.flagged], fh)

    n_adm_events = sum(1 for e in (hook.events if hook else []) if e["event"] == "admission")
    flagged = [r for r in per_frame if r["flagged"]]
    summary = {
        "n_frames": len(frames), "n_scored": len(per_frame), "n_flagged": len(flagged),
        "n_flagged_at_admission": sum(r["flag_phase"] == "admission" for r in flagged),
        "n_flagged_during_relaxation": sum(r["flag_phase"] == "step" for r in flagged),
        "n_converged": sum(r["converged"] for r in per_frame),
        "n_budget_exhausted": sum(r["reason"] == "dynamics" and not r["converged"] for r in per_frame),
        "n_unfinished": sum(r["reason"] == "unfinished" for r in per_frame),
        "refill_events": max(n_adm_events - 1, 0),
        "systems_admitted_by_refill": sum(e["n"] for e in (hook.events if hook else [])[1:]
                                          if e["event"] == "admission"),
        "wall_seconds": t_wall,
        "atom_steps_per_s": atom_steps / t_wall if t_wall else None,
        "frames_per_s": len(per_frame) / t_wall if t_wall else None,
        "hook": hook.summary() if hook else None,
        "hook_gather_fraction": (hook.gather_seconds / t_wall) if (hook and t_wall) else None,
        "sigma_admission_quantiles": (np.quantile([r["sigma_admission"] for r in per_frame
                                                   if r["sigma_admission"] is not None],
                                                  [0.1, 0.5, 0.7, 0.9]).tolist() if per_frame else None),
    }
    if a.score_labels and per_frame and "force_rmse_admission" in per_frame[0]:
        from trustswarm.config import WRONG_F

        def mean_err(rows):
            v = [np.mean(list(r["force_rmse_admission"].values())) for r in rows if "force_rmse_admission" in r]
            return float(np.mean(v)) if v else None
        summary["label_check"] = {
            "mean_committee_force_rmse_flagged": mean_err(flagged),
            "mean_committee_force_rmse_unflagged": mean_err([r for r in per_frame if not r["flagged"]]),
            "note": f"post-hoc only; WRONG_F={WRONG_F}",
        }
    out["summary"] = summary
    out["frames"] = per_frame
    for n, c in clients.items():
        try:
            c.close()
        except Exception as exc:
            out["errors"].append(f"close {n}: {exc}")
    if hook is not None:
        hook.close()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(summary, indent=1, default=float), flush=True)
    print("SCREEN_DONE", flush=True)
    return 0 if not out["errors"] else 2


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    raise SystemExit(main())
