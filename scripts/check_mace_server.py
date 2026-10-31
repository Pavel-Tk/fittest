"""Smoke-check the MACE model server (Toolkit MACEWrapper backend) on Transition1x test frames.

    python scripts/check_mace_server.py --base-port 6100 --n-reactions 5 --n-path 2

Prints force RMSE vs DFT (wB97x/6-31G(d)), relative-energy MAE within reactions and throughput;
writes ``results/check_mace_server.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trustswarm import paths, t1x  # noqa: E402
from trustswarm.model_server import spawn  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default=paths.t1x_path(), help="Transition1x HDF5 [$TRUSTSWARM_T1X]")
    ap.add_argument("--base-port", "--port", dest="port", type=int, default=6100)
    ap.add_argument("--model", default="mace_mpa0")
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--n-reactions", type=int, default=5)
    ap.add_argument("--n-path", type=int, default=2)
    ap.add_argument("--log-dir", default=paths.log_dir(), help="server logs [$TRUSTSWARM_LOG_DIR]")
    ap.add_argument("--python", default=paths.core_python(),
                    help="server interpreter [$TRUSTSWARM_CORE_PYTHON]")
    ap.add_argument("--out", default=str(ROOT / "results" / "check_mace_server.json"))
    a = ap.parse_args()

    frames = t1x.sample_split(a.h5, "test", a.n_reactions, a.n_path, seed=0)
    Z = np.concatenate([f.Z for f in frames])
    R = np.concatenate([f.R for f in frames])
    b = np.concatenate([np.full(len(f.Z), i) for i, f in enumerate(frames)])
    Fd = np.concatenate([f.F for f in frames])

    t0 = time.perf_counter()
    c = spawn(a.model, a.port, python=a.python, log_dir=a.log_dir, model_path=a.model_path)
    try:
        c.connect(timeout=900)
        t_up = time.perf_counter() - t0
        c.predict(Z, R, b)  # warm-up (CUDA kernels, cuEq)
        times = []
        for _ in range(3):
            o = c.predict(Z, R, b)
            times.append(o["seconds"])
    except Exception as exc:
        log = Path(a.log_dir) / f"server_{a.model}.log"
        print("FAIL", exc)
        print(log.read_text()[-3000:] if log.exists() else "(no log)")
        return 1
    finally:
        c.close()

    F, E = o["forces"], o["energy"]
    rmse_all = float(np.sqrt(np.mean((F - Fd) ** 2)))
    per_frame = [float(np.sqrt(np.mean((F[b == i] - f.F) ** 2))) for i, f in enumerate(frames)]
    # Relative energies within each reaction (reactant-referenced): level-of-theory offset cancels.
    rel_err = []
    by_rxn: dict[str, int] = {f.reaction: i for i, f in enumerate(frames) if f.role == "reactant"}
    for i, f in enumerate(frames):
        r = by_rxn.get(f.reaction)
        if r is not None and i != r:
            rel_err.append(abs((E[i] - E[r]) - (f.E - frames[r].E)))
    sec = float(np.median(times))
    res = {
        "model": a.model, "model_path": a.model_path, "n_frames": len(frames), "n_atoms": int(len(Z)),
        "force_rmse_all": rmse_all, "force_rmse_frame_mean": float(np.mean(per_frame)),
        "force_rmse_frame_median": float(np.median(per_frame)),
        "rel_energy_mae": float(np.mean(rel_err)) if rel_err else None,
        "predict_seconds_median": sec, "atoms_per_s": len(Z) / sec, "systems_per_s": len(frames) / sec,
        "server_startup_seconds": t_up, "finite": bool(np.isfinite(F).all() and np.isfinite(E).all()),
    }
    print(json.dumps(res, indent=1))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print("MACE_SERVER_CHECK_DONE")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    raise SystemExit(main())
