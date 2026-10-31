"""Build the Fittest frame set from Transition1x and cache every model's predictions.

Each model agent runs in its own process (``fittest.model_server``); this script
only ships coordinates to them and stores what comes back (DESIGN.md §9).

    python scripts/predict_pool.py --h5 $FITTEST_T1X --build-frames \
        --models aimnet2,aimnet2_m0,aimnet2_m1,aimnet2_m2,aimnet2_m3,orb,mace_mpa0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fittest import paths, t1x  # noqa: E402
from fittest.cache import load_frames, save_frames, save_predictions  # noqa: E402
from fittest.data import assign_splits  # noqa: E402
from fittest.model_server import spawn  # noqa: E402


def build_frames(h5: str, n_path: int, seed: int, max_test: int | None, max_val: int | None):
    frames = (t1x.sample_split(h5, "test", max_test or 10**9, n_path, seed=seed)
              + t1x.sample_split(h5, "val", max_val or 10**9, n_path, seed=seed + 1))
    return assign_splits(frames, seed=seed)


def predict_model(model: str, frames, port: int, batch_atoms: int, log_dir: str):
    python = paths.model_python(model)  # $FITTEST_ORB_PYTHON / $FITTEST_CORE_PYTHON
    client = spawn(model, port, python=python, log_dir=log_dir).connect(timeout=900)
    info = client.info()
    energies, forces, seconds = [], [], 0.0
    t_wall = time.perf_counter()
    i = 0
    try:
        while i < len(frames):
            chunk, n = [], 0
            while i < len(frames) and (n + frames[i].n_atoms <= batch_atoms or not chunk):
                chunk.append(frames[i]); n += frames[i].n_atoms; i += 1
            Z = np.concatenate([f.Z for f in chunk])
            R = np.concatenate([f.R for f in chunk])
            b = np.concatenate([np.full(f.n_atoms, k) for k, f in enumerate(chunk)])
            out = client.predict(Z, R, b)
            seconds += out["seconds"]
            energies.append(out["energy"])
            offs = np.cumsum([0] + [f.n_atoms for f in chunk])
            forces.extend(out["forces"][offs[k]:offs[k + 1]] for k in range(len(chunk)))
    finally:
        client.close()
    n_atoms = int(sum(f.n_atoms for f in frames))
    meta = {"level_of_theory": info.get("level_of_theory"), "server_pid": info.get("pid"),
            "python": info.get("python"), "model_seconds": seconds,
            "wall_seconds": time.perf_counter() - t_wall, "n_frames": len(frames),
            "n_atoms": n_atoms, "atoms_per_second": n_atoms / max(seconds, 1e-9)}
    return np.concatenate(energies), forces, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default=paths.t1x_path(), help="Transition1x HDF5 [$FITTEST_T1X]")
    ap.add_argument("--frames", default="data/frames.pkl")
    ap.add_argument("--preds", default="data/preds")
    ap.add_argument("--build-frames", action="store_true")
    ap.add_argument("--n-path", type=int, default=8)
    ap.add_argument("--max-test", type=int, default=None)
    ap.add_argument("--max-val", type=int, default=None)
    ap.add_argument("--models", default="aimnet2,aimnet2_m0,aimnet2_m1,aimnet2_m2,aimnet2_m3,orb,mace_mpa0")
    ap.add_argument("--base-port", "--port", dest="port", type=int, default=6000,
                    help="model k of --models listens on base-port + k")
    ap.add_argument("--batch-atoms", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-dir", default=paths.log_dir(), help="server logs [$FITTEST_LOG_DIR]")
    a = ap.parse_args()

    if a.build_frames or not os.path.exists(a.frames):
        frames = build_frames(a.h5, a.n_path, a.seed, a.max_test, a.max_val)
        save_frames(a.frames, frames)
        counts = {}
        for f in frames:
            counts[f.split] = counts.get(f.split, 0) + 1
        print("frames", len(frames), json.dumps(counts), flush=True)
    frames = load_frames(a.frames)

    status = {}
    for k, model in enumerate(m for m in a.models.split(",") if m):
        t0 = time.time()
        try:
            E, F, meta = predict_model(model, frames, a.port + k, a.batch_atoms, a.log_dir)
            save_predictions(os.path.join(a.preds, f"{model}.npz"), model,
                             [f.uid for f in frames], E, F, meta)
            Fd = np.concatenate([f.F for f in frames]); Fp = np.concatenate(F)
            status[model] = {"ok": True, "force_rmse_all": float(np.sqrt(((Fp - Fd) ** 2).mean())),
                             **{k2: meta[k2] for k2 in ("atoms_per_second", "wall_seconds")}}
        except Exception as exc:  # report and continue with the other models
            status[model] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        print(model, json.dumps(status[model]), f"{time.time() - t0:.0f}s", flush=True)
    os.makedirs("results", exist_ok=True)
    with open("results/predict_status.json", "w") as fh:
        json.dump(status, fh, indent=2)


if __name__ == "__main__":
    main()
