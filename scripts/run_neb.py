"""Batched NEB on the 8 bundled Transition1x reactions with the DisagreementHook watching.

Follows ``nvalchemi-toolkit/examples/advanced/11_batched_neb.py``: one ``Batch`` holds all
8 bands x 10 images; IDPP pre-relaxation; climbing-image NEB driven by a Toolkit model
(default: the full AIMNet2 calculator, the same potential as the ``aimnet2`` committee
member). ``DisagreementHook(graduate=False)`` observes the band at admission and every
``--every`` steps (it never removes images).

The scientific question: does cross-model disagreement sigma_F peak at the transition
state, where model errors are largest? We answer it on two sets of geometries:

1. the **DFT reference band** (final T1x v4 NEB images, wB97x/6-31G(d)): sigma_F of the
   committee per image vs each model's |dE_model - dE_DFT| (reactant-referenced), and per-
   image force RMSE vs DFT where the image is found in the T1x HDF5 (exact geometry match);
2. the **model's own converged band**: sigma_F per image (label-free) and the NEB barrier
   error vs the DFT barrier.

    python3 scripts/run_neb.py --driver aimnet2

Outputs ``results/neb.json`` and ``results/neb_sigma.png``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trustswarm import paths  # noqa: E402



def spearman(x, y) -> float | None:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return None
    rx = np.argsort(np.argsort(x[ok])).astype(float)
    ry = np.argsort(np.argsort(y[ok])).astype(float)
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def match_dft_forces(h5_path, reaction_id, Z, images, tol=2e-3):
    """Find each reference image in the T1x HDF5 path arrays; return DFT forces or None per image."""
    try:
        import h5py

        from trustswarm.t1x import FORCES_KEY

        formula, rxn = reaction_id.split("/")
        with h5py.File(h5_path, "r") as f:
            g = f["data"][formula][rxn]
            if not np.array_equal(np.asarray(g["atomic_numbers"][()]), np.asarray(Z)):
                return [None] * len(images)
            P = np.asarray(g["positions"][()], dtype=np.float64)
            Fd = np.asarray(g[FORCES_KEY][()], dtype=np.float64)
        out = []
        for R in images:
            d = np.abs(P - np.asarray(R)[None]).reshape(len(P), -1).max(1)
            j = int(np.argmin(d))
            out.append(Fd[j] if d[j] < tol else None)
        return out
    except Exception:
        return [None] * len(images)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--neb-dir", default=paths.neb_dir(),
                    help="toolkit examples/advanced/batched_neb_endpoints [$TRUSTSWARM_NEB_DIR]")
    ap.add_argument("--h5", default=paths.t1x_path(), help="Transition1x HDF5 [$TRUSTSWARM_T1X]")
    ap.add_argument("--driver", default="aimnet2", choices=["aimnet2", "mace"])
    ap.add_argument("--models", default="mace_mpa0,aimnet2,orb")
    ap.add_argument("--images", type=int, default=10)
    ap.add_argument("--every", type=int, default=25)
    ap.add_argument("--n-steps", type=int, default=500)
    ap.add_argument("--fmax", type=float, default=0.05)
    ap.add_argument("--tau", type=float, default=0.30, help="flag threshold (recorded only)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--log-dir", default=paths.log_dir(), help="server logs [$TRUSTSWARM_LOG_DIR]")
    ap.add_argument("--base-port", type=int, default=6103,
                    help="committee member k of --models listens on base-port + k")
    ap.add_argument("--out", default=str(ROOT / "results" / "neb.json"))
    ap.add_argument("--fig", default=str(ROOT / "results" / "neb_sigma.png"))
    a = ap.parse_args()

    import torch
    from nvalchemi.data import AtomicData, Batch
    from nvalchemi.dynamics.mep import NEB, ClimbingImageConfig, IDPPModel, interpolate_paths, prepare_idpp_targets

    from trustswarm.hook import DisagreementHook, DisagreementMonitor
    from trustswarm.model_server import spawn

    dev = torch.device(a.device)
    endpoints = json.loads((Path(a.neb_dir) / "endpoints.json").read_text())["paths"]
    refs = json.loads((Path(a.neb_dir) / "dft_reference.json").read_text())["paths"]
    rids = [r["reaction_id"] for r in endpoints]
    assert [r["reaction_id"] for r in refs] == rids, "reference order mismatch"
    out: dict = {"config": vars(a), "servers": {}, "errors": [], "reactions": []}

    # ---------------------------------------------------------------- committee
    procs = {}
    for k, n in enumerate(m for m in a.models.split(",") if m):
        port, py = a.base_port + k, paths.model_python(n)
        try:
            procs[n] = spawn(n, port, python=py, log_dir=a.log_dir)
        except Exception as exc:
            out["servers"][n] = f"spawn failed: {exc}"
    clients = {}
    for n, c in procs.items():
        try:
            clients[n] = c.connect(timeout=900)
            out["servers"][n] = "ok"
        except Exception as exc:
            out["servers"][n] = f"connect failed: {exc}"
    print("[neb] servers:", out["servers"], flush=True)
    if len(clients) < 2:
        out["errors"].append("fewer than two committee members")
        Path(a.out).write_text(json.dumps(out, indent=1))
        return 1
    mon = DisagreementMonitor(clients, tau=a.tau, keep_forces="last")

    # ---------------------------------------------------------------- driver
    if a.driver == "aimnet2":
        from trustswarm.models.aimnet2_calc import AIMNet2CalculatorWrapper

        model = AIMNet2CalculatorWrapper("aimnet2", device=dev).eval()
    else:
        from trustswarm.models.mace_batched import BatchedMACE

        model = BatchedMACE(device=a.device).model
    out["config"]["driver_model"] = type(model).__name__

    def ad(Z, R, q):
        return AtomicData(positions=torch.tensor(R, dtype=torch.float32),
                          atomic_numbers=torch.tensor(Z, dtype=torch.long),
                          charge=torch.tensor([[float(q)]], dtype=torch.float32))

    def init_fields(b):
        b.energy = torch.zeros(b.num_graphs, 1, dtype=b.positions.dtype, device=b.device)
        b.forces = torch.zeros_like(b.positions)
        b.velocities = torch.zeros_like(b.positions)

    P, n_img = len(endpoints), a.images
    hook = None
    t_neb = 0.0
    try:
        initial = Batch.from_data_list([ad(r["atomic_numbers"], r["reactant_positions"], r["charge"]) for r in endpoints], device=dev)
        final = Batch.from_data_list([ad(r["atomic_numbers"], r["product_positions"], r["charge"]) for r in endpoints], device=dev)
        linear = interpolate_paths(initial, final, n_img)
        idpp_band = prepare_idpp_targets(linear.clone())
        init_fields(idpp_band)
        idpp_band = NEB(model=IDPPModel(), fmax=0.1, n_steps=100).run(idpp_band)
        band = linear.clone()
        band.positions.copy_(idpp_band.positions)
        init_fields(band)

        hook = DisagreementHook(clients, tau=a.tau, every=a.every, stage="both", graduate=False,
                                keep_forces="last")
        neb = NEB(model=model, spring=0.1, method="improved_tangent", fmax=a.fmax, n_steps=a.n_steps,
                  climbing=ClimbingImageConfig(regular_fmax=0.5), extra_hooks=[hook])
        torch.cuda.synchronize() if dev.type == "cuda" else None
        t0 = time.perf_counter()
        band = neb.run(band)
        torch.cuda.synchronize() if dev.type == "cuda" else None
        t_neb = time.perf_counter() - t0
        status = band.status.view(-1).cpu().numpy().reshape(P, n_img)
        E_band = band.energy.detach().double().cpu().numpy().reshape(P, n_img)
        R_band = band.positions.detach().double().cpu().numpy()
        ptr = band.batch_ptr.cpu().numpy()
    except Exception as exc:
        out["errors"].append(f"NEB failed: {type(exc).__name__}: {exc}")
        out["traceback"] = traceback.format_exc()
        print(out["traceback"], flush=True)
        R_band = None

    # ---------------------------------------------------------------- committee on both bands
    def committee(images_Z, images_R, keys, phase):
        Z = np.concatenate(images_Z)
        R = np.concatenate(images_R)
        b = np.concatenate([np.full(len(z), i) for i, z in enumerate(images_Z)])
        sig, _ = mon.check(keys, Z, R, b, step=0, phase=phase)
        recs = [mon.records[k]["checks"][-1] for k in keys]
        return sig, [r.get("energy", {}) for r in recs]

    for p, (ep, ref) in enumerate(zip(endpoints, refs)):
        Z = np.asarray(ref["atomic_numbers"])
        ref_R = [np.asarray(x, float) for x in ref["positions"]]
        e_dft = np.asarray(ref["relative_energies_ev"], float)
        ts = int(ref["transition_state_index"])
        row: dict = {"reaction_id": ref["reaction_id"], "dft_ts_index": ts, "dft_barrier": float(e_dft[ts]),
                     "dft_relative_energies": e_dft.tolist()}
        try:
            sig_ref, E_ref = committee([Z] * len(ref_R), ref_R, [("dft", p, i) for i in range(len(ref_R))], "dft_band")
            names = sorted(E_ref[0])
            rel = {m: np.array([e[m] for e in E_ref]) - E_ref[0][m] for m in names}
            err = {m: np.abs(rel[m] - e_dft) for m in names}
            Fd = match_dft_forces(a.h5, ref["reaction_id"], Z, ref_R)
            frmse = {m: [None if Fd[i] is None else
                         float(np.sqrt(np.mean((mon.records[("dft", p, i)]["last"]["forces"][m] - Fd[i]) ** 2)))
                         if "last" in mon.records[("dft", p, i)] else None
                         for i in range(len(ref_R))] for m in names}
            mean_err = np.mean([err[m] for m in names], axis=0)
            row.update({
                "sigma_dft_band": sig_ref.tolist(),
                "model_rel_energy_dft_band": {m: rel[m].tolist() for m in names},
                "abs_rel_energy_error_dft_band": {m: err[m].tolist() for m in names},
                "force_rmse_dft_band": frmse,
                "n_images_with_dft_forces": int(sum(x is not None for x in Fd)),
                "argmax_sigma_dft_band": int(np.argmax(sig_ref)),
                "argmax_mean_error_dft_band": int(np.argmax(mean_err)),
                "sigma_rank_of_ts": int((sig_ref > sig_ref[ts]).sum()),  # 0 = TS has the largest sigma
                "spearman_sigma_vs_mean_abs_rel_e_err": spearman(sig_ref[1:], mean_err[1:]),
            })
            fr_mean = [np.mean([frmse[m][i] for m in names]) if all(frmse[m][i] is not None for m in names)
                       else np.nan for i in range(len(ref_R))]
            row["spearman_sigma_vs_mean_force_rmse"] = spearman(sig_ref, fr_mean)
            row["mean_force_rmse_dft_band"] = [None if not np.isfinite(v) else float(v) for v in fr_mean]
            row["argmax_mean_force_rmse_dft_band"] = (int(np.nanargmax(fr_mean))
                                                      if np.isfinite(fr_mean).any() else None)
        except Exception as exc:
            out["errors"].append(f"{ref['reaction_id']} dft band: {type(exc).__name__}: {exc}")
        if R_band is not None:
            try:
                g0 = p * n_img
                imgs = [R_band[ptr[g0 + i]:ptr[g0 + i + 1]] for i in range(n_img)]
                sig_m, E_m = committee([Z] * n_img, imgs, [("model", p, i) for i in range(n_img)], "model_band")
                rel_drv = E_band[p] - E_band[p, 0]
                mts = int(np.argmax(rel_drv))
                row.update({
                    "sigma_model_band": sig_m.tolist(),
                    "driver_rel_energies": rel_drv.tolist(), "driver_ts_index": mts,
                    "driver_barrier": float(rel_drv[mts]),
                    "driver_barrier_error": float(rel_drv[mts] - e_dft[ts]),
                    "committee_barriers_on_model_band": {m: float(max(E_m[i][m] - E_m[0][m] for i in range(n_img)))
                                                         for m in sorted(E_m[0])},
                    "argmax_sigma_model_band": int(np.argmax(sig_m)),
                    "neb_status": status[p].tolist(), "neb_converged": bool((status[p] >= 2).all()),
                    "sigma_trace_hook": {str(g - g0): [[c["step"], c["sigma"]] for c in hook.records[g]["checks"]]
                                         for g in range(g0, g0 + n_img) if g in hook.records},
                })
            except Exception as exc:
                out["errors"].append(f"{ref['reaction_id']} model band: {type(exc).__name__}: {exc}")
        out["reactions"].append(row)

    rx = out["reactions"]
    ok = [r for r in rx if "sigma_dft_band" in r]
    out["summary"] = {
        "n_reactions": len(rx), "neb_seconds": t_neb,
        "n_neb_converged": sum(bool(r.get("neb_converged")) for r in rx),
        "ts_has_max_sigma_dft_band": sum(r["sigma_rank_of_ts"] == 0 for r in ok),
        "ts_within_1_image_of_max_sigma": sum(abs(r["argmax_sigma_dft_band"] - r["dft_ts_index"]) <= 1 for r in ok),
        "max_sigma_coincides_with_max_error": sum(r["argmax_sigma_dft_band"] == r["argmax_mean_error_dft_band"] for r in ok),
        "max_sigma_coincides_with_max_force_rmse": sum(r.get("argmax_mean_force_rmse_dft_band") == r["argmax_sigma_dft_band"] for r in ok),
        "ts_has_max_force_rmse": sum(r.get("argmax_mean_force_rmse_dft_band") == r["dft_ts_index"] for r in ok),
        "median_spearman_sigma_vs_force_rmse": (float(np.median([r["spearman_sigma_vs_mean_force_rmse"] for r in ok
                                                                 if r.get("spearman_sigma_vs_mean_force_rmse") is not None]))
                                                if any(r.get("spearman_sigma_vs_mean_force_rmse") is not None for r in ok) else None),
        "median_spearman_sigma_vs_rel_e_err": (float(np.median([r["spearman_sigma_vs_mean_abs_rel_e_err"] for r in ok
                                                                if r["spearman_sigma_vs_mean_abs_rel_e_err"] is not None]))
                                               if ok else None),
        "pooled_spearman_sigma_vs_rel_e_err": spearman(
            np.concatenate([r["sigma_dft_band"][1:] for r in ok]) if ok else [],
            np.concatenate([np.mean(list(r["abs_rel_energy_error_dft_band"].values()), axis=0)[1:] for r in ok]) if ok else []),
        "pooled_spearman_sigma_vs_force_rmse": spearman(
            np.concatenate([r["sigma_dft_band"] for r in ok]) if ok else [],
            np.concatenate([[np.mean([r["force_rmse_dft_band"][m][i] for m in r["force_rmse_dft_band"]])
                             if all(r["force_rmse_dft_band"][m][i] is not None for m in r["force_rmse_dft_band"])
                             else np.nan for i in range(len(r["sigma_dft_band"]))] for r in ok]) if ok else []),
        "mean_abs_driver_barrier_error": (float(np.mean([abs(r["driver_barrier_error"]) for r in rx
                                                         if "driver_barrier_error" in r])) if R_band is not None else None),
        "hook": hook.summary() if hook else None,
        "committee": mon.summary(),
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out["summary"], indent=1, default=float), flush=True)

    try:
        plot(ok, a.fig, out["config"])
    except Exception as exc:
        out["errors"].append(f"plot: {exc}")
        Path(a.out).write_text(json.dumps(out, indent=1, default=float))
    for c in clients.values():
        try:
            c.close()
        except Exception:
            pass
    mon.close()
    if hook:
        hook.close()
    print("NEB_DONE", flush=True)
    return 0 if not out["errors"] else 2


def plot(rows, path, cfg):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(rows)
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.3 * nrow), squeeze=False)
    colors = {"mace_mpa0": "#1f77b4", "aimnet2": "#2ca02c", "orb": "#9467bd"}
    for ax, r in zip(axes.flat, rows):
        x = np.arange(len(r["sigma_dft_band"]))
        ax.plot(x, r["sigma_dft_band"], "-o", color="black", lw=2, ms=4, label="σ_F (DFT band)")
        if "sigma_model_band" in r:
            ax.plot(np.arange(len(r["sigma_model_band"])), r["sigma_model_band"], "--", color="gray",
                    lw=1.2, label="σ_F (model NEB band)")
        ax.axvline(r["dft_ts_index"], color="red", ls=":", lw=1.2, label="DFT TS")
        ax.set_xlabel("image")
        ax.set_ylabel("σ_F  [eV/Å]")
        ax2 = ax.twinx()
        for m, e in r["abs_rel_energy_error_dft_band"].items():
            ax2.plot(x, e, "-", color=colors.get(m, None), alpha=0.75, lw=1, label=f"|ΔE err| {m}")
        fr = r.get("force_rmse_dft_band", {})
        if fr and all(v is not None for vals in fr.values() for v in vals):
            ax2.plot(x, np.mean([fr[m] for m in fr], axis=0), "-s", color="orange", ms=3, lw=1.2,
                     label="mean force RMSE vs DFT [eV/Å]")
        ax2.set_ylabel("error  [eV | eV/Å]")
        title = r["reaction_id"]
        if r.get("driver_barrier_error") is not None:
            title += f"\nNEB barrier err {r['driver_barrier_error']:+.2f} eV"
        ax.set_title(title, fontsize=9)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    h1, l1 = axes.flat[0].get_legend_handles_labels()
    twins = [t for t in fig.axes if t not in list(axes.flat)]
    h2, l2 = twins[0].get_legend_handles_labels() if twins else ([], [])
    fig.legend(h1 + h2, l1 + l2, loc="lower center", ncol=6, fontsize=8, frameon=False)
    fig.suptitle("Cross-model force disagreement σ_F (MACE-MPA-0, AIMNet2, Orb) along Transition1x NEB bands "
                 "vs per-model relative-energy error (ωB97x/6-31G(d))", fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    raise SystemExit(main())
