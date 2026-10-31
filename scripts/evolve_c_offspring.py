"""Give birth to one run-C niche specialist (AIMNet2, Orb-v3 or MACE-MPA-0 head fine-tune, GPU).

    python scripts/evolve_c_offspring.py --frames data/frames.pkl --spec <job>.json --device cuda

The spec (written by ``scripts/evolve_c.py``) holds: ``id``, ``family`` (aimnet2 | orb | mace),
``parent_ckpt`` (None = the foundation checkpoint, else the head-state file of an earlier
offspring), ``recipe`` (lr, energy_weight, force_weight, steps, reactions_per_batch), ``seed``,
``train_uids`` (the niche's breed labels), ``predict_uids`` (pool + eval) and output paths.

Loss (pre-registered, results/evolution_c/PREREG.md):
    w_F * mean_atoms |F - F_DFT|^2  +  w_E * mean_frames ((E_i - E_r) - (E_i - E_r)_DFT)^2
where r is the reactant of frame i's reaction (its reference energy is supplied by the Oracle
contract, as in runs A/B; its forces are not used). Only the readout/head parameters train:
AIMNet2 ``mlps.2`` + ``outputs``; Orb ``heads``; MACE ``readouts``.

Writes predictions (``fittest.cache`` npz), the trainable head state (``<ckpt>.pt``) and a result
JSON with training curves. Imports only numpy-level fittest modules, so it runs in the Orb venv.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fittest.cache import load_frames, save_predictions  # noqa: E402
from fittest.data import assign_leakfree_splits, reactant_index  # noqa: E402

HEAD_PREFIX = {"aimnet2": ("mlps.2", "outputs"), "orb": ("heads",), "mace": ("model.readouts",)}


# ------------------------------------------------------------------ family adapters
class AIMNet2Adapter:
    def __init__(self, device: str):
        import torch
        from aimnet.calculators import AIMNet2Calculator

        self.torch, self.device = torch, device
        self.calc = AIMNet2Calculator("aimnet2", device=device, train=True)
        self.module = self.calc.model

    def ef(self, frames, train: bool):
        torch = self.torch
        self.calc._train = train
        Z = np.concatenate([f.Z for f in frames])
        R = np.concatenate([f.R for f in frames])
        b = np.repeat(np.arange(len(frames)), [len(f.Z) for f in frames])
        out = self.calc({"coord": torch.as_tensor(R, dtype=torch.float32, device=self.device),
                         "numbers": torch.as_tensor(Z, dtype=torch.long, device=self.device),
                         "charge": torch.zeros(len(frames), dtype=torch.float32, device=self.device),
                         "mol_idx": torch.as_tensor(b, dtype=torch.long, device=self.device)}, forces=True)
        return out["energy"].reshape(-1), out["forces"].reshape(-1, 3)


class OrbAdapter:
    def __init__(self, device: str):
        import torch
        from orb_models.forcefield import pretrained

        self.torch, self.device = torch, device
        loaded = pretrained.orb_v3_conservative_inf_omat(device=device, precision="float32-highest",
                                                         compile=False)
        self.module, self.adapter = loaded if isinstance(loaded, tuple) else (loaded, None)

    def ef(self, frames, train: bool):
        from ase import Atoms

        atoms = [Atoms(numbers=f.Z, positions=f.R) for f in frames]
        if self.adapter is not None:
            g = self.adapter.from_ase_atoms_list(atoms, device=self.device)
        else:
            from orb_models.forcefield.atomic_system import ase_atoms_to_atom_graphs
            from orb_models.forcefield.base import batch_graphs

            g = batch_graphs([ase_atoms_to_atom_graphs(a, self.module.system_config, device=self.device)
                              for a in atoms])
        self.module.train(train)
        out = self.module(g)
        m = self.module
        E = m.heads[m.energy_name].denormalize(out[m.energy_name], g).reshape(-1)
        return E, out[m.grad_forces_name].reshape(-1, 3)


class MACEAdapter:
    def __init__(self, device: str):
        import torch
        from nvalchemi.models.mace import MACEWrapper

        self.torch, self.device = torch, device
        self.module = MACEWrapper.from_checkpoint("medium-mpa-0", device=torch.device(device),
                                                  dtype=torch.float32, enable_cueq=False,
                                                  compile_model=False)
        self.module.model_config.active_outputs = {"energy", "forces"}
        self.nc = self.module.model_config.neighbor_config

    def ef(self, frames, train: bool):
        from nvalchemi.neighbors import compute_neighbors

        from fittest.models.toolkit_io import batch_from_arrays

        Z = np.concatenate([f.Z for f in frames]).astype(np.int64)
        R = np.concatenate([f.R for f in frames]).astype(np.float64)
        b = np.repeat(np.arange(len(frames)), [len(f.Z) for f in frames])
        bt = batch_from_arrays(Z, R, b, device=self.torch.device(self.device))
        compute_neighbors(bt, config=self.nc)
        self.module.train(train)
        with self.torch.enable_grad():
            out = self.module(bt)
        return out["energy"].reshape(-1), out["forces"].reshape(-1, 3)


ADAPTERS = {"aimnet2": AIMNet2Adapter, "orb": OrbAdapter, "mace": MACEAdapter}


def head_params(module, family):
    pre = HEAD_PREFIX[family]
    named = [(n, p) for n, p in module.named_parameters() if n.startswith(pre)]
    if not named:
        raise RuntimeError(f"{family}: no parameters with prefix {pre}; "
                           f"first names: {[n for n, _ in list(module.named_parameters())[:8]]}")
    return named


def predict(ad, frames, max_atoms: int = 3000):
    E, F = [], []
    i = 0
    while i < len(frames):
        j, n = i, 0
        while j < len(frames) and (n + frames[j].n_atoms <= max_atoms or j == i):
            n += frames[j].n_atoms
            j += 1
        e, f = ad.ef(frames[i:j], train=False)
        E.append(e.detach().double().cpu().numpy())
        sizes = np.cumsum([0] + [fr.n_atoms for fr in frames[i:j]])
        fa = f.detach().double().cpu().numpy()
        F.extend(fa[sizes[k]:sizes[k + 1]] for k in range(j - i))
        i = j
    return np.concatenate(E), F


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="data/frames.pkl")
    ap.add_argument("--spec", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true", help="few steps, predict 50 frames")
    args = ap.parse_args(argv)
    spec = json.loads(Path(args.spec).read_text())
    t0 = time.time()
    import torch

    torch.manual_seed(int(spec["seed"]))
    rng = np.random.default_rng(int(spec["seed"]))
    frames, _ = assign_leakfree_splits(load_frames(args.frames), 0)
    by_uid = {f.uid: f for f in frames}
    reactant = reactant_index(frames)
    train = [by_uid[u] for u in spec["train_uids"]]
    assert all(f.split == "pool" for f in train), "offspring train on pool labels only"
    fam, rec = spec["family"], spec["recipe"]
    ad = ADAPTERS[fam](args.device)
    for p in ad.module.parameters():
        p.requires_grad_(False)
    named = head_params(ad.module, fam)
    if spec.get("parent_ckpt"):
        state = torch.load(spec["parent_ckpt"], map_location=args.device)
        missing = [n for n, _ in named if n not in state]
        if missing:
            raise KeyError(f"parent head state lacks {missing[:3]}")
        with torch.no_grad():
            for n, p in named:
                p.copy_(state[n].to(p.dtype))
    params = [p for _, p in named]
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.Adam(params, lr=float(rec["lr"]))
    rx = sorted({f.reaction for f in train})
    lab_by_rx = {r: [f for f in train if f.reaction == r] for r in rx}
    steps = 3 if args.smoke else int(rec["steps"])
    k = int(rec.get("reactions_per_batch", 4))
    wF, wE = float(rec["force_weight"]), float(rec["energy_weight"])
    curve = []
    for step in range(steps):
        pick = rng.choice(len(rx), size=min(k, len(rx)), replace=False)
        batch, ref_pos, lab_pos = [], [], []
        for r in (rx[i] for i in pick):
            rf = reactant[r]
            labs = lab_by_rx[r]
            batch.append(rf)
            rp = len(batch) - 1
            for f in labs:
                if f.uid == rf.uid:
                    lab_pos.append((rp, rp))
                else:
                    batch.append(f)
                    lab_pos.append((len(batch) - 1, rp))
        E, F = ad.ef(batch, train=True)
        sizes = np.cumsum([0] + [f.n_atoms for f in batch])
        fl, el, n_at = 0.0, 0.0, 0
        for (i, rp) in lab_pos:
            fr = batch[i]
            Ft = torch.as_tensor(fr.F, dtype=F.dtype, device=F.device)
            fl = fl + ((F[sizes[i]:sizes[i + 1]] - Ft) ** 2).sum()
            n_at += fr.n_atoms * 3
            d_true = float(fr.E) - float(batch[rp].E)
            el = el + ((E[i] - E[rp]) - d_true) ** 2
        loss = wF * fl / n_at + wE * el / len(lab_pos)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 10.0)
        opt.step()
        if step % 20 == 0 or step == steps - 1:
            curve.append({"step": step, "loss": float(loss.detach()), "force_mse": float(fl.detach() / n_at),
                          "rel_e_mse": float(el / len(lab_pos))})
            print(json.dumps(curve[-1]), flush=True)
    for p in params:
        p.requires_grad_(False)
    Path(spec["ckpt"]).parent.mkdir(parents=True, exist_ok=True)
    torch.save({n: p.detach().cpu() for n, p in named}, spec["ckpt"])
    uids = spec["predict_uids"][:50] if args.smoke else spec["predict_uids"]
    pf = [by_uid[u] for u in uids]
    t1 = time.time()
    E, F = predict(ad, pf)
    save_predictions(spec["out"], spec["id"], uids, E, F,
                     {"family": fam, "recipe": rec, "parent_ckpt": spec.get("parent_ckpt"),
                      "n_train": len(train)})
    res = {"id": spec["id"], "ckpt": spec["ckpt"], "curve": curve, "n_train": len(train),
           "n_train_reactions": len(rx), "n_head_params": int(sum(p.numel() for p in params)),
           "train_seconds": t1 - t0, "predict_seconds": time.time() - t1, "wall_seconds": time.time() - t0,
           "toolkit": {}}
    Path(spec["result"]).write_text(json.dumps(res, indent=1))
    print("DONE", json.dumps({k: v for k, v in res.items() if k != "curve"}), flush=True)


if __name__ == "__main__":
    main()
