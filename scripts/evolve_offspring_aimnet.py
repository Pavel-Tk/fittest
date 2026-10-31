"""Give birth to one AIMNet2 offspring for the evolutionary loop (run B; plain torch, CPU).

    python scripts/evolve_offspring_aimnet.py --frames data/frames.pkl --spec <job>.json --threads 1

Fine-tunes AIMNet2 (``aimnet2`` = ensemble member 0, ωB97M-D3) on the breed labels with
``AIMNet2Calculator(train=True)`` (forces by autograd, double backward for the force loss).

Loss per batch of labelled frames (plus, prediction only, the reactant geometry of each frame's
reaction)::

    L = w_F * mean_components (F_hat - F)^2
      + w_rel * mean_frames ((E_hat_i - E_hat_reactant) - (E_i - E_reactant))^2

The relative-energy target uses the reactant reference that the Oracle returns with every label
(DESIGN.md: "the reactant reference is computed once per reaction and not charged to the
budget"), so every label has a same-reaction partner; the job also reports how many labels have a
same-reaction partner *among the labelled frames themselves*. Absolute energies are never fitted
(AIMNet2 is ωB97M-D3/def2-TZVPP, T1x is ωB97x/6-31G(d): constant per-composition offsets cancel in
relative energies; part of any gain is learning the T1x level of theory from budgeted labels).

``recipe.trainable``: ``head`` (mlps.2.* + outputs.*), ``outputs`` (outputs.* only) or ``all``
(every parameter except the AEV basis). The checkpoint is the model ``state_dict``
(``<ckpt>/aimnet2_state.pt``); a child of an offspring loads its parent's state first.
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
from fittest.data import EVAL, POOL, assign_leakfree_splits, reactant_index  # noqa: E402

STATE_FILE = "aimnet2_state.pt"


def trainable(name: str, mode: str) -> bool:
    if name.startswith("aev."):
        return False
    if mode == "all":
        return True
    if mode == "outputs":
        return name.startswith("outputs.")
    return name.startswith("outputs.") or name.startswith("mlps.2.")  # head


def make_calc(parent: str, train: bool, device: str = "cpu"):
    import torch
    from aimnet.calculators import AIMNet2Calculator

    calc = AIMNet2Calculator("aimnet2", device=device, train=train)
    if parent != "aimnet2":
        state = torch.load(Path(parent) / STATE_FILE, map_location=device)
        calc.model.load_state_dict(state)
    return calc


def run_batch(calc, frames):
    import torch

    Z = np.concatenate([f.Z for f in frames])
    R = np.concatenate([f.R for f in frames])
    b = np.repeat(np.arange(len(frames)), [len(f.Z) for f in frames])
    out = calc({"coord": torch.tensor(R, dtype=torch.float32), "numbers": torch.tensor(Z, dtype=torch.long),
                "charge": torch.zeros(len(frames)), "mol_idx": torch.tensor(b, dtype=torch.long)}, forces=True)
    return out["energy"].reshape(-1), out["forces"].reshape(-1, 3)


def predict(calc, frames, chunk: int = 64):
    E, F = [], []
    for i in range(0, len(frames), chunk):
        fs = frames[i:i + chunk]
        e, f = run_batch(calc, fs)
        E.append(e.detach().double().numpy())
        off = np.cumsum([0] + [len(x.Z) for x in fs])
        f = f.detach().double().numpy()
        F.extend(f[off[k]:off[k + 1]].copy() for k in range(len(fs)))
    return np.concatenate(E), F


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="data/frames.pkl")
    ap.add_argument("--spec", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--data-seed", type=int, default=0)
    ap.add_argument("--predict-limit", type=int, default=0)
    args = ap.parse_args(argv)
    spec = json.loads(Path(args.spec).read_text())
    t0 = time.perf_counter()
    import torch

    torch.set_num_threads(max(1, args.threads))
    torch.manual_seed(int(spec["seed"]))
    rng = np.random.default_rng(int(spec["seed"]))
    frames, _ = assign_leakfree_splits(load_frames(args.frames), args.data_seed)
    by_uid = {f.uid: f for f in frames}
    train = [by_uid[u] for u in dict.fromkeys(spec["train_uids"])]
    hold = {u for u in spec["holdout_uids"]}
    if any(f.split not in (POOL,) for f in train) or hold & {f.uid for f in train}:
        raise ValueError("training labels must be pool frames disjoint from the hold-out")
    if any(f.E is None or f.F is None for f in train):
        raise ValueError("unlabelled training frame")
    react = reactant_index(frames)
    rel_true = {f.uid: float(f.E - react[f.reaction].E) for f in train}
    rx_count: dict[str, int] = {}
    for f in train:
        rx_count[f.reaction] = rx_count.get(f.reaction, 0) + 1
    n_partner = sum(1 for f in train if rx_count[f.reaction] >= 2)
    r = spec["recipe"]
    calc = make_calc(spec["parent"], train=True, device=args.device)
    params = []
    for name, p in calc.model.named_parameters():
        on = trainable(name, r["trainable"])
        p.requires_grad_(on)
        if on:
            params.append(p)
    n_params = int(sum(p.numel() for p in params))
    opt = torch.optim.Adam(params, lr=float(r["lr"]))
    bs = min(int(r["batch_size"]), len(train))
    steps_per_epoch = max(1, len(train) // bs)
    epochs = max(int(r["epochs"]), -(-int(r.get("min_steps", 0)) // steps_per_epoch))
    total = epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total, eta_min=float(r["lr"]) * 0.05)
    wF, wR = float(r["force_weight"]), float(r["rel_weight"])
    print(f"=== {spec['id']}: parent {spec['parent']}, {len(train)} train ({n_partner} with a labelled "
          f"same-reaction partner), {n_params} trainable params ({r['trainable']}), {epochs} epochs x "
          f"{steps_per_epoch} steps, recipe {r}", flush=True)
    history = []
    t_train = time.perf_counter()
    for ep in range(epochs):
        order = rng.permutation(len(train))
        tot = 0.0
        for s in range(steps_per_epoch):
            fs = [train[i] for i in order[s * bs:(s + 1) * bs]]
            reactants = list({react[f.reaction].uid: react[f.reaction] for f in fs}.values())
            ridx = {fr.uid: len(fs) + k for k, fr in enumerate(reactants)}
            E, F = run_batch(calc, fs + reactants)
            n_lab = sum(len(f.Z) for f in fs)
            F_true = torch.tensor(np.concatenate([f.F for f in fs]), dtype=torch.float32)
            lf = ((F[:n_lab] - F_true) ** 2).mean()
            e_lab = E[:len(fs)]
            e_ref = E[[ridx[react[f.reaction].uid] for f in fs]]
            rt = torch.tensor([rel_true[f.uid] for f in fs], dtype=torch.float32)
            lr_ = (((e_lab - e_ref) - rt) ** 2).mean()
            loss = wF * lf + wR * lr_
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss)
        history.append(tot / steps_per_epoch)
        if ep == 0 or (ep + 1) % max(1, epochs // 8) == 0:
            print(f"  epoch {ep + 1:4d}  loss {history[-1]:.5f}  ({time.perf_counter() - t_train:.1f} s)", flush=True)
    train_seconds = time.perf_counter() - t_train
    ckpt = Path(spec["ckpt"])
    ckpt.mkdir(parents=True, exist_ok=True)
    torch.save(calc.model.state_dict(), ckpt / STATE_FILE)
    (ckpt / "fittest_aimnet2.json").write_text(json.dumps({
        "format": "fittest.aimnet2_ft/1", "base": "aimnet2", "parent": spec["parent"], "recipe": r,
        "n_train": len(train), "n_with_labelled_partner": n_partner, "trainable_params": n_params,
        "epochs_used": epochs, "loss_history": history}, indent=1))
    del calc
    pcalc = make_calc(str(ckpt), train=False, device=args.device)
    targets = [f for f in frames if f.split in (POOL, EVAL)]
    if args.predict_limit:
        targets = targets[: args.predict_limit]
    t1 = time.perf_counter()
    E, F = predict(pcalc, targets)
    predict_seconds = time.perf_counter() - t1
    meta = {"base_model": "aimnet2", "parent": spec["parent"], "recipe": r, "checkpoint": str(ckpt),
            "train_uids": [f.uid for f in train], "n_with_labelled_partner": n_partner,
            "n_with_reactant_reference": len(train), "trainable_params": n_params, "epochs_used": epochs,
            "train_seconds": train_seconds, "predict_wall_seconds": predict_seconds, "loss_history": history,
            "level_of_theory_note": "AIMNet2 wB97M-D3 fine-tuned on wB97x/6-31G(d) T1x labels"}
    save_predictions(spec["out"], model=spec["id"], uids=[f.uid for f in targets], energies=E, forces=F, meta=meta)
    toolkit = {"accepted": None, "worst_ratio": None,
               "detail": "ALCHEMI acceptance report not applicable (AIMNet2 is not an ALCHEMI-wrapped model)"}
    result = {"id": spec["id"], "out": spec["out"], "ckpt": str(ckpt), "toolkit": toolkit,
              "wall_seconds": time.perf_counter() - t0, "train_seconds": train_seconds,
              "predict_seconds": predict_seconds, "n_with_labelled_partner": n_partner, "n_train": len(train)}
    Path(spec["result"]).write_text(json.dumps(result, indent=1))
    print(f"--- {spec['id']}: train {train_seconds:.0f} s, predict {predict_seconds:.0f} s, "
          f"total {result['wall_seconds']:.0f} s", flush=True)


if __name__ == "__main__":
    main()
