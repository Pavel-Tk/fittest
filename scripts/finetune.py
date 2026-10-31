"""Curator repair step: LoRA fine-tune MACE-MPA-0 on oracle labels, gate it, re-predict.

DESIGN.md §4 (Curator) / §9 (prediction cache). ALCHEMI Toolkit 0.3.0-rc APIs used:
``FineTuningStrategy`` + ``LoRAConfig`` (``nvalchemi.training``), ``fit_atomic_reference_energies``,
``evaluate_accuracy`` + ``build_acceptance_report`` (``nvalchemi.training.distillation.evaluation``).

    python3 scripts/finetune.py --frames data/frames.pkl \\
        --labelled-uids results/oracle_labels_armD.json --model mace_mpa0 \\
        --out data/preds/mace_lora_curated.npz --ckpt data/ckpt/mace_lora_curated \\
        --epochs 60 --seed 0 --predict-uids all

    # "ensemble of one fine-tuned model" baseline: 4 LoRA runs, same data, different seeds
    python3 scripts/finetune.py ... --seeds 0,1,2,3     # -> data/preds/mace_seed{k}.npz

Pipeline
--------
1. Training pool = frames whose uid is in ``--labelled-uids`` (oracle labels; E, F read from
   frames.pkl). Frames of split ``eval`` (or ``skeptic_ref``) are refused outright.
2. 15 % of the labelled *reactions* are held out for the acceptance report (by reaction, never
   by frame; seeded by ``--split-seed`` so every ``--seeds`` member sees identical data).
3. Per-element reference energies are re-fitted on the training frames
   (``fit_atomic_reference_energies`` on the residual ωB97x − MACE-MPA-0, added to the
   checkpoint E0s): T1x labels are ωB97x/6-31G(d) *total* energies, MACE-MPA-0 is PBE.
   "Before" = base MACE-MPA-0 **with the same refitted E0s**, so the comparison isolates
   what LoRA adds and energies are on one scale. The fit uses training frames only.
4. LoRA fine-tune (e3nn layout, recipe of ALCHEMI example intermediate/08) on energy + forces.
5. Acceptance: ``evaluate_accuracy`` for base(+E0) and fine-tuned on the held-out labelled
   slice, ``build_acceptance_report`` with ``max_from_scratch_ratio = --max-ratio`` (1.0) and
   the base model as baseline: fine-tuned must not regress energy/atom MAE *or* force MAE.
   ``report.accepted`` -> ``shipped``.
6. Predict E/F for ``--predict-uids`` (default all frames) with the *reloaded* checkpoint and
   write ``trustswarm.cache.save_predictions(--out, model=...)``.

Loss (forces are what TrustSwarm scores; relative energies are scored too)::

    L = w_E * EnergyHuber(per-atom, delta=0.01 eV/atom) + w_F * ForceHuber(per-component, delta=0.01 eV/Å)

Huber terms as in the ALCHEMI LoRA example. Errors here are ≫ delta, so both terms act as L1
and w_E : w_F is the exchange rate between eV/atom of energy error and eV/Å of force error.
Defaults (--energy-weight, --force-weight) are set from the dev sweep in docs/STATUS_finetune.md.

Checkpoint format (consumed by ``BatchedMACE(model_path=<ckpt dir>)``)
--------------------------------------------------------------------
See :mod:`trustswarm.mace_lora` (single source of truth)::

    <ckpt>/model.pt              merged plain mace-torch module (e3nn, float32, E0s baked in)
    <ckpt>/trustswarm.json       format "trustswarm.mace_lora/1", E0s, training config
    <ckpt>/atomic_energies.json  refitted E0s {Z: eV}
    <ckpt>/manifest.json ...     the ALCHEMI PEFT checkpoint (trainable-only) at the same root
    <ckpt>/acceptance_report.json

    torch.load("<ckpt>/model.pt", weights_only=False)            # plain mace-torch
    trustswarm.mace_lora.load_mace_lora("<ckpt>", enable_cueq=True)  # ALCHEMI MACEWrapper

Predictions meta (read by ``trustswarm.agents.curator.ReplayFinetune``): ``base_model``,
``train_uids``, ``holdout_uids``, ``acceptance_report`` (``verdict`` "pass"/"fail"),
``shipped``, ``epochs``, ``n_train``, ``wall_seconds``, ``atoms_per_second``, ``weights``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
import time
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trustswarm import mace_lora  # noqa: E402
from trustswarm.cache import load_frames, save_predictions  # noqa: E402
from trustswarm.config import CURATED_BASE_MODEL, CURATED_MODEL, WRONG_F  # noqa: E402
from trustswarm.data import EVAL, PLANTED, POOL, SKEPTIC_REF, Frame, split_reactions  # noqa: E402

BASE_CHECKPOINTS = {"mace_mpa0": "medium-mpa-0"}
ALLOWED_TRAIN_SPLITS = (POOL, PLANTED)  # oracle labels come from the pool stream (planted inside it)
FORBIDDEN_SPLITS = (EVAL, SKEPTIC_REF)  # final-scoring / Skeptic-private labels: never trained on

# --- Recipe (ALCHEMI examples/intermediate/08_lora_finetuning.py, e3nn module names checked
# --- against MACE-MPA-0 with enable_cueq=False on the VM) -------------------------------------
LORA_RANK = 16
LORA_ALPHA = 2.0
LORA_TARGET_PATTERNS = (
    "main.model.node_embedding.linear",
    "main.model.interactions.0.linear",
    "main.model.interactions.*.linear_up",
    "main.model.interactions.*.conv_tp_weights.layer[123]",
    "main.model.products.*.linear",
    "main.model.readouts.*.linear",
    "main.model.readouts.*.linear_1",
    "main.model.readouts.*.linear_2",
)
LORA_EXTRA_TRAINABLE = (
    "main.model.interactions.1.linear.weight",
    "main.model.interactions.*.density_fn.layer0.weight",
)
LAST_LAYERS_TRAINABLE = ("main.model.products.*", "main.model.readouts.*")
ENERGY_WEIGHT, FORCE_WEIGHT, HUBER_DELTA = 10.0, 10.0, 0.01  # defaults of --energy/--force-weight
WEIGHT_DECAY = 1e-4
LR_FLOOR_FRACTION = 0.05  # cosine decay to 5 % of the peak learning rate


# ============================================================ label-free helpers (no torch)
def read_uid_list(path: str | Path) -> list[str]:
    """Accept a JSON list of uids, a list of {"uid": ...} records, a {"uids"/"labelled_uids"/
    "labels"/"oracle_labels": ...} wrapper, or a {uid: ...} mapping. Order-preserving, deduped."""
    raw = json.loads(Path(path).read_text())

    def walk(x: Any) -> list[str]:
        if isinstance(x, str):
            return [x]
        if isinstance(x, list):
            return [u for item in x for u in walk(item)]
        if isinstance(x, dict):
            if "uid" in x and isinstance(x["uid"], str):
                return [x["uid"]]
            for key in ("uids", "labelled_uids", "labels", "oracle_labels", "frames"):
                if key in x:
                    return walk(x[key])
            return [k for k in x if isinstance(k, str)]
        raise ValueError(f"cannot read uids from {type(x).__name__} in {path}")

    return list(dict.fromkeys(walk(raw)))


def select_labelled(frames: Sequence[Frame], uids: Sequence[str],
                    allowed_splits: Sequence[str] = ALLOWED_TRAIN_SPLITS) -> list[Frame]:
    """Frames for ``uids`` with labels; refuses eval/skeptic_ref and unknown uids."""
    by_uid = {f.uid: f for f in frames}
    missing = [u for u in uids if u not in by_uid]
    if missing:
        raise KeyError(f"{len(missing)} labelled uids not in frames file, e.g. {missing[:3]}")
    chosen = [by_uid[u] for u in uids]
    bad = sorted({f.split for f in chosen} & set(FORBIDDEN_SPLITS))
    if bad:
        raise ValueError(f"refusing to train on frames of split(s) {bad}: eval labels are for "
                         "final scoring only, skeptic_ref labels are private to the Skeptic")
    other = sorted({f.split for f in chosen} - set(allowed_splits))
    if other:
        raise ValueError(f"labelled frames from unexpected split(s) {other}; allowed {allowed_splits}")
    unlabelled = [f.uid for f in chosen if f.E is None or f.F is None]
    if unlabelled:
        raise ValueError(f"{len(unlabelled)} labelled uids have no E/F in the frames file")
    return chosen


def holdout_by_reaction(frames: Sequence[Frame], fraction: float,
                        seed: int) -> tuple[list[Frame], list[Frame]]:
    """(train, holdout) with ≈``fraction`` of *reactions* (≥ 1) held out."""
    reactions = sorted({f.reaction for f in frames})
    if len(reactions) < 2:
        raise ValueError("need labelled frames from at least two reactions to hold one out")
    frac = max(fraction, 1.0 / len(reactions))  # always hold out at least one reaction
    held, _ = split_reactions(reactions, frac, np.random.default_rng(seed))
    held = set(held)
    return [f for f in frames if f.reaction not in held], [f for f in frames if f.reaction in held]


def error_metrics(frames: Sequence[Frame], energy: np.ndarray,
                  forces: Sequence[np.ndarray]) -> dict[str, float]:
    """Force RMSE/MAE (global, per-component), mean per-frame force RMSE, fraction of frames
    with force RMSE > WRONG_F, and the relative-energy MAE: energies centred on their
    reaction mean (reactions with ≥ 2 frames), which cancels level-of-theory offsets without
    needing labels outside the set (the reactant may not be oracle-labelled)."""
    dF = np.concatenate([np.asarray(p) - f.F for f, p in zip(frames, forces)])
    per_frame = np.array([math.sqrt(float(np.mean((np.asarray(p) - f.F) ** 2)))
                          for f, p in zip(frames, forces)])
    out = {"n_frames": len(frames), "n_atoms": int(len(dF)),
           "force_rmse": float(np.sqrt(np.mean(dF ** 2))), "force_mae": float(np.mean(np.abs(dF))),
           "frame_force_rmse_mean": float(per_frame.mean()),
           "frac_force_wrong": float(np.mean(per_frame > WRONG_F))}
    groups: dict[str, list[int]] = defaultdict(list)
    for i, f in enumerate(frames):
        groups[f.reaction].append(i)
    errs = []
    for idx in groups.values():
        if len(idx) < 2:
            continue
        e_hat = energy[idx] - energy[idx].mean()
        e_ref = np.array([frames[i].E for i in idx]); e_ref = e_ref - e_ref.mean()
        errs.extend(np.abs(e_hat - e_ref))
    out["rel_energy_mae"] = float(np.mean(errs)) if errs else float("nan")
    out["n_rel_energy"] = len(errs)
    return out


# ============================================================ torch / ALCHEMI part
def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_atomic_data(f: Frame, with_labels: bool = True):
    import torch
    from nvalchemi.data import AtomicData

    n = f.n_atoms
    return AtomicData(
        atomic_numbers=torch.as_tensor(f.Z, dtype=torch.long),
        positions=torch.as_tensor(f.R, dtype=torch.float32),
        charge=torch.zeros(1, 1),
        energy=torch.tensor([[float(f.E)]]) if with_labels else torch.zeros(1, 1),
        forces=(torch.as_tensor(f.F, dtype=torch.float32) if with_labels else torch.zeros(n, 3)),
    )


def make_loader(frames: Sequence[Frame], batch_size: int, shuffle: bool, device):
    from nvalchemi.data import Batch
    from nvalchemi.data.datapipes import DataLoader, InMemoryDataset

    batch = Batch.from_data_list([to_atomic_data(f) for f in frames])
    return DataLoader(InMemoryDataset(in_memory_batch=batch, device=device),
                      batch_size=batch_size, shuffle=shuffle, drop_last=shuffle,
                      prefetch_factor=0, num_streams=1, use_streams=False, pin_memory=True)


def load_base(checkpoint: str, device, atomic_energies: dict[int, float] | None = None,
              train: bool = False, enable_cueq: bool = False):
    """Foundation MACE. Training always uses the e3nn layout (enable_cueq=False) so the merged
    model can be exported as a plain mace-torch module; inference may use cuEq kernels."""
    import torch
    from nvalchemi.models.mace import MACEWrapper

    return MACEWrapper.from_checkpoint(checkpoint, device=device, dtype=torch.float32,
                                       enable_cueq=enable_cueq, compile_model=False,
                                       atomic_energies=atomic_energies, train=train)


def use_cueq(flag: str, device) -> bool:
    import importlib.util

    available = (getattr(device, "type", str(device)) == "cuda"
                 and importlib.util.find_spec("cuequivariance_torch") is not None
                 and importlib.util.find_spec("cuequivariance_ops_torch") is not None)
    if flag == "on" and not available:
        raise RuntimeError("--cueq on but cuequivariance(-ops)-torch or CUDA is unavailable")
    return available if flag == "auto" else flag == "on"


def refit_reference_energies(checkpoint: str, train: Sequence[Frame], device) -> tuple[dict[int, float], dict]:
    """Checkpoint E0 + least-squares residual (ωB97x − MACE) per element, on training frames."""
    import torch
    from nvalchemi.hooks import HookContext, NeighborListHook
    from nvalchemi.training import TrainingStage
    from nvalchemi.training.reference_energies import fit_atomic_reference_energies

    base = load_base(checkpoint, device)
    base.model_config.active_outputs = {"energy"}
    nl = NeighborListHook(base.model_config.neighbor_config, stage=TrainingStage.BEFORE_FORWARD)

    def baseline(b):
        nl(HookContext(batch=b, model=base), TrainingStage.BEFORE_FORWARD)
        return base(b)["energy"]

    ckpt_e0 = dict(zip((int(z) for z in base.model.atomic_numbers.tolist()),
                       (float(e) for e in base.model.atomic_energies_fn.atomic_energies.reshape(-1).tolist())))
    loader = make_loader(train, 64, False, device)
    try:
        fit = fit_atomic_reference_energies(loader, baseline_energy_fn=baseline, device=device)
    finally:
        loader.dataset.close()
        del base
        torch.cuda.empty_cache()
    e0 = {int(z): ckpt_e0[int(z)] + float(d) for z, d in fit.reference_energies.items()}
    info = {"residual_corrections": {str(z): float(d) for z, d in fit.reference_energies.items()},
            "rank": int(fit.rank), "n_elements": len(fit.atomic_numbers),
            "num_structures": int(fit.num_structures),
            "rms_residual_per_structure_eV": float(math.sqrt(fit.residual_sum_squares / max(fit.num_structures, 1)))}
    return e0, info


def predict(model, frames: Sequence[Frame], device, max_atoms: int = 4096) -> tuple[np.ndarray, list[np.ndarray], float]:
    """Batched E/F for ``frames`` (order kept). Returns (E (N,), [F_i (n_i,3)], model seconds)."""
    import torch
    from nvalchemi.data import Batch
    from nvalchemi.neighbors import compute_neighbors

    model.eval()
    model.model_config.active_outputs = {"energy", "forces"}
    nc = model.model_config.neighbor_config
    energies: list[np.ndarray] = []
    forces: list[np.ndarray] = []
    seconds, i = 0.0, 0
    while i < len(frames):
        chunk, n = [], 0
        while i < len(frames) and (not chunk or n + frames[i].n_atoms <= max_atoms):
            chunk.append(frames[i]); n += frames[i].n_atoms; i += 1
        b = Batch.from_data_list([to_atomic_data(f, with_labels=False) for f in chunk], device=device)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        compute_neighbors(b, config=nc)
        with torch.enable_grad():
            out = model(b)
        e = out["energy"].detach().double().cpu().numpy().reshape(-1)
        F = out["forces"].detach().double().cpu().numpy().reshape(-1, 3)
        seconds += time.perf_counter() - t0
        offs = np.cumsum([0] + [f.n_atoms for f in chunk])
        energies.append(e)
        forces.extend(F[offs[k]:offs[k + 1]].copy() for k in range(len(chunk)))
    return np.concatenate(energies), forces, seconds


def holdout_batches(frames: Sequence[Frame], neighbor_config, device) -> list:
    """Re-iterable held-out set; every batch already carries its neighbor list."""
    from nvalchemi.neighbors import compute_neighbors

    loader = make_loader(frames, 64, False, device)
    out = []
    for b in loader:
        compute_neighbors(b, config=neighbor_config)
        out.append(b)
    return out


def alchemi_accuracy(model, batches):
    from nvalchemi.training.distillation.evaluation import evaluate_accuracy

    model.model_config.active_outputs = {"energy", "forces"}
    return evaluate_accuracy(model, batches, targets="reference", quantities=("energy", "forces"))


class EpochLossLog:
    """Training hook: mean total loss per epoch (AFTER_LOSS accumulate, AFTER_EPOCH flush)."""

    frequency = 1

    def __init__(self, print_every: int):
        from nvalchemi.training import TrainingStage

        self.stage = TrainingStage.AFTER_LOSS
        self._after_epoch = TrainingStage.AFTER_EPOCH
        self.print_every = max(1, print_every)
        self.history: list[float] = []
        self._sum, self._n, self._t0 = 0.0, 0, time.perf_counter()

    def _runs_on_stage(self, stage) -> bool:
        return stage in (self.stage, self._after_epoch)

    def __call__(self, ctx, stage) -> None:
        if stage == self.stage:
            if ctx.loss is not None:
                self._sum += float(ctx.loss.detach()); self._n += 1
            return
        self.history.append(self._sum / max(self._n, 1))
        self._sum, self._n = 0.0, 0
        ep = len(self.history)
        if ep == 1 or ep % self.print_every == 0:
            print(f"  epoch {ep:4d}  train loss {self.history[-1]:.5f}  "
                  f"({time.perf_counter() - self._t0:.1f} s)", flush=True)


def finetune(checkpoint: str, e0: dict[int, float], train: Sequence[Frame], args, seed: int,
             peft_dir: Path) -> tuple[Any, dict]:
    """Run the FineTuningStrategy; returns (merged MACEWrapper, training info)."""
    import torch
    from nvalchemi.hooks import NeighborListHook
    from nvalchemi.training import (EnergyHuberLoss, FineTuningStrategy, ForceHuberLoss,
                                    OptimizerConfig, TrainingStage, default_training_fn)
    from nvalchemi.training.peft.lora import LoRAConfig, is_lora_layer, merge_lora_into_model

    device = torch.device(args.device)
    seed_everything(seed)
    model = load_base(checkpoint, device, atomic_energies=e0, train=True)
    model.model_config.active_outputs = {"energy", "forces"}
    model.train()

    bs = min(args.batch_size, len(train))
    steps_per_epoch = len(train) // bs
    total_steps = steps_per_epoch * args.epochs
    loss_fn = (args.energy_weight * EnergyHuberLoss(per_atom=True, delta=HUBER_DELTA, ignore_nonfinite=True)
               + args.force_weight * ForceHuberLoss(normalize_by_atom_count=False, delta=HUBER_DELTA,
                                               ignore_nonfinite=True))
    loss_fn.normalize_weights = False
    log = EpochLossLog(print_every=max(1, args.epochs // 10))
    lora = not args.full_finetune_last_layers
    strategy = FineTuningStrategy(
        models=model,
        optimizer_configs=OptimizerConfig(
            optimizer_cls=torch.optim.AdamW,
            optimizer_kwargs={"lr": args.lr, "weight_decay": WEIGHT_DECAY},
            scheduler_cls=torch.optim.lr_scheduler.CosineAnnealingLR,
            scheduler_kwargs={"T_max": max(total_steps, 1), "eta_min": args.lr * LR_FLOOR_FRACTION}),
        num_epochs=args.epochs, num_steps=None, training_fn=default_training_fn, loss_fn=loss_fn,
        devices=[device],
        hooks=[NeighborListHook(model.model_config.neighbor_config, stage=TrainingStage.BEFORE_FORWARD), log],
        peft_config=(LoRAConfig(rank=args.lora_rank, alpha=args.lora_alpha, lora_dropout=0.0,
                                lora_target_patterns=LORA_TARGET_PATTERNS) if lora else None),
        trainable_patterns=LORA_EXTRA_TRAINABLE if lora else LAST_LAYERS_TRAINABLE,
    )
    tuned = strategy.models["main"]
    n_lora = sum(1 for _, m in tuned.named_modules() if is_lora_layer(m))
    if lora and n_lora == 0:
        raise RuntimeError("no LoRA adapters were inserted; check LORA_TARGET_PATTERNS")
    loader = make_loader(train, bs, True, device)
    t0 = time.perf_counter()
    try:
        strategy.run(loader)
    finally:
        loader.dataset.close()
    train_seconds = time.perf_counter() - t0
    strategy.save_checkpoint(peft_dir, save_trainable_state_only=True)
    summary = strategy.trainable_parameter_summary
    if lora:
        merge_lora_into_model(tuned)
    tuned.eval()
    for p in tuned.parameters():
        p.requires_grad_(False)
    info = {"mode": "lora" if lora else "last_layers", "epochs": args.epochs, "batch_size": bs,
            "steps": total_steps, "lr": args.lr, "scheduler": "cosine", "weight_decay": WEIGHT_DECAY,
            "train_seconds": train_seconds, "loss_history": log.history, "n_lora_layers": n_lora,
            "trainable_parameters": {k: int(v.get("parameter_count", 0)) for k, v in summary.items()}
            if isinstance(summary, dict) else str(summary),
            "lora": {"rank": args.lora_rank, "alpha": args.lora_alpha,
                     "target_patterns": list(LORA_TARGET_PATTERNS),
                     "extra_trainable": list(LORA_EXTRA_TRAINABLE)} if lora else None,
            "trainable_patterns": None if lora else list(LAST_LAYERS_TRAINABLE),
            "loss": {"energy": {"kind": "EnergyHuberLoss", "per_atom": True, "delta": HUBER_DELTA,
                                "weight": args.energy_weight},
                     "forces": {"kind": "ForceHuberLoss", "delta": HUBER_DELTA, "weight": args.force_weight}},
            "seed": seed}
    del strategy
    return tuned, info


def acceptance(base_acc, tuned_acc, name: str, max_ratio: float) -> tuple[bool, dict]:
    """ALCHEMI acceptance report. The bar is the toolkit's ``max_from_scratch_ratio`` with the
    *base model* (same refitted E0s, same hold-out batches) as the baseline: the worst of
    MAE_tuned / MAE_base over energy-per-atom and forces must be <= ``max_ratio``, i.e. the
    repair must not trade energies for forces or vice versa. No absolute bar: the Curator
    replaces the worst model, so "better than what it replaces" is the shipping criterion."""
    from nvalchemi.training.distillation.evaluation import (AcceptanceThresholds, StudentEvaluation,
                                                            build_acceptance_report)

    thresholds = AcceptanceThresholds(max_from_scratch_ratio=max_ratio)
    report = build_acceptance_report(
        [StudentEvaluation(name=name, accuracy=tuned_acc, baseline_accuracy=base_acc, weights="raw")],
        thresholds)
    accepted = bool(report.accepted)
    checks = [{"name": c.name, "value": c.value, "limit": c.limit, "comparison": c.comparison,
               "passed": bool(c.passed), "detail": c.detail} for c in report.verdicts[0].checks]
    return accepted, {"verdict": "pass" if accepted else "fail", "accepted": accepted,
                      "checks": checks,
                      "thresholds": {"max_from_scratch_ratio": max_ratio},
                      "baseline": "base MACE-MPA-0 with the same refitted E0s",
                      "alchemi_report": report.to_dict()}


def _jsonable(x: Any) -> Any:
    return json.loads(json.dumps(x, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


def _fmt_check(c: dict) -> str:
    value = "n/a" if c["value"] is None else f"{c['value']:.4g}"
    return f"{c['name']} {value} {c['comparison']} {c['limit']}: {'ok' if c['passed'] else 'FAIL'}"


def _fmt(before: dict, after: dict) -> str:
    rows = [("force RMSE  [eV/Å]", "force_rmse"), ("force MAE   [eV/Å]", "force_mae"),
            ("frame F-RMSE mean ", "frame_force_rmse_mean"), ("frac F-RMSE>WRONG_F", "frac_force_wrong"),
            ("rel-E MAE   [eV]  ", "rel_energy_mae")]
    return "\n".join(f"    {label}  {before[k]:8.4f} -> {after[k]:8.4f}" for label, k in rows)


def run_one(*, name: str, seed: int, frames: list[Frame], train: list[Frame], hold: list[Frame],
            e0: dict[int, float], e0_info: dict, base_eval: dict, args, out: Path, ckpt: Path) -> dict:
    import torch

    device = torch.device(args.device)
    t_start = time.perf_counter()
    print(f"\n=== {name}: seed {seed}, {len(train)} train / {len(hold)} hold-out frames ===", flush=True)
    if ckpt.exists():
        if not (ckpt / mace_lora.MANIFEST_FILE).exists() and any(ckpt.iterdir()):
            raise FileExistsError(f"{ckpt} exists and is not a {mace_lora.FORMAT} checkpoint; refusing to overwrite")
        shutil.rmtree(ckpt)  # a previous run of ours: start clean (no stale PEFT checkpoint indices)
    tuned, info = finetune(BASE_CHECKPOINTS[args.model], e0, train, args, seed, ckpt)

    manifest = {"base_model": args.model, "base_checkpoint": BASE_CHECKPOINTS[args.model],
                "layout": "e3nn", "dtype": "float32", "name": name,
                "atomic_energies": {str(z): e for z, e in sorted(e0.items())}, "e0_fit": e0_info,
                "training": info, "n_train": len(train), "n_holdout": len(hold)}
    mace_lora.write_checkpoint(ckpt, tuned.model, _jsonable(manifest))
    del tuned
    torch.cuda.empty_cache()

    # Everything downstream uses the checkpoint as shipped (reload == what BatchedMACE serves).
    shipped_model = mace_lora.load_mace_lora(ckpt, device=device, enable_cueq=use_cueq(args.cueq, device))
    tuned_acc = alchemi_accuracy(shipped_model, holdout_batches(hold, shipped_model.model_config.neighbor_config, device))
    accepted, report = acceptance(base_eval["alchemi"], tuned_acc, name, args.max_ratio)
    E_h, F_h, _ = predict(shipped_model, hold, device)
    E_t, F_t, _ = predict(shipped_model, train, device)
    after = {"holdout": error_metrics(hold, E_h, F_h), "train": error_metrics(train, E_t, F_t),
             "alchemi": tuned_acc.to_dict()}
    report.update({"holdout_metrics": {"before": base_eval["holdout"], "after": after["holdout"]},
                   "train_metrics": {"before": base_eval["train"], "after": after["train"]},
                   "alchemi_accuracy": {"before": base_eval["alchemi"].to_dict(), "after": after["alchemi"]},
                   "n_train": len(train), "n_holdout": len(hold),
                   "n_holdout_reactions": len({f.reaction for f in hold})})
    (ckpt / mace_lora.REPORT_FILE).write_text(json.dumps(_jsonable(report), indent=2))
    finetune_seconds = time.perf_counter() - t_start

    targets = select_predict(frames, train, hold, args.predict_uids)
    t0 = time.perf_counter()
    E, F, model_seconds = predict(shipped_model, targets, device, max_atoms=args.predict_max_atoms)
    predict_wall = time.perf_counter() - t0
    n_atoms = int(sum(f.n_atoms for f in targets))
    slim_report = {k: report[k] for k in ("verdict", "accepted", "checks", "thresholds", "baseline",
                                          "holdout_metrics", "n_train", "n_holdout")}
    meta = {"base_model": args.model, "base_checkpoint": BASE_CHECKPOINTS[args.model],
            "weights": str(ckpt), "checkpoint": str(ckpt), "checkpoint_format": mace_lora.FORMAT,
            "shipped": accepted, "acceptance_report": slim_report,
            "epochs": args.epochs, "seed": seed, "n_train": len(train), "n_holdout": len(hold),
            "train_uids": [f.uid for f in train], "holdout_uids": [f.uid for f in hold],
            "finetune_wall_seconds": finetune_seconds, "train_seconds": info["train_seconds"],
            "predict_wall_seconds": predict_wall, "model_seconds": model_seconds,
            "wall_seconds": finetune_seconds + predict_wall, "n_frames": len(targets), "n_atoms": n_atoms,
            "atoms_per_second": n_atoms / max(model_seconds, 1e-9),
            "level_of_theory": "MACE-MPA-0 + LoRA on ωB97x/6-31G(d) oracle labels (E0 refitted)",
            "training": {k: v for k, v in info.items() if k != "loss_history"}}
    save_predictions(out, model=name, uids=[f.uid for f in targets], energies=E, forces=F,
                     meta=_jsonable(meta))
    del shipped_model
    torch.cuda.empty_cache()

    checks = ", ".join(_fmt_check(c) for c in report["checks"])
    print(f"--- {name}: acceptance {report['verdict'].upper()} ({checks})")
    print("  hold-out (labelled, by reaction)  before -> after")
    print(_fmt(base_eval["holdout"], after["holdout"]))
    print("  train                             before -> after")
    print(_fmt(base_eval["train"], after["train"]))
    print(f"  fine-tune wall {finetune_seconds:.1f} s (train {info['train_seconds']:.1f} s, "
          f"{info['steps']} steps); predict {len(targets)} frames in {predict_wall:.1f} s "
          f"({meta['atoms_per_second']:.0f} atoms/s) -> {out}", flush=True)
    return {"name": name, "verdict": report["verdict"], "out": str(out), "ckpt": str(ckpt),
            "holdout_before": base_eval["holdout"], "holdout_after": after["holdout"],
            "finetune_wall_seconds": finetune_seconds}


def select_predict(frames: list[Frame], train: list[Frame], hold: list[Frame], spec: str) -> list[Frame]:
    if spec == "all":
        return list(frames)
    if spec == "labelled":
        return list(train) + list(hold)
    by_uid = {f.uid: f for f in frames}
    return [by_uid[u] for u in read_uid_list(spec)]


def evaluate_base(checkpoint: str, e0: dict[int, float], train, hold, device, cueq: bool) -> dict:
    """Before-metrics: base MACE with the refitted E0s, on hold-out (ALCHEMI + ours) and train."""
    import torch

    base = load_base(checkpoint, device, atomic_energies=e0, enable_cueq=cueq)
    acc = alchemi_accuracy(base, holdout_batches(hold, base.model_config.neighbor_config, device))
    E_h, F_h, _ = predict(base, hold, device)
    E_t, F_t, _ = predict(base, train, device)
    out = {"alchemi": acc, "holdout": error_metrics(hold, E_h, F_h), "train": error_metrics(train, E_t, F_t)}
    del base
    torch.cuda.empty_cache()
    return out


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--frames", default="data/frames.pkl")
    ap.add_argument("--labelled-uids", required=True, help="JSON with the oracle-labelled uids")
    ap.add_argument("--model", default=CURATED_BASE_MODEL, choices=sorted(BASE_CHECKPOINTS))
    ap.add_argument("--out", default=f"data/preds/{CURATED_MODEL}.npz")
    ap.add_argument("--ckpt", default=f"data/ckpt/{CURATED_MODEL}")
    ap.add_argument("--name", default=CURATED_MODEL, help="model name stored in the predictions file")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--energy-weight", type=float, default=ENERGY_WEIGHT,
                    help="weight of the per-atom energy Huber term")
    ap.add_argument("--force-weight", type=float, default=FORCE_WEIGHT,
                    help="weight of the per-component force Huber term")
    ap.add_argument("--lora-rank", type=int, default=LORA_RANK)
    ap.add_argument("--lora-alpha", type=float, default=LORA_ALPHA)
    ap.add_argument("--seed", type=int, default=0, help="training seed (LoRA init, shuffling)")
    ap.add_argument("--split-seed", type=int, default=0, help="seed of the by-reaction hold-out split")
    ap.add_argument("--holdout-fraction", type=float, default=0.15)
    ap.add_argument("--max-ratio", type=float, default=1.0,
                    help="acceptance: worst MAE ratio fine-tuned/base over energy/atom and forces")
    ap.add_argument("--predict-uids", default="all", help="'all', 'labelled', or a JSON uid list")
    ap.add_argument("--predict-max-atoms", type=int, default=4096, help="atoms per inference batch")
    ap.add_argument("--seeds", default=None,
                    help="comma list, e.g. 0,1,2,3: seed-ensemble mode, writes mace_seed{k}.npz next to --out")
    ap.add_argument("--full-finetune-last-layers", action="store_true",
                    help="instead of LoRA, fully train products.* + readouts.* (lower --lr, e.g. 1e-4)")
    ap.add_argument("--cueq", choices=("auto", "on", "off"), default="auto",
                    help="cuEquivariance kernels for evaluation/prediction (training is always e3nn)")
    ap.add_argument("--device", default="cuda")
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    t0 = time.perf_counter()
    frames = load_frames(args.frames)
    labelled = select_labelled(frames, read_uid_list(args.labelled_uids))
    train, hold = holdout_by_reaction(labelled, args.holdout_fraction, args.split_seed)
    if len(train) < 8 or not hold:
        raise ValueError(f"too few labelled frames to fine-tune: {len(train)} train / {len(hold)} hold-out")
    print(f"{len(labelled)} labelled frames from {len({f.reaction for f in labelled})} reactions "
          f"-> {len(train)} train / {len(hold)} hold-out ({len({f.reaction for f in hold})} reactions); "
          f"max {max(f.n_atoms for f in labelled)} atoms", flush=True)

    import torch

    device = torch.device(args.device)
    checkpoint = BASE_CHECKPOINTS[args.model]
    e0, e0_info = refit_reference_energies(checkpoint, train, device)
    print(f"E0 refit on {e0_info['num_structures']} structures: rank {e0_info['rank']}/"
          f"{e0_info['n_elements']}, rms residual {e0_info['rms_residual_per_structure_eV']:.3f} eV; "
          f"corrections {e0_info['residual_corrections']}", flush=True)
    base_eval = evaluate_base(checkpoint, e0, train, hold, device, use_cueq(args.cueq, device))

    out, ckpt = Path(args.out), Path(args.ckpt)
    if args.seeds:
        runs = [(f"mace_seed{k}", int(k), out.parent / f"mace_seed{k}.npz", ckpt.parent / f"mace_seed{k}")
                for k in args.seeds.split(",")]
    else:
        runs = [(args.name, args.seed, out, ckpt)]
    results = [run_one(name=n, seed=s, frames=frames, train=train, hold=hold, e0=e0, e0_info=e0_info,
                       base_eval=base_eval, args=args, out=o, ckpt=c) for n, s, o, c in runs]
    print(f"\nDONE {len(results)} run(s) in {time.perf_counter() - t0:.1f} s: "
          + ", ".join(f"{r['name']}={r['verdict']}" for r in results), flush=True)


if __name__ == "__main__":
    main()
