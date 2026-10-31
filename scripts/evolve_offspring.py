"""Give birth to one MACE offspring for the evolutionary loop (torch side, CPU or GPU).

    python scripts/evolve_offspring.py --frames data/frames.pkl --spec results/evolution/jobs/g1_a.json \
        --device cpu --threads 2

The spec (written by ``scripts/evolve.py``) names the child, its parent (``mace_mpa0`` = the
foundation checkpoint, or the checkpoint directory of an earlier offspring), its recipe
(energy/force loss weights, LoRA rank/alpha, lr, epochs, batch size), the training uids (breed
labels) and the gate hold-out uids (select labels), plus the output paths.

Reuses the fine-tune machinery of ``scripts/finetune.py`` (FineTuningStrategy + LoRAConfig,
E0 refit, ALCHEMI acceptance report). Differences:

* the hold-out is given (the select labels), not carved here; component- and geometry-
  disjointness from the training labels is asserted;
* a child of an offspring starts from its parent's merged weights (new LoRA adapters on top,
  E0s inherited); a child of the foundation model refits E0s on its training labels;
* the ALCHEMI report compares the child with its *parent* on the hold-out; the evolution
  loop adds the force-RMSE / relative-energy gates from cached predictions;
* predictions are written for pool + eval frames only (all the loop needs).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fittest import mace_lora  # noqa: E402
from fittest.cache import load_frames, save_predictions  # noqa: E402
from fittest.data import (ENDPOINT_ROLES, EVAL, POOL, assign_leakfree_splits,  # noqa: E402
                          geometry_hash, reaction_components)


def _load_finetune_module():
    spec = importlib.util.spec_from_file_location("fittest_finetune_script", ROOT / "scripts" / "finetune.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def assert_disjoint(train, hold, frames) -> dict:
    rxn = {f.reaction for f in train} | {f.reaction for f in hold}
    comp = reaction_components([f for f in frames if f.reaction in rxn])
    shared_c = {comp[f.reaction] for f in train} & {comp[f.reaction] for f in hold}
    if shared_c:
        raise AssertionError(f"train and hold-out share {len(shared_c)} endpoint-geometry components")

    def geoms(fs):
        r = {f.reaction for f in fs}
        ends = {geometry_hash(f.Z, f.R) for f in frames if f.reaction in r and f.role in ENDPOINT_ROLES}
        return ends | {geometry_hash(f.Z, f.R) for f in fs}

    shared = geoms(train) & geoms(hold)
    if shared:
        raise AssertionError(f"train and hold-out share {len(shared)} geometries")
    return {"shared_components": 0, "shared_geometries": 0, "passed": True,
            "n_train": len(train), "n_holdout": len(hold)}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="data/frames.pkl")
    ap.add_argument("--spec", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--data-seed", type=int, default=0)
    args = ap.parse_args(argv)
    spec = json.loads(Path(args.spec).read_text())
    t0 = time.perf_counter()

    import torch

    if args.threads:
        torch.set_num_threads(args.threads)
    ft = _load_finetune_module()
    device = torch.device(args.device)
    frames, split_report = assign_leakfree_splits(load_frames(args.frames), args.data_seed)
    train = ft.select_labelled(frames, list(dict.fromkeys(spec["train_uids"])))
    hold = ft.select_labelled(frames, list(dict.fromkeys(spec["holdout_uids"])))
    train, leak = ft.exclude_eval_geometry(train, frames)
    hold, leak_h = ft.exclude_eval_geometry(hold, frames)
    disjoint = assert_disjoint(train, hold, frames)
    recipe = spec["recipe"]
    ns = SimpleNamespace(device=args.device, batch_size=int(recipe["batch_size"]),
                         epochs=int(recipe["epochs"]), energy_weight=float(recipe["energy_weight"]),
                         force_weight=float(recipe["force_weight"]), lr=float(recipe["lr"]),
                         lora_rank=int(recipe["lora_rank"]), lora_alpha=float(recipe["lora_alpha"]),
                         full_finetune_last_layers=False)
    name, parent = spec["id"], spec["parent"]
    ckpt, out = Path(spec["ckpt"]), Path(spec["out"])
    print(f"=== {name}: parent {parent}, {len(train)} train / {len(hold)} hold-out, recipe {recipe}", flush=True)

    from nvalchemi.models.mace import MACEWrapper

    if parent in ft.BASE_CHECKPOINTS:
        checkpoint = ft.BASE_CHECKPOINTS[parent]
        e0, e0_info = ft.refit_reference_energies(checkpoint, train, device)
        parent_model = ft.load_base(checkpoint, device, atomic_energies=e0)
        inheritance = "foundation checkpoint + E0 refit on training labels"
    else:
        checkpoint = mace_lora.read_manifest(parent).get("base_checkpoint", "medium-mpa-0")
        pman = mace_lora.read_manifest(parent)
        e0 = {int(z): float(v) for z, v in pman["atomic_energies"].items()}
        e0_info = {"inherited_from": parent}
        parent_model = mace_lora.load_mace_lora(parent, device=device, enable_cueq=False)
        inheritance = "parent's merged weights (new LoRA adapters on top), E0s inherited"

        def load_parent_for_training(*_a, **_k):
            module = mace_lora.load_mace_module(parent, device=device, dtype=torch.float32)
            module.train()
            return MACEWrapper(module, train=True)

        ft.load_base = load_parent_for_training  # finetune() builds its model through load_base

    base_acc = ft.alchemi_accuracy(parent_model, ft.holdout_batches(hold, parent_model.model_config.neighbor_config, device))
    del parent_model
    if ckpt.exists():
        import shutil
        shutil.rmtree(ckpt)
    tuned, info = ft.finetune(checkpoint, e0, train, ns, int(spec["seed"]), ckpt)
    manifest = {"base_model": "mace_mpa0", "base_checkpoint": checkpoint, "parent": parent,
                "inheritance": inheritance, "layout": "e3nn", "dtype": "float32", "name": name,
                "atomic_energies": {str(z): e for z, e in sorted(e0.items())}, "e0_fit": e0_info,
                "training": info, "n_train": len(train), "n_holdout": len(hold), "recipe": recipe}
    mace_lora.write_checkpoint(ckpt, tuned.model, ft._jsonable(manifest))
    del tuned
    child = mace_lora.load_mace_lora(ckpt, device=device, enable_cueq=False)
    tuned_acc = ft.alchemi_accuracy(child, ft.holdout_batches(hold, child.model_config.neighbor_config, device))
    accepted, report = ft.acceptance(base_acc, tuned_acc, name, 1.0)
    ratios = [c["value"] for c in report["checks"] if "ratio" in c["name"] and isinstance(c["value"], (int, float))]
    finetune_seconds = time.perf_counter() - t0

    targets = [f for f in frames if f.split in (POOL, EVAL)]
    t1 = time.perf_counter()
    E, F, model_seconds = ft.predict(child, targets, device, max_atoms=2048)
    predict_seconds = time.perf_counter() - t1
    toolkit = {"accepted": bool(accepted), "worst_ratio": max(ratios) if ratios else None, "max_ratio": 1.0,
               "baseline": "parent on the same hold-out batches (same E0s)",
               "detail": "; ".join(f"{c['name']}={c['value']}" for c in report["checks"])}
    meta = {"base_model": "mace_mpa0", "parent": parent, "inheritance": inheritance, "recipe": recipe,
            "checkpoint": str(ckpt), "toolkit_acceptance": toolkit, "train_uids": [f.uid for f in train],
            "holdout_uids": [f.uid for f in hold], "geometry_leak_check": leak, "holdout_leak_check": leak_h,
            "disjoint_check": disjoint, "train_seconds": info["train_seconds"],
            "finetune_wall_seconds": finetune_seconds, "predict_wall_seconds": predict_seconds,
            "n_frames": len(targets), "threads": torch.get_num_threads(),
            "training": {k: v for k, v in info.items() if k != "loss_history"},
            "loss_history": info.get("loss_history")}
    save_predictions(out, model=name, uids=[f.uid for f in targets], energies=E, forces=F, meta=ft._jsonable(meta))
    result = {"id": name, "out": str(out), "ckpt": str(ckpt), "toolkit": toolkit,
              "alchemi_report": ft._jsonable(report), "wall_seconds": time.perf_counter() - t0,
              "train_seconds": info["train_seconds"], "predict_seconds": predict_seconds}
    Path(spec["result"]).write_text(json.dumps(ft._jsonable(result), indent=1))
    print(f"--- {name}: toolkit {'ACCEPTED' if accepted else 'rejected'} (worst ratio {toolkit['worst_ratio']}), "
          f"train {info['train_seconds']:.0f} s, predict {predict_seconds:.0f} s, total {result['wall_seconds']:.0f} s", flush=True)


if __name__ == "__main__":
    main()
