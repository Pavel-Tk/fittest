"""Fresh-frame predictions from cached run-C/E/F children (results/evolution_f/PREREG_fresh_frames.md, amendment 1).

    python scripts/evolve_c_fresh_predict.py --cache results/evolution_f_gpu1/cache --families orb --device cuda
    python scripts/evolve_c_fresh_predict.py --cache ... --families aimnet2,mace --verify 3

For every cached child (``<family>_<key>.spec.json`` + ``.pt`` + ``.npz``) it builds the family's model exactly as
``scripts/evolve_c_offspring.py`` does, copies the saved head state onto it and predicts the leak-free ``planted`` +
``skeptic_ref`` frames with the same ``predict()``. Output: ``<family>_<key>.fresh.npz`` next to the cache entry.
Cache keys, specs and the cached ``.npz`` are not touched. ``--verify N``: for the first N children per family it
also re-predicts the child's full ``predict_uids`` list (same batching as at birth) and reports max |dE|, |dF| against
the cached ``.npz`` (written to ``<cache>/fresh_verify.json``).
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
sys.path.insert(0, str(ROOT / "scripts"))

from evolve_c_offspring import ADAPTERS, head_params, predict  # noqa: E402

from fittest.cache import load_frames, load_predictions, save_predictions  # noqa: E402
from fittest.data import PLANTED, SKEPTIC_REF, assign_leakfree_splits  # noqa: E402

FRESH_SPLITS = (PLANTED, SKEPTIC_REF)


def fresh_path(ckpt: str | Path) -> Path:
    """Where the fresh-frame predictions of the child with head state ``ckpt`` live."""
    p = Path(ckpt)
    return p.with_name(p.name[:-len(".pt")] + ".fresh.npz") if p.name.endswith(".pt") else p.with_suffix(".fresh.npz")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="data/frames.pkl")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--families", default="aimnet2,orb,mace")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--verify", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    import torch

    frames, _ = assign_leakfree_splits(load_frames(args.frames), 0)
    by_uid = {f.uid: f for f in frames}
    fresh = [f for f in frames if f.split in FRESH_SPLITS]
    cache = Path(args.cache)
    vfile = cache / "fresh_verify.json"
    verify = json.loads(vfile.read_text()) if vfile.exists() else {}
    for fam in [f for f in args.families.split(",") if f]:
        specs = sorted(cache.glob(f"{fam}_*.spec.json"))
        todo = []
        for sp in specs:
            s = json.loads(sp.read_text())
            if not (Path(s["ckpt"]).exists() and Path(s["out"]).exists() and Path(s["result"]).exists()):
                print(f"[fresh] skip {sp.name}: incomplete cache entry", flush=True)
                continue
            todo.append(s)
        print(f"[fresh] {fam}: {len(todo)} children in {cache}", flush=True)
        if not todo:
            continue
        ad = ADAPTERS[fam](args.device)
        for p in ad.module.parameters():
            p.requires_grad_(False)
        named = head_params(ad.module, fam)
        n_ver = sum(1 for v in verify.values() if v["family"] == fam)
        for s in todo:
            out = fresh_path(s["ckpt"])
            need_ver = n_ver < args.verify and s["key"] not in verify
            if out.exists() and not args.force and not need_ver:
                continue
            t0 = time.time()
            state = torch.load(s["ckpt"], map_location=args.device)
            missing = [n for n, _ in named if n not in state]
            if missing:
                raise KeyError(f"{s['ckpt']}: head state lacks {missing[:3]}")
            with torch.no_grad():
                for n, p in named:
                    p.copy_(state[n].to(p.dtype))
            mx = 1000 if fam == "mace" else 3000
            if not out.exists() or args.force:
                E, F = predict(ad, fresh, max_atoms=mx)
                save_predictions(str(out), s["id"], [f.uid for f in fresh], E, F,
                                 {"family": fam, "key": s["key"], "ckpt": s["ckpt"], "splits": list(FRESH_SPLITS)})
            if need_ver:
                uids = s["predict_uids"]
                E, F = predict(ad, [by_uid[u] for u in uids], max_atoms=mx)
                ref = load_predictions(s["out"])
                dE = max(abs(float(E[i]) - float(ref.energy[u])) for i, u in enumerate(uids))
                dF = max(float(np.abs(np.asarray(F[i]) - np.asarray(ref.forces[u])).max()) for i, u in enumerate(uids))
                verify[s["key"]] = {"family": fam, "id": s["id"], "n_frames": len(uids), "max_abs_dE": dE,
                                    "max_abs_dF": dF}
                vfile.write_text(json.dumps(verify, indent=1))
                print(f"[fresh] verify {fam}_{s['key']}: max|dE| {dE:.3g} eV, max|dF| {dF:.3g} eV/A", flush=True)
                n_ver += 1
            print(f"[fresh] {fam}_{s['key']} ({s['id']}) {time.time() - t0:.1f} s", flush=True)
    print("FRESH_PREDICT_DONE", cache, flush=True)


if __name__ == "__main__":
    main()
