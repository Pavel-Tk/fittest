"""Generate a synthetic frames.pkl + preds/*.npz so the whole pipeline runs on CPU.

    python scripts/make_synthetic.py --out data/synthetic --seed 0
"""

from __future__ import annotations

import argparse
from collections import Counter

from trustswarm.cache import load_frames
from trustswarm.synthetic import SyntheticSpec, write_synthetic


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/synthetic", help="output directory")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-test", type=int, default=200, help="test reactions (→ pool/eval)")
    ap.add_argument("--n-val", type=int, default=80, help="val reactions (→ skeptic_ref/planted)")
    ap.add_argument("--n-train", type=int, default=20, help="train reactions (→ seed_train)")
    ap.add_argument("--no-mace-seeds", action="store_true", help="omit the optional mace_seed0..3 files")
    args = ap.parse_args(argv)
    spec = SyntheticSpec(n_train=args.n_train, n_val=args.n_val, n_test=args.n_test, seed=args.seed)
    paths = write_synthetic(args.out, spec, include_mace_seeds=not args.no_mace_seeds)
    frames = load_frames(paths["frames"])
    print(f"wrote {paths['frames']}: {len(frames)} frames, splits {dict(Counter(f.split for f in frames))}")
    print("models:", ", ".join(sorted(k for k in paths if k != "frames")))


if __name__ == "__main__":
    main()
