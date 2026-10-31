"""Shared fixtures: one small synthetic dataset and one ablation run per test session."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from fittest.ablation import AblationConfig, run_ablation
from fittest.cache import load_frames
from fittest.data import Frame
from fittest.synthetic import SyntheticSpec, write_synthetic

SMALL_SPEC = SyntheticSpec(n_train=4, n_val=30, n_test=60, n_path=4, seed=0)
SMALL_CFG = AblationConfig(budget_per_round=12, rounds=3, seed=0, n_bootstrap=60, max_planted=15)


@pytest.fixture(scope="session")
def synthetic_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("synthetic")
    write_synthetic(out, SMALL_SPEC)
    return out


@pytest.fixture(scope="session")
def frames(synthetic_dir: Path) -> list[Frame]:
    return load_frames(synthetic_dir / "frames.pkl")


@pytest.fixture(scope="session")
def ablation_run(synthetic_dir: Path, frames: list[Frame],
                 tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    out = tmp_path_factory.mktemp("results")
    results = run_ablation(frames, synthetic_dir / "preds", SMALL_CFG,
                           blackboard_path=out / "blackboard_armD.jsonl",
                           snapshot_path=out / "trust_snapshot.json")
    return {"results": results, "out": out}
