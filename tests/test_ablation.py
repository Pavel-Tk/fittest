"""End-to-end ablation on synthetic data, leakage guarantees, robustness and determinism."""

from __future__ import annotations

import dataclasses
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from conftest import SMALL_CFG, SMALL_SPEC

from trustswarm.ablation import (AblationConfig, arm_specs, auroc_mann_whitney, brier, build_trustswarm, topk_stream,
                                 expected_calibration_error, plot_ablation, prepare_data,
                                 run_ablation, trustswarm_score)
from trustswarm.agents import (Calibrator, FinetuneReport, ModelAgent, NullFinetune,
                               ReplayFinetune)
from trustswarm.agents.base import iter_rows
from trustswarm.blackboard import ORACLE_RESULT, Blackboard, Event, load_jsonl
from trustswarm.cache import load_models, load_predictions
from trustswarm.candidates import candidate_members
from trustswarm.data import Frame, arrival_rounds
from trustswarm.round_driver import RoundDriver
from trustswarm.synthetic import write_curated

ARMS = [name for name, _, _ in arm_specs(SMALL_CFG)]
HEADLINE = ("auroc_pool", "auroc_common", "auroc_eval", "spearman_pool", "recall_at_k",
            "planted_caught", "eval_force_rmse_pre", "eval_force_rmse_post", "eval_rel_e_mae_pre")


def test_all_arms_run_and_report_every_metric(ablation_run: dict[str, Any]) -> None:
    results = ablation_run["results"]
    assert list(results["arms"]) == ARMS
    for name, arm in results["arms"].items():
        assert arm["status"] == "ok", (name, arm.get("reason"))
        for key in HEADLINE:
            m = arm["metrics"][key]
            assert m["value"] is not None, (name, key)
            lo, hi = m["ci"]
            assert lo <= m["value"] + 1e-9 and m["value"] <= hi + 1e-9, (name, key, m)
        for key in ("auroc_pool", "auroc_eval", "recall_at_k", "planted_caught"):
            assert 0.0 <= arm["metrics"][key]["value"] <= 1.0
        assert len(arm["by_threshold"]) == len(SMALL_CFG.wrong_thresholds)
    json.dumps(results, allow_nan=False)  # JSON-serialisable, no NaN


def test_same_oracle_budget_exactly_k_per_round(ablation_run: dict[str, Any]) -> None:
    k, rounds = SMALL_CFG.budget_per_round, SMALL_CFG.rounds
    for arm in ablation_run["results"]["arms"].values():
        assert arm["oracle_calls"] == k * rounds
        assert arm["oracle_calls_per_round"] == {str(r): k for r in range(rounds)}
    log = load_jsonl(ablation_run["out"] / "blackboard_armD.jsonl")
    per_round = [sum(e.payload["bid_round"] == r for e in log if e.kind == ORACLE_RESULT)
                 for r in range(rounds)]
    assert per_round == [k] * rounds


def test_artifacts_written(ablation_run: dict[str, Any], tmp_path: Path) -> None:
    out = ablation_run["out"]
    log = load_jsonl(out / "blackboard_armD.jsonl")
    assert len(log) == ablation_run["results"]["arms"]["D_trustswarm"]["blackboard"]["n_events"]
    assert [e.id for e in log] == list(range(len(log)))
    snap = json.loads((out / "trust_snapshot.json").read_text())
    assert set(snap) == {"models", "candidates", "candidate_members", "p_global", "tiebreak_order",
                         "tau", "sigma_edges", "fbar_edges", "p_correct", "counts", "reputation",
                         "wrong_thresholds", "budget", "created"}
    assert snap["tiebreak_order"][0] == "consensus" and set(snap["p_global"]) == set(snap["candidates"])
    assert snap["models"] == list(SMALL_CFG.hetero_models)
    assert snap["candidates"] == list(candidate_members(SMALL_CFG.hetero_models))
    assert len(snap["candidates"]) == 7 and snap["candidates"][-1] == "consensus"
    assert snap["candidate_members"]["mean(aimnet2+orb)"] == ["aimnet2", "orb"]
    assert set(snap["p_correct"]) == set(snap["counts"]) == set(snap["candidates"])
    # exploration audits give the ledger labels in the low-σ buckets too
    assert sum(snap["counts"]["consensus"][0]) + sum(snap["counts"]["consensus"][1]) > 0
    # the audit log never carries arrays: forces, labels and coordinates are summarised
    for e in log:
        for key in ("forces", "F", "R"):
            if key in e.payload:
                assert set(e.payload[key]) == {"omitted", "shape"}, (e.kind, key)
    plot_ablation(ablation_run["results"], tmp_path / "ablation.png")
    assert (tmp_path / "ablation.png").stat().st_size > 10_000


def test_audits_target_only_trusted_systems_within_budget(ablation_run: dict[str, Any]) -> None:
    log = load_jsonl(ablation_run["out"] / "blackboard_armD.jsonl")
    k, quota = SMALL_CFG.budget_per_round, SMALL_CFG.audit_quota
    assert quota > 0
    audit_bids = [(e, uid) for e in log if e.kind == "bid" and e.payload.get("audit")
                  for uid in e.payload["uid"]]
    assert audit_bids and all(e.author == "skeptic" for e, _ in audit_bids)
    for bid, uid in audit_bids:
        before = [e for e in log if e.id < bid.id]
        assert not any(e.kind == "flag" and e.payload["uid"] == uid for e in before), uid
        assert not any(e.kind == "oracle_result" and e.payload["uid"] == uid for e in before), uid
        tau = [e.payload["tau"] for e in before if e.kind == "threshold"][-1]
        sigma = [row["sigma_F"] for e in before if e.kind == "verdict"
                 for _, row in iter_rows(e.payload, ("uid", "sigma_F")) if row["uid"] == uid][-1]
        assert sigma < tau
    results = [e.payload for e in log if e.kind == ORACLE_RESULT]
    for r in range(SMALL_CFG.rounds):
        this_round = [p for p in results if p["bid_round"] == r]
        assert len(this_round) == k  # audits come out of the same budget
        assert sum(p["audit"] for p in this_round) <= quota
    d = ablation_run["results"]["arms"]["D_trustswarm"]
    by_source = d["oracle_labels_by_source"]
    assert len(by_source["audit"]) + len(by_source["bid"]) == k * SMALL_CFG.rounds
    assert d["metrics"]["audit_hit_rate"]["value"] is not None
    # every oracle result reports the error of every subset candidate
    assert all(set(p["candidate_errors"]) == set(candidate_members(SMALL_CFG.hetero_models))
               for p in results)


def test_topk_stream_audits_trusted_systems_within_budget() -> None:
    scores = {f"u{i:02d}": float(i) for i in range(40)}
    arrival = {u: 0 for u in scores}
    checked, audited = topk_stream(scores, arrival, k=5, rounds=3, audit_quota=2,
                                   trusted_below=20.0, rng=np.random.default_rng(0))
    assert len(checked) == 15 and len(audited) == 6
    assert all(scores[u] < 20.0 for u in audited)
    for r in range(3):
        assert sum(v == r for v in checked.values()) == 5
    plain, none = topk_stream(scores, arrival, k=5, rounds=1)
    assert none == set() and sorted(plain) == [f"u{i}" for i in range(35, 40)]


def test_calibration_and_silent_failure_metrics(ablation_run: dict[str, Any]) -> None:
    for name, arm in ablation_run["results"]["arms"].items():
        m = arm["metrics"]
        for key in ("ece_eval", "brier_eval", "silent_failures_pool", "silent_failures_eval"):
            assert m[key]["value"] is not None and 0.0 <= m[key]["value"] <= 1.0, (name, key)


def test_expected_calibration_error_and_brier() -> None:
    y = np.array([0, 0, 1, 1], dtype=bool)
    assert expected_calibration_error(np.array([0.0, 0.0, 1.0, 1.0]), y) == 0.0
    assert expected_calibration_error(np.full(4, 0.5), y) == pytest.approx(0.0)  # one bin, 0.5 vs 0.5
    assert expected_calibration_error(np.array([0.9, 0.9, 0.9, 0.9]), y) == pytest.approx(0.4)
    assert brier(np.array([1.0, 0.0]), np.array([1, 1])) == pytest.approx(0.5)


def test_fast_auroc_matches_sklearn() -> None:
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(0)
    for _ in range(20):
        y = rng.random(200) < 0.3
        s = np.round(rng.normal(size=200) + y, 1)  # rounding creates ties
        assert auroc_mann_whitney(y, s) == pytest.approx(roc_auc_score(y, s), abs=1e-12)
    assert auroc_mann_whitney(np.zeros(5, bool), np.arange(5.0)) is None


def test_trustswarm_score_is_sigma_order_unless_skeptic_speaks() -> None:
    uids = ["a", "b", "c", "d"]
    sigma = {"a": 0.1, "b": 0.2, "c": 0.3, "d": 0.4}

    def plateau(s: float) -> float:  # an isotonic fit with a flat step
        return 0.2 if s < 0.35 else 0.8

    plain = trustswarm_score(uids, sigma, {}, plateau)
    assert sorted(uids, key=plain.get) == ["a", "b", "c", "d"]  # ties broken by σ
    vetoed = trustswarm_score(uids, sigma, {"a": 0.9}, plateau)
    assert max(uids, key=vetoed.get) == "a"  # the Skeptic's p_S lifts a vetoed system
    fallback = trustswarm_score(uids, sigma, {}, lambda s: None)
    assert sorted(uids, key=fallback.get) == ["a", "b", "c", "d"]


def test_finetune_without_offline_replay_is_reported(ablation_run: dict[str, Any]) -> None:
    ft = ablation_run["results"]["arms"]["D_trustswarm"]["finetune"]
    assert ft["model"] == max(ft["ledger_force_rmse"], key=ft["ledger_force_rmse"].get)
    assert not ft["shipped"]
    assert any("fine-tune not shipped" in f for f in ablation_run["results"]["failures"])


class Recorder:
    """Records the Curator's request (as scripts/run_ablation.py does for the GPU script)."""

    def __init__(self) -> None:
        self.request: tuple[str, list[str]] | None = None

    def __call__(self, model: str, frames: list[Frame]) -> FinetuneReport:
        self.request = (model, [f.uid for f in frames])
        return NullFinetune()(model, frames)


def test_two_pass_finetune_ships_only_with_matching_provenance(
        synthetic_dir: Path, frames: list[Frame], tmp_path: Path) -> None:
    shutil.copytree(synthetic_dir / "preds", tmp_path / "preds")
    cfg = dataclasses.replace(SMALL_CFG, n_bootstrap=10, mace_seed_models=None)
    rec = Recorder()
    run_ablation(frames, tmp_path / "preds", cfg, finetune_fn=rec)      # pass 1: labels
    assert rec.request is not None
    model, uids = rec.request
    write_curated(tmp_path, uids, SMALL_SPEC)                           # "GPU" fine-tune
    again = run_ablation(frames, tmp_path / "preds", cfg)               # pass 2: replay
    ft = again["arms"]["D_trustswarm"]["finetune"]
    if model == cfg.curated_base_model:
        assert ft["shipped"], ft["report"]
        assert ft["eval"]["finetuned_force_rmse"]["value"] is not None
    else:
        assert not ft["shipped"]
    # a different budget means different oracle labels: the replay must not ship
    other = run_ablation(frames, tmp_path / "preds", dataclasses.replace(cfg, budget_per_round=6))
    ft_other = other["arms"]["D_trustswarm"]["finetune"]
    assert not ft_other["shipped"]
    if model == cfg.curated_base_model:
        assert "not re-run" in ft_other["report"]["reason"]


def test_missing_model_skips_arm_without_crashing(synthetic_dir: Path, frames: list[Frame],
                                                   tmp_path: Path) -> None:
    preds = tmp_path / "preds"
    shutil.copytree(synthetic_dir / "preds", preds)
    (preds / "orb.npz").unlink()
    for f in preds.glob("mace_seed*.npz"):
        f.unlink()
    cfg = AblationConfig(**{**SMALL_CFG.__dict__, "n_bootstrap": 20})
    results = run_ablation(frames, preds, cfg, finetune_fn=NullFinetune())
    arms = results["arms"]
    for name in ("C_collective", "D_trustswarm", "B_mace_seeds"):
        assert arms[name]["status"] == "skipped" and "missing" in arms[name]["reason"]
    assert arms["A_single"]["status"] == arms["B_seed"]["status"] == "ok"
    assert any(f.startswith("C_collective skipped") and "orb.npz" in f for f in results["failures"])
    assert "D_trustswarm did not run: no comparisons possible" in results["failures"]
    json.dumps(results, allow_nan=False)
    plot_ablation(results, tmp_path / "partial.png")  # still plots, with "skipped" markers


def test_null_finetune_is_reported(synthetic_dir: Path, frames: list[Frame]) -> None:
    cfg = AblationConfig(**{**SMALL_CFG.__dict__, "n_bootstrap": 10, "mace_seed_models": None})
    results = run_ablation(frames, synthetic_dir / "preds", cfg, finetune_fn=NullFinetune())
    d = results["arms"]["D_trustswarm"]
    assert d["finetune"]["shipped"] is False
    assert d["metrics"]["eval_force_rmse_pre"] == d["metrics"]["eval_force_rmse_post"]
    assert any("NullFinetune" in f for f in results["failures"])


def test_replay_enforces_provenance(frames: list[Frame], tmp_path: Path) -> None:
    pool = [f for f in frames if f.split == "pool"][:40]
    path = write_curated(tmp_path, [f.uid for f in pool], SMALL_SPEC)
    assert "fine-tuned from 'mace_mpa0'" in ReplayFinetune({"orb": path})("orb", pool).report["reason"]
    trained = load_predictions(path).meta["train_uids"]
    leak = ReplayFinetune({"mace_mpa0": path}, held_out_uids=trained[:1])("mace_mpa0", pool)
    assert not leak.shipped and "held-out" in leak.report["reason"]
    foreign = ReplayFinetune({"mace_mpa0": path})("mace_mpa0", pool[:5])
    assert not foreign.shipped and "different label set" in foreign.report["reason"]
    ok = ReplayFinetune({"mace_mpa0": path})("mace_mpa0", pool)
    assert ok.shipped == (load_predictions(path).meta["acceptance_report"]["verdict"] == "pass")
    assert not ReplayFinetune({})("aimnet2", []).shipped


def test_service_reproduces_calibrator_recommendations(synthetic_dir: Path, frames: list[Frame],
                                                       tmp_path: Path) -> None:
    """The trust service, given the exported snapshot and the same predictions, must pick the
    same served candidate as the Calibrator did for every eval system."""
    service = pytest.importorskip("trustswarm.service")
    cfg = dataclasses.replace(SMALL_CFG, mace_seed_models=None)
    preds = load_models(synthetic_dir / "preds", cfg.hetero_models)
    data = prepare_data(frames, load_predictions(
        synthetic_dir / "preds" / f"{cfg.planted_model}.npz"), cfg)
    arrival = arrival_rounds(data.pool, cfg.rounds, cfg.seed)
    arrival.update({f.uid: 0 for f in data.injected})
    bb = Blackboard()
    driver, agents = build_trustswarm(data, preds, arrival, cfg, NullFinetune(), bb)
    driver.run(cfg.rounds + 2)
    (tmp_path / "snap.json").write_text(json.dumps(agents["calibrator"].export_snapshot()))
    snapshot = service.TrustSnapshot.load(tmp_path / "snap.json")

    served: dict[str, str] = {}
    for e in bb.read("verdict"):
        for _, row in iter_rows(e.payload, ("uid", "phase", "recommended_model")):
            if row["phase"] != "triage":
                served[row["uid"]] = row["recommended_model"]
    order = [f.uid for f in data.eval if f.uid in served]
    systems = [{"Z": data.by_uid[u].Z.tolist(), "R": data.by_uid[u].R.tolist()} for u in order]

    def cached(model: str) -> Any:  # a "predictor" replaying what the ModelAgent posted
        def predict(Z: np.ndarray, R: np.ndarray, batch: np.ndarray) -> dict[str, np.ndarray]:
            return {"energy": np.array([preds[model].energy[u] for u in order]),
                    "forces": np.concatenate([np.round(preds[model].forces[u], 6) for u in order])}
        return predict

    out = service.score(systems, {m: cached(m) for m in cfg.hetero_models}, snapshot)
    mismatches = [(u, r["recommended_model"], served[u]) for u, r in zip(order, out)
                  if r["recommended_model"] != served[u]]
    assert len(order) == len(data.eval) and not mismatches, mismatches[:5]


def test_deterministic_under_fixed_seed(synthetic_dir: Path, frames: list[Frame],
                                        ablation_run: dict[str, Any]) -> None:
    again = json.loads(json.dumps(run_ablation(frames, synthetic_dir / "preds", SMALL_CFG)))
    first = json.loads(json.dumps(ablation_run["results"]))
    for name in ARMS:
        a, b = first["arms"][name], again["arms"][name]
        a.pop("blackboard", None)  # only the log path differs
        b.pop("blackboard", None)
        assert a == b, name
    assert first["failures"] == again["failures"]


# --- leakage ------------------------------------------------------------------------------

LABEL_KEYS = {"E", "F", "rel_E"}


def _spy(agent: Any, seen: list[Event]) -> None:
    original = agent.handle

    def handle(bb: Blackboard, events: list[Event]) -> None:
        seen.extend(events)
        original(bb, events)

    agent.handle = handle


def _labelled_frames_in(obj: Any, depth: int = 0) -> list[str]:
    """Recursively find labelled Frames reachable from an agent's attributes."""
    if depth > 4:
        return []
    if isinstance(obj, Frame):
        return [obj.uid] if obj.has_labels else []
    if isinstance(obj, dict):
        return [u for v in obj.values() for u in _labelled_frames_in(v, depth + 1)]
    if isinstance(obj, (list, tuple, set)):
        return [u for v in obj for u in _labelled_frames_in(v, depth + 1)]
    if hasattr(obj, "__dict__") and not isinstance(obj, type):
        return _labelled_frames_in(vars(obj), depth + 1)
    return []


def test_labels_reach_agents_only_through_oracle_results(synthetic_dir: Path,
                                                         frames: list[Frame]) -> None:
    cfg = SMALL_CFG
    preds = load_models(synthetic_dir / "preds", cfg.hetero_models)
    data = prepare_data(frames, load_predictions(synthetic_dir / "preds" / f"{cfg.planted_model}.npz"), cfg)
    arrival = arrival_rounds(data.pool, cfg.rounds, cfg.seed)
    arrival.update({f.uid: 0 for f in data.injected})
    bb = Blackboard()
    driver, agents = build_trustswarm(data, preds, arrival, cfg, NullFinetune(), bb)
    watched = {name: a for name, a in agents.items() if isinstance(a, (ModelAgent, Calibrator))}
    seen: dict[str, list[Event]] = {name: [] for name in watched}
    for name, agent in watched.items():
        _spy(agent, seen[name])
    driver.run(cfg.rounds + 2)

    oracle_events = {e.payload["uid"]: e.id for e in bb.read(ORACLE_RESULT)}
    assert 0 < len(oracle_events) <= cfg.total_budget
    pool_and_eval = {f.uid for f in [*data.pool, *data.eval]}
    for name, events in seen.items():
        for e in events:
            if e.kind != ORACLE_RESULT:
                assert not (LABEL_KEYS & set(e.payload)), (name, e.kind)
            else:
                assert e.payload["uid"] in oracle_events
        revealed = {e.payload["uid"] for e in events if e.kind == ORACLE_RESULT}
        assert not (revealed & {f.uid for f in data.eval}), "eval labels must never be revealed"
        assert revealed <= pool_and_eval | {f.uid for f in data.injected}
        assert _labelled_frames_in(watched[name]) == [], name
    calibrator: Calibrator = agents["calibrator"]
    # every ledger entry was learned from the oracle_result that revealed it — never before
    assert calibrator.labelled == {u: oracle_events[u] for u in calibrator.labelled}
    assert set(calibrator.labelled) == set(oracle_events)
    # the stream and model agents only ever published label-free structures
    for e in bb.read(["system", "plant", "prediction"]):
        assert not (LABEL_KEYS & set(e.payload))
    assert isinstance(driver, RoundDriver)


def test_planted_and_reference_are_disjoint_by_formula(synthetic_dir: Path,
                                                        frames: list[Frame]) -> None:
    model = load_predictions(synthetic_dir / "preds" / f"{SMALL_CFG.planted_model}.npz")
    data = prepare_data(frames, model, SMALL_CFG)
    planted = {f.reaction for f in data.injected}
    assert planted and not planted & {f.reaction for f in data.refs}
    assert not {f.formula for f in data.injected} & {f.formula for f in data.refs}
    assert not planted & {f.reaction for f in [*data.pool, *data.eval]}
    assert not {f.reaction for f in data.pool} & {f.reaction for f in data.eval}
    assert 0 < len(data.planted_cases) <= SMALL_CFG.max_planted
    # planted cases are, by construction, confidently-wrong cases of a model every arm uses
    for f in data.planted_cases:
        assert np.sqrt(np.mean((model.forces[f.uid] - f.F) ** 2)) > SMALL_CFG.planted_force_rmse
