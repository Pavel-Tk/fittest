from __future__ import annotations

import numpy as np
import pytest

from trustswarm import config
from trustswarm.agents import (Calibrator, DisagreementMonitor, ModelAgent, Oracle, PoolStream,
                               Skeptic)
from trustswarm.agents.base import Agent, batch_payload, iter_rows
from trustswarm.agents.calibrator import bucket_of, isotonic_threshold
from trustswarm.blackboard import (BID, FLAG, ORACLE_RESULT, PASS, PREDICTION, REPUTATION,
                                   THRESHOLD, VERDICT, Blackboard, Event)
from trustswarm.cache import Predictions
from trustswarm.candidates import candidate_members
from trustswarm.data import Frame
from trustswarm.round_driver import QuiescenceError, RoundDriver


def labelled(uid: str, reaction: str = "r1", role: str = "path", n: int = 3) -> Frame:
    rng = np.random.default_rng(sum(map(ord, uid)))
    return Frame(uid=uid, reaction=reaction, formula="CH2", split="pool", role=role,
                 Z=np.array([6, 1, 1][:n]), R=rng.normal(size=(n, 3)), E=-10.0,
                 F=rng.normal(size=(n, 3)))


# --- Oracle --------------------------------------------------------------------------------

def _bid_round(bb: Blackboard, uids: list[str], bidders: int = 2) -> None:
    for b in range(bidders):
        rows = [{"uid": u, "value": 1.0 / (1 + i)} for i, u in enumerate(uids)]
        bb.post(BID, f"bidder{b}", batch_payload(rows))


def test_oracle_pays_exactly_k_per_round_and_never_exceeds_budget() -> None:
    frames = [labelled(f"u{i:02d}") for i in range(30)]
    oracle = Oracle(frames, budget_per_round=4, n_bid_rounds=3)
    bb = Blackboard()
    for _ in range(5):  # bids keep coming after the auction closes
        _bid_round(bb, [f.uid for f in frames])
        oracle.step(bb)
        bb.tick()
    oracle.step(bb)
    results = bb.read(ORACLE_RESULT)
    per_round = [sum(e.payload["bid_round"] == r for e in results) for r in range(5)]
    assert per_round == [4, 4, 4, 0, 0]
    assert oracle.spent == oracle.total_budget == 12
    assert len({e.payload["uid"] for e in results}) == 12  # never pays twice for one system
    # the top summed bids win, ties broken by uid
    assert [e.payload["uid"] for e in results[:4]] == ["u00", "u01", "u02", "u03"]


def test_oracle_carries_unspent_allowance_and_rejects_unlabelled() -> None:
    frames = [labelled(f"u{i:02d}") for i in range(10)]
    oracle = Oracle(frames, budget_per_round=3, n_bid_rounds=2)
    bb = Blackboard()
    _bid_round(bb, ["u00", "eval-only"])  # one labelable system, one the oracle cannot label
    oracle.step(bb)
    bb.tick()
    _bid_round(bb, [f.uid for f in frames])
    oracle.step(bb)
    bb.tick()
    oracle.step(bb)
    assert oracle.spent_per_round == {0: 1, 1: 5}  # 3 + carry 2
    assert oracle.spent == oracle.total_budget
    assert oracle.rejected_bids == 2


def test_oracle_pays_audits_from_the_same_budget() -> None:
    frames = [labelled(f"u{i:02d}") for i in range(30)]
    oracle = Oracle(frames, budget_per_round=4, n_bid_rounds=2, audit_quota=1)
    bb = Blackboard()
    for _ in range(3):
        _bid_round(bb, [f.uid for f in frames[:20]])             # regular bids
        bb.post(BID, "skeptic", batch_payload(                  # three audit requests
            [{"uid": f.uid, "value": 1.0} for f in frames[25:28]], audit=True))
        oracle.step(bb)
        bb.tick()
    oracle.step(bb)
    results = bb.read(ORACLE_RESULT)
    assert oracle.spent_per_round == {0: 4, 1: 4} and oracle.spent == oracle.total_budget
    assert oracle.audits_per_round == {0: 1, 1: 1}  # quota, not the number requested
    audits = [e.payload for e in results if e.payload["audit"]]
    assert [p["uid"] for p in audits] == ["u25", "u26"]  # first unrevealed audit request
    assert all(p["bidders"] == {"skeptic": 1.0} for p in audits)
    with pytest.raises(ValueError):
        Oracle(frames, budget_per_round=2, n_bid_rounds=1, audit_quota=3)


def test_candidate_members_names_every_subset() -> None:
    members = candidate_members(["orb", "aimnet2", "mace_mpa0"])
    assert list(members) == ["orb", "aimnet2", "mace_mpa0", "mean(aimnet2+mace_mpa0)",
                             "mean(aimnet2+orb)", "mean(mace_mpa0+orb)", "consensus"]
    assert members["consensus"] == ("aimnet2", "mace_mpa0", "orb")
    assert list(candidate_members(["a", "b"])) == ["a", "b", "consensus"]
    with pytest.raises(ValueError):
        candidate_members(["a", "consensus"])


def test_calibrator_can_serve_a_subset_mean() -> None:
    bb = Blackboard()
    cal = Calibrator(["a", "b", "c"])
    _prediction(bb, "x", "a", np.zeros((1, 3)))
    _prediction(bb, "x", "b", np.zeros((1, 3)))
    _prediction(bb, "x", "c", np.zeros((1, 3)))
    _prediction(bb, "y", "a", np.zeros((1, 3)))
    _prediction(bb, "y", "b", np.zeros((1, 3)))
    _prediction(bb, "y", "c", np.zeros((1, 3)))
    cal.step(bb)
    wrong = {name: name != "mean(a+b)" for name in candidate_members(["a", "b", "c"])}
    errors = {n: {"force_rmse": 0.0, "rel_e_err": 0.0, "wrong": w} for n, w in wrong.items()}
    bb.post(ORACLE_RESULT, "oracle", {"uid": "x", "bidders": {}, "candidate_errors": errors,
                                      "errors": {m: errors[m] for m in "abc"},
                                      "consensus_error": errors["consensus"]})
    cal.step(bb)
    v = _verdicts(bb)[-1]
    assert (v["uid"], v["recommended_model"]) == ("y", "mean(a+b)")
    assert set(v["prediction_events"]) == {"a", "b"}


def test_oracle_requires_labels() -> None:
    f = labelled("a")
    f.E = None
    with pytest.raises(ValueError):
        Oracle([f], 1, 1)


# --- Stream / ModelAgent ----------------------------------------------------------------------

def test_stream_refuses_labelled_frames() -> None:
    with pytest.raises(ValueError):
        PoolStream([labelled("a")], {"a": 0})


def test_model_agent_bids_on_flags_only_until_revealed() -> None:
    agent = ModelAgent(Predictions("m", {"a": 1.0}, {"a": np.zeros((3, 3))}))
    bb = Blackboard()
    bb.post(FLAG, "hook", {"uid": "a", "phase": "triage", "sigma_F": 0.5,
                           "deviation": {"m": 0.3}})
    agent.step(bb)
    agent.step(bb)  # at most one bid per system per round
    bids = bb.read(BID)
    assert len(bids) == 1 and bids[0].payload["uid"] == ["a"]
    expected = 0.5 * (1 - np.exp(-0.8 / (2 * config.WRONG_F)))  # rep_0 · g(d, σ)
    assert bids[0].payload["value"][0] == pytest.approx(expected)
    bb.post(ORACLE_RESULT, "oracle", {"uid": "a", "errors": {}})
    bb.tick()
    agent.step(bb)
    assert len(bb.read(BID)) == 1


# --- Monitor --------------------------------------------------------------------------------

def _prediction(bb: Blackboard, uid: str, model: str, forces: np.ndarray,
                phase: str = "triage") -> int:
    """A one-row prediction batch, as a ModelAgent would post it."""
    row = {"uid": uid, "phase": phase, "energy": 0.0, "rel_energy": 0.0, "forces": forces.tolist()}
    return bb.post(PREDICTION, model, batch_payload([row], model=model, weights=model, version=0))


def _verdicts(bb: Blackboard) -> list[dict]:
    keys = ("uid", "phase", "trust", "risk", "recommended_model", "prediction_events", "bucket")
    return [row for e in bb.read(VERDICT) for _, row in iter_rows(e.payload, keys)]


def test_monitor_waits_for_threshold_then_flags_high_sigma() -> None:
    bb = Blackboard()
    hook = DisagreementMonitor(["a", "b"])
    _prediction(bb, "calm", "a", np.zeros((1, 3)))
    _prediction(bb, "calm", "b", np.zeros((1, 3)))
    _prediction(bb, "hot", "a", np.array([[1.0, 0, 0]]))
    _prediction(bb, "hot", "b", np.array([[-1.0, 0, 0]]))
    hook.step(bb)
    assert bb.read([FLAG, PASS]) == []
    bb.post(THRESHOLD, "calibrator", {"tau": 0.5})
    hook.step(bb)
    (passed,) = bb.read(PASS)  # passes come as one batch
    assert passed.payload["uid"] == ["calm"] and passed.payload["tau"] == 0.5
    assert [e.payload["uid"] for e in bb.read(FLAG)] == ["hot"]
    flag = bb.read(FLAG)[0].payload
    assert flag["sigma_F"] == pytest.approx(1.0)
    assert flag["deviation"] == {"a": 1.0, "b": 1.0}


# --- Calibrator -----------------------------------------------------------------------------

def test_bucket_of_uses_searchsorted_right() -> None:
    edges, fedges = [0.1, 0.2, 0.3], [1.0, 2.0]
    assert bucket_of(0.05, 0.5, edges, fedges) == (0, 0)
    assert bucket_of(0.1, 1.0, edges, fedges) == (1, 1)  # on an edge -> upper bin
    assert bucket_of(9.0, 9.0, edges, fedges) == (3, 2)


def test_isotonic_threshold() -> None:
    sigma = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    wrong = np.array([0, 0, 0, 1, 1, 1], dtype=bool)
    assert isotonic_threshold(sigma, wrong, 0.5) == pytest.approx(0.1)  # whole set: 3/6
    assert isotonic_threshold(sigma, wrong, 0.7) == pytest.approx(0.3)  # tail {.3, ...}: 3/4
    assert isotonic_threshold(sigma, np.zeros(6, bool), 0.5) is None
    assert isotonic_threshold(sigma, np.ones(6, bool), 0.5) == pytest.approx(0.1)


def _oracle_result(bb: Blackboard, uid: str, wrong: dict[str, bool],
                   bidders: dict[str, float], consensus_wrong: bool | None = None) -> None:
    errors = {m: {"force_rmse": 1.0 if w else 0.0, "rel_e_err": 0.0, "wrong": w}
              for m, w in wrong.items()}
    consensus = None if consensus_wrong is None else {"force_rmse": 0.0, "rel_e_err": 0.0,
                                                       "wrong": consensus_wrong}
    bb.post(ORACLE_RESULT, "oracle", {"uid": uid, "bidders": bidders, "errors": errors,
                                      "consensus_error": consensus, "E": 0.0, "F": [[0, 0, 0]]})


def test_calibrator_beta_updates_settlement_and_snapshot() -> None:
    bb = Blackboard()
    cal = Calibrator(["a", "b"])
    for uid in ("x", "y"):
        _prediction(bb, uid, "a", np.zeros((1, 3)))
        _prediction(bb, uid, "b", np.zeros((1, 3)))
    cal.step(bb)
    assert bb.read(THRESHOLD) and len(_verdicts(bb)) == 2
    v = _verdicts(bb)[0]
    a0, b0 = config.CORRECTNESS_PRIOR
    # prior tie over {a, b, consensus}: with no evidence the ensemble mean is served
    assert v["recommended_model"] == "consensus"
    assert set(v["prediction_events"]) == {"a", "b"}
    assert v["trust"] == pytest.approx(a0 / (a0 + b0))
    assert v["risk"] is None  # too few labels for the isotonic risk
    bucket = tuple(v["bucket"])

    # x: served consensus is wrong (a wrong, b right) -> the bidders were right to ask
    _oracle_result(bb, "x", {"a": True, "b": False}, {"a": 0.6, "skeptic": 0.4},
                   consensus_wrong=True)
    cal.step(bb)
    assert cal.p_correct("a", bucket) == pytest.approx(a0 / (a0 + b0 + 1))
    assert cal.p_correct("b", bucket) == pytest.approx((a0 + 1) / (a0 + b0 + 1))
    assert cal.p_correct("consensus", bucket) == pytest.approx(a0 / (a0 + b0 + 1))
    ra, rb = config.REPUTATION_PRIOR
    assert cal.reputation("a") == pytest.approx((ra + 0.6) / (ra + rb + 0.6))
    assert cal.reputation("skeptic") == pytest.approx((ra + 0.4) / (ra + rb + 0.4))
    last = _verdicts(bb)[-1]  # y is re-verdicted with the updated ledger
    assert (last["uid"], last["recommended_model"]) == ("y", "b")
    assert set(last["prediction_events"]) == {"b"}

    # y: served model (now b) is right -> false alarm costs lambda * share
    _oracle_result(bb, "y", {"a": True, "b": False}, {"a": 1.0}, consensus_wrong=False)
    cal.step(bb)
    lam = config.FALSE_ALARM_PENALTY
    assert cal.reputation("a") == pytest.approx((ra + 0.6) / (ra + rb + 0.6 + lam))
    assert bb.read(REPUTATION)[-1].payload["reputation"]["a"] == pytest.approx(cal.reputation("a"))
    assert set(cal.labelled) == {"x", "y"}

    snap = cal.export_snapshot(budget=(40, 5))
    assert snap["models"] == ["a", "b"]
    assert snap["candidates"] == ["a", "b", "consensus"]
    assert len(snap["sigma_edges"]) == config.SIGMA_BINS - 1
    assert len(snap["fbar_edges"]) == config.FBAR_BINS - 1
    for c in snap["candidates"]:
        assert np.array(snap["p_correct"][c]).shape == (config.SIGMA_BINS, config.FBAR_BINS)
        assert sum(map(sum, snap["counts"][c])) == 2
    assert snap["budget"] == {"per_round": 40, "rounds": 5}


def test_calibrator_risk_is_isotonic_in_sigma() -> None:
    bb = Blackboard()
    cal = Calibrator(["a", "b"])
    n = config.RISK_MIN_LABELLED + 10
    for i in range(n):  # σ grows with i; the served answer is wrong for the upper half
        _prediction(bb, f"s{i}", "a", np.zeros((1, 3)))
        _prediction(bb, f"s{i}", "b", np.array([[0.01 * (i + 1), 0.0, 0.0]]))
    cal.step(bb)
    for i in range(n):
        _oracle_result(bb, f"s{i}", {"a": i >= n // 2, "b": i >= n // 2}, {},
                       consensus_wrong=i >= n // 2)
    cal.step(bb)
    risks = [cal.risk(0.005 * (i + 1)) for i in range(n)]
    assert all(r is not None for r in risks)
    assert risks == sorted(risks) and risks[0] == 0.0 and risks[-1] == 1.0


def test_calibrator_reputation_is_clipped() -> None:
    cal = Calibrator(["a", "b"])
    bb = Blackboard()
    _prediction(bb, "x", "a", np.zeros((1, 3)))
    _prediction(bb, "x", "b", np.zeros((1, 3)))
    cal.step(bb)
    for i in range(200):
        _oracle_result(bb, f"z{i}", {"a": True, "b": True}, {"lucky": 1.0}, consensus_wrong=True)
    cal.step(bb)
    assert cal.reputation("lucky") == config.REPUTATION_CLIP[1]


# --- Skeptic --------------------------------------------------------------------------------

def test_calibrator_needs_two_models() -> None:
    with pytest.raises(ValueError):
        Calibrator(["only"])


def test_skeptic_rejects_overlapping_reference_and_planted_reactions() -> None:
    ref = labelled("ref", reaction="shared")
    preds = {m: Predictions(m, {"ref": -10.0}, {"ref": np.zeros((3, 3))}) for m in ("a", "b")}
    with pytest.raises(ValueError, match="share reactions"):
        Skeptic([ref], preds, [labelled("plant", reaction="shared")])
    sk = Skeptic([ref], preds, [labelled("plant", reaction="other")])
    assert sk.reference_reactions.isdisjoint(sk.planted_reactions)


# --- Round driver ---------------------------------------------------------------------------

class Chatterbox(Agent):
    def handle(self, bb: Blackboard, events: list[Event]) -> None:
        self.post(bb, "noise", {})


class Once(Agent):
    def __init__(self) -> None:
        super().__init__("once")
        self.rounds: list[int] = []

    def handle(self, bb: Blackboard, events: list[Event]) -> None:
        if bb.round not in self.rounds:
            self.rounds.append(bb.round)
            self.post(bb, "hello", {"round": bb.round})


def test_round_driver_only_advances_clock() -> None:
    bb = Blackboard()
    once = Once()
    RoundDriver(bb, [once]).run(3)
    assert bb.round == 3 and once.rounds == [0, 1, 2]
    with pytest.raises(QuiescenceError):
        RoundDriver(Blackboard(), [Chatterbox("x")], max_substeps=5).run(1)
