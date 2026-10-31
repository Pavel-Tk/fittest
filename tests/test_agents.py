from __future__ import annotations

import numpy as np
import pytest

from trustswarm import config
from trustswarm.agents import (Calibrator, DisagreementMonitor, ModelAgent, Oracle, PoolStream,
                               Skeptic)
from trustswarm.agents.base import Agent
from trustswarm.agents.calibrator import bucket_of, isotonic_threshold
from trustswarm.blackboard import (BID, FLAG, ORACLE_RESULT, PASS, PREDICTION, REPUTATION,
                                   THRESHOLD, VERDICT, Blackboard, Event)
from trustswarm.cache import Predictions
from trustswarm.data import Frame
from trustswarm.round_driver import QuiescenceError, RoundDriver


def labelled(uid: str, reaction: str = "r1", role: str = "path", n: int = 3) -> Frame:
    rng = np.random.default_rng(sum(map(ord, uid)))
    return Frame(uid=uid, reaction=reaction, formula="CH2", split="pool", role=role,
                 Z=np.array([6, 1, 1][:n]), R=rng.normal(size=(n, 3)), E=-10.0,
                 F=rng.normal(size=(n, 3)))


# --- Oracle --------------------------------------------------------------------------------

def _bid_round(bb: Blackboard, uids: list[str], bidders: int = 2) -> None:
    for i, u in enumerate(uids):
        for b in range(bidders):
            bb.post(BID, f"bidder{b}", {"uid": u, "value": 1.0 / (1 + i)})


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
    assert len(bids) == 1
    expected = 0.5 * (1 - np.exp(-0.8 / (2 * config.WRONG_F)))  # rep_0 · g(d, σ)
    assert bids[0].payload["value"] == pytest.approx(expected)
    bb.post(ORACLE_RESULT, "oracle", {"uid": "a", "errors": {}})
    bb.tick()
    agent.step(bb)
    assert len(bb.read(BID)) == 1


# --- Monitor --------------------------------------------------------------------------------

def _prediction(bb: Blackboard, uid: str, model: str, forces: np.ndarray,
                phase: str = "triage") -> int:
    return bb.post(PREDICTION, model, {"uid": uid, "model": model, "weights": model, "version": 0,
                                       "phase": phase, "energy": 0.0, "rel_energy": 0.0,
                                       "forces": forces.tolist()})


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
    assert {e.payload["uid"]: e.kind for e in bb.read([FLAG, PASS])} == {"calm": PASS, "hot": FLAG}
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
                   bidders: dict[str, float]) -> None:
    errors = {m: {"force_rmse": 1.0 if w else 0.0, "rel_e_err": 0.0, "wrong": w}
              for m, w in wrong.items()}
    bb.post(ORACLE_RESULT, "oracle", {"uid": uid, "bidders": bidders, "errors": errors,
                                      "consensus_error": None, "E": 0.0, "F": [[0, 0, 0]]})


def test_calibrator_beta_updates_settlement_and_snapshot() -> None:
    bb = Blackboard()
    cal = Calibrator(["a", "b"])
    for uid in ("x", "y"):
        _prediction(bb, uid, "a", np.zeros((1, 3)))
        _prediction(bb, uid, "b", np.zeros((1, 3)))
    cal.step(bb)
    assert bb.read(THRESHOLD) and len(bb.read(VERDICT)) == 2
    v = bb.read(VERDICT)[0].payload
    a0, b0 = config.CORRECTNESS_PRIOR
    assert v["recommended_model"] == "a"  # prior tie -> model order
    assert v["trust"] == pytest.approx(a0 / (a0 + b0))
    bucket = tuple(v["bucket"])

    # x: served model a is wrong, b right -> the bidders were right to ask
    _oracle_result(bb, "x", {"a": True, "b": False}, {"a": 0.6, "skeptic": 0.4})
    cal.step(bb)
    assert cal.p_correct("a", bucket) == pytest.approx(a0 / (a0 + b0 + 1))
    assert cal.p_correct("b", bucket) == pytest.approx((a0 + 1) / (a0 + b0 + 1))
    ra, rb = config.REPUTATION_PRIOR
    assert cal.reputation("a") == pytest.approx((ra + 0.6) / (ra + rb + 0.6))
    assert cal.reputation("skeptic") == pytest.approx((ra + 0.4) / (ra + rb + 0.4))
    last = bb.read(VERDICT)[-1].payload  # y is re-verdicted with the updated ledger
    assert (last["uid"], last["recommended_model"]) == ("y", "b")

    # y: served model (now b) is right -> false alarm costs lambda * share
    _oracle_result(bb, "y", {"a": True, "b": False}, {"a": 1.0})
    cal.step(bb)
    lam = config.FALSE_ALARM_PENALTY
    assert cal.reputation("a") == pytest.approx((ra + 0.6) / (ra + rb + 0.6 + lam))
    assert bb.read(REPUTATION)[-1].payload["reputation"]["a"] == pytest.approx(cal.reputation("a"))
    assert set(cal.labelled) == {"x", "y"}

    snap = cal.export_snapshot(budget=(40, 5))
    assert snap["models"] == ["a", "b"]
    assert len(snap["sigma_edges"]) == config.SIGMA_BINS - 1
    assert len(snap["fbar_edges"]) == config.FBAR_BINS - 1
    for m in ("a", "b"):
        assert np.array(snap["p_correct"][m]).shape == (config.SIGMA_BINS, config.FBAR_BINS)
    assert sum(map(sum, snap["counts"]["a"])) == 2
    assert snap["budget"] == {"per_round": 40, "rounds": 5}


def test_calibrator_reputation_is_clipped() -> None:
    cal = Calibrator(["a", "b"])
    bb = Blackboard()
    _prediction(bb, "x", "a", np.zeros((1, 3)))
    _prediction(bb, "x", "b", np.zeros((1, 3)))
    cal.step(bb)
    for i in range(200):
        _oracle_result(bb, f"z{i}", {"a": True, "b": True}, {"lucky": 1.0})
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
