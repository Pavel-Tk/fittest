"""Oracle: the DFT engine. Holds labels; pays the top-k summed bids of each closed round.

Auction semantics: bids posted during round r are sealed until the clock leaves round r; at
its first step in round r+1 the Oracle pays the ``k + carry`` highest summed bids (ties
broken by uid), never exceeding the total budget ``K = k × n_bid_rounds``. Unused allowance
carries over within the run and is reported, never silently dropped.

``oracle_result`` is the *only* event that carries labels. An oracle call on frame x returns
F(x), E(x) and the relative energy E(x) - E(reactant of x); the reactant reference is computed
once per reaction and not charged to the budget (identical rule in every ablation arm).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any, ClassVar

import numpy as np

from trustswarm.agents.base import Agent, to_list
from trustswarm.blackboard import BID, ORACLE_RESULT, PREDICTION, Blackboard, Event
from trustswarm.data import Frame, reactant_index
from trustswarm.signals import frame_error


class Oracle(Agent):
    subscriptions: ClassVar[frozenset[str]] = frozenset({BID, PREDICTION})

    def __init__(self, labelled: Sequence[Frame], budget_per_round: int, n_bid_rounds: int,
                 name: str = "oracle") -> None:
        super().__init__(name)
        if any(f.E is None or f.F is None for f in labelled):
            raise ValueError("Oracle needs labelled frames")
        self._labels: dict[str, Frame] = {f.uid: f for f in labelled}  # private
        self._reactant_E = {r: float(f.E) for r, f in reactant_index(labelled).items()}  # type: ignore[arg-type]
        self.budget_per_round = int(budget_per_round)
        self.n_bid_rounds = int(n_bid_rounds)
        self.total_budget = self.budget_per_round * self.n_bid_rounds
        self._bids: dict[int, dict[tuple[str, str], float]] = defaultdict(dict)
        self._latest_pred: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        self._revealed: set[str] = set()
        self._next_round_to_settle = 0
        self._carry = 0
        self.spent_per_round: dict[int, int] = {}
        self.rejected_bids = 0  # bids on systems the Oracle cannot label (e.g. eval)

    @property
    def spent(self) -> int:
        return sum(self.spent_per_round.values())

    @property
    def remaining(self) -> int:
        return self.total_budget - self.spent

    def handle(self, bb: Blackboard, events: list[Event]) -> None:
        for e in events:
            if e.kind == BID:
                # latest bid per (bidder, system) within a round counts
                self._bids[e.round][(e.author, e.payload["uid"])] = float(e.payload["value"])
            elif e.kind == PREDICTION:
                self._latest_pred[e.payload["uid"]][e.payload["model"]] = e.payload
        while self._next_round_to_settle < bb.round:
            self._settle(bb, self._next_round_to_settle)
            self._next_round_to_settle += 1

    def _settle(self, bb: Blackboard, r: int) -> None:
        if r >= self.n_bid_rounds:
            return  # auction closed: budget K has been offered in full
        totals: dict[str, float] = defaultdict(float)
        bidders: dict[str, dict[str, float]] = defaultdict(dict)
        for (author, uid), value in sorted(self._bids.pop(r, {}).items()):
            if uid in self._revealed or value <= 0.0:
                continue
            if uid not in self._labels:
                self.rejected_bids += 1
                continue
            totals[uid] += value
            bidders[uid][author] = value
        allowance = min(self.budget_per_round + self._carry, self.remaining)
        winners = sorted(totals, key=lambda u: (-totals[u], u))[:allowance]
        self._carry = allowance - len(winners)
        self.spent_per_round[r] = len(winners)
        assert self.spent <= self.total_budget, "oracle budget exceeded"
        for uid in winners:
            self._revealed.add(uid)
            self.post(bb, ORACLE_RESULT, self._result(uid, r, totals[uid], bidders[uid]))

    def _result(self, uid: str, r: int, total: float, bidders: dict[str, float]) -> dict[str, Any]:
        f = self._labels[uid]
        assert f.E is not None and f.F is not None
        e_r = self._reactant_E.get(f.reaction)
        rel_true = None if e_r is None else float(f.E) - e_r
        preds = self._latest_pred.get(uid, {})
        errors = {m: frame_error(np.asarray(p["forces"]), f.F, p["rel_energy"], rel_true).to_dict()
                  for m, p in sorted(preds.items())}
        consensus = None
        if preds:
            mean_f = np.mean([np.asarray(p["forces"]) for p in preds.values()], axis=0)
            rels = [p["rel_energy"] for p in preds.values()]
            mean_rel = float(np.mean(rels)) if all(x is not None for x in rels) else None
            consensus = frame_error(mean_f, f.F, mean_rel, rel_true).to_dict()
        return {"uid": uid, "reaction": f.reaction, "role": f.role, "bid_round": r,
                "bid_total": total, "bidders": bidders, "E": float(f.E), "F": to_list(f.F),
                "rel_E": rel_true, "errors": errors, "consensus_error": consensus,
                "budget_remaining": self.remaining}
