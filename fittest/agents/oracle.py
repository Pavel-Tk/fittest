"""Oracle: the DFT engine. Holds labels; pays the top-k summed bids of each closed round.

Auction semantics: bids posted during round r are sealed until the clock leaves round r; at
its first step in round r+1 the Oracle spends the allowance ``k + carry``: first on up to
``audit_quota`` exploration audits (bids tagged ``audit``), then on the highest summed bids
(ties broken by uid), never exceeding the total budget ``K = k × n_bid_rounds``. Unused
allowance carries over within the run and is reported, never silently dropped.

``oracle_result`` is the *only* event that carries labels, plus the error of every served-answer
candidate (each non-empty subset mean of the models, :mod:`fittest.candidates`). An oracle call on frame x returns
F(x), E(x) and the relative energy E(x) - E(reactant of x); the reactant reference is computed
once per reaction and not charged to the budget (identical rule in every ablation arm).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any, ClassVar

import numpy as np

from fittest.agents.base import Agent, iter_rows, to_list
from fittest.blackboard import BID, ORACLE_RESULT, PREDICTION, Blackboard, Event
from fittest.candidates import CONSENSUS, candidate_members
from fittest.data import Frame, reactant_index
from fittest.signals import frame_error


class Oracle(Agent):
    subscriptions: ClassVar[frozenset[str]] = frozenset({BID, PREDICTION})

    def __init__(self, labelled: Sequence[Frame], budget_per_round: int, n_bid_rounds: int,
                 models: Sequence[str] | None = None, audit_quota: int = 0,
                 name: str = "oracle") -> None:
        super().__init__(name)
        if any(f.E is None or f.F is None for f in labelled):
            raise ValueError("Oracle needs labelled frames")
        if not 0 <= audit_quota <= budget_per_round:
            raise ValueError("audit_quota must lie in [0, budget_per_round]")
        self._labels: dict[str, Frame] = {f.uid: f for f in labelled}  # private
        self._reactant_E = {r: float(f.E) for r, f in reactant_index(labelled).items()}  # type: ignore[arg-type]
        self.budget_per_round = int(budget_per_round)
        self.n_bid_rounds = int(n_bid_rounds)
        self.total_budget = self.budget_per_round * self.n_bid_rounds
        self.audit_quota = int(audit_quota)
        self._candidates = candidate_members(models) if models else None
        self._bids: dict[int, dict[tuple[str, str], float]] = defaultdict(dict)
        self._audit_bids: dict[int, dict[str, tuple[str, float]]] = defaultdict(dict)
        self._latest_pred: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        self._revealed: set[str] = set()
        self._next_round_to_settle = 0
        self._carry = 0
        self.spent_per_round: dict[int, int] = {}
        self.audits_per_round: dict[int, int] = {}
        self.rejected_bids = 0  # bids on systems the Oracle cannot label (e.g. eval)

    @property
    def spent(self) -> int:
        return sum(self.spent_per_round.values())

    @property
    def remaining(self) -> int:
        return self.total_budget - self.spent

    def handle(self, bb: Blackboard, events: list[Event]) -> None:
        for e in events:
            if e.kind == BID:  # a batch of bids from one bidder
                for _, row in iter_rows(e.payload, ("uid", "value")):
                    uid, value = row["uid"], float(row["value"])
                    if e.payload.get("audit"):
                        self._audit_bids[e.round].setdefault(uid, (e.author, value))
                    else:  # latest bid per (bidder, system) within a round counts
                        self._bids[e.round][(e.author, uid)] = value
            elif e.kind == PREDICTION:
                for _, row in iter_rows(e.payload, ("uid", "rel_energy", "forces")):
                    self._latest_pred[row["uid"]][e.payload["model"]] = row
        while self._next_round_to_settle < bb.round:
            self._settle(bb, self._next_round_to_settle)
            self._next_round_to_settle += 1

    def _payable(self, uid: str) -> bool:
        if uid in self._revealed:
            return False
        if uid not in self._labels:
            self.rejected_bids += 1
            return False
        return True

    def _settle(self, bb: Blackboard, r: int) -> None:
        """Pay exploration audits first (up to ``audit_quota``), then the top summed bids,
        all out of the same per-round allowance ``k + carry``."""
        if r >= self.n_bid_rounds:
            return  # auction closed: budget K has been offered in full
        allowance = min(self.budget_per_round + self._carry, self.remaining)
        audits = [(u, a) for u, a in self._audit_bids.pop(r, {}).items() if self._payable(u)]
        audits = audits[: min(self.audit_quota, allowance)]
        audited = {u for u, _ in audits}
        totals: dict[str, float] = defaultdict(float)
        bidders: dict[str, dict[str, float]] = defaultdict(dict)
        for (author, uid), value in sorted(self._bids.pop(r, {}).items()):
            if value <= 0.0 or uid in audited or not self._payable(uid):
                continue
            totals[uid] += value
            bidders[uid][author] = value
        winners = sorted(totals, key=lambda u: (-totals[u], u))[: allowance - len(audits)]
        self._carry = allowance - len(audits) - len(winners)
        self.spent_per_round[r] = len(audits) + len(winners)
        self.audits_per_round[r] = len(audits)
        assert self.spent <= self.total_budget, "oracle budget exceeded"
        for uid, (author, value) in audits:
            self._revealed.add(uid)
            self.post(bb, ORACLE_RESULT, self._result(uid, r, value, {author: value}, audit=True))
        for uid in winners:
            self._revealed.add(uid)
            self.post(bb, ORACLE_RESULT, self._result(uid, r, totals[uid], bidders[uid], audit=False))

    def _result(self, uid: str, r: int, total: float, bidders: dict[str, float],
                audit: bool) -> dict[str, Any]:
        f = self._labels[uid]
        assert f.E is not None and f.F is not None
        e_r = self._reactant_E.get(f.reaction)
        rel_true = None if e_r is None else float(f.E) - e_r
        preds = self._latest_pred.get(uid, {})
        candidates = self._candidates or candidate_members(sorted(preds))
        candidate_errors = {}
        for name, members in candidates.items():
            if not all(m in preds for m in members):
                continue
            mean_f = np.mean([np.asarray(preds[m]["forces"]) for m in members], axis=0)
            rels = [preds[m]["rel_energy"] for m in members]
            mean_rel = float(np.mean(rels)) if all(x is not None for x in rels) else None
            candidate_errors[name] = frame_error(mean_f, f.F, mean_rel, rel_true).to_dict()
        errors = {m: candidate_errors[m] for m in sorted(preds) if m in candidate_errors}
        return {"uid": uid, "reaction": f.reaction, "role": f.role, "bid_round": r,
                "audit": audit, "bid_total": total, "bidders": bidders, "E": float(f.E),
                "F": to_list(f.F), "rel_E": rel_true, "errors": errors,
                "candidate_errors": candidate_errors,
                "consensus_error": candidate_errors.get(CONSENSUS),
                "budget_remaining": self.remaining}
