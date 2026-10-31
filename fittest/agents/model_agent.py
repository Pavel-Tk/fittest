"""ModelAgent: one MLIP. Holds only its own predictions; bids for oracle calls (DESIGN.md §4)."""

from __future__ import annotations

import math
from typing import ClassVar

import numpy as np

from fittest.agents.base import Agent, batch_payload, to_list
from fittest.agents.stream import TRIAGE
from fittest.blackboard import (BID, FLAG, MODEL_UPDATE, ORACLE_RESULT, PLANT, PREDICTION,
                                   REPUTATION, SYSTEM, Blackboard, Event)
from fittest.cache import Predictions, load_predictions
from fittest.config import WRONG_F

MIN_HISTORY_FOR_RECALIBRATION = 5


class ModelAgent(Agent):
    """Posts ``prediction`` for every arriving structure it can predict, and ``bid`` on flagged
    triage systems: ``bid_m(x) = rep_m · g(d_m(x), σ(x))`` with
    ``g = 1 - exp(-(σ + d_m) / (2 s_m))``. ``s_m`` is the agent's private calibration scale,
    learned from its own oracle-revealed errors (how much disagreement it takes before *it*
    is wrong)."""

    subscriptions: ClassVar[frozenset[str]] = frozenset(
        {SYSTEM, PLANT, FLAG, ORACLE_RESULT, REPUTATION, MODEL_UPDATE})

    def __init__(self, predictions: Predictions, name: str | None = None) -> None:
        super().__init__(name or predictions.model)
        self._preds = predictions                  # private: own predictions only
        self._weights = predictions.model
        self._version = 0
        self._reaction_of: dict[str, str] = {}
        self._phase_of: dict[str, str] = {}
        self._reactant_uid: dict[str, str] = {}
        self._flagged: dict[str, tuple[float, float]] = {}  # uid -> (σ, own deviation)
        self._revealed: set[str] = set()
        self._last_bid_round: dict[str, int] = {}
        self._reputation = 0.5
        self._scale = WRONG_F
        self._history: list[tuple[float, float]] = []  # ((σ+d)/2, own force RMSE)
        self.missing: list[str] = []                    # structures it could not predict

    # --- event handling ------------------------------------------------------------------
    def handle(self, bb: Blackboard, events: list[Event]) -> None:
        arrivals = [e for e in events if e.kind in (SYSTEM, PLANT)]
        for e in arrivals:
            p = e.payload
            self._reaction_of[p["uid"]] = p["reaction"]
            self._phase_of[p["uid"]] = p["phase"]
            if p["role"] == "reactant":
                self._reactant_uid[p["reaction"]] = p["uid"]
        self._predict(bb, [e.payload["uid"] for e in arrivals])

        for e in events:
            if e.kind == FLAG:
                self._on_flag(e)
            elif e.kind == ORACLE_RESULT:
                self._on_oracle(e)
            elif e.kind == REPUTATION and self.name in e.payload["reputation"]:
                self._reputation = float(e.payload["reputation"][self.name])
            elif e.kind == MODEL_UPDATE and e.payload["model"] == self.name:
                self._on_update(bb, e)
        self._bid(bb)

    def _predict(self, bb: Blackboard, uids: list[str]) -> None:
        """Post one ``prediction`` batch (column-wise) for the structures it can predict."""
        rows = []
        for uid in uids:
            if uid not in self._preds:
                self.missing.append(uid)
                continue
            energy = self._preds.energy[uid]
            r_uid = self._reactant_uid.get(self._reaction_of[uid])
            rel = energy - self._preds.energy[r_uid] if r_uid in self._preds else None
            rows.append({"uid": uid, "phase": self._phase_of[uid], "energy": energy,
                         "rel_energy": rel, "forces": to_list(self._preds.forces[uid])})
        if rows:
            self.post(bb, PREDICTION, batch_payload(rows, model=self.name, weights=self._weights,
                                                    version=self._version))

    def _on_flag(self, e: Event) -> None:
        p = e.payload
        if p["phase"] != TRIAGE or p["uid"] in self._revealed or self.name not in p["deviation"]:
            return
        self._flagged[p["uid"]] = (float(p["sigma_F"]), float(p["deviation"][self.name]))

    def _on_oracle(self, e: Event) -> None:
        uid = e.payload["uid"]
        self._revealed.add(uid)
        flagged = self._flagged.pop(uid, None)
        own = e.payload["errors"].get(self.name)
        if flagged is None or own is None:
            return
        sigma, dev = flagged
        self._history.append(((sigma + dev) / 2.0, float(own["force_rmse"])))
        if len(self._history) >= MIN_HISTORY_FOR_RECALIBRATION:
            ratios = [x / max(err, 1e-3) for x, err in self._history]
            self._scale = float(np.clip(WRONG_F * np.median(ratios), WRONG_F / 4, 4 * WRONG_F))

    def _on_update(self, bb: Blackboard, e: Event) -> None:
        """Curator shipped new weights: swap private predictions and re-predict."""
        self._preds = load_predictions(e.payload["predictions_path"])
        self._weights = str(e.payload["weights"])
        self._version += 1
        self._predict(bb, sorted(self._reaction_of))

    # --- bidding -------------------------------------------------------------------------
    def wrongness_estimate(self, sigma: float, deviation: float) -> float:
        """g(d, σ): private estimate that the served answer on this system is wrong."""
        return 1.0 - math.exp(-(sigma + deviation) / (2.0 * self._scale))

    def _bid(self, bb: Blackboard) -> None:
        """One bid batch per step: every flagged, unrevealed system not yet bid on this round."""
        rows = []
        for uid in sorted(self._flagged):
            if self._last_bid_round.get(uid) == bb.round:
                continue
            sigma, dev = self._flagged[uid]
            self._last_bid_round[uid] = bb.round
            rows.append({"uid": uid, "value": self._reputation * self.wrongness_estimate(sigma, dev)})
        if rows:
            self.post(bb, BID, batch_payload(rows))
