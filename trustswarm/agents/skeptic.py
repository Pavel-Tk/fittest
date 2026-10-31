"""Skeptic: audits trusted verdicts against private held-out references (DESIGN.md §4).

Private info: ``skeptic_ref`` frames with labels (and the models' predictions on them, from
which it knows where the consensus was wrong), and the ``planted`` cases with labels.

* Round 0: injects the planted structures (label-free) into the stream as ``plant`` events,
  together with their reactions' reactant frames so relative energies can be formed.
* On every triage ``verdict`` with trust ≥ TRUST_HIGH: ``p_S`` = fraction of the k=7 nearest
  references (standardised label-free descriptor) whose consensus was wrong; if
  ``p_S > 0.5`` it posts ``veto`` and, every round until the system is revealed, a ``bid``
  ``rep_S · p_S``.
* Oracle results become new references (the reference set grows online).

The Skeptic deliberately does **not** use its knowledge of which systems are planted when
vetoing or bidding — otherwise "planted failures caught" would be true by construction.
Its planted labels are used only by :meth:`planted_report` for scoring.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

import numpy as np

from trustswarm import config
from trustswarm.agents.base import Agent
from trustswarm.agents.stream import TRIAGE, structure_payload
from trustswarm.blackboard import (BID, ORACLE_RESULT, PLANT, REPUTATION, SYSTEM, VERDICT, VETO,
                                   Blackboard, Event)
from trustswarm.cache import Predictions
from trustswarm.data import Frame, reactant_index, skeptic_descriptor, strip_labels
from trustswarm.signals import force_spread, frame_error, mean_abs_force


def reference_table(refs: Sequence[Frame], predictions: Mapping[str, Predictions],
                    wrong_f: float = config.WRONG_F, wrong_e: float = config.WRONG_E
                    ) -> tuple[list[str], np.ndarray, np.ndarray]:
    """(uids, descriptors (N, d), consensus_wrong (N,)) for labelled reference frames.

    Consensus = ensemble mean of the given models (forces and relative energies)."""
    models = sorted(predictions)
    reactants = reactant_index(refs)
    uids, X, y = [], [], []
    for f in refs:
        if f.F is None or any(f.uid not in predictions[m] for m in models):
            continue
        forces = np.stack([predictions[m].forces[f.uid] for m in models])
        sigma = force_spread(forces)[0]
        fbar = forces.mean(axis=0)
        r = reactants.get(f.reaction)
        rel_hat = rel_true = None
        if r is not None and r.E is not None and f.E is not None \
                and all(r.uid in predictions[m] for m in models):
            rel_hat = float(np.mean([predictions[m].energy[f.uid] - predictions[m].energy[r.uid]
                                     for m in models]))
            rel_true = f.E - r.E
        wrong = frame_error(fbar, f.F, rel_hat, rel_true).wrong(wrong_f, wrong_e)
        uids.append(f.uid)
        X.append(skeptic_descriptor(f.Z, f.R, mean_abs_force(fbar), sigma))
        y.append(wrong)
    return uids, np.array(X, dtype=float).reshape(len(uids), -1), np.array(y, dtype=bool)


class Skeptic(Agent):
    subscriptions: ClassVar[frozenset[str]] = frozenset(
        {SYSTEM, PLANT, VERDICT, ORACLE_RESULT, REPUTATION})

    def __init__(self, refs: Sequence[Frame], ref_predictions: Mapping[str, Predictions],
                 planted_cases: Sequence[Frame], planted_context: Sequence[Frame] = (),
                 k: int = config.KNN_K, name: str = "skeptic") -> None:
        super().__init__(name)
        ref_reactions = {f.reaction for f in refs}
        planted_reactions = {f.reaction for f in [*planted_cases, *planted_context]}
        overlap = ref_reactions & planted_reactions
        if overlap:
            raise ValueError(f"planted and skeptic_ref share reactions: {sorted(overlap)[:5]}")
        self.reference_reactions: frozenset[str] = frozenset(ref_reactions)
        self.planted_reactions: frozenset[str] = frozenset(planted_reactions)
        self._ref_uids, X, self._ref_wrong = reference_table(refs, ref_predictions)
        if len(self._ref_uids) == 0:
            raise ValueError("Skeptic needs at least one usable reference frame")
        self._mu = X.mean(axis=0)
        self._sd = np.where(X.std(axis=0) > 1e-9, X.std(axis=0), 1.0)
        self._X = (X - self._mu) / self._sd
        self._k = k
        self._planted = {f.uid: f for f in planted_cases}            # private, labelled
        planted_uids = set(self._planted)
        self._context = [f for f in planted_context if f.uid not in planted_uids]
        self._planted_posted = False
        self._geometry: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._features: dict[str, tuple[float, float]] = {}
        self._vetoed: dict[str, float] = {}
        self._revealed: set[str] = set()
        self._last_bid_round: dict[str, int] = {}
        self._reputation = 0.5

    @property
    def planted_uids(self) -> list[str]:
        return sorted(self._planted)

    def p_wrong(self, descriptor: np.ndarray) -> tuple[float, list[int]]:
        z = (descriptor - self._mu) / self._sd
        d = np.linalg.norm(self._X - z, axis=1)
        nn = np.argsort(d, kind="stable")[: min(self._k, len(d))]
        return float(np.mean(self._ref_wrong[nn])), nn.tolist()

    def handle(self, bb: Blackboard, events: list[Event]) -> None:
        if not self._planted_posted:
            self._planted_posted = True
            for f in sorted([*self._planted.values(), *self._context], key=lambda f: f.uid):
                self.post(bb, PLANT, structure_payload(strip_labels(f), TRIAGE))
        for e in events:
            p = e.payload
            if e.kind in (SYSTEM, PLANT):
                self._geometry[p["uid"]] = (np.asarray(p["Z"]), np.asarray(p["R"], dtype=float))
            elif e.kind == VERDICT:
                self._audit(bb, p)
            elif e.kind == ORACLE_RESULT:
                self._learn(p)
            elif e.kind == REPUTATION and self.name in p["reputation"]:
                self._reputation = float(p["reputation"][self.name])
        for uid in sorted(self._vetoed):
            if uid in self._revealed or self._last_bid_round.get(uid) == bb.round:
                continue
            self._last_bid_round[uid] = bb.round
            self.post(bb, BID, {"uid": uid, "value": self._reputation * self._vetoed[uid]})

    def _descriptor(self, uid: str) -> np.ndarray:
        Z, R = self._geometry[uid]
        sigma, fbar = self._features[uid]
        return skeptic_descriptor(Z, R, fbar, sigma)

    def _audit(self, bb: Blackboard, p: dict[str, Any]) -> None:
        uid = p["uid"]
        self._features[uid] = (float(p["sigma_F"]), float(p["mean_abs_fbar"]))
        if p["phase"] != TRIAGE or uid in self._revealed or uid in self._vetoed \
                or p["trust"] < config.TRUST_HIGH or uid not in self._geometry:
            return
        p_s, nn = self.p_wrong(self._descriptor(uid))
        if p_s > config.VETO_P:
            self._vetoed[uid] = p_s
            self.post(bb, VETO, {"uid": uid, "p_S": p_s, "trust": p["trust"],
                                 "neighbours": [self._ref_uids[i] for i in nn]})

    def _learn(self, p: dict[str, Any]) -> None:
        uid = p["uid"]
        self._revealed.add(uid)
        if uid in self._features and uid in self._geometry and p["consensus_error"] is not None:
            self._X = np.vstack([self._X, (self._descriptor(uid) - self._mu) / self._sd])
            self._ref_wrong = np.append(self._ref_wrong, bool(p["consensus_error"]["wrong"]))
            self._ref_uids.append(uid)

    def planted_report(self, caught: set[str]) -> dict[str, Any]:
        """Scoring hook: which planted cases did the collective catch?"""
        ids = self.planted_uids
        hit = [u for u in ids if u in caught]
        return {"n_planted": len(ids), "n_caught": len(hit),
                "fraction": len(hit) / len(ids) if ids else None}
