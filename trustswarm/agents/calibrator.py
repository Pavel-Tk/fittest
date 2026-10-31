"""Calibrator: keeps the ledger; posts ``threshold``, ``verdict`` and ``reputation`` (DESIGN.md §4).

* Ledger: per (model, context bucket) Beta posterior of correctness from oracle results.
  The bucket is label-free: (σ_F quantile bin [4], mean |F̄| bin [3]); bin edges are fixed
  from the σ distribution the Calibrator sees in round 0.
* Verdict: ``recommended_model = argmax_m P(m correct | bucket)``; the served answer is that
  model's prediction; ``trust = P(served answer correct)``.
* Threshold: smallest τ with ledger-estimated P(served wrong | σ ≥ τ) ≥ 0.5 (isotonic fit on
  labelled points), floored at the 70th percentile of round-0 σ. If no τ reaches 0.5 the
  floor is kept (the hook never flags *less* than the floor).
* Settlement: a winning bid on a system whose served answer is revealed wrong earns
  ``+bid share``; a false alarm costs ``λ · bid share``. Reputation = Beta mean, clipped.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
from sklearn.isotonic import IsotonicRegression

from trustswarm import config
from trustswarm.agents.base import Agent
from trustswarm.agents.stream import TRIAGE
from trustswarm.blackboard import (ORACLE_RESULT, PREDICTION, REPUTATION, THRESHOLD, VERDICT,
                                   Blackboard, Event)
from trustswarm.signals import force_spread, mean_abs_force


@dataclass
class BetaCounter:
    alpha: float
    beta: float

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)


@dataclass(frozen=True)
class Verdict:
    recommended_model: str
    trust: float
    prediction_event: int
    bucket: tuple[int, int]


def bucket_of(sigma_F: float, mean_abs_fbar: float, sigma_edges: Sequence[float],
              fbar_edges: Sequence[float]) -> tuple[int, int]:
    """Label-free context bucket ``(i, j)``.

    ``i = np.searchsorted(sigma_edges, sigma_F, side="right")`` and likewise ``j`` for
    ``mean_abs_fbar``: with inner edges ``e_0 < e_1 < ...`` a value ``v`` lands in bin
    ``#{e_k <= v}``, so ``v < e_0`` → 0, ``e_0 <= v < e_1`` → 1, ..., ``v >= e_last`` → len(edges).
    Shared by the Calibrator and the trust service so both bucket identically.
    """
    return (int(np.searchsorted(np.asarray(sigma_edges, dtype=float), sigma_F, side="right")),
            int(np.searchsorted(np.asarray(fbar_edges, dtype=float), mean_abs_fbar, side="right")))


def isotonic_threshold(sigma: np.ndarray, wrong: np.ndarray, target: float) -> float | None:
    """Smallest σ_i such that mean isotonic P(wrong) over {σ_j ≥ σ_i} ≥ target (None if none)."""
    if len(sigma) == 0 or len(np.unique(wrong)) < 2:
        return None if len(sigma) == 0 or not wrong.all() else float(np.min(sigma))
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(sigma, wrong.astype(float))
    order = np.argsort(sigma)
    s, p = sigma[order], iso.predict(sigma[order])
    tail_mean = np.cumsum(p[::-1])[::-1] / np.arange(len(p), 0, -1)
    ok = np.flatnonzero(tail_mean >= target)
    return float(s[ok[0]]) if ok.size else None


class Calibrator(Agent):
    subscriptions: ClassVar[frozenset[str]] = frozenset({PREDICTION, ORACLE_RESULT})

    def __init__(self, models: Sequence[str], name: str = "calibrator",
                 correctness_prior: tuple[float, float] = config.CORRECTNESS_PRIOR,
                 reputation_prior: tuple[float, float] = config.REPUTATION_PRIOR,
                 false_alarm_penalty: float = config.FALSE_ALARM_PENALTY) -> None:
        super().__init__(name)
        if len(models) < 2:
            raise ValueError("the Calibrator needs at least two models to compare")
        self._models = tuple(models)
        self._prior = correctness_prior
        self._rep_prior = reputation_prior
        self._lambda = false_alarm_penalty
        self._preds: dict[str, dict[str, tuple[int, dict[str, Any]]]] = defaultdict(dict)
        self._sigma: dict[str, tuple[float, float]] = {}  # uid -> (σ_F, mean |F̄|)
        self._phase: dict[str, str] = {}
        self._sigma_edges: np.ndarray | None = None
        self._fbar_edges: np.ndarray | None = None
        self._tau_floor: float | None = None
        self._tau: float | None = None
        self._tau_round: int | None = None
        # ledger (private): built exclusively from oracle_result events
        self._ledger: dict[tuple[str, tuple[int, int]], BetaCounter] = {}
        self._counts: dict[tuple[str, tuple[int, int]], int] = defaultdict(int)
        self._global: dict[str, BetaCounter] = {m: BetaCounter(*self._prior) for m in self._models}
        self.labelled: dict[str, int] = {}  # uid -> id of the oracle_result that revealed it
        self._labelled_points: list[tuple[float, bool]] = []
        self._reputation: dict[str, BetaCounter] = {}
        self._verdicts: dict[str, Verdict] = {}

    # --- public read-only views (used by tests and scoring) -------------------------------
    def reputation(self, agent: str) -> float:
        lo, hi = config.REPUTATION_CLIP
        c = self._reputation.get(agent, BetaCounter(*self._rep_prior))
        return float(np.clip(c.mean, lo, hi))

    def p_correct(self, model: str, bucket: tuple[int, int]) -> float:
        return self._ledger.get((model, bucket), BetaCounter(*self._prior)).mean

    @property
    def tau(self) -> float | None:
        return self._tau

    def export_snapshot(self, wrong_thresholds: tuple[float, float] = (config.WRONG_F, config.WRONG_E),
                        budget: tuple[int, int] | None = None) -> dict[str, Any]:
        """JSON-serialisable final state for the trust service (``results/trust_snapshot.json``)."""
        if self._sigma_edges is None or self._fbar_edges is None or self._tau is None:
            raise RuntimeError("Calibrator has not seen any system yet; nothing to export")

        def grid(value: Any) -> list[list[Any]]:
            return [[value((i, j)) for j in range(config.FBAR_BINS)]
                    for i in range(config.SIGMA_BINS)]

        return {
            "models": list(self._models),
            "tau": float(self._tau),
            "sigma_edges": [float(x) for x in self._sigma_edges],
            "fbar_edges": [float(x) for x in self._fbar_edges],
            "p_correct": {m: grid(lambda b, m=m: float(self.p_correct(m, b))) for m in self._models},
            "counts": {m: grid(lambda b, m=m: int(self._counts[(m, b)])) for m in self._models},
            "reputation": {a: self.reputation(a) for a in sorted(self._reputation)},
            "wrong_thresholds": {"force_rmse": float(wrong_thresholds[0]),
                                 "rel_e": float(wrong_thresholds[1])},
            "budget": {"per_round": int(budget[0]) if budget else None,
                       "rounds": int(budget[1]) if budget else None},
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }

    # --- main loop -------------------------------------------------------------------------
    def handle(self, bb: Blackboard, events: list[Event]) -> None:
        dirty: set[str] = set()
        ledger_changed = False
        reputations_changed = False
        for e in events:
            if e.kind == PREDICTION:
                p = e.payload
                if p["model"] in self._models:
                    self._preds[p["uid"]][p["model"]] = (e.id, p)
                    self._phase[p["uid"]] = p["phase"]
                    if len(self._preds[p["uid"]]) == len(self._models):
                        self._sigma[p["uid"]] = self._measure(p["uid"])
                        dirty.add(p["uid"])
        self._ensure_bucket_edges()
        for e in events:
            if e.kind == ORACLE_RESULT:
                reputations_changed |= self._settle(e)
                ledger_changed = True
        self._update_threshold(bb)
        targets = (set(self._sigma) - set(self.labelled)) if ledger_changed else dirty - set(self.labelled)
        for uid in sorted(targets):
            self._maybe_post_verdict(bb, uid)
        if reputations_changed:
            self.post(bb, REPUTATION, {"reputation": {a: self.reputation(a)
                                                      for a in sorted(self._reputation)}})

    # --- features / buckets -------------------------------------------------------------------
    def _measure(self, uid: str) -> tuple[float, float]:
        forces = np.stack([np.asarray(self._preds[uid][m][1]["forces"]) for m in self._models])
        return force_spread(forces)[0], mean_abs_force(forces.mean(axis=0))

    def _ensure_bucket_edges(self) -> None:
        if self._sigma_edges is not None or not self._sigma:
            return
        sig = np.array([s for s, _ in self._sigma.values()])
        fbar = np.array([f for _, f in self._sigma.values()])
        self._sigma_edges = np.quantile(sig, np.linspace(0, 1, config.SIGMA_BINS + 1)[1:-1])
        self._fbar_edges = np.quantile(fbar, np.linspace(0, 1, config.FBAR_BINS + 1)[1:-1])
        self._tau_floor = float(np.quantile(sig, config.TAU_FLOOR_QUANTILE))

    def bucket(self, uid: str) -> tuple[int, int]:
        assert self._sigma_edges is not None and self._fbar_edges is not None
        s, f = self._sigma[uid]
        return int(np.searchsorted(self._sigma_edges, s, side="right")), \
            int(np.searchsorted(self._fbar_edges, f, side="right"))

    # --- ledger + settlement ------------------------------------------------------------------
    def _settle(self, e: Event) -> bool:
        p = e.payload
        uid = p["uid"]
        if uid in self.labelled:
            return False
        self.labelled[uid] = e.id
        verdict = self._verdicts.get(uid)
        served = verdict.recommended_model if verdict else self._best_global()
        served_err = p["errors"].get(served) or p["consensus_error"]
        served_wrong = bool(served_err["wrong"]) if served_err else False
        if uid in self._sigma:
            b = self.bucket(uid)
            for m in self._models:
                if m in p["errors"]:
                    correct = not p["errors"][m]["wrong"]
                    c = self._ledger.setdefault((m, b), BetaCounter(*self._prior))
                    c.alpha += correct
                    c.beta += not correct
                    self._counts[(m, b)] += 1
                    self._global[m].alpha += correct
                    self._global[m].beta += not correct
            if self._phase.get(uid) == TRIAGE:
                self._labelled_points.append((self._sigma[uid][0], served_wrong))
        total = sum(p["bidders"].values())
        for agent, value in p["bidders"].items():
            share = value / total if total > 0 else 0.0
            c = self._reputation.setdefault(agent, BetaCounter(*self._rep_prior))
            if served_wrong:
                c.alpha += share
            else:
                c.beta += self._lambda * share
        return bool(p["bidders"])

    # --- threshold ------------------------------------------------------------------------------
    def _update_threshold(self, bb: Blackboard) -> None:
        if self._tau_floor is None:
            return
        tau = self._tau_floor
        if len(self._labelled_points) >= config.TAU_MIN_LABELLED:
            sig = np.array([s for s, _ in self._labelled_points])
            wrong = np.array([w for _, w in self._labelled_points])
            t = isotonic_threshold(sig, wrong, config.TAU_TARGET_P_WRONG)
            if t is not None:
                tau = max(tau, t)
        if tau != self._tau or self._tau_round != bb.round:
            self._tau, self._tau_round = tau, bb.round
            self.post(bb, THRESHOLD, {"tau": tau, "floor": self._tau_floor,
                                      "n_labelled": len(self._labelled_points)})

    # --- verdicts -------------------------------------------------------------------------------
    def _best_global(self) -> str:
        return max(self._models, key=lambda m: (self._global[m].mean, -self._models.index(m)))

    def recommend(self, uid: str) -> Verdict:
        b = self.bucket(uid)
        rec = max(self._models, key=lambda m: (self.p_correct(m, b), self._global[m].mean,
                                               -self._models.index(m)))
        return Verdict(rec, self.p_correct(rec, b), self._preds[uid][rec][0], b)

    def _maybe_post_verdict(self, bb: Blackboard, uid: str) -> None:
        v = self.recommend(uid)
        old = self._verdicts.get(uid)
        if old is not None and old.recommended_model == v.recommended_model \
                and old.prediction_event == v.prediction_event \
                and abs(old.trust - v.trust) < config.VERDICT_REPOST_DELTA:
            return
        self._verdicts[uid] = v
        sigma, fbar = self._sigma[uid]
        self.post(bb, VERDICT, {"uid": uid, "phase": self._phase[uid], "trust": v.trust,
                                "p_wrong": 1.0 - v.trust, "recommended_model": v.recommended_model,
                                "prediction_event": v.prediction_event, "bucket": list(v.bucket),
                                "sigma_F": sigma, "mean_abs_fbar": fbar})
