"""Calibrator: keeps the ledger; posts ``threshold``, ``verdict`` and ``reputation`` (DESIGN.md §4).

* Candidates: every non-empty subset mean of the models (singletons by model name, pairs as
  ``mean(a+b)``, the full set as ``consensus``; :mod:`trustswarm.candidates`), each with its
  own per-bucket Beta ledger and the same prior. Oracle results carry every candidate's error.
  Exploration audits (``audit`` oracle results) update the ledger like any label but do not
  settle reputations: an audit is not a claim that the system is wrong.
* Ledger: per (candidate, context bucket) Beta posterior of correctness from oracle results.
  The bucket is label-free: (σ_F quantile bin [4], mean |F̄| bin [3]); bin edges are fixed
  from the σ distribution the Calibrator sees in round 0 (see :func:`bucket_of`).
* Verdict: ``recommended = argmax_c P(c correct | bucket)`` over candidates (ties: global
  posterior, then consensus first, then model order — with no evidence the ensemble mean is
  served); ``trust = P(served answer correct | bucket)``; ``risk`` = isotonic
  P(served wrong | σ_F) fitted on the oracle-labelled points (continuous; ``None`` until
  ``RISK_MIN_LABELLED`` labels exist).
* Threshold: smallest τ with ledger-estimated P(served wrong | σ ≥ τ) ≥ 0.5 (isotonic fit on
  labelled points), floored at the 70th percentile of round-0 σ. If no τ reaches 0.5 the
  floor is kept (the hook never flags *less* than the floor).
* Settlement: a winning bid on a system whose served answer is revealed wrong earns
  ``+bid share``; a false alarm costs ``λ · bid share``. Reputation = Beta mean, clipped.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, ClassVar

import numpy as np
from sklearn.isotonic import IsotonicRegression

from trustswarm import config
from trustswarm.agents.base import Agent, batch_payload, iter_rows
from trustswarm.agents.stream import TRIAGE
from trustswarm.blackboard import (ORACLE_RESULT, PREDICTION, REPUTATION, THRESHOLD, VERDICT,
                                   Blackboard, Event)
from trustswarm.candidates import CONSENSUS, candidate_members
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
    recommended: str
    trust: float
    prediction_events: dict[str, tuple[int, int]]
    bucket: tuple[int, int]
    risk: float | None


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


def isotonic_probability(x: np.ndarray, wrong: np.ndarray) -> IsotonicRegression:
    """Monotone (non-decreasing) P(wrong | x), clipped to [0, 1] and to the observed range."""
    return IsotonicRegression(increasing=True, out_of_bounds="clip", y_min=0.0,
                              y_max=1.0).fit(np.asarray(x, float), np.asarray(wrong, float))


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
        self.members = candidate_members(self._models)  # raises on reserved/duplicate names
        self.candidates: tuple[str, ...] = tuple(self.members)
        # prior ties: the full ensemble mean first, then larger subsets, then single models
        self._tiebreak = {c: i for i, c in enumerate(
            sorted(self.candidates, key=lambda c: (-len(self.members[c]), self.candidates.index(c))))}
        self._prior = correctness_prior
        self._rep_prior = reputation_prior
        self._lambda = false_alarm_penalty
        # uid -> model -> ((prediction event id, row index), row)
        self._preds: dict[str, dict[str, tuple[tuple[int, int], dict[str, Any]]]] = defaultdict(dict)
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
        self._global = {c: BetaCounter(*self._prior) for c in self.candidates}
        self.labelled: dict[str, int] = {}  # uid -> id of the oracle_result that revealed it
        self._labelled_points: list[tuple[float, bool]] = []
        self._risk_fit: IsotonicRegression | None = None
        self._reputation: dict[str, BetaCounter] = {}
        self._verdicts: dict[str, Verdict] = {}

    # --- public read-only views (used by tests, scoring and the service snapshot) ------
    @property
    def models(self) -> tuple[str, ...]:
        return self._models

    @property
    def tau(self) -> float | None:
        return self._tau

    def reputation(self, agent: str) -> float:
        lo, hi = config.REPUTATION_CLIP
        c = self._reputation.get(agent, BetaCounter(*self._rep_prior))
        return float(np.clip(c.mean, lo, hi))

    def p_correct(self, candidate: str, bucket: tuple[int, int]) -> float:
        return self._ledger.get((candidate, bucket), BetaCounter(*self._prior)).mean

    def risk(self, sigma_F: float) -> float | None:
        """Isotonic P(served answer wrong | σ_F) from oracle labels; None if too few labels."""
        if self._risk_fit is None:
            return None
        # == IsotonicRegression.predict with out_of_bounds="clip", without per-call validation
        return float(np.interp(sigma_F, self._risk_fit.X_thresholds_, self._risk_fit.y_thresholds_))

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
            "candidates": list(self.candidates),
            "candidate_members": {c: list(m) for c, m in self.members.items()},
            # recommend() ranks by (p_correct[c][bucket], p_global[c], -index in tiebreak_order)
            "p_global": {c: float(self._global[c].mean) for c in self.candidates},
            "tiebreak_order": sorted(self.candidates, key=self._tiebreak.__getitem__),
            "tau": float(self._tau),
            "sigma_edges": [float(x) for x in self._sigma_edges],
            "fbar_edges": [float(x) for x in self._fbar_edges],
            "p_correct": {c: grid(lambda b, c=c: float(self.p_correct(c, b))) for c in self.candidates},
            "counts": {c: grid(lambda b, c=c: int(self._counts[(c, b)])) for c in self.candidates},
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
            if e.kind == PREDICTION and e.payload["model"] in self._models:
                for i, row in iter_rows(e.payload, ("uid", "phase", "forces")):
                    uid = row["uid"]
                    self._preds[uid][e.payload["model"]] = ((e.id, i), row)
                    self._phase[uid] = row["phase"]
                    if len(self._preds[uid]) == len(self._models):
                        self._sigma[uid] = self._measure(uid)
                        dirty.add(uid)
        self._ensure_bucket_edges()
        for e in events:
            if e.kind == ORACLE_RESULT:
                reputations_changed |= self._settle(e)
                ledger_changed = True
        if ledger_changed:
            self._refit_risk()
        self._update_threshold(bb)
        targets = (set(self._sigma) if ledger_changed else dirty) - set(self.labelled)
        rows = [r for r in (self._verdict_row(uid) for uid in sorted(targets)) if r is not None]
        if rows:  # one verdict batch per step
            self.post(bb, VERDICT, batch_payload(rows))
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
        return bucket_of(s, f, self._sigma_edges, self._fbar_edges)

    # --- ledger + settlement ------------------------------------------------------------------
    @staticmethod
    def _candidate_error(payload: dict[str, Any], candidate: str) -> dict[str, Any] | None:
        errors = payload.get("candidate_errors")
        if errors is not None:
            return errors.get(candidate)
        if candidate == CONSENSUS:
            return payload.get("consensus_error")
        return payload["errors"].get(candidate)

    def _settle(self, e: Event) -> bool:
        p = e.payload
        uid = p["uid"]
        if uid in self.labelled:
            return False
        self.labelled[uid] = e.id
        verdict = self._verdicts.get(uid)
        served = verdict.recommended if verdict else self._best_global()
        served_err = self._candidate_error(p, served) or p.get("consensus_error")
        served_wrong = bool(served_err["wrong"]) if served_err else False
        if uid in self._sigma:
            b = self.bucket(uid)
            for c in self.candidates:
                err = self._candidate_error(p, c)
                if err is None:
                    continue
                correct = not err["wrong"]
                counter = self._ledger.setdefault((c, b), BetaCounter(*self._prior))
                counter.alpha += correct
                counter.beta += not correct
                self._counts[(c, b)] += 1
                self._global[c].alpha += correct
                self._global[c].beta += not correct
            if self._phase.get(uid) == TRIAGE:
                self._labelled_points.append((self._sigma[uid][0], served_wrong))
        if p.get("audit"):
            return False  # exploration: no reputation at stake
        total = sum(p["bidders"].values())
        for agent, value in p["bidders"].items():
            share = value / total if total > 0 else 0.0
            c = self._reputation.setdefault(agent, BetaCounter(*self._rep_prior))
            if served_wrong:
                c.alpha += share
            else:
                c.beta += self._lambda * share
        return bool(p["bidders"])

    def _refit_risk(self) -> None:
        if len(self._labelled_points) < config.RISK_MIN_LABELLED:
            self._risk_fit = None
            return
        sig = np.array([s for s, _ in self._labelled_points])
        wrong = np.array([w for _, w in self._labelled_points])
        self._risk_fit = isotonic_probability(sig, wrong)

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
        return max(self.candidates, key=lambda c: (self._global[c].mean, -self._tiebreak[c]))

    def recommend(self, uid: str) -> Verdict:
        b = self.bucket(uid)
        rec = max(self.candidates, key=lambda c: (self.p_correct(c, b), self._global[c].mean,
                                                  -self._tiebreak[c]))
        served = {m: self._preds[uid][m][0] for m in self.members[rec]}
        return Verdict(rec, self.p_correct(rec, b), served, b, self.risk(self._sigma[uid][0]))

    def _verdict_row(self, uid: str) -> dict[str, Any] | None:
        """The verdict for ``uid`` if it changed materially since the last one, else None."""
        v = self.recommend(uid)
        old = self._verdicts.get(uid)
        if old is not None and old.recommended == v.recommended \
                and old.prediction_events == v.prediction_events \
                and abs(old.trust - v.trust) < config.VERDICT_REPOST_DELTA \
                and (old.risk is None) == (v.risk is None) \
                and abs((old.risk or 0.0) - (v.risk or 0.0)) < config.VERDICT_REPOST_DELTA:
            return None
        self._verdicts[uid] = v
        sigma, fbar = self._sigma[uid]
        return {"uid": uid, "phase": self._phase[uid], "trust": v.trust, "risk": v.risk,
                "recommended_model": v.recommended,
                "prediction_events": {m: list(ref) for m, ref in v.prediction_events.items()},
                "bucket": list(v.bucket), "sigma_F": sigma, "mean_abs_fbar": fbar}
