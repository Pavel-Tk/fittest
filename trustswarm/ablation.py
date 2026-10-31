"""Ablation: arms A-D at the same oracle budget K = rounds × k (DESIGN.md §5, §10).

=================  ===========================  ====================  ================================
Arm                Models                       Triage score          Served answer
=================  ===========================  ====================  ================================
A_single           aimnet2                      max |F|               that model
A_random           aimnet2                      uniform random        that model
B_seed             aimnet2_m0..3                seed σ_F              ensemble mean
B_mace_seeds       mace_seed0..3 (optional)     seed σ_F              ensemble mean
C_collective       mace_mpa0, aimnet2, orb      σ_F                   ensemble mean
D_trustswarm       mace_mpa0, aimnet2, orb      σ_F + reputation bids Calibrator's recommended model
=================  ===========================  ====================  ================================

All arms see the same stream: pool reactions arrive in ``rounds`` chunks, the planted
cases (and their reactants) are injected at round 0, and each round at most ``k`` systems
are sent to the oracle. Arms A-C pick top-k by score among arrived, unchecked systems;
arm D runs the agent collective on a Blackboard and the Oracle pays the top-k summed bids.

Metrics (pool = triage, eval = held-out generalisation), each with a 95% bootstrap CI over
reactions (paired across arms: the same resampled reactions are used for every arm):

1. ``auroc_pool``: AUROC(triage score → served answer wrong) on pool; ``auroc_common``: same
   score vs a common, arm-independent target (MACE-MPA-0 wrong); ``auroc_eval``: AUROC on
   eval (arm D uses the Calibrator's P(served wrong) there); ``spearman_*``: threshold-free
   Spearman(score, served force RMSE).
2. ``recall_at_k``: fraction of wrong served pool answers that received an oracle call.
3. ``planted_caught``: fraction of planted cases flagged, vetoed or oracle-checked.
4. ``eval_force_rmse_{pre,post}``, ``eval_rel_e_mae_{pre,post}``: served-answer error on eval,
   before/after the Curator's fine-tune (A-C: no fine-tune, pre = post).
5. ``finetune``: fine-tuned model vs base model error on eval (if a fine-tune shipped).

Failures — skipped arms, undefined metrics, unshipped fine-tunes, unspent budget, and
comparisons where TrustSwarm does *not* beat a baseline — are collected in
``results["failures"]``; nothing is dropped silently.
"""

from __future__ import annotations

import dataclasses
import json
import zlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from trustswarm import blackboard as bbk
from trustswarm import config
from trustswarm.agents import (Calibrator, Curator, DisagreementMonitor, FinetuneFn, ModelAgent,
                               NullFinetune, Oracle, PoolStream, ReplayFinetune, Skeptic)
from trustswarm.agents.stream import TRIAGE
from trustswarm.blackboard import Blackboard
from trustswarm.cache import Predictions, load_models, load_predictions, prediction_path
from trustswarm.data import (EVAL, PLANTED, POOL, SKEPTIC_REF, Frame, arrival_rounds, by_split,
                             reactant_index, strip_all)
from trustswarm.round_driver import RoundDriver
from trustswarm.signals import force_rmse, force_spread, max_abs_force

Served = dict[str, tuple[np.ndarray, float | None]]  # uid -> (forces, relative energy)


@dataclass(frozen=True)
class AblationConfig:
    budget_per_round: int = config.BUDGET_PER_ROUND
    rounds: int = config.N_ROUNDS
    seed: int = 0
    n_bootstrap: int = config.N_BOOTSTRAP
    single_model: str = config.SINGLE_MODEL
    seed_models: tuple[str, ...] = config.SEED_MODELS
    mace_seed_models: tuple[str, ...] | None = config.MACE_SEED_MODELS
    hetero_models: tuple[str, ...] = config.HETERO_MODELS
    common_target: str = config.COMMON_TARGET_MODEL
    curated_model: str = config.CURATED_MODEL
    curated_base_model: str = config.CURATED_BASE_MODEL
    wrong_f: float = config.WRONG_F
    wrong_e: float = config.WRONG_E
    wrong_thresholds: tuple[tuple[float, float], ...] = config.WRONG_THRESHOLDS
    flag_quantile: float = config.TAU_FLOOR_QUANTILE
    max_planted: int | None = 40
    """Cap on planted cases (seeded subsample of the filtered set); None = all."""

    @property
    def total_budget(self) -> int:
        return self.budget_per_round * self.rounds


# --- data ---------------------------------------------------------------------------------------

@dataclass
class AblationData:
    pool: list[Frame]
    eval: list[Frame]
    refs: list[Frame]
    planted_cases: list[Frame]
    planted_context: list[Frame]
    by_uid: dict[str, Frame]
    rel_true: dict[str, float | None]
    reactant_uid: dict[str, str]

    @property
    def injected(self) -> list[Frame]:
        return [*self.planted_cases, *self.planted_context]

    @property
    def candidates(self) -> list[Frame]:
        """Everything the oracle may be asked about: pool + injected planted structures."""
        return [*self.pool, *self.injected]


def _model_rel(p: Predictions, uid: str, reactant_uid: str | None) -> float | None:
    if reactant_uid is None or reactant_uid not in p or uid not in p:
        return None
    return p.energy[uid] - p.energy[reactant_uid]


def select_planted(planted_split: Sequence[Frame], target: Predictions, cfg: AblationConfig
                   ) -> tuple[list[Frame], list[Frame]]:
    """Planted cases = frames of the ``planted`` split on which the common-target model is
    wrong (known failures); context = the reactant frames of those reactions."""
    reactants = reactant_index(planted_split)
    cases = []
    for f in planted_split:
        r = reactants.get(f.reaction)
        if f.F is None or f.uid not in target or f.role == "reactant":
            continue
        rel_true = None if r is None or r.E is None or f.E is None else f.E - r.E
        err = force_rmse(target.forces[f.uid], f.F)
        rel = _model_rel(target, f.uid, r.uid if r else None)
        rel_err = None if rel is None or rel_true is None else abs(rel - rel_true)
        if err > cfg.wrong_f or (rel_err is not None and rel_err > cfg.wrong_e):
            cases.append(f)
    cases.sort(key=lambda f: f.uid)
    if cfg.max_planted is not None and len(cases) > cfg.max_planted:
        rng = np.random.default_rng([cfg.seed, 17])
        keep = np.sort(rng.choice(len(cases), size=cfg.max_planted, replace=False))
        cases = [cases[i] for i in keep]
    context = sorted({reactants[f.reaction].uid: reactants[f.reaction] for f in cases
                      if f.reaction in reactants}.values(), key=lambda f: f.uid)
    return cases, context


def prepare_data(frames: Sequence[Frame], target: Predictions, cfg: AblationConfig) -> AblationData:
    splits = by_split(frames)
    pool, eval_, refs = (sorted(splits.get(s, []), key=lambda f: f.uid) for s in (POOL, EVAL, SKEPTIC_REF))
    if not pool or not eval_:
        raise ValueError(f"need non-empty pool and eval splits, got {sorted(splits)}")
    cases, context = select_planted(splits.get(PLANTED, []), target, cfg)
    reactants = reactant_index(frames)
    rel_true: dict[str, float | None] = {}
    for f in frames:
        r = reactants.get(f.reaction)
        rel_true[f.uid] = None if r is None or r.E is None or f.E is None else f.E - r.E
    return AblationData(pool=pool, eval=eval_, refs=refs, planted_cases=cases,
                        planted_context=context, by_uid={f.uid: f for f in frames},
                        rel_true=rel_true, reactant_uid={k: v.uid for k, v in reactants.items()})


# --- statistics -------------------------------------------------------------------------------

class Bootstrap:
    """Reaction-level bootstrap. Resamples depend only on (seed, set of reactions), so every
    metric on the same split — in every arm, before and after fine-tuning — sees the same
    resampled reactions, and all differences are paired."""

    def __init__(self, n: int, seed: int) -> None:
        self.n, self.seed = n, seed

    def samples(self, stat: Callable[[np.ndarray], float | None],
                groups: Sequence[str]) -> np.ndarray:
        uniq, inv = np.unique(np.asarray(groups), return_inverse=True)
        members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
        rng = np.random.default_rng([self.seed, zlib.crc32("\n".join(uniq).encode())])
        out = np.full(self.n, np.nan)
        for b in range(self.n):
            idx = np.concatenate([members[g] for g in rng.integers(len(uniq), size=len(uniq))])
            v = stat(idx)
            out[b] = np.nan if v is None else v
        return out


def _ci(samples: np.ndarray, level: float = config.CI_LEVEL) -> list[float] | None:
    ok = samples[np.isfinite(samples)]
    if ok.size < 0.5 * samples.size or ok.size == 0:
        return None
    lo, hi = np.percentile(ok, [50 * (1 - level), 50 * (1 + level)])
    return [float(lo), float(hi)]


def _auroc(y: np.ndarray, s: np.ndarray) -> float | None:
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else None


def _spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    return float(spearmanr(a, b).statistic)


def _mean(x: np.ndarray) -> float | None:
    x = x[np.isfinite(x)]
    return float(x.mean()) if x.size else None


# --- arm outcomes --------------------------------------------------------------------------------

@dataclass
class ArmOutcome:
    name: str
    models: list[str]
    score: str
    served: str
    pool_score: dict[str, float]
    eval_score: dict[str, float]
    served_pool: Served
    served_eval_pre: Served
    served_eval_post: Served
    checked: dict[str, int]
    flagged: set[str]
    vetoed: set[str] = field(default_factory=set)
    oracle_calls_per_round: dict[int, int] = field(default_factory=dict)
    finetune: dict[str, Any] | None = None
    notes: list[str] = field(default_factory=list)


def topk_stream(scores: Mapping[str, float], arrival: Mapping[str, int], k: int,
                rounds: int) -> dict[str, int]:
    """Each round, check the k highest-scoring arrived, unchecked systems (ties by uid)."""
    checked: dict[str, int] = {}
    for r in range(rounds):
        cand = [u for u in scores if arrival[u] <= r and u not in checked]
        for u in sorted(cand, key=lambda u: (-scores[u], u))[:k]:
            checked[u] = r
    return checked


def _flag_by_round0_quantile(scores: Mapping[str, float], arrival: Mapping[str, int],
                             q: float) -> set[str]:
    """Baseline "flag": score ≥ q-quantile of round-0 scores (mirror of the hook's τ floor)."""
    r0 = [s for u, s in scores.items() if arrival[u] == 0]
    tau = float(np.quantile(r0, q)) if r0 else np.inf
    return {u for u, s in scores.items() if s >= tau}


def _ensemble_served(preds: Mapping[str, Predictions], uids: Sequence[str],
                     data: AblationData) -> Served:
    out: Served = {}
    for u in uids:
        r_uid = data.reactant_uid.get(data.by_uid[u].reaction)
        forces = np.mean([p.forces[u] for p in preds.values()], axis=0)
        rels = [_model_rel(p, u, r_uid) for p in preds.values()]
        out[u] = (forces, float(np.mean(rels)) if all(x is not None for x in rels) else None)
    return out


def _check_coverage(preds: Mapping[str, Predictions], uids: Sequence[str]) -> str | None:
    for m, p in preds.items():
        missing = [u for u in uids if u not in p]
        if missing:
            return f"{m} has no prediction for {len(missing)} required frames (e.g. {missing[0]})"
    return None


def run_baseline_arm(name: str, preds: Mapping[str, Predictions], score_name: str,
                     data: AblationData, arrival: Mapping[str, int], cfg: AblationConfig
                     ) -> ArmOutcome:
    models = list(preds)
    cand = [f.uid for f in data.candidates]
    eval_uids = [f.uid for f in data.eval]
    if score_name == "max|F|":
        (p,) = preds.values()
        score = {u: max_abs_force(p.forces[u]) for u in [*cand, *eval_uids]}
    elif score_name == "random":
        rng = np.random.default_rng([cfg.seed, zlib.crc32(name.encode())])
        allu = [*cand, *eval_uids]
        score = dict(zip(allu, rng.random(len(allu)).tolist()))
    elif score_name == "sigma_F":
        score = {u: force_spread(np.stack([p.forces[u] for p in preds.values()]))[0]
                 for u in [*cand, *eval_uids]}
    else:
        raise ValueError(score_name)
    cand_scores = {u: score[u] for u in cand}
    checked = topk_stream(cand_scores, arrival, cfg.budget_per_round, cfg.rounds)
    served_eval = _ensemble_served(preds, eval_uids, data)
    per_round = {r: sum(1 for v in checked.values() if v == r) for r in range(cfg.rounds)}
    return ArmOutcome(
        name=name, models=models, score=score_name,
        served=models[0] if len(models) == 1 else "ensemble mean",
        pool_score=cand_scores, eval_score={u: score[u] for u in eval_uids},
        served_pool=_ensemble_served(preds, cand, data), served_eval_pre=served_eval,
        served_eval_post=served_eval, checked=checked,
        flagged=_flag_by_round0_quantile(cand_scores, arrival, cfg.flag_quantile),
        oracle_calls_per_round=per_round)


def build_trustswarm(data: AblationData, preds: Mapping[str, Predictions],
                     arrival: Mapping[str, int], cfg: AblationConfig, finetune_fn: FinetuneFn,
                     bb: Blackboard) -> tuple[RoundDriver, dict[str, Any]]:
    """Wire arm D's agents. Each gets only its private data; everything else is on ``bb``."""
    models = list(cfg.hetero_models)
    streamed = [f.uid for f in [*data.candidates, *data.eval]]
    agents: dict[str, Any] = {
        "oracle": Oracle(data.candidates, cfg.budget_per_round, cfg.rounds),
        "stream": PoolStream(strip_all(data.pool), arrival, strip_all(data.eval),
                             serve_round=cfg.rounds),
        "skeptic": Skeptic(data.refs, {m: preds[m].subset(f.uid for f in data.refs) for m in models},
                           data.planted_cases, data.planted_context),
        **{m: ModelAgent(preds[m].subset(streamed)) for m in models},
        "calibrator": Calibrator(models),
        "hook": DisagreementMonitor(models),
        "curator": Curator(models, finetune_fn, repair_round=cfg.rounds + 1),
    }
    return RoundDriver(bb, list(agents.values())), agents


def run_trustswarm_arm(data: AblationData, preds: Mapping[str, Predictions],
                       arrival: Mapping[str, int], cfg: AblationConfig, finetune_fn: FinetuneFn,
                       blackboard_path: str | Path | None = None
                       ) -> tuple[ArmOutcome, Blackboard, dict[str, Any]]:
    with Blackboard(blackboard_path) as bb:
        driver, agents = build_trustswarm(data, preds, arrival, cfg, finetune_fn, bb)
        driver.run(cfg.rounds + 2)  # triage rounds, settle + serve eval, repair + re-predict
    events = bb.read()
    pred_by_id = {e.id: e.payload for e in events if e.kind == bbk.PREDICTION}
    update_ids = [e.id for e in events if e.kind == bbk.MODEL_UPDATE]
    cutoff = update_ids[0] if update_ids else len(events)

    def served_from(verdicts: list[bbk.Event]) -> Served:
        last: dict[str, bbk.Event] = {}
        for e in verdicts:
            last[e.payload["uid"]] = e
        out: Served = {}
        for u, e in last.items():
            p = pred_by_id[e.payload["prediction_event"]]
            out[u] = (np.asarray(p["forces"], dtype=float), p["rel_energy"])
        return out

    verdicts = [e for e in events if e.kind == bbk.VERDICT]
    pre = [e for e in verdicts if e.id < cutoff]
    served_pool = served_from([e for e in pre if e.payload["phase"] == TRIAGE])
    p_wrong_eval: dict[str, float] = {}
    for e in pre:
        if e.payload["phase"] != TRIAGE:
            p_wrong_eval[e.payload["uid"]] = float(e.payload["p_wrong"])
    sigma: dict[str, float] = {}
    flagged: set[str] = set()
    for e in events:
        if e.kind in bbk.MEASUREMENT_KINDS and e.id < cutoff:
            sigma.setdefault(e.payload["uid"], float(e.payload["sigma_F"]))
            if e.kind == bbk.FLAG and e.payload["phase"] == TRIAGE:
                flagged.add(e.payload["uid"])
    oracle: Oracle = agents["oracle"]
    curator: Curator = agents["curator"]
    notes = []
    missing = sorted({u for m in cfg.hetero_models for u in agents[m].missing})
    if missing:
        notes.append(f"{len(missing)} streamed systems lacked a prediction from some model")
    if oracle.rejected_bids:
        notes.append(f"oracle rejected {oracle.rejected_bids} bids on systems it cannot label")
    cand = [f.uid for f in data.candidates]
    outcome = ArmOutcome(
        name="D_trustswarm", models=list(cfg.hetero_models), score="sigma_F (+ reputation bids)",
        served="Calibrator recommended model",
        pool_score={u: sigma[u] for u in cand if u in sigma},
        eval_score=p_wrong_eval, served_pool=served_pool,
        served_eval_pre=served_from([e for e in pre if e.payload["phase"] != TRIAGE]),
        served_eval_post=served_from([e for e in verdicts if e.payload["phase"] != TRIAGE]),
        checked={e.payload["uid"]: int(e.payload["bid_round"]) for e in events
                 if e.kind == bbk.ORACLE_RESULT},
        flagged=flagged, vetoed={e.payload["uid"] for e in events if e.kind == bbk.VETO},
        oracle_calls_per_round=dict(oracle.spent_per_round),
        finetune=_finetune_summary(curator, events), notes=notes)
    return outcome, bb, agents


def _finetune_summary(curator: Curator, events: list[bbk.Event]) -> dict[str, Any]:
    reports = [e.payload for e in events if e.kind == bbk.FINETUNE_REPORT]
    if not reports:
        return {"shipped": False, "reason": "Curator never ran"}
    p = reports[-1]
    return {"model": p["model"], "shipped": p["shipped"], "n_labels": p.get("n_labels"),
            "ledger_force_rmse": p["ledger_force_rmse"], "report": p["report"],
            "new_predictions_path": p.get("new_predictions_path")}


# --- metrics ------------------------------------------------------------------------------------

@dataclass
class ErrorTable:
    uids: list[str]
    groups: np.ndarray
    force_rmse: np.ndarray
    rel_err: np.ndarray  # NaN where the relative energy is undefined

    def wrong(self, wf: float, we: float) -> np.ndarray:
        return (self.force_rmse > wf) | (np.nan_to_num(self.rel_err, nan=-np.inf) > we)


def error_table(served: Served, uids: Sequence[str], data: AblationData) -> ErrorTable:
    fr, re = [], []
    for u in uids:
        f = data.by_uid[u]
        assert f.F is not None
        F_hat, rel_hat = served[u]
        fr.append(force_rmse(F_hat, f.F))
        rt = data.rel_true[u]
        re.append(np.nan if rel_hat is None or rt is None else abs(rel_hat - rt))
    return ErrorTable(list(uids), np.array([data.by_uid[u].reaction for u in uids]),
                      np.array(fr), np.array(re))


def _metric(value: float | None, samples: np.ndarray | None, n: int) -> dict[str, Any]:
    return {"value": value, "ci": None if samples is None else _ci(samples), "n": n}


class ArmScorer:
    """Computes every metric for one arm; keeps bootstrap samples for paired comparisons."""

    def __init__(self, data: AblationData, cfg: AblationConfig, common: Predictions) -> None:
        self.data, self.cfg = data, cfg
        self.boot = Bootstrap(cfg.n_bootstrap, cfg.seed)
        self.pool_uids = [f.uid for f in data.pool]
        self.eval_uids = [f.uid for f in data.eval]
        common_served = {u: (common.forces[u], _model_rel(common, u, data.reactant_uid.get(
            data.by_uid[u].reaction))) for u in self.pool_uids}
        self.common = error_table(common_served, self.pool_uids, data)
        self.samples: dict[tuple[str, str], np.ndarray] = {}

    def _boot(self, arm: str, key: str, stat: Callable[[np.ndarray], float | None],
              groups: Sequence[str]) -> np.ndarray:
        s = self.boot.samples(stat, groups)
        self.samples[(arm, key)] = s
        return s

    def score(self, o: ArmOutcome) -> dict[str, Any]:
        cfg, d = self.cfg, self.data
        missing = [u for u in self.pool_uids if u not in o.served_pool or u not in o.pool_score]
        if missing:
            raise RuntimeError(f"{o.name}: no served answer/score for {len(missing)} pool frames")
        pool = error_table(o.served_pool, self.pool_uids, d)
        s_pool = np.array([o.pool_score[u] for u in self.pool_uids])
        y = pool.wrong(cfg.wrong_f, cfg.wrong_e)
        y_common = self.common.wrong(cfg.wrong_f, cfg.wrong_e)
        checked = np.array([u in o.checked for u in self.pool_uids])
        g = pool.groups
        m: dict[str, Any] = {}

        def add(key: str, stat: Callable[[np.ndarray], float | None], groups: np.ndarray,
                n: int) -> None:
            m[key] = _metric(stat(np.arange(len(groups))), self._boot(o.name, key, stat, groups), n)

        add("auroc_pool", lambda i: _auroc(y[i], s_pool[i]), g, len(y))
        add("auroc_common", lambda i: _auroc(y_common[i], s_pool[i]), g, len(y))
        add("spearman_pool", lambda i: _spearman(s_pool[i], pool.force_rmse[i]), g, len(y))
        add("recall_at_k", lambda i: float(checked[i][y[i]].mean()) if y[i].any() else None, g,
            int(y.sum()))
        add("served_wrong_rate_pool", lambda i: float(y[i].mean()), g, len(y))

        # held-out eval
        eval_uids = [u for u in self.eval_uids if u in o.served_eval_pre]
        if len(eval_uids) < len(self.eval_uids):
            o.notes.append(f"{len(self.eval_uids) - len(eval_uids)} eval frames had no served answer")
        pre = error_table(o.served_eval_pre, eval_uids, d)
        post = error_table(o.served_eval_post, eval_uids, d)
        s_eval = np.array([o.eval_score.get(u, np.nan) for u in eval_uids])
        y_eval = pre.wrong(cfg.wrong_f, cfg.wrong_e)
        ge = pre.groups
        add("auroc_eval", lambda i: _auroc(y_eval[i], s_eval[i]), ge, len(eval_uids))
        add("spearman_eval", lambda i: _spearman(s_eval[i], pre.force_rmse[i]), ge, len(eval_uids))
        for tag, t in (("pre", pre), ("post", post)):
            add(f"eval_force_rmse_{tag}", lambda i, t=t: _mean(t.force_rmse[i]), ge, len(eval_uids))
            add(f"eval_rel_e_mae_{tag}", lambda i, t=t: _mean(t.rel_err[i]), ge, len(eval_uids))
            add(f"eval_wrong_rate_{tag}",
                lambda i, t=t: float(t.wrong(cfg.wrong_f, cfg.wrong_e)[i].mean()), ge, len(eval_uids))

        # planted
        planted = d.planted_cases
        caught = np.array([f.uid in o.checked or f.uid in o.flagged or f.uid in o.vetoed
                           for f in planted])
        gp = np.array([f.reaction for f in planted])
        if len(planted):
            add("planted_caught", lambda i: float(caught[i].mean()), gp, len(planted))
        else:
            m["planted_caught"] = _metric(None, None, 0)
        m["planted_breakdown"] = {
            "oracle_checked": int(sum(f.uid in o.checked for f in planted)),
            "flagged": int(sum(f.uid in o.flagged for f in planted)),
            "vetoed": int(sum(f.uid in o.vetoed for f in planted)), "n": len(planted)}

        by_threshold = {}
        for wf, we in cfg.wrong_thresholds:
            yt, yc, ye = pool.wrong(wf, we), self.common.wrong(wf, we), pre.wrong(wf, we)
            by_threshold[f"F{wf:g}_E{we:g}"] = {
                "wrong_f": wf, "wrong_e": we, "auroc_pool": _auroc(yt, s_pool),
                "auroc_common": _auroc(yc, s_pool), "auroc_eval": _auroc(ye, s_eval),
                "recall_at_k": float(checked[yt].mean()) if yt.any() else None,
                "served_wrong_rate_pool": float(yt.mean())}
        return {"metrics": m, "by_threshold": by_threshold}

    def finetune_eval(self, base: Predictions, tuned: Predictions) -> dict[str, Any]:
        """Model-level error on eval: fine-tuned vs base (metric 5)."""
        d = self.data
        uids = [u for u in self.eval_uids if u in base and u in tuned]
        tabs = {}
        for tag, p in (("base", base), ("finetuned", tuned)):
            served = {u: (p.forces[u], _model_rel(p, u, d.reactant_uid.get(d.by_uid[u].reaction)))
                      for u in uids}
            tabs[tag] = error_table(served, uids, d)
        g = tabs["base"].groups
        out: dict[str, Any] = {"base_model": base.model, "finetuned_model": tuned.model}
        for tag, t in tabs.items():
            for key, stat in (("force_rmse", lambda i, t=t: _mean(t.force_rmse[i])),
                              ("rel_e_mae", lambda i, t=t: _mean(t.rel_err[i]))):
                s = self.boot.samples(stat, g)
                self.samples[("finetune", f"{tag}_{key}")] = s
                out[f"{tag}_{key}"] = _metric(stat(np.arange(len(uids))), s, len(uids))
        for key in ("force_rmse", "rel_e_mae"):
            diff = self.samples[("finetune", f"finetuned_{key}")] - self.samples[("finetune", f"base_{key}")]
            out[f"delta_{key}_ci"] = _ci(diff)
        return out

    def paired_difference(self, arm_a: str, arm_b: str, key: str) -> list[float] | None:
        a, b = self.samples.get((arm_a, key)), self.samples.get((arm_b, key))
        return None if a is None or b is None else _ci(a - b)


# --- top level ---------------------------------------------------------------------------------

HEADLINE: tuple[tuple[str, bool], ...] = (
    ("auroc_pool", True), ("auroc_eval", True), ("recall_at_k", True),
    ("planted_caught", True), ("eval_force_rmse_post", False))
"""(metric, higher_is_better) pairs compared between D and every other arm."""


def arm_specs(cfg: AblationConfig) -> list[tuple[str, tuple[str, ...], str]]:
    specs = [("A_single", (cfg.single_model,), "max|F|"),
             ("A_random", (cfg.single_model,), "random"),
             ("B_seed", cfg.seed_models, "sigma_F")]
    if cfg.mace_seed_models:
        specs.append(("B_mace_seeds", cfg.mace_seed_models, "sigma_F"))
    specs += [("C_collective", cfg.hetero_models, "sigma_F"),
              ("D_trustswarm", cfg.hetero_models, "trustswarm")]
    return specs


def default_finetune(preds_dir: str | Path, cfg: AblationConfig) -> ReplayFinetune:
    """Replay ``{curated_model}.npz`` for its base model, plus any ``{model}_lora_curated.npz``."""
    replay = {m: prediction_path(preds_dir, f"{m}_lora_curated") for m in cfg.hetero_models}
    replay[cfg.curated_base_model] = prediction_path(preds_dir, cfg.curated_model)
    return ReplayFinetune({m: p for m, p in replay.items() if p.exists()})


def run_ablation(frames: Sequence[Frame], preds_dir: str | Path, cfg: AblationConfig = AblationConfig(),
                 finetune_fn: FinetuneFn | None = None,
                 blackboard_path: str | Path | None = None,
                 snapshot_path: str | Path | None = None) -> dict[str, Any]:
    """Run every arm; return a JSON-serialisable results dict.

    Never raises for a missing model or a failing arm: the arm is reported as ``skipped`` /
    ``failed`` with the reason. ``blackboard_path`` receives arm D's JSONL event log and
    ``snapshot_path`` the final Calibrator state (:meth:`Calibrator.export_snapshot`)."""
    failures: list[str] = []
    results: dict[str, Any] = {
        "schema": "trustswarm.ablation/v1",
        "config": {**dataclasses.asdict(cfg), "total_budget": cfg.total_budget},
        "arms": {}, "failures": failures, "comparisons": {}}
    try:
        (common,) = load_models(preds_dir, [cfg.common_target]).values()
    except FileNotFoundError as exc:
        results["status"] = "failed"
        failures.append(f"cannot run ablation: common target unavailable ({exc})")
        return results
    data = prepare_data(frames, common, cfg)
    arrival = arrival_rounds(data.pool, cfg.rounds, cfg.seed)
    arrival.update({f.uid: 0 for f in data.injected})
    results["data"] = {
        "n_pool": len(data.pool), "n_eval": len(data.eval), "n_skeptic_ref": len(data.refs),
        "n_planted": len(data.planted_cases), "n_planted_context": len(data.planted_context),
        "n_pool_reactions": len({f.reaction for f in data.pool}),
        "n_eval_reactions": len({f.reaction for f in data.eval}),
        "pool_eval_shared_formulas": len({f.formula for f in data.pool} & {f.formula for f in data.eval}),
        "planted_filter": f"{cfg.common_target} wrong (F>{cfg.wrong_f}, E>{cfg.wrong_e})",
    }
    if not data.planted_cases:
        failures.append("no planted cases: the common-target model is never wrong on the planted split")
    scorer = ArmScorer(data, cfg, common)
    needed = [f.uid for f in [*data.candidates, *data.eval]]
    finetune_fn = finetune_fn or default_finetune(preds_dir, cfg)

    for name, models, score in arm_specs(cfg):
        entry: dict[str, Any] = {"models": list(models), "score": score}
        results["arms"][name] = entry
        try:
            preds = load_models(preds_dir, models)
        except FileNotFoundError as exc:
            entry.update(status="skipped", reason=str(exc))
            failures.append(f"{name} skipped: {exc}")
            continue
        gap = _check_coverage(preds, needed)
        if gap:
            entry.update(status="skipped", reason=gap)
            failures.append(f"{name} skipped: {gap}")
            continue
        try:
            _run_arm(name, score, preds, data, arrival, cfg, scorer, finetune_fn, entry,
                     failures, blackboard_path, snapshot_path)
        except Exception as exc:  # noqa: BLE001 — an arm failure is a result, not a crash
            entry.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
            failures.append(f"{name} failed: {type(exc).__name__}: {exc}")

    _compare(results, scorer, failures)
    return results


def _run_arm(name: str, score: str, preds: Mapping[str, Predictions], data: AblationData,
             arrival: Mapping[str, int], cfg: AblationConfig, scorer: ArmScorer,
             finetune_fn: FinetuneFn, entry: dict[str, Any], failures: list[str],
             blackboard_path: str | Path | None, snapshot_path: str | Path | None) -> None:
    if score == "trustswarm":
        outcome, bb, agents = run_trustswarm_arm(data, preds, arrival, cfg, finetune_fn,
                                                 blackboard_path)
        entry["blackboard"] = {"n_events": len(bb),
                               "path": None if blackboard_path is None else str(blackboard_path)}
        if snapshot_path is not None:
            snap = agents["calibrator"].export_snapshot((cfg.wrong_f, cfg.wrong_e),
                                                        (cfg.budget_per_round, cfg.rounds))
            Path(snapshot_path).parent.mkdir(parents=True, exist_ok=True)
            Path(snapshot_path).write_text(json.dumps(snap, indent=2), encoding="utf-8")
    else:
        outcome = run_baseline_arm(name, preds, score, data, arrival, cfg)
    entry.update(status="ok", served=outcome.served, score=outcome.score,
                 oracle_calls=len(outcome.checked),
                 oracle_calls_per_round={str(k): v for k, v in sorted(outcome.oracle_calls_per_round.items())},
                 **scorer.score(outcome))
    if outcome.finetune is not None:
        entry["finetune"] = outcome.finetune
        if outcome.finetune["shipped"]:
            tuned = load_predictions(outcome.finetune["new_predictions_path"])
            entry["finetune"]["eval"] = scorer.finetune_eval(preds[outcome.finetune["model"]], tuned)
        else:
            failures.append(f"{name}: fine-tune not shipped "
                            f"({outcome.finetune['report'].get('reason', 'no reason given')})")
    if len(outcome.checked) < cfg.total_budget:
        failures.append(f"{name}: used {len(outcome.checked)}/{cfg.total_budget} oracle calls")
    entry["notes"] = outcome.notes
    failures.extend(f"{name}: {n}" for n in outcome.notes)
    for key, val in entry["metrics"].items():
        if isinstance(val, dict) and "value" in val and val["value"] is None:
            failures.append(f"{name}: {key} undefined (single-class target or empty set)")


def _compare(results: dict[str, Any], scorer: ArmScorer, failures: list[str]) -> None:
    arms = results["arms"]
    if arms.get("D_trustswarm", {}).get("status") != "ok":
        failures.append("D_trustswarm did not run: no comparisons possible")
        return
    d = arms["D_trustswarm"]["metrics"]
    for name, entry in arms.items():
        if name == "D_trustswarm" or entry.get("status") != "ok":
            continue
        comp = {}
        for key, higher in HEADLINE:
            ci = scorer.paired_difference("D_trustswarm", name, key)
            dv, bv = d[key]["value"], entry["metrics"][key]["value"]
            diff = None if dv is None or bv is None else dv - bv
            better = None if ci is None else (ci[0] > 0 if higher else ci[1] < 0)
            worse = None if ci is None else (ci[1] < 0 if higher else ci[0] > 0)
            comp[key] = {"diff": diff, "ci": ci, "higher_is_better": higher,
                         "d_significantly_better": better, "d_significantly_worse": worse}
            if better is False and ci is not None:
                verdict = "significantly WORSE" if worse else "not significantly better"
                shown = "n/a" if diff is None else f"{diff:+.3f}"
                failures.append(f"D vs {name}: {key} {verdict} "
                                f"(diff {shown}, paired 95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}])")
        results["comparisons"][f"D_vs_{name}"] = comp


# --- figure ---------------------------------------------------------------------------------------

ARM_LABELS = {"A_single": "A\nsingle", "A_random": "A\nrandom", "B_seed": "B\nseeds",
              "B_mace_seeds": "B′\nMACE", "C_collective": "C\nhetero", "D_trustswarm": "D\nswarm"}
ACCENT, ACCENT_LIGHT = "#2a78d6", "#86b6ef"
NEUTRAL, NEUTRAL_LIGHT = "#6e6d68", "#c3c2b7"
INK, INK_MUTED = "#0b0b0b", "#52514e"


def _failure_lines(results: Mapping[str, Any], max_lines: int = 8) -> tuple[list[str], list[str]]:
    """(run failures, D-vs-baseline shortfalls) as short lines for the figure."""
    run = [f for f in results.get("failures", []) if not f.startswith("D vs ")]
    left = [f"• {f[:100]}" for f in run[:max_lines]] or ["• no run failures (all arms ran, full budget used)"]
    if len(run) > max_lines:
        left[-1] = f"• … and {len(run) - max_lines + 1} more"
    right = ["• D NOT significantly better (paired 95% CI) than:   (* = significantly worse)"]
    for name, comp in results.get("comparisons", {}).items():
        worse = [k + ("*" if c["d_significantly_worse"] else "") for k, c in comp.items()
                 if c["d_significantly_better"] is False]
        if worse:
            right.append(f"    {name[5:]:<13} {', '.join(worse)}")
    if len(right) == 1:
        right = ["• D significantly better than every baseline on every headline metric"]
    return left, right[: max_lines + 1]


def plot_ablation(results: Mapping[str, Any], path: str | Path) -> None:
    """One multi-panel figure: AUROC, recall@K, planted caught, served eval error (pre/post),
    each with 95% bootstrap CIs, plus an explicit failures panel."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    arms = list(results["arms"])
    entries = [results["arms"][a] for a in arms]
    labels = [ARM_LABELS.get(a, a) for a in arms]
    x = np.arange(len(arms))
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK_MUTED, "axes.labelcolor": INK,
                         "xtick.color": INK_MUTED, "ytick.color": INK_MUTED})
    fig = plt.figure(figsize=(15, 8.2))
    gs = fig.add_gridspec(2, 4, height_ratios=[3.2, 1.0], hspace=0.38, wspace=0.28)
    axes = [fig.add_subplot(gs[0, i]) for i in range(4)]

    def val(e: Mapping[str, Any], key: str) -> tuple[float, list[float] | None]:
        if e.get("status") != "ok":
            return np.nan, None
        m = e["metrics"].get(key, {})
        v = m.get("value")
        return (np.nan if v is None else v), m.get("ci")

    def bars(ax: Any, key: str, offset: float, width: float, light: bool) -> None:
        for i, (name, e) in enumerate(zip(arms, entries)):
            v, ci = val(e, key)
            d = name.startswith("D_")
            color = (ACCENT_LIGHT if light else ACCENT) if d else (NEUTRAL_LIGHT if light else NEUTRAL)
            if e.get("status") != "ok":
                if offset <= 0:
                    ax.text(i, 0.02, "skipped", rotation=90, ha="center", va="bottom",
                            color=INK_MUTED, fontsize=8, transform=ax.get_xaxis_transform())
                continue
            if not np.isfinite(v):
                ax.text(i + offset, 0.02, "n/a", ha="center", va="bottom", color=INK_MUTED,
                        fontsize=7, transform=ax.get_xaxis_transform())
                continue
            ax.bar(i + offset, v, width=width, color=color, edgecolor="white", linewidth=1.0, zorder=2)
            if ci:
                ax.errorbar(i + offset, v, yerr=[[v - ci[0]], [ci[1] - v]], fmt="none",
                            ecolor=INK, elinewidth=1.0, capsize=2.5, zorder=3)

    def style(ax: Any, title: str, ylabel: str, ylim: tuple[float, float] | None = None) -> None:
        ax.set_title(title, loc="left", fontsize=10, color=INK, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x, labels, fontsize=7)
        ax.grid(axis="y", color="#e6e5e0", linewidth=0.8, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        if ylim:
            ax.set_ylim(*ylim)

    legend_kw = {"frameon": False, "fontsize": 7.5, "loc": "upper left"}
    bars(axes[0], "auroc_pool", -0.2, 0.38, light=False)
    bars(axes[0], "auroc_eval", 0.2, 0.38, light=True)
    axes[0].axhline(0.5, color=INK_MUTED, linestyle=(0, (3, 3)), linewidth=0.9, zorder=1)
    style(axes[0], "1  AUROC (score → served wrong)", "AUROC", (0.0, 1.22))
    axes[0].legend(handles=[Patch(color=NEUTRAL, label="pool (triage score)"),
                            Patch(color=NEUTRAL_LIGHT, label="eval (held-out)"),
                            Line2D([], [], color=INK_MUTED, linestyle=(0, (3, 3)), label="chance")],
                   **legend_kw)

    bars(axes[1], "recall_at_k", 0.0, 0.6, light=False)
    k = results["config"]
    style(axes[1], f"2  Recall@K (K = {k['total_budget']})", "fraction of wrong served answers checked", (0, 1.08))

    bars(axes[2], "planted_caught", 0.0, 0.6, light=False)
    n_pl = results.get("data", {}).get("n_planted", 0)
    style(axes[2], f"3  Planted caught (n = {n_pl})", "fraction flagged, vetoed or checked", (0, 1.08))

    bars(axes[3], "eval_force_rmse_pre", -0.2, 0.38, light=True)
    bars(axes[3], "eval_force_rmse_post", 0.2, 0.38, light=False)
    style(axes[3], "4  Served force RMSE, eval", "eV/Å (lower is better)")
    tops = [hi for e in entries for key in ("eval_force_rmse_pre", "eval_force_rmse_post")
            for hi in (val(e, key)[1] or [0.0])]
    axes[3].set_ylim(0, 1.65 * max(tops + [0.1]))
    ft = results["arms"].get("D_trustswarm", {}).get("finetune")
    if ft is not None:
        if ft.get("shipped") and "eval" in ft:
            b, t = ft["eval"]["base_force_rmse"]["value"], ft["eval"]["finetuned_force_rmse"]["value"]
            note = f"fine-tuned {ft['model']} alone on eval:\n{b:.3f} → {t:.3f} eV/Å"
        else:
            note = f"fine-tune not shipped ({ft.get('model')})"
        axes[3].text(0.98, 0.80, note, transform=axes[3].transAxes, ha="right", va="top",
                     fontsize=7.5, color=INK_MUTED)
    axes[3].legend(handles=[Patch(color=NEUTRAL_LIGHT, label="before fine-tune"),
                            Patch(color=NEUTRAL, label="after fine-tune (D only)")], **legend_kw)

    ax_f = fig.add_subplot(gs[1, :])
    ax_f.axis("off")
    ax_f.text(0.0, 1.0, "Failures and caveats (reported, not hidden; full list in ablation.json)",
              fontsize=10, fontweight="bold", color=INK, va="top", transform=ax_f.transAxes)
    left, right = _failure_lines(results)
    for col, lines in ((0.0, left), (0.55, right)):
        ax_f.text(col, 0.8, "\n".join(lines), fontsize=7.6, color=INK_MUTED, va="top",
                  transform=ax_f.transAxes, family="monospace")
    cfg = results["config"]
    fig.suptitle(f"TrustSwarm ablation — same oracle budget K = {cfg['budget_per_round']} × "
                 f"{cfg['rounds']} rounds; wrong ⇔ F-RMSE > {cfg['wrong_f']} eV/Å or "
                 f"|ΔE_rel| > {cfg['wrong_e']} eV; bars = point estimate, whiskers = 95% "
                 f"reaction-bootstrap CI", x=0.01, ha="left", fontsize=10.5, color=INK)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
