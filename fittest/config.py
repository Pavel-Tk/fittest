"""Fixed-before-running configuration (DESIGN.md sections 2-5).

Every threshold that decides what counts as "wrong", how the oracle budget is spent,
or how agents update beliefs lives here, so the ablation cannot be tuned post hoc by
editing code paths. Module-level constants are the contract values; ``AblationConfig``
in :mod:`fittest.ablation` copies them as defaults so a run records what it used.
"""

from __future__ import annotations

# --- Section 2: error definition ------------------------------------------------------
WRONG_F: float = 0.25
"""Force RMSE threshold [eV/Å] above which a prediction is wrong."""
WRONG_E: float = 0.25
"""Relative-energy error threshold [eV] (≈ 5.8 kcal/mol) above which a prediction is wrong."""
WRONG_THRESHOLDS: tuple[tuple[float, float], ...] = ((0.15, 0.15), (0.25, 0.25), (0.40, 0.40))
"""(WRONG_F, WRONG_E) pairs at which AUROC/recall are reported: the pre-registered primary
(WRONG_F, WRONG_E) and the DESIGN.md §2 sensitivity values 0.15 and 0.4; a threshold-free
Spearman(score, force RMSE) is reported too."""

# --- Section 4/5: oracle budget ---------------------------------------------------------
BUDGET_PER_ROUND: int = 40
N_ROUNDS: int = 5

# --- Section 4: models ------------------------------------------------------------------
SINGLE_MODEL: str = "aimnet2"
SEED_MODELS: tuple[str, ...] = ("aimnet2_m0", "aimnet2_m1", "aimnet2_m2", "aimnet2_m3")
"""AIMNet2's four shipped ensemble members (DESIGN.md §10)."""
MACE_SEED_MODELS: tuple[str, ...] = ("mace_seed0", "mace_seed1", "mace_seed2", "mace_seed3")
"""Optional extra arm B' (LoRA seeds of MACE-MPA-0); skipped with a reason if absent."""
HETERO_MODELS: tuple[str, ...] = ("mace_mpa0", "aimnet2", "orb")
CURATED_MODEL: str = "mace_lora_curated"
CURATED_BASE_MODEL: str = "mace_mpa0"
"""The offline GPU fine-tune script writes ``{CURATED_MODEL}.npz`` for this base model."""
COMMON_TARGET_MODEL: str = "aimnet2"
"""Model whose wrongness is the arm-independent AUROC target (present in every arm)."""

# --- Section 4: Calibrator ----------------------------------------------------------------
CORRECTNESS_PRIOR: tuple[float, float] = (2.0, 1.0)
"""Beta(α, β) prior on P(model correct | bucket): foundation models are mostly right."""
REPUTATION_PRIOR: tuple[float, float] = (1.0, 1.0)
REPUTATION_CLIP: tuple[float, float] = (0.05, 0.95)
FALSE_ALARM_PENALTY: float = 0.5
"""λ: a winning bid on a system revealed correct costs λ × bid share."""
SIGMA_BINS: int = 4
FBAR_BINS: int = 3
TAU_FLOOR_QUANTILE: float = 0.70
TAU_TARGET_P_WRONG: float = 0.5
TAU_MIN_LABELLED: int = 10
"""Below this many oracle labels the threshold stays at its round-0 floor."""
VERDICT_REPOST_DELTA: float = 0.02
"""A verdict is re-posted only when trust moves by at least this much (or the model changes)."""

# --- Section 4: Skeptic -------------------------------------------------------------------
KNN_K: int = 7
TRUST_HIGH: float = 0.5
"""Verdicts with trust ≥ this are audited by the Skeptic."""
VETO_P: float = 0.5

# --- Section 5: statistics ----------------------------------------------------------------
N_BOOTSTRAP: int = 1000
CI_LEVEL: float = 0.95

# --- Revisions (coordinator, after the first synthetic run) -------------------------------
CONSENSUS: str = "consensus"
"""Served-answer candidate = ensemble mean of the model agents' predictions."""
RISK_MIN_LABELLED: int = 20
"""Below this many oracle labels the continuous risk score falls back to the σ rank."""
SILENT_F: float = 0.4
"""'Badly wrong' force RMSE [eV/Å] for the silent-failure metric."""
ECE_BINS: int = 10
AUDIT_FRACTION: float = 0.25
"""Share of each round's oracle budget the Skeptic spends on uniform audits of trusted systems."""
PLANTED_MODEL: str = "aimnet2"
"""Planted cases are arm-independent: frames where this model (in every arm) is badly wrong."""
PLANTED_FORCE_RMSE: float = 0.4
"""'Confidently wrong' = force RMSE above this [eV/Å] for PLANTED_MODEL."""
