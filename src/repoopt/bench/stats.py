"""Paired A/B/A/B timing statistics, bootstrap confidence intervals, and RL reward scoring."""

from __future__ import annotations

import math
import random
import statistics
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class OutcomeLabel(str, Enum):
    """Canonical RepoOpt trajectory classification labels (Section 6)."""

    VERIFIED_SPEEDUP = "VERIFIED_SPEEDUP"
    CORRECT_NO_SPEEDUP = "CORRECT_NO_SPEEDUP"
    CORRECT_BUT_SLOWER = "CORRECT_BUT_SLOWER"
    INCORRECT = "INCORRECT"
    INCONCLUSIVE = "INCONCLUSIVE"
    TIMEOUT = "TIMEOUT"
    ENVIRONMENT_FAILURE = "ENVIRONMENT_FAILURE"


class PairedTimingSummary(BaseModel):
    """Summary statistics from interleaved paired baseline vs. candidate timing runs."""

    n_pairs: int
    baseline_ms: list[float]
    candidate_ms: list[float]
    median_baseline_ms: float
    median_candidate_ms: float
    speedup: float
    ci95_low: float
    ci95_high: float
    baseline_cv: float
    candidate_cv: float


class EvaluationResult(BaseModel):
    """Structured verifier output stored in evaluation.json and reward.json."""

    label: OutcomeLabel
    reward: float
    correct: bool
    is_noop_patch: bool
    build_ok: bool = True
    tests_ok: bool = True
    max_abs_err: float | None = None
    max_rel_err: float | None = None
    timing: PairedTimingSummary | None = None
    reason: str = ""
    details: dict[str, Any] = Field(default_factory=dict)


def _coefficient_of_variation(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = statistics.fmean(values)
    if mean <= 0.0:
        return 0.0
    return statistics.stdev(values) / mean


def compute_paired_stats(
    baseline_ms: list[float],
    candidate_ms: list[float],
    *,
    n_bootstrap: int = 2000,
    seed: int = 0,
) -> PairedTimingSummary:
    """Compute median times, paired speedup, CV, and 95% bootstrap CI over paired b_i / c_i ratios."""
    if not baseline_ms or not candidate_ms:
        raise ValueError("baseline_ms and candidate_ms must be non-empty")
    if len(baseline_ms) != len(candidate_ms):
        raise ValueError(
            f"Paired timing lengths must match: got {len(baseline_ms)} baseline vs {len(candidate_ms)} candidate"
        )
    if any(t <= 0.0 or not math.isfinite(t) for t in baseline_ms + candidate_ms):
        raise ValueError("All timing measurements must be finite and strictly positive")

    n = len(baseline_ms)
    ratios = [b / c for b, c in zip(baseline_ms, candidate_ms)]
    median_base = statistics.median(baseline_ms)
    median_cand = statistics.median(candidate_ms)
    # Use median paired ratio (invariant to paired order and robust to single outliers)
    speedup = statistics.median(ratios)

    if n == 1:
        ci95_low = speedup
        ci95_high = speedup
    else:
        rng = random.Random(seed)
        boot: list[float] = []
        for _ in range(n_bootstrap):
            sample = [ratios[rng.randrange(n)] for _ in range(n)]
            boot.append(statistics.median(sample))
        boot.sort()
        lo_idx = max(0, int(math.floor(0.025 * (n_bootstrap - 1))))
        hi_idx = min(n_bootstrap - 1, int(math.ceil(0.975 * (n_bootstrap - 1))))
        ci95_low = boot[lo_idx]
        ci95_high = boot[hi_idx]

    return PairedTimingSummary(
        n_pairs=n,
        baseline_ms=[round(x, 4) for x in baseline_ms],
        candidate_ms=[round(x, 4) for x in candidate_ms],
        median_baseline_ms=round(median_base, 4),
        median_candidate_ms=round(median_cand, 4),
        speedup=round(speedup, 6),
        ci95_low=round(ci95_low, 6),
        ci95_high=round(ci95_high, 6),
        baseline_cv=round(_coefficient_of_variation(baseline_ms), 6),
        candidate_cv=round(_coefficient_of_variation(candidate_ms), 6),
    )


def classify_and_score(
    *,
    correct: bool,
    is_noop_patch: bool,
    stats: PairedTimingSummary | None = None,
    build_ok: bool = True,
    tests_ok: bool = True,
    max_abs_err: float | None = None,
    max_rel_err: float | None = None,
    timed_out: bool = False,
    env_error: bool = False,
    max_cv_threshold: float = 0.25,
    min_speedup_threshold: float = 1.05,
    min_ci95_low_threshold: float = 1.02,
    reward_cap: float = 4.0,
    reason: str = "",
    details: dict[str, Any] | None = None,
) -> EvaluationResult:
    """Classify optimization outcome and compute anti-collapse RL scalar reward."""
    extra = details or {}

    if env_error:
        return EvaluationResult(
            label=OutcomeLabel.ENVIRONMENT_FAILURE,
            reward=0.0,
            correct=False,
            is_noop_patch=is_noop_patch,
            build_ok=build_ok,
            tests_ok=tests_ok,
            max_abs_err=max_abs_err,
            max_rel_err=max_rel_err,
            timing=stats,
            reason=reason or "Environment failure during verification",
            details=extra,
        )

    if timed_out:
        return EvaluationResult(
            label=OutcomeLabel.TIMEOUT,
            reward=0.0,
            correct=False,
            is_noop_patch=is_noop_patch,
            build_ok=build_ok,
            tests_ok=tests_ok,
            max_abs_err=max_abs_err,
            max_rel_err=max_rel_err,
            timing=stats,
            reason=reason or "Rollout or verification timed out",
            details=extra,
        )

    if not build_ok or not tests_ok or not correct:
        return EvaluationResult(
            label=OutcomeLabel.INCORRECT,
            reward=0.0,
            correct=False,
            is_noop_patch=is_noop_patch,
            build_ok=build_ok,
            tests_ok=tests_ok,
            max_abs_err=max_abs_err,
            max_rel_err=max_rel_err,
            timing=stats,
            reason=reason or "Candidate failed build, unit tests, or numerical equivalence checks",
            details=extra,
        )

    if is_noop_patch:
        return EvaluationResult(
            label=OutcomeLabel.CORRECT_NO_SPEEDUP,
            reward=0.0,
            correct=True,
            is_noop_patch=True,
            build_ok=True,
            tests_ok=True,
            max_abs_err=max_abs_err,
            max_rel_err=max_rel_err,
            timing=stats,
            reason=reason or "No-op candidate patch receives 0.0 RL reward",
            details=extra,
        )

    if stats is None:
        return EvaluationResult(
            label=OutcomeLabel.INCONCLUSIVE,
            reward=0.0,
            correct=True,
            is_noop_patch=False,
            build_ok=True,
            tests_ok=True,
            max_abs_err=max_abs_err,
            max_rel_err=max_rel_err,
            timing=None,
            reason=reason or "Missing timing statistics",
            details=extra,
        )

    if stats.speedup > min_speedup_threshold and stats.ci95_low > min_ci95_low_threshold:
        raw_reward = min(reward_cap, max(0.0, math.log2(stats.speedup)))
        return EvaluationResult(
            label=OutcomeLabel.VERIFIED_SPEEDUP,
            reward=round(raw_reward, 6),
            correct=True,
            is_noop_patch=False,
            build_ok=True,
            tests_ok=True,
            max_abs_err=max_abs_err,
            max_rel_err=max_rel_err,
            timing=stats,
            reason=reason or f"Verified speedup {stats.speedup:.3f}x (95% CI low {stats.ci95_low:.3f}x)",
            details=extra,
        )

    if max(stats.baseline_cv, stats.candidate_cv) > max_cv_threshold:
        return EvaluationResult(
            label=OutcomeLabel.INCONCLUSIVE,
            reward=0.05,
            correct=True,
            is_noop_patch=False,
            build_ok=True,
            tests_ok=True,
            max_abs_err=max_abs_err,
            max_rel_err=max_rel_err,
            timing=stats,
            reason=reason or f"Timing noise too high (CV base={stats.baseline_cv:.3f}, cand={stats.candidate_cv:.3f})",
            details=extra,
        )

    if stats.ci95_high < 0.98:
        return EvaluationResult(
            label=OutcomeLabel.CORRECT_BUT_SLOWER,
            reward=0.05,
            correct=True,
            is_noop_patch=False,
            build_ok=True,
            tests_ok=True,
            max_abs_err=max_abs_err,
            max_rel_err=max_rel_err,
            timing=stats,
            reason=reason or f"Correct but slower ({stats.speedup:.3f}x, 95% CI high {stats.ci95_high:.3f}x)",
            details=extra,
        )

    return EvaluationResult(
        label=OutcomeLabel.CORRECT_NO_SPEEDUP,
        reward=0.05,
        correct=True,
        is_noop_patch=False,
        build_ok=True,
        tests_ok=True,
        max_abs_err=max_abs_err,
        max_rel_err=max_rel_err,
        timing=stats,
        reason=reason or f"Correct without significant speedup ({stats.speedup:.3f}x, CI low {stats.ci95_low:.3f}x)",
        details=extra,
    )
