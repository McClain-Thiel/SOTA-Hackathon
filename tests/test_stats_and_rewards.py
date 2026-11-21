"""Unit tests for paired timing statistics and anti-collapse RL reward calculations."""

from __future__ import annotations

import math

import pytest

from repoopt.bench.stats import OutcomeLabel, classify_and_score, compute_paired_stats


def test_paired_stats_basic_speedup() -> None:
    base = [100.0, 102.0, 99.0, 101.0, 100.5]
    cand = [20.0, 20.5, 19.8, 20.1, 20.0]
    stats = compute_paired_stats(base, cand, seed=42)
    assert stats.n_pairs == 5
    assert 4.9 <= stats.speedup <= 5.1
    assert stats.ci95_low > 4.8
    assert stats.ci95_high < 5.2
    assert stats.baseline_cv < 0.05
    assert stats.candidate_cv < 0.05


def test_noop_patch_gets_zero_reward_even_if_correct() -> None:
    base = [100.0, 100.0, 100.0, 100.0, 100.0]
    cand = [99.0, 99.0, 99.0, 99.0, 99.0]
    stats = compute_paired_stats(base, cand)
    res = classify_and_score(correct=True, is_noop_patch=True, stats=stats)
    assert res.label == OutcomeLabel.CORRECT_NO_SPEEDUP
    assert res.reward == 0.0


def test_incorrect_gets_zero_reward_even_if_fast() -> None:
    stats = compute_paired_stats([100.0, 100.0, 100.0], [1.0, 1.0, 1.0])
    res = classify_and_score(correct=False, is_noop_patch=False, stats=stats)
    assert res.label == OutcomeLabel.INCORRECT
    assert res.reward == 0.0


def test_verified_speedup_log2_reward_and_cap() -> None:
    stats_4x = compute_paired_stats([100.0, 101.0, 99.0, 100.0, 100.0], [25.0, 25.2, 24.8, 25.0, 25.0])
    res_4x = classify_and_score(correct=True, is_noop_patch=False, stats=stats_4x)
    assert res_4x.label == OutcomeLabel.VERIFIED_SPEEDUP
    assert math.isclose(res_4x.reward, math.log2(stats_4x.speedup), rel_tol=1e-5)

    stats_100x = compute_paired_stats([1000.0] * 5, [10.0] * 5)
    res_100x = classify_and_score(correct=True, is_noop_patch=False, stats=stats_100x)
    assert res_100x.label == OutcomeLabel.VERIFIED_SPEEDUP
    assert res_100x.reward == 4.0


def test_correct_but_slower_and_no_speedup() -> None:
    slower_stats = compute_paired_stats([100.0] * 5, [120.0] * 5)
    slower = classify_and_score(correct=True, is_noop_patch=False, stats=slower_stats)
    assert slower.label == OutcomeLabel.CORRECT_BUT_SLOWER
    assert slower.reward == 0.05

    flat_stats = compute_paired_stats([100.0] * 5, [99.0] * 5)
    flat = classify_and_score(correct=True, is_noop_patch=False, stats=flat_stats)
    assert flat.label == OutcomeLabel.CORRECT_NO_SPEEDUP
    assert flat.reward == 0.05


def test_timeout_and_env_failure_and_bad_inputs() -> None:
    assert classify_and_score(correct=True, is_noop_patch=False, timed_out=True).label == OutcomeLabel.TIMEOUT
    assert classify_and_score(correct=True, is_noop_patch=False, env_error=True).label == OutcomeLabel.ENVIRONMENT_FAILURE
    with pytest.raises(ValueError):
        compute_paired_stats([], [])
    with pytest.raises(ValueError):
        compute_paired_stats([1.0, 2.0], [1.0])
    with pytest.raises(ValueError):
        compute_paired_stats([0.0, 1.0], [1.0, 1.0])
