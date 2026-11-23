"""Probability of backtest overfitting by CSCV.

The tolerances in the two statistical tests come from the simulation that
``scripts/reproduce.py`` writes to ``results/pbo_sampling.csv`` (see PAPER.md
section 9): with 50 strategies, 1,000 observations and 10 blocks, a single PBO
estimate on pure noise has a standard deviation of about 0.18, so only the
average over many datasets is tested against 0.5.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from procrustes import cscv, n_cscv_combinations, sharpe_ratio

N_STRATEGIES, N_OBS, N_BLOCKS = 50, 1000, 10


def _pbo_sample(drift: float, reps: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.empty(reps)
    for i in range(reps):
        r = rng.standard_normal((N_OBS, N_STRATEGIES))
        r[:, 0] += drift
        out[i] = cscv(r, N_BLOCKS).pbo
    return out


def test_pbo_is_half_on_pure_noise() -> None:
    """Mean of 200 estimates: standard error about 0.18/sqrt(200) = 0.013, tolerance 4 SE."""
    pbo = _pbo_sample(0.0, 200, seed=101)
    assert pbo.mean() == pytest.approx(0.5, abs=0.05)
    # a single estimate is noisy; it should still not pile up at 0 or 1
    assert 0.1 < pbo.std(ddof=1) < 0.25


def test_pbo_is_near_zero_with_real_drift() -> None:
    """One strategy with a per-observation Sharpe ratio of 0.3 (t-statistic about 9.5)."""
    pbo = _pbo_sample(0.3, 50, seed=202)
    assert pbo.max() < 0.01


def test_pbo_falls_as_drift_grows() -> None:
    means = [_pbo_sample(d, 60, seed=303).mean() for d in (0.0, 0.1, 0.2)]
    assert means[0] > means[1] > means[2]
    assert means[2] < 0.05


def test_combination_counts() -> None:
    """BBLZ17 eq. (2.3). The paper prints 12,780 for S = 16; C(16, 8) is 12,870."""
    assert n_cscv_combinations(4) == 6
    assert n_cscv_combinations(12) == 924
    assert n_cscv_combinations(16) == 12_870
    assert n_cscv_combinations(24) == 2_704_156
    for bad in (0, 3, 7):
        with pytest.raises(ValueError):
            n_cscv_combinations(bad)


def test_result_shapes_and_determinism() -> None:
    rng = np.random.default_rng(4)
    r = rng.standard_normal((803, 12))
    a = cscv(r, 8)
    b = cscv(r, 8)
    assert a.n_combinations == 70 == a.logits.size
    assert a.rows_used == 800
    assert a.n_strategies == 12
    assert np.array_equal(a.logits, b.logits)
    assert a.pbo == b.pbo
    assert a.prob_oos_loss == pytest.approx(np.mean(a.oos_performance < 0))
    # every in-sample winner has the highest in-sample Sharpe ratio seen
    assert np.all(a.is_performance > 0)


def test_custom_metric_matches_fast_sharpe_path() -> None:
    rng = np.random.default_rng(6)
    r = rng.standard_normal((240, 9)) * 0.02 + 0.001
    fast = cscv(r, 6)
    slow = cscv(r, 6, metric=sharpe_ratio)
    assert np.array_equal(fast.selected, slow.selected)
    np.testing.assert_allclose(fast.logits, slow.logits)
    np.testing.assert_allclose(fast.is_performance, slow.is_performance)
    np.testing.assert_allclose(fast.oos_performance, slow.oos_performance)
    assert fast.pbo == slow.pbo


def test_hand_worked_two_blocks() -> None:
    """S = 2, N = 3, mean as the metric: the logits can be written down by hand.

    Block means are A = (3, 2, 1) and B = (1, 2, 3). In sample on A the winner
    is column 0, which is worst on B: rank 1 of 3, ω = 1/4, λ = ln(1/3). In
    sample on B the winner is column 2, worst on A: the same logit. PBO = 1.
    """
    block_a = np.array([[3.0, 2.0, 1.0], [3.0, 2.0, 1.0]])
    block_b = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    res = cscv(np.vstack([block_a, block_b]), 2, metric=lambda x: x.mean(axis=0))
    np.testing.assert_allclose(res.logits, [math.log(1 / 3)] * 2)
    assert res.selected.tolist() == [0, 2]
    assert res.pbo == 1.0
    assert res.degradation_slope != res.degradation_slope  # nan: no in-sample spread


def test_median_rank_counts_half_for_odd_n() -> None:
    """Winner on A is column 0, which is the median of three on B: λ = 0."""
    block_a = np.array([[3.0, 2.0, 1.0], [3.0, 2.0, 1.0]])
    block_b = np.array([[2.0, 3.0, 1.0], [2.0, 3.0, 1.0]])
    res = cscv(np.vstack([block_a, block_b]), 2, metric=lambda x: x.mean(axis=0))
    # split 1: IS = A, winner 0, OOS rank 2 of 3 -> λ = 0 -> counts 0.5
    # split 2: IS = B, winner 1, OOS rank 2 of 3 -> λ = 0 -> counts 0.5
    np.testing.assert_allclose(res.logits, [0.0, 0.0], atol=1e-12)
    assert res.pbo == 0.5


def test_column_order_does_not_matter() -> None:
    rng = np.random.default_rng(9)
    r = rng.standard_normal((400, 15))
    perm = rng.permutation(15)
    assert cscv(r, 8).pbo == cscv(r[:, perm], 8).pbo


def test_degradation_slope_is_negative_under_noise_with_fixed_total() -> None:
    """With iid noise the selected strategy's IS and OOS halves share a fixed full-sample
    mean, so a higher in-sample statistic goes with a lower out-of-sample one."""
    rng = np.random.default_rng(12)
    res = cscv(rng.standard_normal((1000, 50)), 10)
    assert res.degradation_slope < 0


def test_input_validation() -> None:
    with pytest.raises(ValueError):
        cscv(np.zeros(10), 2)
    with pytest.raises(ValueError):
        cscv(np.zeros((10, 1)), 2)
    with pytest.raises(ValueError):
        cscv(np.zeros((2, 3)), 2)
