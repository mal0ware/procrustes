"""Sharpe ratio inference: closed forms, limits and the papers' worked examples."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm

from procrustes import (
    average_correlation,
    deflated_sharpe_ratio,
    deflated_sharpe_ratio_from_returns,
    effective_number_of_trials,
    expected_max_sharpe,
    expected_max_standard_normal,
    min_backtest_length,
    min_backtest_length_upper_bound,
    min_track_record_length,
    probabilistic_sharpe_ratio,
    return_moments,
    sharpe_ratio,
    sharpe_ratio_std,
)

# --- probabilistic Sharpe ratio: Bailey and López de Prado (2012) -----------------


def test_psr_hedge_fund_example() -> None:
    """BLdP12 section 3 and Figure 6: monthly SR 0.458 over two years."""
    assert probabilistic_sharpe_ratio(0.458, 24) == pytest.approx(0.982, abs=5e-4)
    assert probabilistic_sharpe_ratio(0.458, 24, 0.0, -2.448, 10.164) == pytest.approx(
        0.913, abs=5e-4
    )
    # "should we have 3 years instead of 2, PSR(0) = 0.953"
    assert probabilistic_sharpe_ratio(0.458, 36, 0.0, -2.448, 10.164) == pytest.approx(
        0.953, abs=6e-4
    )


@pytest.mark.parametrize(
    ("periods", "skew", "kurt", "years"),
    [
        (250, 0.0, 3.0, 2.73),  # Figure 8, daily
        (52, 0.0, 3.0, 2.83),  # Figure 9, weekly
        (12, 0.0, 3.0, 3.24),  # Figure 10, monthly
        (12, -0.72, 5.78, 4.99),  # Figure 11, monthly, HFR moments
    ],
)
def test_min_track_record_length_examples(
    periods: int, skew: float, kurt: float, years: float
) -> None:
    """BLdP12 section 5: annualised SR 2 against a benchmark of 1, 95% confidence."""
    n = min_track_record_length(2 / math.sqrt(periods), 1 / math.sqrt(periods), skew, kurt)
    assert n / periods == pytest.approx(years, abs=5e-3)


def test_min_track_record_length_appendix_value() -> None:
    """BLdP12 appendix A.3: MinTRL = 59.895 months and PSR at that length is 0.95."""
    sr, ref = 2 / math.sqrt(12), 1 / math.sqrt(12)
    n = min_track_record_length(sr, ref, -0.72, 5.78, 0.95)
    assert n == pytest.approx(59.895, abs=5e-4)
    assert probabilistic_sharpe_ratio(sr, 59.895, ref, -0.72, 5.78) == pytest.approx(0.95, abs=1e-6)


@pytest.mark.parametrize("confidence", [0.9, 0.95, 0.99])
def test_min_track_record_length_inverts_psr(confidence: float) -> None:
    sr, ref, skew, kurt = 0.12, 0.03, -0.8, 6.0
    n = min_track_record_length(sr, ref, skew, kurt, confidence)
    assert probabilistic_sharpe_ratio(sr, n, ref, skew, kurt) == pytest.approx(confidence)


def test_psr_limits_and_monotonicity() -> None:
    assert probabilistic_sharpe_ratio(0.1, 500, 0.1) == pytest.approx(0.5)
    assert probabilistic_sharpe_ratio(0.1, 500) > probabilistic_sharpe_ratio(0.1, 100)
    # negative skew and fat tails lower the confidence in a positive Sharpe ratio
    base = probabilistic_sharpe_ratio(0.1, 500)
    assert probabilistic_sharpe_ratio(0.1, 500, skew=-1.0) < base
    assert probabilistic_sharpe_ratio(0.1, 500, kurt=8.0) < base
    assert min_track_record_length(0.05, 0.05) == math.inf


def test_sharpe_std_matches_simulation() -> None:
    """Mertens' standard error against the spread of simulated Normal Sharpe ratios."""
    rng = np.random.default_rng(11)
    n, true_sr = 250, 0.15
    srs = sharpe_ratio(rng.standard_normal((n, 20_000)) + true_sr)
    assert srs.std(ddof=1) == pytest.approx(sharpe_ratio_std(true_sr, n), rel=0.02)


def test_impossible_moments_rejected() -> None:
    with pytest.raises(ValueError, match="not positive"):
        sharpe_ratio_std(2.0, 100, skew=3.0, kurt=1.0)


def test_return_moments_and_sharpe() -> None:
    rng = np.random.default_rng(3)
    r = rng.standard_normal(100_000) * 0.01 + 0.0005
    sr, skew, kurt, n = return_moments(r)
    assert n == 100_000
    assert sr == pytest.approx(float(sharpe_ratio(r)))
    assert skew == pytest.approx(0.0, abs=0.03)
    assert kurt == pytest.approx(3.0, abs=0.06)
    assert sharpe_ratio(np.ones((10, 2))).tolist() == [0.0, 0.0]


# --- expected maximum: BLdP14 eq. (1) and (5), BBLZ14 Proposition 1 ---------------


def test_expected_max_paper_values() -> None:
    """BBLZ14: N = 10 gives 1.57; seven binary parameters (N = 128) give 'above 2.6'."""
    assert expected_max_standard_normal(10) == pytest.approx(1.57, abs=5e-3)
    assert 2.6 < expected_max_standard_normal(128) < 2.65


def _exact_expected_max(n: int) -> float:
    """E[max of n iid N(0,1)] by numerical integration of x·n·φ(x)·Φ(x)^(n-1)."""
    value, _ = quad(lambda x: x * n * norm.pdf(x) * norm.cdf(x) ** (n - 1), -12, 12, limit=400)
    return float(value)


def test_expected_max_against_exact_integral() -> None:
    """The approximation overshoots the exact value by a few hundredths.

    BLdP14 Exhibit 3.1 reports an error "as small as 0.006" at 1,000 trials.
    The exact integral gives 0.0137 there; see PAPER.md section 8.
    """
    assert _exact_expected_max(2) == pytest.approx(1 / math.sqrt(math.pi), abs=1e-9)
    errors = {n: expected_max_standard_normal(n) - _exact_expected_max(n) for n in (10, 100, 1000)}
    assert errors[10] == pytest.approx(0.0358, abs=5e-4)
    assert errors[100] == pytest.approx(0.0230, abs=5e-4)
    assert errors[1000] == pytest.approx(0.0137, abs=5e-4)
    rng = np.random.default_rng(5)
    sim = rng.standard_normal((40_000, 100)).max(axis=1).mean()
    assert sim == pytest.approx(_exact_expected_max(100), abs=0.01)


def test_expected_max_small_n_and_bound() -> None:
    assert expected_max_standard_normal(1) == 0.0
    values = [expected_max_standard_normal(n) for n in (1, 1.25, 1.5, 2, 3, 10, 100, 1e6)]
    assert all(b > a for a, b in itertools.pairwise(values))
    for n in (2, 10, 1000, 10**6):
        assert expected_max_standard_normal(n) < math.sqrt(2 * math.log(n))
    with pytest.raises(ValueError):
        expected_max_standard_normal(0.5)


def test_expected_max_sharpe_scales() -> None:
    base = expected_max_standard_normal(50)
    assert expected_max_sharpe(50, 4.0) == pytest.approx(2 * base)
    assert expected_max_sharpe(50, 1.0, sr_mean=0.3) == pytest.approx(0.3 + base)


# --- deflated Sharpe ratio: BLdP14 -------------------------------------------------


def test_dsr_numerical_example() -> None:
    """BLdP14 'A numerical example': SR0 = 0.1132 and DSR = 0.9004."""
    sr = 2.5 / math.sqrt(250)
    var = 0.5 / 250  # V = 1/2 in annualised units
    assert expected_max_sharpe(100, var) == pytest.approx(0.1132, abs=5e-5)
    assert deflated_sharpe_ratio(sr, 1250, 100, var, -3.0, 10.0) == pytest.approx(0.9004, abs=5e-5)


def test_dsr_example_alternative_trial_counts() -> None:
    """Same example: DSR = 0.9505 at N = 46, and at N = 88 with Normal returns."""
    sr = 2.5 / math.sqrt(250)
    var = 0.5 / 250
    assert deflated_sharpe_ratio(sr, 1250, 46, var, -3.0, 10.0) == pytest.approx(0.9505, abs=5e-5)
    assert deflated_sharpe_ratio(sr, 1250, 88, var, 0.0, 3.0) == pytest.approx(0.9505, abs=5e-5)


def test_dsr_equals_psr_for_one_trial() -> None:
    rng = np.random.default_rng(2)
    for _ in range(50):
        sr = rng.uniform(-0.3, 0.3)
        n = int(rng.integers(30, 3000))
        skew = rng.uniform(-2, 2)
        kurt = skew**2 + 1 + rng.uniform(0.5, 8)
        var = rng.uniform(0, 0.1)
        dsr = deflated_sharpe_ratio(sr, n, 1, var, skew, kurt)
        assert dsr == probabilistic_sharpe_ratio(sr, n, 0.0, skew, kurt)


def test_dsr_decreases_with_trials_and_variance() -> None:
    args = (0.12, 1000)
    dsrs = [deflated_sharpe_ratio(*args, n, 0.002) for n in (1, 2, 10, 100, 1000)]
    assert all(b < a for a, b in itertools.pairwise(dsrs))
    assert deflated_sharpe_ratio(*args, 100, 0.004) < deflated_sharpe_ratio(*args, 100, 0.002)


def test_effective_number_of_trials() -> None:
    """BLdP14 eq. (8) and (9)."""
    assert effective_number_of_trials(100, 0.0) == 100
    assert effective_number_of_trials(100, 1.0) == 1
    assert effective_number_of_trials(100, 0.25) == pytest.approx(0.25 + 0.75 * 100)
    assert effective_number_of_trials(100, -0.01) == 100
    m, rho = 7, 0.3
    corr = np.full((m, m), rho) + (1 - rho) * np.eye(m)
    assert average_correlation(corr) == pytest.approx(rho)


def test_correlated_trials_deflate_less() -> None:
    sr, var = 0.1, 0.001
    independent = deflated_sharpe_ratio(sr, 1250, 500, var)
    correlated = deflated_sharpe_ratio(sr, 1250, 500, var, avg_correlation=0.8)
    assert correlated > independent
    assert correlated == pytest.approx(deflated_sharpe_ratio(sr, 1250, 0.8 + 0.2 * 500, var))


def test_dsr_from_returns_consistent_with_scalar_api() -> None:
    rng = np.random.default_rng(8)
    r = rng.standard_normal((600, 40)) * 0.01
    res = deflated_sharpe_ratio_from_returns(r)
    srs = sharpe_ratio(r)
    assert res.selected == int(np.argmax(srs))
    assert res.n_trials == 40
    assert res.n_effective == 40
    assert res.dsr == pytest.approx(
        deflated_sharpe_ratio(res.sharpe, 600, 40, float(np.var(srs, ddof=1)), res.skew, res.kurt)
    )
    assert res.dsr < res.psr_zero
    adjusted = deflated_sharpe_ratio_from_returns(r + r[:, [0]], adjust_for_correlation=True)
    assert adjusted.n_effective < adjusted.n_trials
    assert adjusted.avg_correlation > 0.3


# --- minimum backtest length: BBLZ14 Theorem 2 -------------------------------------


def test_min_backtest_length_paper_values() -> None:
    """BBLZ14 Figure 2: 45 trials need five years, seven trials need about two."""
    assert min_backtest_length(45) == pytest.approx(5.0, abs=0.01)
    assert min_backtest_length(46) > 5.0
    assert min_backtest_length(7) == pytest.approx(2.0, abs=0.1)


def test_min_backtest_length_scaling_and_bound() -> None:
    assert min_backtest_length(100, 2.0) == pytest.approx(min_backtest_length(100) / 4)
    for n in (2, 7, 45, 1000):
        assert min_backtest_length(n) < min_backtest_length_upper_bound(n)
    assert min_backtest_length_upper_bound(45) == pytest.approx(2 * math.log(45))


def test_min_backtest_length_is_where_expected_max_hits_target() -> None:
    """With y = MinBTL years, the expected best annualised Sharpe ratio is the target."""
    n, target = 200, 1.5
    years = min_backtest_length(n, target)
    assert expected_max_sharpe(n, 1 / years) == pytest.approx(target)
    assert norm.cdf(0) == 0.5
