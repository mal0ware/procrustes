"""Sharpe ratio inference: probabilistic and deflated Sharpe ratio, minimum lengths.

Sources (see PAPER.md for equation-by-equation references):

* Bailey and López de Prado (2012), "The Sharpe Ratio Efficient Frontier".
* Bailey and López de Prado (2014), "The Deflated Sharpe Ratio".
* Bailey, Borwein, López de Prado and Zhu (2014), "Pseudo-Mathematics and
  Financial Charlatanism".

Conventions used throughout this module:

* Sharpe ratios are per observation (not annualised) unless a function says
  otherwise. Returns are assumed to be excess returns already.
* Kurtosis is the raw fourth standardised moment: a Normal distribution has
  kurtosis 3.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm

FloatArray = NDArray[np.float64]

EULER_GAMMA: float = 0.5772156649015329


def sharpe_ratio(returns: ArrayLike, axis: int = 0, ddof: int = 1) -> FloatArray:
    """Per-observation Sharpe ratio: mean over standard deviation.

    Parameters
    ----------
    returns
        Excess returns. A 1-D series, or a 2-D array with one strategy per column.
    axis
        Axis that indexes time.
    ddof
        Degrees of freedom removed in the standard deviation.

    Returns
    -------
    numpy.ndarray
        One Sharpe ratio per strategy (0-d for a 1-D input). A strategy with
        zero variance gets a Sharpe ratio of 0.
    """
    r = np.asarray(returns, dtype=np.float64)
    mean = r.mean(axis=axis)
    std = r.std(axis=axis, ddof=ddof)
    out = np.zeros_like(mean)
    np.divide(mean, std, out=out, where=std > 0)
    return out


def return_moments(returns: ArrayLike) -> tuple[float, float, float, int]:
    """Sharpe ratio, skewness, kurtosis and length of one return series.

    Skewness and kurtosis are the plain moment estimators (no small-sample
    correction); kurtosis is not excess kurtosis.
    """
    r = np.asarray(returns, dtype=np.float64).ravel()
    n = r.size
    if n < 3:
        raise ValueError("need at least 3 observations")
    mean = float(r.mean())
    dev = r - mean
    m2 = float(np.mean(dev**2))
    if m2 <= 0:
        raise ValueError("returns have zero variance")
    skew = float(np.mean(dev**3)) / m2**1.5
    kurt = float(np.mean(dev**4)) / m2**2
    sr = mean / float(r.std(ddof=1))
    return sr, skew, kurt, n


def _sr_variance_term(sr: float, skew: float, kurt: float) -> float:
    """1 - γ3·SR + (γ4 - 1)/4·SR², the numerator of the Mertens variance."""
    term = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2
    if term <= 0:
        raise ValueError(
            "1 - skew*SR + (kurt-1)/4*SR^2 is not positive; "
            "skewness and kurtosis are not those of a real distribution "
            "(they must satisfy kurt >= skew^2 + 1)"
        )
    return term


def sharpe_ratio_std(sr: float, n_obs: float, skew: float = 0.0, kurt: float = 3.0) -> float:
    """Standard error of an estimated Sharpe ratio (Mertens 2002).

    ``sqrt((1 - γ3·SR + (γ4-1)/4·SR²) / (n-1))``, BLdP12 section 2.5.
    """
    if n_obs <= 1:
        raise ValueError("n_obs must exceed 1")
    return math.sqrt(_sr_variance_term(sr, skew, kurt) / (n_obs - 1.0))


def probabilistic_sharpe_ratio(
    sr: float,
    n_obs: float,
    sr_benchmark: float = 0.0,
    skew: float = 0.0,
    kurt: float = 3.0,
) -> float:
    """Probability that the true Sharpe ratio exceeds ``sr_benchmark``.

    BLdP12 eq. (11). ``sr`` and ``sr_benchmark`` are per observation, at the
    same frequency as the ``n_obs`` returns they were estimated from.
    """
    z = (sr - sr_benchmark) / sharpe_ratio_std(sr, n_obs, skew, kurt)
    return float(norm.cdf(z))


def min_track_record_length(
    sr: float,
    sr_benchmark: float = 0.0,
    skew: float = 0.0,
    kurt: float = 3.0,
    confidence: float = 0.95,
) -> float:
    """Observations needed for ``PSR(sr_benchmark)`` to reach ``confidence``.

    BLdP12 eq. (13). Requires ``sr > sr_benchmark``; returns ``inf`` otherwise,
    because no track record length makes a Sharpe ratio at or below the
    benchmark significantly above it.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    if sr <= sr_benchmark:
        return math.inf
    z = float(norm.ppf(confidence))
    return 1.0 + _sr_variance_term(sr, skew, kurt) * (z / (sr - sr_benchmark)) ** 2


def _expected_max_formula(n: float) -> float:
    return float(
        (1.0 - EULER_GAMMA) * norm.ppf(1.0 - 1.0 / n)
        + EULER_GAMMA * norm.ppf(1.0 - 1.0 / (n * math.e))
    )


def expected_max_standard_normal(n_trials: float) -> float:
    """Approximate expected maximum of ``n_trials`` independent N(0, 1) draws.

    BLdP14 eq. (5) / BBLZ14 eq. (4):
    ``(1-γ)·Z⁻¹[1 - 1/N] + γ·Z⁻¹[1 - 1/(N·e)]``, an approximation for large N.

    Two departures from the paper, both for small N (see PAPER.md section 3):
    the function returns the exact value 0 at ``N = 1`` where the formula is
    minus infinity, and interpolates linearly between ``N = 1`` and ``N = 2``.
    ``n_trials`` need not be an integer, so an effective number of trials can
    be passed.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1")
    if n_trials == 1:
        return 0.0
    if n_trials < 2:
        return (n_trials - 1.0) * _expected_max_formula(2.0)
    return _expected_max_formula(float(n_trials))


def expected_max_sharpe(n_trials: float, sr_variance: float, sr_mean: float = 0.0) -> float:
    """Expected maximum Sharpe ratio among ``n_trials`` independent trials.

    BLdP14 eq. (1): ``E[SR] + sqrt(V[SR])·E[max_N]``. ``sr_variance`` is the
    variance of the estimated Sharpe ratios across trials, in the same units
    as the Sharpe ratio the result will be compared with.
    """
    if sr_variance < 0:
        raise ValueError("sr_variance must be non-negative")
    return sr_mean + math.sqrt(sr_variance) * expected_max_standard_normal(n_trials)


def average_correlation(corr: ArrayLike) -> float:
    """Equal-weighted average off-diagonal correlation, BLdP14 eq. (8)."""
    c = np.asarray(corr, dtype=np.float64)
    if c.ndim != 2 or c.shape[0] != c.shape[1]:
        raise ValueError("corr must be a square matrix")
    m = c.shape[0]
    if m < 2:
        raise ValueError("need at least two trials")
    return float((c.sum() - np.trace(c)) / (m * (m - 1)))


def effective_number_of_trials(n_trials: float, avg_correlation: float) -> float:
    """Implied number of independent trials, BLdP14 eq. (9).

    ``N̂ = ρ̂ + (1 - ρ̂)·M``. A negative average correlation is clipped to 0,
    since the interpolation is only defined between ρ = 0 and ρ = 1.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1")
    if avg_correlation > 1:
        raise ValueError("avg_correlation cannot exceed 1")
    rho = max(avg_correlation, 0.0)
    return rho + (1.0 - rho) * n_trials


def deflated_sharpe_ratio(
    sr: float,
    n_obs: float,
    n_trials: float,
    sr_variance: float,
    skew: float = 0.0,
    kurt: float = 3.0,
    avg_correlation: float | None = None,
) -> float:
    """Deflated Sharpe ratio, BLdP14 eq. (2).

    The probabilistic Sharpe ratio of the selected strategy measured against
    ``SR0 = sqrt(V[SR])·E[max_N]``, the Sharpe ratio the best of ``n_trials``
    skill-less trials is expected to show.

    Parameters
    ----------
    sr
        Per-observation Sharpe ratio of the selected strategy.
    n_obs
        Number of return observations behind ``sr``.
    n_trials
        Number of trials the strategy was selected from.
    sr_variance
        Variance of the per-observation Sharpe ratios across the trials.
    skew, kurt
        Skewness and (non-excess) kurtosis of the selected strategy's returns.
    avg_correlation
        If given, ``n_trials`` is first reduced to the implied number of
        independent trials with :func:`effective_number_of_trials`.
    """
    n_eff = (
        n_trials
        if avg_correlation is None
        else effective_number_of_trials(n_trials, avg_correlation)
    )
    sr0 = expected_max_sharpe(n_eff, sr_variance)
    return probabilistic_sharpe_ratio(sr, n_obs, sr0, skew, kurt)


@dataclass(frozen=True)
class DeflatedSharpeResult:
    """Outcome of :func:`deflated_sharpe_ratio_from_returns`.

    All Sharpe ratios are per observation.
    """

    selected: int
    """Column index of the selected strategy."""
    sharpe: float
    """Its estimated Sharpe ratio."""
    skew: float
    kurt: float
    n_obs: int
    n_trials: int
    """Number of columns."""
    n_effective: float
    """Trials used in the expected maximum (equals ``n_trials`` unless adjusted)."""
    avg_correlation: float
    """Average pairwise correlation between the trials' returns."""
    sr_variance: float
    """Sample variance of the trials' Sharpe ratios."""
    expected_max_sharpe: float
    """SR0, the expected best Sharpe ratio among skill-less trials."""
    psr_zero: float
    """PSR against a benchmark of zero, ignoring the number of trials."""
    dsr: float
    """Deflated Sharpe ratio."""


def deflated_sharpe_ratio_from_returns(
    returns: ArrayLike,
    selected: int | None = None,
    adjust_for_correlation: bool = False,
) -> DeflatedSharpeResult:
    """Deflated Sharpe ratio of one column of a matrix of trial returns.

    Parameters
    ----------
    returns
        ``T × N`` array, one trial per column.
    selected
        Column to evaluate. Defaults to the column with the highest Sharpe ratio.
    adjust_for_correlation
        If true, replace N by ``ρ̂ + (1 - ρ̂)·N`` (BLdP14 eq. 9), with ρ̂ the
        average pairwise correlation of the columns.
    """
    r = np.asarray(returns, dtype=np.float64)
    if r.ndim != 2 or r.shape[1] < 2:
        raise ValueError("returns must be a T x N array with N >= 2")
    n_obs, n_trials = r.shape
    srs = sharpe_ratio(r)
    pick = int(np.argmax(srs)) if selected is None else int(selected)
    sr, skew, kurt, _ = return_moments(r[:, pick])
    sr_var = float(np.var(srs, ddof=1))
    rho = average_correlation(np.corrcoef(r, rowvar=False))
    n_eff = effective_number_of_trials(n_trials, rho) if adjust_for_correlation else float(n_trials)
    sr0 = expected_max_sharpe(n_eff, sr_var)
    return DeflatedSharpeResult(
        selected=pick,
        sharpe=sr,
        skew=skew,
        kurt=kurt,
        n_obs=n_obs,
        n_trials=n_trials,
        n_effective=n_eff,
        avg_correlation=rho,
        sr_variance=sr_var,
        expected_max_sharpe=sr0,
        psr_zero=probabilistic_sharpe_ratio(sr, n_obs, 0.0, skew, kurt),
        dsr=probabilistic_sharpe_ratio(sr, n_obs, sr0, skew, kurt),
    )


def min_backtest_length(n_trials: float, target_sharpe: float = 1.0) -> float:
    """Minimum backtest length in years, BBLZ14 eq. (6).

    The number of years below which the best of ``n_trials`` independent
    skill-less strategies is expected to show an annualised in-sample Sharpe
    ratio of ``target_sharpe`` or more.
    """
    if target_sharpe <= 0:
        raise ValueError("target_sharpe must be positive")
    return (expected_max_standard_normal(n_trials) / target_sharpe) ** 2


def min_backtest_length_upper_bound(n_trials: float, target_sharpe: float = 1.0) -> float:
    """The bound ``2·ln(N) / E[max_N]²`` on :func:`min_backtest_length`, BBLZ14 eq. (6)."""
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1")
    if target_sharpe <= 0:
        raise ValueError("target_sharpe must be positive")
    return 2.0 * math.log(n_trials) / target_sharpe**2
