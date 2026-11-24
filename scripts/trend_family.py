"""A real family of correlated trials: moving-average rules on the US market.

Each trial is a long/short rule on the daily market excess return. At the
close of day t it compares a fast and a slow trailing mean of the log market
index (excess-return index) and holds +1 if the fast mean is above the slow
one, -1 if below. Its return on day t+1 is that position times the market
excess return on day t+1. The grid of (fast, slow) lengths is fixed below.
No trading costs are charged: the family is a test bed for the statistics, not
a strategy being proposed.

Null: each window's daily returns are multiplied by independent random signs,
the index is rebuilt from the flipped returns and every rule is re-run. A
rule's position on day t+1 uses only days up to t, and the sign on day t+1 is
independent of everything before it, so every rule has a true mean of exactly
zero. The size of each daily move, and so the volatility clustering and fat
tails of the real series, is kept. The rules stay as correlated with each
other as they are on the real data, because they still read the same kind of
index.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq

from procrustes import (
    average_correlation,
    deflated_sharpe_ratio,
    effective_number_of_trials,
    expected_max_sharpe,
    expected_max_standard_normal,
)

FloatArray = NDArray[np.float64]

FAST = (1, 2, 3, 5, 8, 10, 15, 20, 30, 40, 50, 75, 100)
SLOW = (10, 20, 30, 50, 75, 100, 125, 150, 200, 250, 300)
RULES: tuple[tuple[int, int], ...] = tuple((f, s) for s in SLOW for f in FAST if f < s)
WARMUP = max(SLOW)  # days of history needed before a rule's first position
WINDOW = 1260  # five years of daily returns, as in the simulated experiment
PERIODS = 252
PLANTED = (1.0, 2.0, 3.0)  # annualised true Sharpe ratios planted in the power runs


def trailing_mean(x: FloatArray, length: int) -> FloatArray:
    """Mean of x[t-length+1 .. t] at each t; NaN where the window is incomplete."""
    out = np.full(x.shape, np.nan)
    if length == 1:
        return x.astype(np.float64, copy=True)
    c = np.concatenate([[0.0], np.cumsum(x)])
    out[length - 1 :] = (c[length:] - c[:-length]) / length
    return out


def rule_returns(excess: FloatArray, rules: Sequence[tuple[int, int]] = RULES) -> FloatArray:
    """Daily returns of every rule, shape ``(len(excess) - WARMUP, len(rules))``.

    Row i is the return on day ``WARMUP + i``, earned with the position set at
    the close of day ``WARMUP + i - 1`` from data up to that day.
    """
    x = np.asarray(excess, dtype=np.float64)
    if x.size <= WARMUP:
        raise ValueError("series shorter than the warm-up")
    logp = np.cumsum(np.log1p(x))
    lengths = sorted({n for rule in rules for n in rule})
    means = {n: trailing_mean(logp, n) for n in lengths}
    nxt = x[WARMUP:]
    out = np.empty((nxt.size, len(rules)))
    for j, (f, s) in enumerate(rules):
        diff = means[f][WARMUP - 1 : -1] - means[s][WARMUP - 1 : -1]
        out[:, j] = np.sign(diff) * nxt
    return out


def column_moments(r: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Sharpe ratio (ddof 1), skewness and raw kurtosis of every column, as in
    :func:`procrustes.return_moments`."""
    mean = r.mean(axis=0)
    dev = r - mean
    m2 = np.mean(dev**2, axis=0)
    skew = np.mean(dev**3, axis=0) / m2**1.5
    kurt = np.mean(dev**4, axis=0) / m2**2
    sr = mean / r.std(axis=0, ddof=1)
    return sr, skew, kurt


@dataclass(frozen=True)
class FamilyStats:
    """What the deflated Sharpe ratio needs from one T x N matrix of trials."""

    n_obs: int
    sr: FloatArray
    skew: FloatArray
    kurt: FloatArray
    avg_corr: float

    @property
    def n_trials(self) -> int:
        return int(self.sr.size)

    @property
    def n_effective(self) -> float:
        return effective_number_of_trials(self.n_trials, self.avg_corr)


def family_stats(r: FloatArray) -> FamilyStats:
    sr, skew, kurt = column_moments(r)
    return FamilyStats(
        r.shape[0], sr, skew, kurt, average_correlation(np.corrcoef(r, rowvar=False))
    )


@dataclass(frozen=True)
class Verdict:
    best: int
    best_sr: float
    sr0_raw: float
    sr0_adj: float
    dsr_raw: float
    dsr_adj: float


def verdict(stats: FamilyStats, shift: tuple[int, float] | None = None) -> Verdict:
    """Deflated Sharpe ratio of the best trial, against N and against N̂.

    ``shift = (j, d)`` adds ``d`` to trial j's per-observation Sharpe ratio,
    which is what adding a constant daily drift of ``d`` standard deviations
    to its returns does. Skewness, kurtosis and correlations do not change
    under a constant shift.
    """
    sr = stats.sr
    if shift is not None:
        sr = sr.copy()
        sr[shift[0]] += shift[1]
    best = int(np.argmax(sr))
    var = float(np.var(sr, ddof=1))
    args = (stats.n_obs,)
    raw = deflated_sharpe_ratio(
        sr[best], *args, stats.n_trials, var, stats.skew[best], stats.kurt[best]
    )
    adj = deflated_sharpe_ratio(
        sr[best], *args, stats.n_effective, var, stats.skew[best], stats.kurt[best]
    )
    return Verdict(
        best=best,
        best_sr=float(sr[best]),
        sr0_raw=expected_max_sharpe(stats.n_trials, var),
        sr0_adj=expected_max_sharpe(stats.n_effective, var),
        dsr_raw=raw,
        dsr_adj=adj,
    )


def windows(n_days: int, n_windows: int | None = None) -> list[tuple[int, int]]:
    """Non-overlapping five-year windows ``(a, b)`` of return days, counted back
    from the last day. Each needs ``WARMUP`` earlier days for its signals."""
    out = []
    b = n_days
    while b - WINDOW - WARMUP >= 0:
        out.append((b - WINDOW, b))
        b -= WINDOW
        if n_windows is not None and len(out) == n_windows:
            break
    return out[::-1]


def implied_trials(sr0: float, sr_variance: float) -> float:
    """N such that ``sqrt(V)·E[max_N]`` equals ``sr0``: how many independent
    trials the family behaves like. NaN if no N between 1 and 10^7 fits."""
    target = sr0 / math.sqrt(sr_variance)

    def gap(n: float) -> float:
        return expected_max_standard_normal(n) - target

    if gap(1.0) > 0 or gap(1e7) < 0:
        return math.nan
    return float(brentq(gap, 1.0, 1e7, xtol=1e-6))


@dataclass
class WindowRun:
    a: int
    b: int
    real: FamilyStats
    real_verdict: Verdict
    null_best_sr: FloatArray
    null_sr_var: FloatArray
    null_avg_corr: FloatArray
    null_n_eff: FloatArray
    null_dsr_raw: FloatArray
    null_dsr_adj: FloatArray
    power_dsr_raw: FloatArray  # shape (levels, reps)
    power_dsr_adj: FloatArray
    power_hit: FloatArray  # planted trial selected


def run_window(
    excess: FloatArray, a: int, b: int, reps: int, rng: np.random.Generator
) -> WindowRun:
    seg = excess[a - WARMUP : b]
    real = family_stats(rule_returns(seg))
    n_levels = len(PLANTED)
    out = {k: np.empty(reps) for k in ("best", "var", "corr", "neff", "raw", "adj")}
    p_raw = np.empty((n_levels, reps))
    p_adj = np.empty((n_levels, reps))
    p_hit = np.empty((n_levels, reps))
    for i in range(reps):
        signs = rng.choice(np.array([-1.0, 1.0]), size=seg.size)
        st = family_stats(rule_returns(seg * signs))
        v = verdict(st)
        out["best"][i] = v.best_sr
        out["var"][i] = float(np.var(st.sr, ddof=1))
        out["corr"][i] = st.avg_corr
        out["neff"][i] = st.n_effective
        out["raw"][i] = v.dsr_raw
        out["adj"][i] = v.dsr_adj
        j = int(rng.integers(st.n_trials))
        for k, level in enumerate(PLANTED):
            pv = verdict(st, (j, level / math.sqrt(PERIODS)))
            p_raw[k, i] = pv.dsr_raw
            p_adj[k, i] = pv.dsr_adj
            p_hit[k, i] = float(pv.best == j)
    return WindowRun(
        a,
        b,
        real,
        verdict(real),
        out["best"],
        out["var"],
        out["corr"],
        out["neff"],
        out["raw"],
        out["adj"],
        p_raw,
        p_adj,
        p_hit,
    )
