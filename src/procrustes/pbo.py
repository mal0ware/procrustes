"""Probability of backtest overfitting by combinatorially symmetric cross-validation.

Source: Bailey, Borwein, López de Prado and Zhu (2017), "The Probability of
Backtest Overfitting", Algorithm 2.3 and section 3. See PAPER.md section 5.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.intp]
Metric = Callable[[FloatArray], FloatArray]

_CHUNK_CELLS = 2_000_000


@dataclass(frozen=True)
class PBOResult:
    """Outcome of :func:`cscv`. Arrays have one entry per combination."""

    pbo: float
    """Share of combinations where the in-sample best is below the out-of-sample median."""
    logits: FloatArray
    """λ_c = ln(ω / (1 - ω)) with ω the relative out-of-sample rank of the in-sample best."""
    selected: IntArray
    """Column chosen in sample in each combination."""
    is_performance: FloatArray
    """In-sample statistic of the selected column."""
    oos_performance: FloatArray
    """Out-of-sample statistic of the selected column."""
    degradation_slope: float
    """β in ``oos = α + β·is + ε`` across combinations (nan if undefined)."""
    degradation_intercept: float
    """α in the same regression."""
    prob_oos_loss: float
    """Share of combinations where the selected column's out-of-sample statistic is negative."""
    n_blocks: int
    n_combinations: int
    n_strategies: int
    rows_used: int
    """Rows kept after trimming to a multiple of ``n_blocks``."""


def n_cscv_combinations(n_blocks: int) -> int:
    """Number of in-sample/out-of-sample splits, ``C(S, S/2)`` (BBLZ17 eq. 2.3)."""
    _check_blocks(n_blocks)
    return math.comb(n_blocks, n_blocks // 2)


def _check_blocks(n_blocks: int) -> None:
    if n_blocks < 2 or n_blocks % 2 != 0:
        raise ValueError("n_blocks must be an even integer of at least 2")


def _sharpe_from_sums(s1: FloatArray, s2: FloatArray, n: int) -> FloatArray:
    mean = s1 / n
    var = (s2 - n * mean**2) / (n - 1)
    out = np.zeros_like(mean)
    ok = var > 1e-300
    np.divide(mean, np.sqrt(var, where=ok, out=np.ones_like(var)), out=out, where=ok)
    return out


def _rank_of_selected(oos: FloatArray, selected: IntArray) -> FloatArray:
    """Average rank (1 = worst, N = best) of each row's selected column."""
    rows = np.arange(oos.shape[0])
    value = oos[rows, selected][:, None]
    below = (oos < value).sum(axis=1)
    equal = (oos == value).sum(axis=1)  # includes the selected column itself
    return np.asarray(below + (equal + 1) / 2.0, dtype=np.float64)


def cscv(
    returns: ArrayLike,
    n_blocks: int = 16,
    metric: Metric | None = None,
) -> PBOResult:
    """Estimate the probability of backtest overfitting.

    The rows of ``returns`` are cut into ``n_blocks`` equal consecutive blocks.
    Every way of choosing half the blocks is used once as the in-sample set,
    with the other half as the out-of-sample set. In each split the column
    that is best in sample is ranked among all columns out of sample.

    Parameters
    ----------
    returns
        ``T × N`` array of returns, one trial per column, rows synchronous.
        It should hold every trial that was run, not only the good ones.
    n_blocks
        Even number of blocks S. The paper suggests 16, which gives 12,870
        splits. If T is not a multiple of S the earliest ``T mod S`` rows are
        dropped.
    metric
        Performance statistic: a function mapping a ``rows × N`` array to N
        values, higher being better. Defaults to the Sharpe ratio, which is
        computed from per-block sums and is much faster than a custom metric.

    Returns
    -------
    PBOResult

    Notes
    -----
    A logit of exactly zero (possible only for odd N) counts as half an
    overfit. Ties out of sample get the average rank.
    """
    _check_blocks(n_blocks)
    r = np.asarray(returns, dtype=np.float64)
    if r.ndim != 2:
        raise ValueError("returns must be a T x N array")
    t, n = r.shape
    if n < 2:
        raise ValueError("need at least two strategies")
    block_len = t // n_blocks
    half_rows = block_len * (n_blocks // 2)
    if half_rows < 2:
        raise ValueError("too few rows: each half needs at least 2 observations")
    used = block_len * n_blocks
    r = r[t - used :]
    blocks = r.reshape(n_blocks, block_len, n)

    combos = np.array(list(combinations(range(n_blocks), n_blocks // 2)), dtype=np.intp)
    n_comb = combos.shape[0]
    mask = np.zeros((n_comb, n_blocks), dtype=np.float64)
    mask[np.arange(n_comb)[:, None], combos] = 1.0

    selected = np.empty(n_comb, dtype=np.intp)
    is_perf = np.empty(n_comb)
    oos_perf = np.empty(n_comb)
    rank = np.empty(n_comb)

    if metric is None:
        b1 = blocks.sum(axis=1)
        b2 = (blocks**2).sum(axis=1)
        tot1 = b1.sum(axis=0)
        tot2 = b2.sum(axis=0)
        step = max(1, _CHUNK_CELLS // n)
        for lo in range(0, n_comb, step):
            hi = min(lo + step, n_comb)
            s1 = mask[lo:hi] @ b1
            s2 = mask[lo:hi] @ b2
            is_stat = _sharpe_from_sums(s1, s2, half_rows)
            oos_stat = _sharpe_from_sums(tot1 - s1, tot2 - s2, half_rows)
            _fill(lo, hi, is_stat, oos_stat, selected, is_perf, oos_perf, rank)
    else:
        all_blocks = np.arange(n_blocks)
        for c in range(n_comb):
            is_rows = blocks[combos[c]].reshape(half_rows, n)
            oos_rows = blocks[np.setdiff1d(all_blocks, combos[c])].reshape(half_rows, n)
            is_stat = np.asarray(metric(is_rows), dtype=np.float64)[None, :]
            oos_stat = np.asarray(metric(oos_rows), dtype=np.float64)[None, :]
            if is_stat.shape != (1, n) or oos_stat.shape != (1, n):
                raise ValueError("metric must return one value per column")
            _fill(c, c + 1, is_stat, oos_stat, selected, is_perf, oos_perf, rank)

    omega = rank / (n + 1.0)
    logits = np.log(omega / (1.0 - omega))
    # exact zero only when rank == (n + 1) / 2; compare ranks to avoid float noise
    mid = (n + 1) / 2.0
    pbo = float((np.sum(rank < mid) + 0.5 * np.sum(rank == mid)) / n_comb)

    if np.ptp(is_perf) > 0:
        slope, intercept = np.polyfit(is_perf, oos_perf, 1)
        slope_f, intercept_f = float(slope), float(intercept)
    else:
        slope_f, intercept_f = math.nan, math.nan

    return PBOResult(
        pbo=pbo,
        logits=logits,
        selected=selected,
        is_performance=is_perf,
        oos_performance=oos_perf,
        degradation_slope=slope_f,
        degradation_intercept=intercept_f,
        prob_oos_loss=float(np.mean(oos_perf < 0)),
        n_blocks=n_blocks,
        n_combinations=n_comb,
        n_strategies=n,
        rows_used=used,
    )


def _fill(
    lo: int,
    hi: int,
    is_stat: FloatArray,
    oos_stat: FloatArray,
    selected: IntArray,
    is_perf: FloatArray,
    oos_perf: FloatArray,
    rank: FloatArray,
) -> None:
    rows = np.arange(hi - lo)
    best = np.argmax(is_stat, axis=1)
    selected[lo:hi] = best
    is_perf[lo:hi] = is_stat[rows, best]
    oos_perf[lo:hi] = oos_stat[rows, best]
    rank[lo:hi] = _rank_of_selected(oos_stat, best)
