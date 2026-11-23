"""Multiple-testing adjustments for t-statistics: Bonferroni, Holm and BHY.

Source: Harvey, Liu and Zhu (2016), "... and the Cross-Section of Expected
Returns", section 4.4. See PAPER.md section 6, including one place where the
code follows the paper's rejection rule and not its printed recursion.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm
from scipy.stats import t as student_t

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
Method = Literal["bonferroni", "holm", "bhy", "bh"]


def p_values_from_t(
    t_stats: ArrayLike, df: float | None = None, two_sided: bool = True
) -> FloatArray:
    """P-values of t-statistics.

    Parameters
    ----------
    t_stats
        The t-statistics.
    df
        Degrees of freedom for a Student t reference distribution. ``None``
        uses the standard Normal, which is what HLZ16 Table 4 is consistent with.
    two_sided
        Two-sided p-values if true, otherwise the upper tail.
    """
    t = np.asarray(t_stats, dtype=np.float64)
    dist = norm if df is None else student_t(df)
    if two_sided:
        return np.asarray(2.0 * dist.sf(np.abs(t)), dtype=np.float64)
    return np.asarray(dist.sf(t), dtype=np.float64)


def _check_p(p_values: ArrayLike) -> FloatArray:
    p = np.asarray(p_values, dtype=np.float64)
    if p.ndim != 1 or p.size == 0:
        raise ValueError("p_values must be a non-empty 1-D array")
    if np.any((p < 0) | (p > 1)) or np.any(np.isnan(p)):
        raise ValueError("p_values must lie in [0, 1]")
    return p


def bonferroni(p_values: ArrayLike) -> FloatArray:
    """Bonferroni adjusted p-values, ``min(M·p, 1)`` (HLZ16 section 4.4.1)."""
    p = _check_p(p_values)
    return np.minimum(p * p.size, 1.0)


def holm(p_values: ArrayLike) -> FloatArray:
    """Holm adjusted p-values (HLZ16 section 4.4.2).

    For the ordered p-values, ``min(max_{j<=i} (M-j+1)·p_(j), 1)``. Returned
    in the original order.
    """
    p = _check_p(p_values)
    m = p.size
    order = np.argsort(p, kind="stable")
    stepped = (m - np.arange(m)) * p[order]
    adjusted = np.minimum(np.maximum.accumulate(stepped), 1.0)
    out = np.empty(m)
    out[order] = adjusted
    return out


def harmonic_number(m: int) -> float:
    """``c(M) = Σ_{j=1..M} 1/j`` (HLZ16 eq. 1)."""
    return float(np.sum(1.0 / np.arange(1, m + 1)))


def bhy(p_values: ArrayLike, independent: bool = False) -> FloatArray:
    """Benjamini-Hochberg-Yekutieli adjusted p-values (HLZ16 section 4.4.3).

    For the ordered p-values, ``min_{j>=i} (M·c(M)/j)·p_(j)``, capped at 1 and
    returned in the original order. A test is rejected at false discovery rate
    α exactly when its adjusted p-value is at most α.

    Parameters
    ----------
    p_values
        Unadjusted p-values.
    independent
        ``False`` uses ``c(M) = Σ 1/j``, valid under any dependence between
        the tests. ``True`` uses ``c(M) = 1``, the Benjamini-Hochberg procedure,
        valid under independence or positive dependence.
    """
    p = _check_p(p_values)
    m = p.size
    c = 1.0 if independent else harmonic_number(m)
    order = np.argsort(p, kind="stable")
    scaled = m * c / np.arange(1, m + 1) * p[order]
    adjusted = np.minimum(np.minimum.accumulate(scaled[::-1])[::-1], 1.0)
    out = np.empty(m)
    out[order] = adjusted
    return out


@dataclass(frozen=True)
class MultipleTestingResult:
    """Outcome of :func:`adjust_t_statistics`. Arrays are in the input order."""

    method: str
    alpha: float
    p_values: FloatArray
    """Unadjusted p-values."""
    adjusted_p_values: FloatArray
    reject: BoolArray
    """True where the null is rejected at level ``alpha`` after adjustment."""
    n_rejected: int
    t_hurdle: float
    """Absolute t-statistic a test had to reach.

    For Bonferroni, the t-ratio whose p-value is ``alpha / M``. For Holm and
    BHY, the smallest absolute t-statistic among the rejected tests, or nan if
    nothing was rejected.
    """


def adjust_t_statistics(
    t_stats: ArrayLike,
    method: Method = "bhy",
    alpha: float = 0.05,
    df: float | None = None,
    two_sided: bool = True,
) -> MultipleTestingResult:
    """Adjust a family of t-statistics for multiple testing.

    Parameters
    ----------
    t_stats
        One t-statistic per test. Include every test that was run.
    method
        ``"bonferroni"`` and ``"holm"`` control the family-wise error rate.
        ``"bhy"`` controls the false discovery rate under any dependence and
        ``"bh"`` under independence or positive dependence.
    alpha
        Significance level (FWER or FDR, depending on the method).
    df
        Degrees of freedom of the t-statistics; ``None`` for the Normal.
    two_sided
        Whether the p-values are two-sided.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    t = np.asarray(t_stats, dtype=np.float64)
    if t.ndim != 1 or t.size == 0:
        raise ValueError("t_stats must be a non-empty 1-D array")
    p = p_values_from_t(t, df=df, two_sided=two_sided)
    if method == "bonferroni":
        adjusted = bonferroni(p)
    elif method == "holm":
        adjusted = holm(p)
    elif method == "bhy":
        adjusted = bhy(p)
    elif method == "bh":
        adjusted = bhy(p, independent=True)
    else:
        raise ValueError(f"unknown method {method!r}")
    reject = adjusted <= alpha
    if method == "bonferroni":
        tail = alpha / t.size / (2.0 if two_sided else 1.0)
        dist = norm if df is None else student_t(df)
        hurdle = float(dist.isf(tail))
    elif reject.any():
        hurdle = float(np.min(np.abs(t[reject]) if two_sided else t[reject]))
    else:
        hurdle = math.nan
    return MultipleTestingResult(
        method=method,
        alpha=alpha,
        p_values=p,
        adjusted_p_values=adjusted,
        reject=reject,
        n_rejected=int(reject.sum()),
        t_hurdle=hurdle,
    )
