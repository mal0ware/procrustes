"""Bonferroni, Holm and BHY adjustments, including HLZ16 Table 4."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from procrustes import (
    adjust_t_statistics,
    bhy,
    bonferroni,
    harmonic_number,
    holm,
    p_values_from_t,
)

# Harvey, Liu and Zhu (2016), Table 4, panel A.
TABLE4_T = np.array([1.99, 2.63, 2.21, 3.43, 2.17, 2.64, 4.56, 5.34, 2.75, 2.49])
TABLE4_P = np.array([4.66, 0.85, 2.71, 0.05, 3.00, 0.84, 0.00, 0.00, 0.60, 1.28]) / 100


def _tests(reject: np.ndarray) -> set[int]:
    """1-based test labels, as in the paper."""
    return {int(i) + 1 for i in np.flatnonzero(reject)}


def test_table4_from_printed_p_values() -> None:
    """Using the p-values as printed: 3, 4 and 6 discoveries, the tests named in the text."""
    assert _tests(bonferroni(TABLE4_P) <= 0.05) == {4, 7, 8}
    assert _tests(holm(TABLE4_P) <= 0.05) == {4, 7, 8, 9}
    assert _tests(bhy(TABLE4_P) <= 0.05) == {2, 4, 6, 7, 8, 9}
    assert _tests(TABLE4_P <= 0.05) == set(range(1, 11))


def test_table4_thresholds() -> None:
    """Panels C and D print the step thresholds in percent."""
    m, alpha = 10, 0.05
    b = np.arange(1, m + 1)
    holm_thresholds = 100 * alpha / (m + 1 - b)
    np.testing.assert_allclose(
        holm_thresholds, [0.50, 0.56, 0.63, 0.71, 0.83, 1.00, 1.25, 1.67, 2.50, 5.00], atol=0.006
    )
    assert harmonic_number(m) == pytest.approx(2.93, abs=0.005)
    bhy_thresholds = 100 * b * alpha / (m * harmonic_number(m))
    np.testing.assert_allclose(
        bhy_thresholds, [0.17, 0.34, 0.51, 0.68, 0.85, 1.02, 1.19, 1.37, 1.54, 1.71], atol=0.006
    )


def test_table4_from_t_statistics() -> None:
    """From the t-statistics, Bonferroni and BHY match the paper; Holm does not.

    The printed p-value of test 6 (t = 2.64) is 0.84%, above Holm's step-5
    threshold of 0.8333%. The exact two-sided Normal p-value of 2.64 is 0.829%,
    just below it, so Holm keeps going and also rejects tests 6 and 2.
    See PAPER.md section 8.
    """
    p = p_values_from_t(TABLE4_T)
    np.testing.assert_allclose(100 * p[[0, 2, 4, 8, 9]], [4.66, 2.71, 3.00, 0.60, 1.28], atol=0.006)
    assert 100 * p[5] == pytest.approx(0.829, abs=5e-4)
    assert _tests(adjust_t_statistics(TABLE4_T, "bonferroni").reject) == {4, 7, 8}
    assert _tests(adjust_t_statistics(TABLE4_T, "bhy").reject) == {2, 4, 6, 7, 8, 9}
    assert _tests(adjust_t_statistics(TABLE4_T, "holm").reject) == {2, 4, 6, 7, 8, 9}


def test_t_hurdles() -> None:
    bon = adjust_t_statistics(TABLE4_T, "bonferroni")
    assert bon.t_hurdle == pytest.approx(norm.isf(0.05 / 10 / 2))
    assert bon.n_rejected == 3
    assert adjust_t_statistics(TABLE4_T, "bhy").t_hurdle == 2.63
    assert adjust_t_statistics([1.96], "bonferroni").t_hurdle == pytest.approx(1.96, abs=1e-3)
    nothing = adjust_t_statistics([0.5, 1.0, -0.2], "holm")
    assert nothing.n_rejected == 0
    assert np.isnan(nothing.t_hurdle)


def _holm_step_down(p: np.ndarray, alpha: float) -> np.ndarray:
    m = p.size
    order = np.argsort(p, kind="stable")
    reject = np.zeros(m, dtype=bool)
    for b, i in enumerate(order, start=1):
        if p[i] > alpha / (m + 1 - b):
            break
        reject[i] = True
    return reject


def _bhy_step_up(p: np.ndarray, alpha: float, c: float) -> np.ndarray:
    m = p.size
    order = np.argsort(p, kind="stable")
    k = 0
    for b, i in enumerate(order, start=1):
        if p[i] <= b * alpha / (m * c):
            k = b
    reject = np.zeros(m, dtype=bool)
    reject[order[:k]] = True
    return reject


@pytest.mark.parametrize("seed", range(25))
def test_adjusted_p_values_agree_with_the_step_procedures(seed: int) -> None:
    """HLZ16 states each method as a rejection rule and as an adjusted p-value."""
    rng = np.random.default_rng(seed)
    m = int(rng.integers(1, 60))
    p = rng.beta(0.3, 2.0, m)
    for alpha in (0.01, 0.05, 0.2):
        assert np.array_equal(bonferroni(p) <= alpha, p <= alpha / m)
        assert np.array_equal(holm(p) <= alpha, _holm_step_down(p, alpha))
        assert np.array_equal(bhy(p) <= alpha, _bhy_step_up(p, alpha, harmonic_number(m)))
        assert np.array_equal(bhy(p, independent=True) <= alpha, _bhy_step_up(p, alpha, 1.0))


@pytest.mark.parametrize("seed", range(10))
def test_ordering_between_methods(seed: int) -> None:
    rng = np.random.default_rng(100 + seed)
    p = rng.uniform(0, 0.2, 30)
    b, h, y, bh = bonferroni(p), holm(p), bhy(p), bhy(p, independent=True)
    assert np.all(h <= b + 1e-15)  # Holm rejects everything Bonferroni does
    assert np.all(bh <= y + 1e-15)
    assert np.all(bh <= h + 1e-15)
    for adjusted in (b, h, y, bh):
        assert np.all(adjusted >= p - 1e-15)
        assert np.all(adjusted <= 1.0)
        # adjustment preserves the order of the p-values
        assert np.all(np.diff(adjusted[np.argsort(p)]) >= -1e-15)


def test_single_test_is_unchanged() -> None:
    p = np.array([0.03])
    for adjusted in (bonferroni(p), holm(p), bhy(p), bhy(p, independent=True)):
        assert adjusted.tolist() == [0.03]


def test_bhy_departure_from_printed_recursion() -> None:
    """The largest p-value is scaled by c(M), in line with the paper's rejection rule."""
    p = np.array([0.001, 0.04])
    assert bhy(p)[1] == pytest.approx(0.04 * 1.5)
    assert bool(bhy(p)[1] <= 0.05) == bool(p[1] <= 0.05 / 1.5)


def test_p_values_from_t() -> None:
    assert p_values_from_t([1.96])[0] == pytest.approx(0.05, abs=1e-4)
    assert p_values_from_t([1.645], two_sided=False)[0] == pytest.approx(0.05, abs=1e-4)
    assert p_values_from_t([-2.0])[0] == p_values_from_t([2.0])[0]
    assert p_values_from_t([2.0], df=10)[0] > p_values_from_t([2.0])[0]


def test_validation() -> None:
    with pytest.raises(ValueError):
        bonferroni([])
    with pytest.raises(ValueError):
        holm([0.2, 1.2])
    with pytest.raises(ValueError):
        adjust_t_statistics([2.0], alpha=1.5)
    with pytest.raises(ValueError):
        adjust_t_statistics([2.0], method="sidak")  # type: ignore[arg-type]
