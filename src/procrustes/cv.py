"""Purged k-fold and combinatorial purged cross-validation.

The ideas are from López de Prado, Advances in Financial Machine Learning
(2018), chapters 7 and 12. I have not read the book. This module is my
implementation of the published idea; the choices it makes are listed in
PAPER.md section 7. No code was taken from any other implementation.

Both splitters follow the scikit-learn cross-validator protocol by duck
typing: ``split(X, y=None, groups=None)`` yields ``(train, test)`` index
arrays and ``get_n_splits`` returns the number of splits. scikit-learn is not
imported.

Each sample ``i`` has a label interval ``[label_start[i], label_end[i]]``: the
period whose information determines its label. Samples must be ordered by
``label_start``. Intervals are closed, so two labels that share an endpoint
overlap.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from itertools import combinations
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

IntArray = NDArray[np.intp]


def _n_samples(x: Any) -> int:
    shape = getattr(x, "shape", None)
    if shape is not None and len(shape) > 0:
        return int(shape[0])
    return len(x)


def purged_train_indices(
    label_start: NDArray[Any],
    label_end: NDArray[Any],
    test_blocks: Sequence[tuple[int, int]],
    embargo: int = 0,
) -> IntArray:
    """Training indices left after purging and embargoing around test blocks.

    Parameters
    ----------
    label_start, label_end
        Label interval of each sample, ordered by ``label_start``.
    test_blocks
        Half-open position ranges ``(a, b)``; samples ``a..b-1`` are tested.
    embargo
        Number of samples to drop after each block, counted from the first
        sample whose label starts after the block's last label ends.

    Returns
    -------
    numpy.ndarray
        Sorted positions of the samples that may be trained on.
    """
    n = label_start.shape[0]
    keep = np.ones(n, dtype=bool)
    for a, b in test_blocks:
        keep[a:b] = False
        t_lo = label_start[a]
        t_hi = label_end[a:b].max()
        keep &= (label_end < t_lo) | (label_start > t_hi)
        if embargo > 0:
            first_after = int(np.searchsorted(label_start, t_hi, side="right"))
            keep[first_after : first_after + embargo] = False
    return np.flatnonzero(keep)


class _PurgedSplitter:
    def __init__(
        self,
        label_start: ArrayLike | None = None,
        label_end: ArrayLike | None = None,
        embargo: int | float = 0,
    ) -> None:
        if (label_start is None) != (label_end is None):
            raise ValueError("give both label_start and label_end, or neither")
        if isinstance(embargo, bool) or embargo < 0:
            raise ValueError("embargo must be a non-negative int or a fraction in [0, 1)")
        if isinstance(embargo, float) and embargo >= 1:
            raise ValueError("a fractional embargo must be below 1; pass an int for a count")
        self.label_start = None if label_start is None else np.asarray(label_start)
        self.label_end = None if label_end is None else np.asarray(label_end)
        self.embargo = embargo

    def _labels(self, n: int) -> tuple[NDArray[Any], NDArray[Any]]:
        if self.label_start is None or self.label_end is None:
            idx = np.arange(n)
            return idx, idx
        start, end = self.label_start, self.label_end
        if start.shape != (n,) or end.shape != (n,):
            raise ValueError("label_start and label_end must have one entry per sample")
        if np.any(end < start):
            raise ValueError("label_end must not precede label_start")
        if np.any(start[1:] < start[:-1]):
            raise ValueError("samples must be ordered by label_start")
        return start, end

    def _embargo_count(self, n: int) -> int:
        if isinstance(self.embargo, float):
            return int(n * self.embargo)
        return int(self.embargo)


class PurgedKFold(_PurgedSplitter):
    """K-fold cross-validation without shuffling, with purging and an embargo.

    Each fold is one contiguous block of samples. Training samples whose label
    interval overlaps the test block's are removed (purging), and so are the
    ``embargo`` samples that follow the block.

    Parameters
    ----------
    n_splits
        Number of folds, at least 2.
    label_start, label_end
        Label interval of each sample (numbers or ``datetime64``), ordered by
        ``label_start``. If omitted, each label covers only its own sample, so
        nothing beyond the test fold is purged.
    embargo
        An int is a number of samples. A float in [0, 1) is a fraction of the
        number of samples, rounded down.
    """

    def __init__(
        self,
        n_splits: int = 5,
        label_start: ArrayLike | None = None,
        label_end: ArrayLike | None = None,
        embargo: int | float = 0,
    ) -> None:
        super().__init__(label_start, label_end, embargo)
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        self.n_splits = n_splits

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        """Number of splits. Arguments are ignored."""
        return self.n_splits

    def split(
        self, X: Any, y: Any = None, groups: Any = None
    ) -> Iterator[tuple[IntArray, IntArray]]:
        """Yield ``(train, test)`` index arrays, one pair per fold."""
        n = _n_samples(X)
        if n < self.n_splits:
            raise ValueError("more splits than samples")
        start, end = self._labels(n)
        h = self._embargo_count(n)
        for fold in np.array_split(np.arange(n), self.n_splits):
            a, b = int(fold[0]), int(fold[-1]) + 1
            yield purged_train_indices(start, end, [(a, b)], h), fold


class CombinatorialPurgedKFold(_PurgedSplitter):
    """Combinatorial purged cross-validation.

    The samples are cut into ``n_groups`` contiguous groups. Every combination
    of ``n_test_groups`` groups is a test set, giving ``C(N, k)`` splits. Since
    each group is tested in ``C(N-1, k-1)`` splits, the out-of-sample forecasts
    can be arranged into ``(k/N)·C(N, k)`` complete backtest paths; see
    :meth:`backtest_paths`.

    Parameters
    ----------
    n_groups
        Number of groups N, at least 2.
    n_test_groups
        Groups per test set k, with ``1 <= k < N``.
    label_start, label_end, embargo
        As for :class:`PurgedKFold`. Purging and the embargo are applied around
        each contiguous run of test groups.
    """

    def __init__(
        self,
        n_groups: int = 6,
        n_test_groups: int = 2,
        label_start: ArrayLike | None = None,
        label_end: ArrayLike | None = None,
        embargo: int | float = 0,
    ) -> None:
        super().__init__(label_start, label_end, embargo)
        if n_groups < 2:
            raise ValueError("n_groups must be at least 2")
        if not 1 <= n_test_groups < n_groups:
            raise ValueError("n_test_groups must satisfy 1 <= k < n_groups")
        self.n_groups = n_groups
        self.n_test_groups = n_test_groups

    @property
    def test_group_combinations(self) -> list[tuple[int, ...]]:
        """Test groups of each split, in the order :meth:`split` yields them."""
        return list(combinations(range(self.n_groups), self.n_test_groups))

    @property
    def n_paths(self) -> int:
        """Number of backtest paths, ``(k/N)·C(N, k) = C(N-1, k-1)``."""
        return math.comb(self.n_groups - 1, self.n_test_groups - 1)

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        """Number of splits, ``C(N, k)``. Arguments are ignored."""
        return math.comb(self.n_groups, self.n_test_groups)

    def group_bounds(self, n_samples: int) -> list[tuple[int, int]]:
        """Half-open position range ``(a, b)`` of each group."""
        if n_samples < self.n_groups:
            raise ValueError("more groups than samples")
        bounds = []
        for g in np.array_split(np.arange(n_samples), self.n_groups):
            bounds.append((int(g[0]), int(g[-1]) + 1))
        return bounds

    def split(
        self, X: Any, y: Any = None, groups: Any = None
    ) -> Iterator[tuple[IntArray, IntArray]]:
        """Yield ``(train, test)`` index arrays, one pair per combination.

        ``test`` lists the test groups in time order, so it is sorted.
        """
        n = _n_samples(X)
        bounds = self.group_bounds(n)
        start, end = self._labels(n)
        h = self._embargo_count(n)
        for combo in self.test_group_combinations:
            blocks: list[tuple[int, int]] = []
            for g in combo:
                a, b = bounds[g]
                if blocks and blocks[-1][1] == a:
                    blocks[-1] = (blocks[-1][0], b)
                else:
                    blocks.append((a, b))
            test = np.concatenate([np.arange(a, b) for a, b in blocks])
            yield purged_train_indices(start, end, blocks, h), test

    def path_splits(self) -> IntArray:
        """Which split supplies each group of each path.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_paths, n_groups)``. Entry ``[p, g]`` is the index of the
            ``p``-th split, in the order of :meth:`split`, whose test set
            contains group ``g``. Every (split, test group) pair appears once.
        """
        out = np.empty((self.n_paths, self.n_groups), dtype=np.intp)
        seen = [0] * self.n_groups
        for s, combo in enumerate(self.test_group_combinations):
            for g in combo:
                out[seen[g], g] = s
                seen[g] += 1
        return out

    def backtest_paths(
        self, predictions: Sequence[ArrayLike], n_samples: int | None = None
    ) -> NDArray[Any]:
        """Arrange per-split test forecasts into complete backtest paths.

        Parameters
        ----------
        predictions
            One array per split, in the order of :meth:`split`. Its first axis
            must line up with that split's ``test`` indices.
        n_samples
            Number of samples. Inferred from the predictions if omitted.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_paths, n_samples, ...)``. Each path holds one
            out-of-sample forecast for every sample.
        """
        combos = self.test_group_combinations
        if len(predictions) != len(combos):
            raise ValueError(f"expected {len(combos)} prediction arrays, got {len(predictions)}")
        preds = [np.asarray(p) for p in predictions]
        if n_samples is None:
            total = sum(p.shape[0] for p in preds)
            n_samples = total // self.n_paths
        bounds = self.group_bounds(n_samples)
        paths = np.empty((self.n_paths, n_samples, *preds[0].shape[1:]), dtype=preds[0].dtype)
        table = self.path_splits()
        for s, combo in enumerate(combos):
            expected = sum(bounds[g][1] - bounds[g][0] for g in combo)
            if preds[s].shape[0] != expected:
                raise ValueError(
                    f"split {s}: expected {expected} predictions, got {preds[s].shape[0]}"
                )
            offset = 0
            for g in combo:
                a, b = bounds[g]
                p = int(np.flatnonzero(table[:, g] == s)[0])
                paths[p, a:b] = preds[s][offset : offset + (b - a)]
                offset += b - a
        return paths
