"""Purged k-fold and combinatorial purged cross-validation."""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np
import pytest

from procrustes import CombinatorialPurgedKFold, PurgedKFold


def _random_labels(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Random label spans: irregular start times, each label lasting a random time."""
    start = np.cumsum(rng.exponential(1.0, n))
    end = start + rng.exponential(rng.uniform(0.5, 8.0), n)
    return start, end


def _contiguous_blocks(test: np.ndarray) -> list[np.ndarray]:
    breaks = np.flatnonzero(np.diff(test) != 1) + 1
    return np.split(test, breaks)


def _check_split(
    train: np.ndarray, test: np.ndarray, start: np.ndarray, end: np.ndarray, embargo: int
) -> None:
    n = start.size
    assert np.intersect1d(train, test).size == 0
    # 1. no training label interval overlaps any test label interval (brute force)
    overlap = (start[train][:, None] <= end[test][None, :]) & (
        start[test][None, :] <= end[train][:, None]
    )
    assert not overlap.any()
    # 2. the embargo: after each test block, the first `embargo` samples whose label
    #    starts after the block's last label ends are not trained on
    in_train = np.zeros(n, dtype=bool)
    in_train[train] = True
    allowed = np.ones(n, dtype=bool)
    allowed[test] = False
    for block in _contiguous_blocks(test):
        block_end = end[block].max()
        after = np.flatnonzero(start > block_end)
        assert not in_train[after[:embargo]].any()
        allowed[after[:embargo]] = False
        allowed &= (end < start[block].min()) | (start > block_end)
    # 3. nothing else was dropped: purging is not more aggressive than it must be
    assert np.array_equal(in_train, allowed)


@pytest.mark.parametrize("seed", range(40))
def test_purged_kfold_random_label_spans(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = int(rng.integers(40, 400))
    k = int(rng.integers(2, 11))
    embargo = int(rng.integers(0, 8))
    start, end = _random_labels(rng, n)
    cv = PurgedKFold(k, start, end, embargo=embargo)
    tested = []
    for train, test in cv.split(np.zeros((n, 3))):
        _check_split(train, test, start, end, embargo)
        tested.append(test)
    assert len(tested) == k == cv.get_n_splits()
    # the folds partition the sample, in order
    assert np.array_equal(np.concatenate(tested), np.arange(n))


@pytest.mark.parametrize("seed", range(40))
def test_combinatorial_random_label_spans(seed: int) -> None:
    rng = np.random.default_rng(1000 + seed)
    n = int(rng.integers(60, 300))
    n_groups = int(rng.integers(3, 8))
    k = int(rng.integers(1, n_groups))
    embargo = int(rng.integers(0, 6))
    start, end = _random_labels(rng, n)
    cv = CombinatorialPurgedKFold(n_groups, k, start, end, embargo=embargo)
    splits = list(cv.split(np.zeros(n)))
    assert len(splits) == math.comb(n_groups, k) == cv.get_n_splits()
    for train, test in splits:
        assert np.all(np.diff(test) > 0)
        _check_split(train, test, start, end, embargo)


def test_fractional_embargo() -> None:
    n = 200
    cv = PurgedKFold(4, embargo=0.05)
    idx = np.arange(n)
    for train, test in cv.split(idx):
        _check_split(train, test, idx, idx, 10)
    train, test = next(iter(cv.split(idx)))
    assert test.tolist() == list(range(50))
    assert train.tolist() == list(range(60, 200))


def test_default_labels_reduce_to_plain_kfold() -> None:
    n, k = 23, 4
    for train, test in PurgedKFold(k).split(np.zeros(n)):
        assert np.array_equal(np.sort(np.concatenate([train, test])), np.arange(n))


def test_datetime_labels() -> None:
    start = np.arange("2020-01-01", "2020-04-10", dtype="datetime64[D]")
    end = start + np.timedelta64(5, "D")
    cv = PurgedKFold(5, start, end, embargo=2)
    for train, test in cv.split(start):
        _check_split(train, test, start, end, 2)
        assert train.size < start.size - test.size  # something was purged


def test_overlapping_labels_purge_both_sides() -> None:
    """Ten samples, each label covering the next two steps; test the middle fold."""
    start = np.arange(9.0)
    end = start + 2
    splits = list(PurgedKFold(3, start, end).split(start))
    train, test = splits[1]
    assert test.tolist() == [3, 4, 5]
    # samples 1 and 2 end at 3 and 4, inside the test span [3, 7]; 6 and 7 start inside it
    assert train.tolist() == [0, 8]


@pytest.mark.parametrize(("n_groups", "k"), [(n, k) for n in range(2, 11) for k in range(1, n)])
def test_number_of_paths(n_groups: int, k: int) -> None:
    """φ[N, k] = (k/N)·C(N, k)."""
    cv = CombinatorialPurgedKFold(n_groups, k)
    assert cv.n_paths * n_groups == k * math.comb(n_groups, k)
    assert cv.get_n_splits() == math.comb(n_groups, k)
    table = cv.path_splits()
    assert table.shape == (cv.n_paths, n_groups)
    combos = cv.test_group_combinations
    for g in range(n_groups):
        expected = [s for s, c in enumerate(combos) if g in c]
        # each group's column lists exactly the splits that test it, each once
        assert table[:, g].tolist() == expected


def test_six_groups_two_tested_is_fifteen_splits_and_five_paths() -> None:
    """N = 6, k = 2: C(6, 2) = 15 splits and (2/6)·15 = 5 paths."""
    cv = CombinatorialPurgedKFold(6, 2)
    assert cv.get_n_splits() == 15
    assert cv.n_paths == 5
    assert cv.test_group_combinations == list(combinations(range(6), 2))
    assert cv.path_splits()[0].tolist() == [0, 0, 1, 2, 3, 4]


@pytest.mark.parametrize(("n_groups", "k", "n"), [(6, 2, 61), (5, 3, 100), (4, 1, 17), (7, 6, 50)])
def test_backtest_paths_cover_every_sample_once(n_groups: int, k: int, n: int) -> None:
    cv = CombinatorialPurgedKFold(n_groups, k)
    x = np.zeros(n)
    splits = list(cv.split(x))
    # "forecast" = the sample's own index: every path must come back as 0..n-1
    paths = cv.backtest_paths([test for _, test in splits], n)
    assert paths.shape == (cv.n_paths, n)
    assert np.array_equal(paths, np.tile(np.arange(n), (cv.n_paths, 1)))
    # "forecast" = the split number: each (split, group) pair is used exactly once
    by_split = cv.backtest_paths([np.full(test.size, s) for s, (_, test) in enumerate(splits)])
    bounds = cv.group_bounds(n)
    table = cv.path_splits()
    for g, (a, b) in enumerate(bounds):
        assert np.array_equal(by_split[:, a:b], np.repeat(table[:, [g]], b - a, axis=1))
        assert len(set(table[:, g].tolist())) == cv.n_paths


def test_backtest_paths_keep_trailing_dimensions() -> None:
    cv = CombinatorialPurgedKFold(4, 2)
    n = 20
    preds = [np.stack([test, -test], axis=1).astype(float) for _, test in cv.split(np.zeros(n))]
    paths = cv.backtest_paths(preds)
    assert paths.shape == (3, n, 2)
    assert np.array_equal(paths[2, :, 1], -np.arange(n))
    with pytest.raises(ValueError):
        cv.backtest_paths(preds[:-1])


def test_matches_documented_skfolio_example() -> None:
    """Cross-check against the output printed in skfolio's documentation for
    ``CombinatorialPurgedCV(n_folds=3, n_test_folds=2, purged_size=1)`` on 12 samples.
    A purge of one sample on each side is a label that reaches one step ahead."""
    start = np.arange(12)
    cv = CombinatorialPurgedKFold(3, 2, start, start + 1)
    trains = [train.tolist() for train, _ in cv.split(start)]
    assert trains == [[9, 10, 11], [5, 6], [0, 1, 2]]
    # and their embargo_size=1 example, with no purging
    cv = CombinatorialPurgedKFold(3, 2, embargo=1)
    trains = [train.tolist() for train, _ in cv.split(start)]
    assert trains == [[9, 10, 11], [5, 6, 7], [0, 1, 2, 3]]


def test_scikit_learn_accepts_the_splitters() -> None:
    model_selection = pytest.importorskip("sklearn.model_selection")
    linear_model = pytest.importorskip("sklearn.linear_model")
    rng = np.random.default_rng(0)
    x = rng.standard_normal((120, 3))
    y = x @ np.array([1.0, -2.0, 0.5]) + 0.1 * rng.standard_normal(120)
    start = np.arange(120)
    for cv in (
        PurgedKFold(5, start, start + 3, embargo=2),
        CombinatorialPurgedKFold(6, 2, start, start + 3, embargo=2),
    ):
        scores = model_selection.cross_val_score(linear_model.LinearRegression(), x, y, cv=cv)
        assert scores.size == cv.get_n_splits()
        assert scores.min() > 0.9
    search = model_selection.GridSearchCV(
        linear_model.Ridge(), {"alpha": [0.1, 1.0]}, cv=PurgedKFold(4, start, start + 3)
    )
    search.fit(x, y)
    assert search.best_score_ > 0.9


def test_validation() -> None:
    with pytest.raises(ValueError):
        PurgedKFold(1)
    with pytest.raises(ValueError):
        PurgedKFold(3, label_start=np.arange(5))
    with pytest.raises(ValueError):
        PurgedKFold(3, embargo=-1)
    with pytest.raises(ValueError):
        PurgedKFold(3, embargo=1.5)
    with pytest.raises(ValueError):
        CombinatorialPurgedKFold(4, 4)
    with pytest.raises(ValueError):
        CombinatorialPurgedKFold(4, 0)
    with pytest.raises(ValueError, match="ordered"):
        list(PurgedKFold(2, np.array([0, 2, 1, 3]), np.array([1, 3, 2, 4])).split(np.zeros(4)))
    with pytest.raises(ValueError, match="precede"):
        list(PurgedKFold(2, np.arange(4), np.arange(4) - 1).split(np.zeros(4)))
    with pytest.raises(ValueError, match="one entry per sample"):
        list(PurgedKFold(2, np.arange(4), np.arange(4)).split(np.zeros(5)))
    with pytest.raises(ValueError):
        list(PurgedKFold(5).split(np.zeros(3)))
