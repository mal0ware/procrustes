"""Leakage through overlapping labels, and what purging and an embargo remove.

Each sample is a trading day i. Its label is the market excess return summed
over the next ``HORIZON`` days, i+1 to i+HORIZON, so the labels of
neighbouring days share all but one day. Its features are trailing market
returns over 5, 20, 60 and 250 days and the trailing 20-day volatility, all
known at the close of day i. The model is a k-nearest-neighbour regression.
Neighbouring days have nearly the same features and nearly the same label, so
a model trained on the days around a test day can look up most of that test
day's label.

Combinatorial purged cross-validation (6 groups, 2 test groups) is run twice
on the same data: once with every training sample kept, and once with
purging (label spans ``[i+1, i+HORIZON]``) and an embargo of ``HORIZON``
days. The score is the out-of-sample correlation between forecast and label
on each of the 5 rebuilt paths.

The same is repeated on sign-randomised copies of the series (each day's
return multiplied by an independent random sign). There the label is
independent of every feature, so the true correlation is zero and anything
above zero is leakage.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from procrustes import CombinatorialPurgedKFold

FloatArray = NDArray[np.float64]

HORIZON = 20
LOOKBACKS = (5, 20, 60, 250)
VOL_LOOKBACK = 20
N_NEIGHBOURS = 25
N_GROUPS, N_TEST_GROUPS = 6, 2


def make_dataset(excess: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Features and labels for every day with a full history and a full label."""
    x = np.asarray(excess, dtype=np.float64)
    c = np.concatenate([[0.0], np.cumsum(x)])
    c2 = np.concatenate([[0.0], np.cumsum(x**2)])
    first = max(*LOOKBACKS, VOL_LOOKBACK) - 1
    days = np.arange(first, x.size - HORIZON)
    feats = [c[days + 1] - c[days + 1 - n] for n in LOOKBACKS]
    m1 = (c[days + 1] - c[days + 1 - VOL_LOOKBACK]) / VOL_LOOKBACK
    m2 = (c2[days + 1] - c2[days + 1 - VOL_LOOKBACK]) / VOL_LOOKBACK
    feats.append(np.sqrt(np.maximum(m2 - m1**2, 0.0)))
    features = np.column_stack(feats)
    labels = c[days + 1 + HORIZON] - c[days + 1]
    return features, labels


@dataclass(frozen=True)
class LeakageResult:
    path_corr: FloatArray
    """Correlation of forecast and label on each rebuilt path."""
    edge_corr: float
    """The same correlation on test samples within ``HORIZON`` days of a
    training group, pooled over splits. Only these can see their own label."""
    n_edge: int
    """Number of such test samples, summed over splits."""
    train_size: float
    """Mean number of training samples per split."""


def _standardised_knn(
    features: FloatArray, labels: FloatArray, train: NDArray[np.intp], test: NDArray[np.intp]
) -> FloatArray:
    from sklearn.neighbors import KNeighborsRegressor

    mu = features[train].mean(axis=0)
    sd = features[train].std(axis=0)
    model = KNeighborsRegressor(n_neighbors=N_NEIGHBOURS)
    model.fit((features[train] - mu) / sd, labels[train])
    return np.asarray(model.predict((features[test] - mu) / sd), dtype=np.float64)


def cpcv_score(features: FloatArray, labels: FloatArray, purge: bool) -> LeakageResult:
    """Combinatorial CV with contiguous groups, with or without purging and embargo."""
    n = labels.size
    idx = np.arange(n)
    if purge:
        cv = CombinatorialPurgedKFold(
            N_GROUPS, N_TEST_GROUPS, label_start=idx + 1, label_end=idx + HORIZON, embargo=HORIZON
        )
    else:
        cv = CombinatorialPurgedKFold(N_GROUPS, N_TEST_GROUPS)
    forecasts = []
    sizes = []
    edge_f: list[FloatArray] = []
    edge_y: list[FloatArray] = []
    for train, test in cv.split(features):
        pred = _standardised_knn(features, labels, train, test)
        forecasts.append(pred)
        sizes.append(train.size)
        in_test = np.zeros(n, dtype=bool)
        in_test[test] = True
        # distance from each test sample to the nearest sample outside the test groups
        outside = np.flatnonzero(~in_test)
        pos = np.searchsorted(outside, test)
        left = np.where(pos > 0, test - outside[np.maximum(pos - 1, 0)], n)
        right = np.where(pos < outside.size, outside[np.minimum(pos, outside.size - 1)] - test, n)
        near = np.minimum(left, right) <= HORIZON
        edge_f.append(pred[near])
        edge_y.append(labels[test][near])
    paths = np.asarray(cv.backtest_paths(forecasts, n), dtype=np.float64)
    corr = np.array([np.corrcoef(p, labels)[0, 1] for p in paths])
    ef, ey = np.concatenate(edge_f), np.concatenate(edge_y)
    return LeakageResult(
        path_corr=corr,
        edge_corr=float(np.corrcoef(ef, ey)[0, 1]),
        n_edge=int(ef.size),
        train_size=float(np.mean(sizes)),
    )


def shuffled_kfold_corr(features: FloatArray, labels: FloatArray, seed: int) -> float:
    """Ordinary 5-fold CV with shuffled rows: every test day has its neighbours in
    the training set. Correlation of the pooled out-of-fold forecasts and labels."""
    n = labels.size
    order = np.random.default_rng(seed).permutation(n)
    pred = np.empty(n)
    for fold in np.array_split(order, 5):
        test = np.sort(fold)
        train = np.setdiff1d(np.arange(n), test)
        pred[test] = _standardised_knn(features, labels, train, test)
    return float(np.corrcoef(pred, labels)[0, 1])
