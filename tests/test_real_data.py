"""The real-data experiments: parser, rule family, null, leakage. No network."""

from __future__ import annotations

import math

import numpy as np
import pytest

import fetch_data
import leakage
import trend_family as tf
from procrustes import deflated_sharpe_ratio_from_returns, expected_max_standard_normal

FRENCH_SAMPLE = """This file was created by using the 202609 CRSP database.
The Tbill return is the simple daily rate.

,Mkt-RF,SMB,HML,RF
19260701,    0.09,   -0.23,   -0.28,    0.01
19260702,    0.45,   -0.34,   -0.03,    0.01
19260706,   -1.17,    0.29,   -0.38,    0.01

Copyright 2026 Eugene F. Fama and Kenneth R. French
"""


def test_parser_reads_percent_and_stops_at_end_date() -> None:
    m = fetch_data.parse_daily_factors(FRENCH_SAMPLE, end=20991231)
    assert m.dates.tolist() == [19260701, 19260702, 19260706]
    assert m.excess == pytest.approx([0.0009, 0.0045, -0.0117])
    cut = fetch_data.parse_daily_factors(FRENCH_SAMPLE, end=19260702)
    assert cut.dates.tolist() == [19260701, 19260702]
    with pytest.raises(ValueError):
        fetch_data.parse_daily_factors("no table here")


def test_trailing_mean_matches_a_loop() -> None:
    x = np.random.default_rng(0).standard_normal(50)
    for n in (1, 3, 10):
        got = tf.trailing_mean(x, n)
        assert np.all(np.isnan(got[: n - 1]))
        for t in range(n - 1, 50):
            assert got[t] == pytest.approx(x[t - n + 1 : t + 1].mean())


def test_rules_use_no_future_data() -> None:
    rng = np.random.default_rng(1)
    x = rng.standard_normal(tf.WARMUP + 200) * 0.01
    base = tf.rule_returns(x)
    k = tf.WARMUP + 120  # a day inside the return rows
    y = x.copy()
    y[k:] = rng.standard_normal(y.size - k) * 0.01
    moved = tf.rule_returns(y)
    row = k - tf.WARMUP
    # rows before day k are unchanged; on day k only the day's return differs
    assert np.array_equal(base[:row], moved[:row])
    assert np.array_equal(np.sign(base[row]) * np.sign(x[k]), np.sign(moved[row]) * np.sign(y[k]))


def test_rule_positions_follow_the_trend() -> None:
    x = np.full(tf.WARMUP + 10, 0.001)  # steady rise: every fast mean is above every slow one
    r = tf.rule_returns(x)
    assert r.shape == (10, len(tf.RULES))
    assert np.allclose(r, 0.001)
    assert np.allclose(tf.rule_returns(-x), 0.001)  # steady fall: short, and earns the fall


def test_family_verdict_matches_the_library() -> None:
    r = np.random.default_rng(2).standard_normal((400, 30)) * 0.01
    r[:, :10] += r[:, [0]]  # some correlation
    st = tf.family_stats(r)
    v = tf.verdict(st)
    for adjust, dsr in ((False, v.dsr_raw), (True, v.dsr_adj)):
        lib = deflated_sharpe_ratio_from_returns(r, adjust_for_correlation=adjust)
        assert lib.selected == v.best
        assert lib.dsr == pytest.approx(dsr, rel=1e-12)
        assert lib.n_effective == pytest.approx(st.n_effective if adjust else 30)


def test_planted_shift_equals_adding_drift() -> None:
    r = np.random.default_rng(3).standard_normal((500, 12)) * 0.01
    j, d = 5, 0.1
    shifted = r.copy()
    shifted[:, j] += d * r[:, j].std(ddof=1)
    a = tf.verdict(tf.family_stats(r), (j, d))
    b = tf.verdict(tf.family_stats(shifted))
    assert a.best == b.best
    assert a.dsr_raw == pytest.approx(b.dsr_raw, rel=1e-9)
    assert a.dsr_adj == pytest.approx(b.dsr_adj, rel=1e-9)


def test_windows_do_not_overlap_and_have_warm_up() -> None:
    w = tf.windows(tf.WARMUP + 3 * tf.WINDOW + 17)
    assert len(w) == 3
    assert w[-1][1] == tf.WARMUP + 3 * tf.WINDOW + 17
    assert all(b - a == tf.WINDOW for a, b in w)
    assert all(w[i][1] == w[i + 1][0] for i in range(len(w) - 1))
    assert w[0][0] >= tf.WARMUP
    assert len(tf.windows(10_000, 2)) == 2


def test_implied_trials_inverts_the_expected_maximum() -> None:
    for n in (5.0, 118.0, 1000.0):
        assert tf.implied_trials(0.3 * expected_max_standard_normal(n), 0.09) == pytest.approx(n)
    assert math.isnan(tf.implied_trials(-1.0, 1.0))


def test_sign_randomisation_removes_a_real_trend_edge() -> None:
    """On a series with a persistent drift the rules earn money; after random sign
    flips their pooled mean return is indistinguishable from zero."""
    rng = np.random.default_rng(4)
    n = tf.WARMUP + 3000
    drift = np.zeros(n)
    for t in range(1, n):
        drift[t] = 0.99 * drift[t - 1] + 0.0003 * rng.standard_normal()
    x = drift + rng.standard_normal(n) * 0.01
    real = tf.rule_returns(x).mean(axis=1)
    assert real.mean() / real.std(ddof=1) * math.sqrt(real.size) > 5
    pooled = []
    for _ in range(20):
        flipped = tf.rule_returns(x * rng.choice(np.array([-1.0, 1.0]), size=n))
        pooled.append(flipped.mean(axis=1))
    p = np.concatenate(pooled)
    # the 20 datasets are independent; the rows within one are close to independent
    assert abs(p.mean() / p.std(ddof=1) * math.sqrt(p.size)) < 4


def test_dataset_labels_and_features() -> None:
    x = np.random.default_rng(5).standard_normal(600) * 0.01
    feats, labels = leakage.make_dataset(x)
    first = max(leakage.LOOKBACKS) - 1
    assert labels.size == 600 - leakage.HORIZON - first
    i = 10  # sample i is day first + i
    day = first + i
    assert labels[i] == pytest.approx(x[day + 1 : day + 1 + leakage.HORIZON].sum())
    assert feats[i, 0] == pytest.approx(x[day - 4 : day + 1].sum())
    assert feats[i, -1] == pytest.approx(x[day - 19 : day + 1].std(), rel=1e-6)


def test_shuffled_cv_leaks_and_purged_cv_does_not() -> None:
    x = np.random.default_rng(6).standard_normal(2500) * 0.01
    feats, labels = leakage.make_dataset(x)
    assert leakage.shuffled_kfold_corr(feats, labels, seed=0) > 0.2
    purged = leakage.cpcv_score(feats, labels, purge=True)
    plain = leakage.cpcv_score(feats, labels, purge=False)
    assert abs(purged.path_corr.mean()) < 0.1
    assert purged.train_size < plain.train_size
