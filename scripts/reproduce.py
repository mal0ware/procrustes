"""Rebuild every figure and table in results/.

    uv run python scripts/reproduce.py            # full run, writes results/
    uv run python scripts/reproduce.py --fast     # small run for CI, writes results_fast/

The simulated experiments use fixed seeds. The real-data experiments use the
daily market excess return from Kenneth French's data library up to
2025-12-31 (downloaded by scripts/fetch_data.py into data/, not committed),
with fixed seeds for the random signs. Two runs on the same download give
identical tables.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import quad
from scipy.stats import norm

from procrustes import (
    CombinatorialPurgedKFold,
    PBOResult,
    adjust_t_statistics,
    cscv,
    deflated_sharpe_ratio,
    deflated_sharpe_ratio_from_returns,
    expected_max_sharpe,
    expected_max_standard_normal,
    harmonic_number,
    min_backtest_length,
    min_track_record_length,
    n_cscv_combinations,
    p_values_from_t,
    probabilistic_sharpe_ratio,
    sharpe_ratio,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fetch_data
import leakage
import trend_family as tf

FloatArray = NDArray[np.float64]

SEED = 20261123
PERIODS = 252  # trading days per year
N_OBS = 5 * PERIODS  # five years of daily returns
N_STRATEGIES = 1000
DAILY_VOL = 0.01
CV_GROUPS, CV_TEST_GROUPS = 6, 2
CONTROL_SHARPE = 2.0  # annualised true Sharpe ratio of the one real strategy in the control


# --------------------------------------------------------------------------------------
# the experiment
# --------------------------------------------------------------------------------------


def simulate(
    rng: np.random.Generator, true_sharpe: float, n_obs: int, n_strategies: int
) -> FloatArray:
    """iid Normal daily returns. Column 0 has an annualised true Sharpe ratio of
    ``true_sharpe``; every other column has none."""
    r = rng.standard_normal((n_obs, n_strategies)) * DAILY_VOL
    r[:, 0] += true_sharpe / math.sqrt(PERIODS) * DAILY_VOL
    return r


def cv_path_returns(returns: FloatArray) -> FloatArray:
    """Out-of-sample return paths of the rule "trade the strategy with the best
    Sharpe ratio on the training set", under combinatorial CV with a 5-day embargo.

    Each sample's label is one day's return, so no two labels overlap and purging
    removes nothing here. Only the embargo drops training days. The demonstration
    of purging is in ``leakage_demo``."""
    cv = CombinatorialPurgedKFold(CV_GROUPS, CV_TEST_GROUPS, embargo=5)
    forecasts = []
    for train, test in cv.split(returns):
        pick = int(np.argmax(sharpe_ratio(returns[train])))
        forecasts.append(returns[test, pick])
    return np.asarray(cv.backtest_paths(forecasts, returns.shape[0]), dtype=np.float64)


def evaluate(returns: FloatArray, n_blocks: int) -> tuple[dict[str, float], PBOResult, FloatArray]:
    """Run every tool on one matrix of trial returns."""
    n_obs, n_strategies = returns.shape
    ann = math.sqrt(PERIODS)
    dsr = deflated_sharpe_ratio_from_returns(returns)
    srs = sharpe_ratio(returns)
    t_stats = srs * math.sqrt(n_obs)
    best = dsr.selected
    pbo = cscv(returns, n_blocks)
    paths = cv_path_returns(returns)
    path_sharpes = sharpe_ratio(paths, axis=1) * ann
    best_ann = dsr.sharpe * ann
    row = {
        "selected_is_real": float(best == 0),
        "best_sharpe_ann": best_ann,
        "t_stat": float(t_stats[best]),
        "p_single_test": float(p_values_from_t(t_stats[[best]])[0]),
        "psr_zero": dsr.psr_zero,
        "expected_max_sharpe_ann": dsr.expected_max_sharpe * ann,
        "dsr": dsr.dsr,
        "min_backtest_years": min_backtest_length(n_strategies, best_ann)
        if best_ann > 0
        else math.inf,
        "pbo": pbo.pbo,
        "prob_oos_loss": pbo.prob_oos_loss,
        "degradation_slope": pbo.degradation_slope,
        "p_bonferroni": float(adjust_t_statistics(t_stats, "bonferroni").adjusted_p_values[best]),
        "p_holm": float(adjust_t_statistics(t_stats, "holm").adjusted_p_values[best]),
        "p_bhy": float(adjust_t_statistics(t_stats, "bhy").adjusted_p_values[best]),
        "cv_path_sharpe_mean_ann": float(path_sharpes.mean()),
        "cv_path_sharpe_min_ann": float(path_sharpes.min()),
        "cv_path_sharpe_max_ann": float(path_sharpes.max()),
    }
    return row, pbo, paths


# --------------------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------------------


def fmt(x: float) -> str:
    if isinstance(x, float) and math.isinf(x):
        return "inf"
    return f"{x:.6g}"


def write_csv(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    lines = [",".join(header)]
    for row in rows:
        lines.append(",".join(fmt(v) if isinstance(v, float) else str(v) for v in row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def write_md(path: Path, header: list[str], rows: list[list[str]]) -> None:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


HEADLINE_LABELS = [
    ("selected_is_real", "Selected strategy is the real one", "{:.0f}"),
    ("best_sharpe_ann", "Annualised Sharpe ratio of the selected strategy", "{:.2f}"),
    ("t_stat", "Its t-statistic", "{:.2f}"),
    ("p_single_test", "p-value as if it were the only test", "{:.2g}"),
    ("psr_zero", "Probabilistic Sharpe ratio, benchmark 0", "{:.4f}"),
    (
        "expected_max_sharpe_ann",
        "Expected best Sharpe ratio if every trial is skill-less",
        "{:.2f}",
    ),
    ("dsr", "Deflated Sharpe ratio", "{:.3f}"),
    ("min_backtest_years", "Minimum backtest length for this Sharpe ratio, years", "{:.1f}"),
    ("pbo", "Probability of backtest overfitting", "{:.3f}"),
    ("prob_oos_loss", "Share of CSCV splits with a negative out-of-sample Sharpe", "{:.3f}"),
    ("p_bonferroni", "Bonferroni adjusted p-value", "{:.3g}"),
    ("p_holm", "Holm adjusted p-value", "{:.3g}"),
    ("p_bhy", "BHY adjusted p-value", "{:.3g}"),
    (
        "cv_path_sharpe_mean_ann",
        "Mean annualised Sharpe ratio of the 5 combinatorial CV paths",
        "{:.2f}",
    ),
    ("cv_path_sharpe_min_ann", "Lowest combinatorial CV path Sharpe ratio", "{:.2f}"),
    ("cv_path_sharpe_max_ann", "Highest combinatorial CV path Sharpe ratio", "{:.2f}"),
]


def headline(out: Path, n_strategies: int, n_blocks: int) -> dict[str, Any]:
    rng_noise, rng_control = (
        np.random.default_rng(s) for s in np.random.SeedSequence(SEED).spawn(2)
    )
    noise = simulate(rng_noise, 0.0, N_OBS, n_strategies)
    control = simulate(rng_control, CONTROL_SHARPE, N_OBS, n_strategies)
    row_n, pbo_n, paths_n = evaluate(noise, n_blocks)
    row_c, pbo_c, _ = evaluate(control, n_blocks)
    keys = [k for k, _, _ in HEADLINE_LABELS] + ["degradation_slope"]
    write_csv(
        out / "headline.csv",
        ["metric", "pure_noise", "one_real_strategy"],
        [[k, row_n[k], row_c[k]] for k in keys],
    )
    write_md(
        out / "headline.md",
        ["", f"{n_strategies:,} strategies, no edge", "Same, plus one with a true Sharpe of 2"],
        [[label, f.format(row_n[k]), f.format(row_c[k])] for k, label, f in HEADLINE_LABELS],
    )
    return {
        "noise": noise,
        "row_noise": row_n,
        "row_control": row_c,
        "pbo_noise": pbo_n,
        "pbo_control": pbo_c,
        "paths_noise": paths_n,
    }


MC_FLAGS: list[tuple[str, Callable[[dict[str, float]], bool]]] = [
    ("single_test_p<0.05", lambda r: r["p_single_test"] < 0.05),
    ("psr_zero>0.95", lambda r: r["psr_zero"] > 0.95),
    ("dsr>0.95", lambda r: r["dsr"] > 0.95),
    ("pbo<0.05", lambda r: r["pbo"] < 0.05),
    ("bonferroni_p<0.05", lambda r: r["p_bonferroni"] < 0.05),
    ("holm_p<0.05", lambda r: r["p_holm"] < 0.05),
    ("bhy_p<0.05", lambda r: r["p_bhy"] < 0.05),
]
MC_MEANS = [
    "selected_is_real",
    "best_sharpe_ann",
    "expected_max_sharpe_ann",
    "psr_zero",
    "dsr",
    "pbo",
    "prob_oos_loss",
    "cv_path_sharpe_mean_ann",
]


def monte_carlo(
    out: Path, levels: list[float], reps: int, n_strategies: int, n_blocks: int
) -> None:
    """Repeat the whole experiment: how often does each tool call the winner real?"""
    seeds = np.random.SeedSequence(SEED + 1).spawn(len(levels))
    header = ["true_sharpe_ann", "reps"]
    header += [f"mean_{k}" for k in MC_MEANS] + ["sd_best_sharpe_ann", "sd_dsr", "sd_pbo"]
    header += [f"share_{name}" for name, _ in MC_FLAGS]
    rows: list[list[Any]] = []
    for level, seed in zip(levels, seeds, strict=True):
        rng = np.random.default_rng(seed)
        runs = [
            evaluate(simulate(rng, level, N_OBS, n_strategies), n_blocks)[0] for _ in range(reps)
        ]
        row: list[Any] = [level, reps]
        row += [float(np.mean([r[k] for r in runs])) for k in MC_MEANS]
        row += [
            float(np.std([r[k] for r in runs], ddof=1)) for k in ("best_sharpe_ann", "dsr", "pbo")
        ]
        row += [float(np.mean([flag(r) for r in runs])) for _, flag in MC_FLAGS]
        rows.append(row)
        print(f"  monte carlo: true Sharpe {level:g} done ({reps} runs)", flush=True)
    write_csv(out / "monte_carlo.csv", header, rows)
    md_rows = []
    for row in rows:
        d = dict(zip(header, row, strict=True))
        md_rows.append(
            [
                f"{d['true_sharpe_ann']:g}",
                f"{d['mean_best_sharpe_ann']:.2f}",
                f"{d['mean_selected_is_real']:.0%}",
                f"{d['share_single_test_p<0.05']:.0%}",
                f"{d['share_psr_zero>0.95']:.0%}",
                f"{d['share_dsr>0.95']:.0%}",
                f"{d['share_bonferroni_p<0.05']:.0%}",
                f"{d['share_bhy_p<0.05']:.0%}",
                f"{d['mean_pbo']:.2f} ({d['sd_pbo']:.2f})",
                f"{d['share_pbo<0.05']:.0%}",
                f"{d['mean_cv_path_sharpe_mean_ann']:.2f}",
            ]
        )
    write_md(
        out / "monte_carlo.md",
        [
            "True Sharpe of the one real strategy",
            "Mean Sharpe of the winner",
            "Winner is the real one",
            "Single test p < 0.05",
            "PSR(0) > 0.95",
            "DSR > 0.95",
            "Bonferroni p < 0.05",
            "BHY p < 0.05",
            "PBO mean (sd)",
            "PBO < 0.05",
            "Mean combinatorial CV path Sharpe",
        ],
        md_rows,
    )


def pbo_sampling(out: Path, reps: int) -> None:
    """Sampling distribution of the PBO estimate, used to set the test tolerances."""
    n_strategies, n_obs, n_blocks = 50, 1000, 10
    drifts = [0.0, 0.1, 0.2, 0.3]
    rows: list[list[Any]] = []
    for drift, seed in zip(
        drifts, np.random.SeedSequence(SEED + 2).spawn(len(drifts)), strict=True
    ):
        rng = np.random.default_rng(seed)
        est = np.empty(reps)
        for i in range(reps):
            r = rng.standard_normal((n_obs, n_strategies))
            r[:, 0] += drift
            est[i] = cscv(r, n_blocks).pbo
        q = np.quantile(est, [0.025, 0.975])
        rows.append(
            [
                drift,
                reps,
                float(est.mean()),
                float(est.std(ddof=1)),
                float(est.min()),
                float(q[0]),
                float(q[1]),
                float(est.max()),
            ]
        )
    write_csv(
        out / "pbo_sampling.csv",
        [
            "per_obs_sharpe_of_strategy_0",
            "reps",
            "mean_pbo",
            "sd_pbo",
            "min",
            "q2.5",
            "q97.5",
            "max",
        ],
        rows,
    )


def exact_expected_max(n: int) -> float:
    value, _ = quad(lambda x: x * n * norm.pdf(x) * norm.cdf(x) ** (n - 1), -12, 12, limit=400)
    return float(value)


def paper_examples(out: Path) -> None:
    """Every worked number the sources print, next to what this library computes."""
    sr = 2.5 / math.sqrt(250)
    var = 0.5 / 250
    t4 = np.array([1.99, 2.63, 2.21, 3.43, 2.17, 2.64, 4.56, 5.34, 2.75, 2.49])
    p4 = np.array([4.66, 0.85, 2.71, 0.05, 3.00, 0.84, 0.00, 0.00, 0.60, 1.28]) / 100

    def n_rej(method: str) -> int:
        return adjust_t_statistics(t4, method).n_rejected  # type: ignore[arg-type]

    from procrustes import bhy, bonferroni, holm

    def trl(q: int, skew: float = 0.0, kurt: float = 3.0) -> float:
        return min_track_record_length(2 / math.sqrt(q), 1 / math.sqrt(q), skew, kurt) / q

    rows: list[tuple[str, str, str, float]] = [
        (
            "BLdP12 s.3",
            "PSR(0), monthly SR 0.458, 24 obs, Normal",
            "0.982",
            probabilistic_sharpe_ratio(0.458, 24),
        ),
        (
            "BLdP12 s.3",
            "same with skew -2.448, kurtosis 10.164",
            "0.913",
            probabilistic_sharpe_ratio(0.458, 24, 0, -2.448, 10.164),
        ),
        (
            "BLdP12 s.3",
            "same with 36 observations",
            "0.953",
            probabilistic_sharpe_ratio(0.458, 36, 0, -2.448, 10.164),
        ),
        ("BLdP12 s.5", "MinTRL years, SR 2 vs 1, daily (250/yr)", "2.73", trl(250)),
        ("BLdP12 s.5", "MinTRL years, weekly", "2.83", trl(52)),
        ("BLdP12 s.5", "MinTRL years, monthly", "3.24", trl(12)),
        (
            "BLdP12 s.5",
            "MinTRL years, monthly, skew -0.72, kurtosis 5.78",
            "4.99",
            trl(12, -0.72, 5.78),
        ),
        ("BLdP12 A.3", "the same in months", "59.895", 12 * trl(12, -0.72, 5.78)),
        (
            "BLdP14 example",
            "SR0, N = 100, V = 1/2 annualised",
            "0.1132",
            expected_max_sharpe(100, var),
        ),
        (
            "BLdP14 example",
            "DSR, N = 100, skew -3, kurtosis 10",
            "0.9004",
            deflated_sharpe_ratio(sr, 1250, 100, var, -3, 10),
        ),
        (
            "BLdP14 example",
            "DSR, N = 46",
            "0.9505",
            deflated_sharpe_ratio(sr, 1250, 46, var, -3, 10),
        ),
        (
            "BLdP14 example",
            "DSR, N = 88, Normal returns",
            "0.9505",
            deflated_sharpe_ratio(sr, 1250, 88, var, 0, 3),
        ),
        ("BBLZ14 p.462", "E[max], N = 10", "1.57", expected_max_standard_normal(10)),
        ("BBLZ14 p.463", "E[max], N = 128", "above 2.6", expected_max_standard_normal(128)),
        ("BBLZ14 Fig. 2", "MinBTL years, N = 45, target SR 1", "5", min_backtest_length(45)),
        ("BBLZ14 Fig. 2", "MinBTL years, N = 7, target SR 1", "2", min_backtest_length(7)),
        (
            "BLdP14 Exh. 3.1",
            "E[max] formula minus exact, N = 1000",
            "0.006",
            expected_max_standard_normal(1000) - exact_expected_max(1000),
        ),
        (
            "BLdP14 Exh. 3.1",
            "E[max] formula minus exact, N = 10",
            "under 0.05",
            expected_max_standard_normal(10) - exact_expected_max(10),
        ),
        ("BBLZ17 eq. 2.3", "CSCV splits, S = 16", "12,780", float(n_cscv_combinations(16))),
        ("BBLZ17 s.4", "CSCV splits, S = 12", "924", float(n_cscv_combinations(12))),
        ("BBLZ17 s.4", "CSCV splits, S = 24", "2,704,156", float(n_cscv_combinations(24))),
        ("HLZ16 Table 4", "c(10)", "2.93", harmonic_number(10)),
        (
            "HLZ16 Table 4",
            "Bonferroni discoveries, printed p-values",
            "3",
            float((bonferroni(p4) <= 0.05).sum()),
        ),
        (
            "HLZ16 Table 4",
            "Holm discoveries, printed p-values",
            "4",
            float((holm(p4) <= 0.05).sum()),
        ),
        ("HLZ16 Table 4", "BHY discoveries, printed p-values", "6", float((bhy(p4) <= 0.05).sum())),
        (
            "HLZ16 Table 4",
            "Bonferroni discoveries, from the t-statistics",
            "3",
            float(n_rej("bonferroni")),
        ),
        ("HLZ16 Table 4", "Holm discoveries, from the t-statistics", "4", float(n_rej("holm"))),
        ("HLZ16 Table 4", "BHY discoveries, from the t-statistics", "6", float(n_rej("bhy"))),
    ]
    write_csv(
        out / "paper_examples.csv",
        ["source", "quantity", "paper", "computed"],
        [[s, q.replace(",", ";"), p.replace(",", ""), v] for s, q, p, v in rows],
    )
    write_md(
        out / "paper_examples.md",
        ["Source", "Quantity", "Paper", "This library"],
        [[s, q, p, f"{v:,.0f}" if v > 1000 else f"{v:.5g}"] for s, q, p, v in rows],
    )


# --------------------------------------------------------------------------------------
# real data: a correlated family of trend rules, and leakage through overlapping labels
# --------------------------------------------------------------------------------------


def data_fingerprint(out: Path, m: fetch_data.MarketSeries) -> None:
    text = "".join(f"{d},{x * 100:.2f}\n" for d, x in zip(m.dates, m.excess, strict=True))
    write_csv(
        out / "data_fingerprint.csv",
        ["first_date", "last_date", "n_days", "sum_excess_pct", "sha256_of_date_value_lines"],
        [
            [
                int(m.dates[0]),
                int(m.dates[-1]),
                int(m.dates.size),
                f"{float(np.round(m.excess * 100, 2).sum()):.2f}",
                hashlib.sha256(text.encode()).hexdigest(),
            ]
        ],
    )


def _share_above(x: FloatArray, level: float = 0.95) -> float:
    return float(np.mean(x > level))


FAMILY_HEADER = [
    "start",
    "end",
    "n_rules",
    "real_avg_corr",
    "real_n_effective",
    "real_best_rule",
    "real_best_sharpe_ann",
    "real_sr0_raw_ann",
    "real_sr0_adj_ann",
    "real_dsr_raw",
    "real_dsr_adj",
    "null_reps",
    "null_avg_corr_mean",
    "null_n_effective_mean",
    "null_best_sharpe_ann_mean",
    "null_sr0_raw_ann_mean",
    "null_sr0_adj_ann_mean",
    "null_implied_trials",
    "size_raw",
    "size_adj",
    "market_lag1_autocorr",
]


def _window_row(m: fetch_data.MarketSeries, r: tf.WindowRun, reps: int) -> list[Any]:
    ann = math.sqrt(PERIODS)
    f, sl = tf.RULES[r.real_verdict.best]
    sd_mean = float(np.mean(np.sqrt(r.null_sr_var)))
    sr0_raw = [expected_max_sharpe(len(tf.RULES), v) for v in r.null_sr_var]
    sr0_adj = [expected_max_sharpe(n, v) for n, v in zip(r.null_n_eff, r.null_sr_var, strict=True)]
    return [
        int(m.dates[r.a]),
        int(m.dates[r.b - 1]),
        len(tf.RULES),
        r.real.avg_corr,
        r.real.n_effective,
        f"{f}/{sl}",
        r.real_verdict.best_sr * ann,
        r.real_verdict.sr0_raw * ann,
        r.real_verdict.sr0_adj * ann,
        r.real_verdict.dsr_raw,
        r.real_verdict.dsr_adj,
        reps,
        float(r.null_avg_corr.mean()),
        float(r.null_n_eff.mean()),
        float(r.null_best_sr.mean()) * ann,
        float(np.mean(sr0_raw)) * ann,
        float(np.mean(sr0_adj)) * ann,
        tf.implied_trials(float(r.null_best_sr.mean()), sd_mean**2),
        _share_above(r.null_dsr_raw),
        _share_above(r.null_dsr_adj),
        float(np.corrcoef(m.excess[r.a : r.b - 1], m.excess[r.a + 1 : r.b])[0, 1]),
    ]


def trend_family(out: Path, m: fetch_data.MarketSeries, fast: bool) -> dict[str, Any]:
    """Size and power of the deflated Sharpe ratio on a real correlated family,
    with N = number of rules and with N̂ from BLdP14 eq. (9)."""
    ann = math.sqrt(PERIODS)
    wins = tf.windows(m.excess.size, 2 if fast else None)
    reps = 10 if fast else 500
    seeds = np.random.SeedSequence(SEED + 3).spawn(len(wins))
    runs = []
    for (a, b), seed in zip(wins, seeds, strict=True):
        runs.append(tf.run_window(m.excess, a, b, reps, np.random.default_rng(seed)))
        print(f"  trend family: window ending {m.dates[b - 1]} done ({reps} datasets)", flush=True)
    rows = [_window_row(m, r, reps) for r in runs]
    write_csv(out / "family_windows.csv", FAMILY_HEADER, rows)

    # the whole span of the windows, real data only
    a0, b0 = runs[0].a, runs[-1].b
    whole = tf.family_stats(tf.rule_returns(m.excess[a0 - tf.WARMUP : b0]))
    wv = tf.verdict(whole)

    # Pooled size and power. The oracle threshold is the 95% quantile of the null
    # DSR (with N) from the first half of each window's datasets; it is applied to
    # the second half.
    half = reps // 2
    null_raw = np.concatenate([r.null_dsr_raw for r in runs])
    null_adj = np.concatenate([r.null_dsr_adj for r in runs])
    thresholds = [float(np.quantile(r.null_dsr_raw[:half], 0.95)) for r in runs]

    def oracle(values: list[FloatArray]) -> float:
        hits = [v[half:] > t for v, t in zip(values, thresholds, strict=True)]
        return float(np.mean(np.concatenate(hits)))

    p_header = [
        "planted_sharpe_ann",
        "datasets",
        "planted_rule_selected",
        "pass_raw",
        "pass_adj",
        "pass_oracle",
    ]
    p_rows: list[list[Any]] = [
        [
            0.0,
            null_raw.size,
            math.nan,
            _share_above(null_raw),
            _share_above(null_adj),
            oracle([r.null_dsr_raw for r in runs]),
        ]
    ]
    for k, level in enumerate(tf.PLANTED):
        raw = np.concatenate([r.power_dsr_raw[k] for r in runs])
        adj = np.concatenate([r.power_dsr_adj[k] for r in runs])
        hit = np.concatenate([r.power_hit[k] for r in runs])
        p_rows.append(
            [
                level,
                raw.size,
                float(hit.mean()),
                _share_above(raw),
                _share_above(adj),
                oracle([r.power_dsr_raw[k] for r in runs]),
            ]
        )
    write_csv(out / "family_power.csv", p_header, p_rows)

    lines = [
        f"{len(tf.RULES)} moving-average rules, {len(runs)} non-overlapping five-year windows "
        f"from {m.dates[a0]} to {m.dates[b0 - 1]}, {reps} sign-randomised datasets per window.",
        "",
        "| Planted true Sharpe (annualised) | Planted rule is the best | DSR > 0.95 with N "
        "| DSR > 0.95 with N̂ (eq. 9) | Oracle threshold |",
        "|---|---|---|---|---|",
    ]
    for lv, _, hit, praw, padj, porc in p_rows:
        shown = "" if math.isnan(hit) else f"{hit:.0%}"
        lines.append(f"| {lv:g} | {shown} | {praw:.1%} | {padj:.1%} | {porc:.1%} |")
    lines += [
        "",
        "| Window | Avg corr | N̂ | Null best SR | SR0 with N | SR0 with N̂ | Implied N "
        "| Size with N | Size with N̂ | Real best rule | Real best SR | Real DSR (N) "
        "| Real DSR (N̂) | Market lag-1 autocorrelation |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        d = dict(zip(FAMILY_HEADER, row, strict=True))
        lines.append(
            f"| {str(d['start'])[:4]}-{str(d['end'])[:4]} | {d['null_avg_corr_mean']:.2f} "
            f"| {d['null_n_effective_mean']:.0f} | {d['null_best_sharpe_ann_mean']:.2f} "
            f"| {d['null_sr0_raw_ann_mean']:.2f} | {d['null_sr0_adj_ann_mean']:.2f} "
            f"| {d['null_implied_trials']:.1f} | {d['size_raw']:.1%} | {d['size_adj']:.1%} "
            f"| {d['real_best_rule']} | {d['real_best_sharpe_ann']:.2f} "
            f"| {d['real_dsr_raw']:.3f} | {d['real_dsr_adj']:.3f} "
            f"| {d['market_lag1_autocorr']:.3f} |"
        )
    f, sl = tf.RULES[wv.best]
    lines += [
        "",
        f"All windows together, real data: average correlation {whole.avg_corr:.2f}, "
        f"N̂ {whole.n_effective:.0f}, best rule {f}/{sl} with an annualised Sharpe ratio of "
        f"{wv.best_sr * ann:.2f}, DSR {wv.dsr_raw:.4f} with N and {wv.dsr_adj:.4f} with N̂.",
    ]
    (out / "family.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return {"runs": runs, "rows": rows, "power": p_rows, "dates": m.dates}


LEAKAGE_HEADER = [
    "data",
    "rep",
    "n_samples",
    "shuffled_kfold_corr",
    "cv_no_purge_path_corr",
    "cv_purged_path_corr",
    "cv_no_purge_edge_corr",
    "cv_purged_edge_corr",
    "edge_samples",
    "train_size_no_purge",
    "train_size_purged",
]


def leakage_demo(out: Path, m: fetch_data.MarketSeries, fast: bool) -> None:
    """Overlapping labels: shuffled k-fold, combinatorial CV without purging, and
    with purging and an embargo, on the real series and on sign-randomised copies."""
    start = 20160101 if fast else 19900101
    x = m.excess[m.dates >= start]
    reps = 2 if fast else 20
    rng = np.random.default_rng(np.random.SeedSequence(SEED + 4))
    cases: list[tuple[str, FloatArray]] = [("real", x)]
    cases += [("null", x * rng.choice(np.array([-1.0, 1.0]), size=x.size)) for _ in range(reps)]
    rows: list[list[Any]] = []
    for i, (name, series) in enumerate(cases):
        feats, labels = leakage.make_dataset(series)
        plain = leakage.cpcv_score(feats, labels, purge=False)
        purged = leakage.cpcv_score(feats, labels, purge=True)
        rows.append(
            [
                name,
                i,
                labels.size,
                leakage.shuffled_kfold_corr(feats, labels, seed=i),
                float(plain.path_corr.mean()),
                float(purged.path_corr.mean()),
                plain.edge_corr,
                purged.edge_corr,
                plain.n_edge,
                plain.train_size,
                purged.train_size,
            ]
        )
    print("  leakage demo done", flush=True)
    write_csv(out / "leakage.csv", LEAKAGE_HEADER, rows)

    null = np.array([r[3:8] for r in rows if r[0] == "null"], dtype=np.float64)
    real = np.array(rows[0][3:8], dtype=np.float64)
    se = null.std(axis=0, ddof=1) / math.sqrt(null.shape[0])
    names = [
        "Shuffled 5-fold CV",
        "Combinatorial CV, no purging, whole paths",
        "Combinatorial CV, purged and embargoed, whole paths",
        "Combinatorial CV, no purging, test days within 20 days of a training group",
        "Combinatorial CV, purged and embargoed, the same test days",
    ]
    md = [
        [name, f"{real[k]:.3f}", f"{null[:, k].mean():.3f} ({se[k]:.3f})"]
        for k, name in enumerate(names)
    ]
    for label, d in (
        ("No purging minus purged, whole paths", null[:, 1] - null[:, 2]),
        ("No purging minus purged, test days near a training group", null[:, 3] - null[:, 4]),
    ):
        md.append([label, "", f"{d.mean():.4f} ({d.std(ddof=1) / math.sqrt(d.size):.4f})"])
    write_md(
        out / "leakage.md",
        [
            "Correlation of forecast and 20-day label",
            f"Real data from {start}",
            f"Sign-randomised, mean of {reps} (standard error)",
        ],
        md,
    )


# --------------------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------------------

BLUE, ORANGE, GREY = "#2b6ca3", "#d9731a", "#6b6b6b"


def figures(out: Path, h: dict[str, Any], n_strategies: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 100,
            "savefig.dpi": 160,
        }
    )
    ann = math.sqrt(PERIODS)
    noise: FloatArray = h["noise"]
    row = h["row_noise"]
    pbo_n: PBOResult = h["pbo_noise"]
    pbo_c: PBOResult = h["pbo_control"]

    # 1. headline: in-sample Sharpe ratios, and what the winner does out of sample
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    srs = sharpe_ratio(noise) * ann
    ax1.hist(srs, bins=40, color=BLUE, alpha=0.85)
    ax1.axvline(
        row["best_sharpe_ann"], color=ORANGE, lw=2, label=f"selected: {row['best_sharpe_ann']:.2f}"
    )
    ax1.axvline(
        row["expected_max_sharpe_ann"],
        color=GREY,
        lw=1.5,
        ls="--",
        label=(
            f"expected best of {n_strategies:,}\n"
            f"with no skill: {row['expected_max_sharpe_ann']:.2f}"
        ),
    )
    ax1.set_xlabel("Annualised Sharpe ratio over the full five years")
    ax1.set_ylabel("Number of strategies")
    ax1.set_title(f"{n_strategies:,} strategies with no edge", loc="left")
    ax1.legend(frameon=False, fontsize=9)
    ax2.scatter(
        pbo_n.is_performance * ann, pbo_n.oos_performance * ann, s=3, color=BLUE, alpha=0.25, lw=0
    )
    ax2.axhline(0, color=GREY, lw=1)
    ax2.set_xlabel(f"In-sample Sharpe ratio of the winner, {pbo_n.n_combinations:,} CSCV splits")
    ax2.set_ylabel("Its out-of-sample Sharpe ratio (annualised)")
    ax2.set_title(
        f"The winner out of sample: PBO {pbo_n.pbo:.2f}, {pbo_n.prob_oos_loss:.0%} lose",
        loc="left",
    )
    fig.tight_layout()
    fig.savefig(out / "headline.png")
    plt.close(fig)

    # 2. logit distributions: noise against one real strategy
    fig, ax = plt.subplots(figsize=(7, 3.8))
    bins = np.linspace(-7.5, 7.5, 61).tolist()
    ax.hist(
        pbo_n.logits,
        bins=bins,
        color=BLUE,
        alpha=0.75,
        label=f"no edge anywhere (PBO {pbo_n.pbo:.2f})",
    )
    ax.hist(
        pbo_c.logits,
        bins=bins,
        color=ORANGE,
        alpha=0.75,
        label=f"one strategy with a true Sharpe of {CONTROL_SHARPE:g} (PBO {pbo_c.pbo:.2f})",
    )
    ax.axvline(0, color=GREY, lw=1)
    ax.set_xlabel(
        "Logit of the out-of-sample rank of the in-sample winner (below 0 = below the median)"
    )
    ax.set_ylabel("Number of CSCV splits")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "pbo_logits.png")
    plt.close(fig)

    # 3. BLdP14 Exhibit 2: threshold and DSR against the number of trials
    fig, ax = plt.subplots(figsize=(7, 3.8))
    trials = list(range(1, 1001))
    sr, var = 2.5 / math.sqrt(250), 0.5 / 250
    sr0 = np.array([expected_max_sharpe(n, var) for n in trials]) * math.sqrt(250)
    dsr = np.array([deflated_sharpe_ratio(sr, 1250, n, var, -3, 10) for n in trials])
    ax.plot(trials, sr0, color=BLUE)
    ax.set_xlabel("Number of independent trials N")
    ax.set_ylabel("Expected best Sharpe ratio under no skill (annualised)", color=BLUE)
    ax.spines["right"].set_visible(True)
    twin = ax.twinx()
    twin.plot(trials, dsr, color=ORANGE)
    twin.axhline(0.95, color=GREY, lw=1, ls="--")
    twin.set_ylabel("Deflated Sharpe ratio", color=ORANGE)
    twin.spines["top"].set_visible(False)
    ax.set_title("The numerical example of Bailey and López de Prado (2014)", loc="left")
    fig.tight_layout()
    fig.savefig(out / "dsr_vs_trials.png")
    plt.close(fig)

    # 4. BBLZ14 Figure 2: minimum backtest length
    fig, ax = plt.subplots(figsize=(7, 3.8))
    trials = list(range(2, 1001))
    ax.plot(
        trials,
        [min_backtest_length(n) for n in trials],
        color=BLUE,
        label="minimum backtest length",
    )
    ax.plot(trials, 2 * np.log(trials), color=GREY, ls="--", label="upper bound 2 ln(N)")
    ax.set_xlabel("Number of independent trials N")
    ax.set_ylabel("Years of backtest")
    ax.set_title(
        "Years needed before the best of N skill-less trials\n"
        "is expected to show a Sharpe ratio below 1",
        loc="left",
    )
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "min_backtest_length.png")
    plt.close(fig)

    # 5. combinatorial CV paths against the in-sample curve of the selected strategy
    fig, ax = plt.subplots(figsize=(7, 3.8))
    years = np.arange(1, N_OBS + 1) / PERIODS
    best = int(np.argmax(srs))
    ax.plot(
        years,
        100 * np.cumsum(noise[:, best]),
        color=ORANGE,
        lw=2,
        label="selected on the full sample",
    )
    paths: FloatArray = h["paths_noise"]
    for i, path in enumerate(paths):
        ax.plot(
            years,
            100 * np.cumsum(path),
            color=BLUE,
            lw=1,
            alpha=0.8,
            label="same selection rule, out of sample (combinatorial CV paths)" if i == 0 else None,
        )
    ax.axhline(0, color=GREY, lw=1)
    ax.set_xlabel("Years")
    ax.set_ylabel("Cumulative return, % (sum of daily returns)")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "cv_paths.png")
    plt.close(fig)


def family_figure(out: Path, fam: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    power = fam["power"]
    labels = ["no edge\n(false positives)"] + [f"true Sharpe {p[0]:g}" for p in power[1:]]
    xs = np.arange(len(power))
    width = 0.27
    for k, (col, name, color) in enumerate(
        [
            (3, "N = number of rules", BLUE),
            (4, "N̂ from the average correlation", ORANGE),
            (5, "threshold from the null itself", GREY),
        ]
    ):
        ax1.bar(xs + (k - 1) * width, [100 * p[col] for p in power], width, color=color, label=name)
    ax1.axhline(5, color="black", lw=0.8, ls=":")
    ax1.set_xticks(xs, labels)
    ax1.set_ylabel("% of datasets where the best rule passes")
    ax1.set_title("Deflated Sharpe ratio > 0.95 on the rule family", loc="left")
    ax1.legend(frameon=False, fontsize=8, loc="upper left")

    rows = fam["rows"]
    idx = {k: i for i, k in enumerate(FAMILY_HEADER)}
    years = [int(str(r[idx["end"]])[:4]) for r in rows]
    ax2.plot(
        years,
        [r[idx["null_best_sharpe_ann_mean"]] for r in rows],
        "o-",
        color="black",
        label="best rule under the null (mean)",
    )
    ax2.plot(
        years,
        [r[idx["null_sr0_raw_ann_mean"]] for r in rows],
        "s--",
        color=BLUE,
        label="SR0 with N",
    )
    ax2.plot(
        years,
        [r[idx["null_sr0_adj_ann_mean"]] for r in rows],
        "^--",
        color=ORANGE,
        label="SR0 with N̂",
    )
    ax2.set_xlabel("Last year of the five-year window")
    ax2.set_ylabel("Annualised Sharpe ratio")
    ax2.set_title("Best rule when no rule has an edge", loc="left")
    ax2.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "family.png")
    plt.close(fig)


# --------------------------------------------------------------------------------------


def run(out: Path, fast: bool = False, make_figures: bool = True, real_data: bool = True) -> None:
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    n_strategies = 200 if fast else N_STRATEGIES
    n_blocks = 10 if fast else 16
    print(f"writing to {out}/ ({'fast' if fast else 'full'} run)", flush=True)
    paper_examples(out)
    h = headline(out, n_strategies, n_blocks)
    print("  headline experiment done", flush=True)
    pbo_sampling(out, reps=20 if fast else 300)
    print("  PBO sampling distribution done", flush=True)
    monte_carlo(
        out,
        levels=[0.0, 2.0] if fast else [0.0, 1.0, 2.0, 3.0],
        reps=5 if fast else 200,
        n_strategies=n_strategies,
        n_blocks=n_blocks,
    )
    fam: dict[str, Any] | None = None
    if real_data:
        market = fetch_data.load_market()
        data_fingerprint(out, market)
        fam = trend_family(out, market, fast)
        leakage_demo(out, market, fast)
    if make_figures:
        figures(out, h, n_strategies)
        if fam is not None:
            family_figure(out, fam)
    print(f"done in {time.time() - started:.0f}s", flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fast", action="store_true", help="small run for CI (under two minutes)")
    parser.add_argument("--out", type=Path, default=None, help="output directory")
    parser.add_argument("--no-figures", action="store_true", help="tables only")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    out = args.out or (root / ("results_fast" if args.fast else "results"))
    run(out, fast=args.fast, make_figures=not args.no_figures)


if __name__ == "__main__":
    main()
