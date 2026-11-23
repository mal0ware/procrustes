"""Tools for detecting backtest overfitting.

See README.md for an overview and PAPER.md for the source of every formula.
"""

from procrustes.cv import CombinatorialPurgedKFold, PurgedKFold, purged_train_indices
from procrustes.multiple_testing import (
    MultipleTestingResult,
    adjust_t_statistics,
    bhy,
    bonferroni,
    harmonic_number,
    holm,
    p_values_from_t,
)
from procrustes.pbo import PBOResult, cscv, n_cscv_combinations
from procrustes.sharpe import (
    DeflatedSharpeResult,
    average_correlation,
    deflated_sharpe_ratio,
    deflated_sharpe_ratio_from_returns,
    effective_number_of_trials,
    expected_max_sharpe,
    expected_max_standard_normal,
    min_backtest_length,
    min_backtest_length_upper_bound,
    min_track_record_length,
    probabilistic_sharpe_ratio,
    return_moments,
    sharpe_ratio,
    sharpe_ratio_std,
)

__version__ = "0.1.0"

__all__ = [
    "CombinatorialPurgedKFold",
    "DeflatedSharpeResult",
    "MultipleTestingResult",
    "PBOResult",
    "PurgedKFold",
    "adjust_t_statistics",
    "average_correlation",
    "bhy",
    "bonferroni",
    "cscv",
    "deflated_sharpe_ratio",
    "deflated_sharpe_ratio_from_returns",
    "effective_number_of_trials",
    "expected_max_sharpe",
    "expected_max_standard_normal",
    "harmonic_number",
    "holm",
    "min_backtest_length",
    "min_backtest_length_upper_bound",
    "min_track_record_length",
    "n_cscv_combinations",
    "p_values_from_t",
    "probabilistic_sharpe_ratio",
    "purged_train_indices",
    "return_moments",
    "sharpe_ratio",
    "sharpe_ratio_std",
]
