"""Forecast and simple sign-strategy evaluation metrics."""

from __future__ import annotations

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)))


def interval_coverage(
    y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> float:
    return float(np.mean((y_true >= lower) & (y_true <= upper)))


def sharpe_ratio(
    y_true: np.ndarray, y_pred: np.ndarray, periods_per_year: int = 1680
) -> float:
    strategy_returns = np.sign(y_pred) * y_true
    std = np.std(strategy_returns)
    if std == 0:
        return 0.0
    return float(np.mean(strategy_returns) / std * np.sqrt(periods_per_year))


def max_drawdown(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    strategy_returns = np.sign(y_pred) * y_true
    cumulative = np.cumsum(strategy_returns)
    running_max = np.maximum.accumulate(cumulative)
    return float(np.min(cumulative - running_max))


def compute_all(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    lo_80: np.ndarray,
    hi_80: np.ndarray,
    lo_90: np.ndarray,
    hi_90: np.ndarray,
) -> dict[str, float]:
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "dir_acc": directional_accuracy(y_true, y_pred),
        "coverage_80": interval_coverage(y_true, lo_80, hi_80),
        "coverage_90": interval_coverage(y_true, lo_90, hi_90),
        "sharpe": sharpe_ratio(y_true, y_pred),
        "max_drawdown": max_drawdown(y_true, y_pred),
    }
