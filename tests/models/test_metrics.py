import numpy as np
import pytest
from scripts.models.metrics import (
    rmse, mae, directional_accuracy,
    interval_coverage, sharpe_ratio, max_drawdown, compute_all,
)


def test_rmse_perfect_forecast():
    y = np.array([1.0, 2.0, 3.0])
    assert rmse(y, y) == pytest.approx(0.0)


def test_rmse_known_value():
    y_true = np.array([0.0, 0.0])
    y_pred = np.array([1.0, 1.0])
    assert rmse(y_true, y_pred) == pytest.approx(1.0)


def test_mae_known_value():
    y_true = np.array([0.0, 0.0, 0.0])
    y_pred = np.array([1.0, -1.0, 2.0])
    assert mae(y_true, y_pred) == pytest.approx(4.0 / 3.0)


def test_directional_accuracy_all_correct():
    y_true = np.array([1.0, -1.0, 1.0])
    y_pred = np.array([0.5, -0.5, 0.3])
    assert directional_accuracy(y_true, y_pred) == pytest.approx(1.0)


def test_directional_accuracy_all_wrong():
    y_true = np.array([1.0, -1.0, 1.0])
    y_pred = np.array([-0.5, 0.5, -0.3])
    assert directional_accuracy(y_true, y_pred) == pytest.approx(0.0)


def test_directional_accuracy_half():
    y_true = np.array([1.0, -1.0])
    y_pred = np.array([1.0, 1.0])
    assert directional_accuracy(y_true, y_pred) == pytest.approx(0.5)


def test_interval_coverage_all_inside():
    y = np.array([0.0, 0.5, -0.5])
    lo = np.array([-1.0, -1.0, -1.0])
    hi = np.array([1.0, 1.0, 1.0])
    assert interval_coverage(y, lo, hi) == pytest.approx(1.0)


def test_interval_coverage_none_inside():
    y = np.array([2.0, 3.0])
    lo = np.array([-1.0, -1.0])
    hi = np.array([1.0, 1.0])
    assert interval_coverage(y, lo, hi) == pytest.approx(0.0)


def test_sharpe_ratio_positive_for_perfect_directional_forecast():
    # Perfect directional forecast: strategy always takes correct side
    y_true = np.array([0.01, -0.005, 0.008, -0.003] * 25)
    y_pred = y_true.copy()  # sign always matches
    assert sharpe_ratio(y_true, y_pred) > 0


def test_sharpe_ratio_negative_for_always_wrong_forecast():
    y_true = np.array([0.01, -0.005, 0.008, -0.003] * 25)
    y_pred = -y_true  # sign always wrong
    assert sharpe_ratio(y_true, y_pred) < 0


def test_max_drawdown_zero_for_monotonic_gains():
    # Strategy always wins: returns are all positive
    y_true = np.array([0.01] * 20)
    y_pred = np.array([0.01] * 20)  # predict positive, actual positive -> +0.01 each step
    assert max_drawdown(y_true, y_pred) == pytest.approx(0.0, abs=1e-10)


def test_max_drawdown_is_non_positive():
    y_true = np.array([0.01, -0.02, 0.01, -0.03])
    y_pred = np.array([0.01, 0.02, 0.01, 0.03])  # wrong on negatives
    assert max_drawdown(y_true, y_pred) <= 0.0


def test_compute_all_returns_all_keys():
    y_true = np.array([0.01, -0.005, 0.008] * 10)
    y_pred = np.array([0.01, -0.005, 0.008] * 10)
    lo_80 = y_pred - 0.02
    hi_80 = y_pred + 0.02
    lo_90 = y_pred - 0.03
    hi_90 = y_pred + 0.03
    result = compute_all(y_true, y_pred, lo_80, hi_80, lo_90, hi_90)
    assert set(result.keys()) == {
        'rmse', 'mae', 'dir_acc', 'coverage_80', 'coverage_90', 'sharpe', 'max_drawdown'
    }
