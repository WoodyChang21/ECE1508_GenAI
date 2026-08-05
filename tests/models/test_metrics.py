import numpy as np

from scripts.models.metrics import compute_all, directional_accuracy, interval_coverage


def test_directional_accuracy():
    actual = np.array([1.0, -1.0, 1.0, -1.0])
    predicted = np.array([0.5, -0.5, -0.5, -0.5])
    assert directional_accuracy(actual, predicted) == 0.75


def test_interval_coverage():
    actual = np.array([0.0, 2.0, 4.0])
    assert interval_coverage(actual, np.array([-1.0, 1.0, 3.0]), np.array([1.0, 3.0, 3.5])) == 2 / 3


def test_compute_all_returns_expected_schema():
    actual = np.array([0.01, -0.01])
    predicted = np.array([0.005, -0.005])
    result = compute_all(
        actual,
        predicted,
        predicted - 0.01,
        predicted + 0.01,
        predicted - 0.02,
        predicted + 0.02,
    )
    assert set(result) == {
        "rmse",
        "mae",
        "dir_acc",
        "coverage_80",
        "coverage_90",
        "sharpe",
        "max_drawdown",
    }
