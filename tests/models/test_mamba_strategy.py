import numpy as np

from scripts.models.mamba_strategy import select_threshold


def test_select_threshold_locks_best_calibration_sharpe():
    actual_timeline = np.array([0.01, 0.01, 0.01, -0.02, -0.02, -0.02])
    actual = np.lib.stride_tricks.sliding_window_view(actual_timeline, 3)
    predicted = np.array(
        [
            [0.002, 0.002, 0.002],
            [0.002, 0.002, 0.002],
            [0.002, 0.002, 0.002],
            [0.0001, 0.0001, 0.0001],
        ]
    )

    policy, results = select_threshold(
        actual,
        predicted,
        thresholds_bps=[0.0, 5.0],
        transaction_cost_bps=1.0,
        periods_per_year=1764,
        position_mode="long_only",
        require_all_steps_agree=True,
    )

    expected = max(
        (row for row in results if row["annualized_net_sharpe"] is not None),
        key=lambda row: row["annualized_net_sharpe"],
    )
    assert policy["threshold_bps"] == expected["threshold_bps"]
    assert policy["calibration_result"] == expected


def test_select_threshold_records_low_sample_fallback():
    actual = np.full((4, 3), 0.001)
    predicted = np.full((4, 3), 0.001)

    policy, _ = select_threshold(
        actual,
        predicted,
        thresholds_bps=[0.0, 20.0],
        transaction_cost_bps=1.0,
        periods_per_year=1764,
        position_mode="long_only",
        require_all_steps_agree=True,
        minimum_trades=30,
    )

    assert policy["selection_metric"] == "fallback_most_trades"
    assert policy["minimum_calibration_trades"] == 30


def test_select_threshold_can_optimize_net_compounded_return():
    actual_timeline = np.array([0.01, 0.01, 0.01, -0.02, -0.02, -0.02])
    actual = np.lib.stride_tricks.sliding_window_view(actual_timeline, 3)
    predicted = np.full_like(actual, 0.001)

    policy, results = select_threshold(
        actual,
        predicted,
        thresholds_bps=[0.0, 100.0],
        transaction_cost_bps=1.0,
        periods_per_year=1764,
        position_mode="long_only",
        require_all_steps_agree=True,
        selection_metric="net_compounded_return",
    )

    expected = max(
        (row for row in results if row["trades"] >= 1),
        key=lambda row: row["net_compounded_return"],
    )
    assert policy["selection_metric"] == "net_compounded_return"
    assert policy["threshold_bps"] == expected["threshold_bps"]
