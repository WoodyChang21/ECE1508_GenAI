import numpy as np

from scripts.models.mamba_strategy import (
    persistent_long_strategy_metrics,
    select_persistent_threshold,
    select_threshold,
)


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


def test_persistent_strategy_does_not_reenter_while_signal_stays_long():
    actual = np.full((5, 3), 0.01)
    predicted = np.full((5, 3), 0.001)

    result = persistent_long_strategy_metrics(
        actual,
        predicted,
        entry_threshold_bps=2.0,
        exit_threshold_bps=0.0,
        transaction_cost_bps=1.0,
        periods_per_year=1764,
    )

    assert result["trades"] == 1
    assert result["active_periods"] == 5
    assert result["position_turnover"] == 2.0
    expected = np.array([0.01 - 0.0001, 0.01, 0.01, 0.01, 0.01 - 0.0001])
    assert result["net_compounded_return"] == np.prod(1.0 + expected) - 1.0


def test_persistent_strategy_uses_only_next_realized_candle():
    actual = np.array([[0.01, -0.50, -0.50], [0.02, -0.50, -0.50]])
    predicted = np.full_like(actual, 0.001)

    result = persistent_long_strategy_metrics(
        actual,
        predicted,
        entry_threshold_bps=0.0,
        exit_threshold_bps=0.0,
        transaction_cost_bps=0.0,
        periods_per_year=1764,
    )

    assert result["gross_compounded_return"] == np.prod([1.01, 1.02]) - 1.0


def test_select_persistent_threshold_locks_best_calibration_return():
    actual = np.array(
        [[0.01, 0.0, 0.0], [0.01, 0.0, 0.0], [-0.01, 0.0, 0.0]]
    )
    predicted = np.array(
        [[0.001, 0.0, 0.0], [0.0001, 0.0, 0.0], [-0.001, 0.0, 0.0]]
    )

    policy, results = select_persistent_threshold(
        actual,
        predicted,
        thresholds_bps=[0.0, 5.0],
        exit_threshold_bps=0.0,
        transaction_cost_bps=0.0,
        periods_per_year=1764,
        minimum_trades=1,
        selection_metric="net_compounded_return",
    )

    expected = max(results, key=lambda row: row["net_compounded_return"])
    assert policy["strategy_type"] == "persistent_long_cash"
    assert policy["entry_threshold_bps"] == expected["entry_threshold_bps"]
