import numpy as np
import pytest

from scripts.models.evaluate_mamba import (
    build_report,
    horizon_arrays,
    load_prediction_frame,
    multi_period_strategy_metrics,
    point_metrics,
    strategy_metrics,
    validate_checkpoint,
)
from scripts.models.mamba_data import FEATURE_COLUMNS


def test_point_metrics_reports_bias_correlation_and_direction():
    actual = np.array([0.01, -0.02, 0.03])
    predicted = np.array([0.02, -0.01, -0.01])

    result = point_metrics(actual, predicted)

    assert result["directional_accuracy"] == pytest.approx(2 / 3)
    assert result["mean_error"] == pytest.approx(-0.02 / 3)
    assert result["correlation"] is not None


def test_strategy_metrics_charges_turnover_and_closing_cost():
    actual = np.array([0.01, -0.02, 0.03])
    predicted = np.array([0.02, -0.01, -0.01])

    result = strategy_metrics(
        actual,
        predicted,
        threshold_bps=0.0,
        transaction_cost_bps=1.0,
        periods_per_year=1,
    )

    # Positions [1, -1, -1]: entry 1 + reversal 2 + final close 1.
    assert result["position_turnover"] == 4.0
    assert result["active_periods"] == 3
    assert result["directional_accuracy_when_active"] == pytest.approx(2 / 3)
    expected_net = np.array([0.01 - 0.0001, 0.02 - 0.0002, -0.03 - 0.0001])
    assert result["net_compounded_return"] == pytest.approx(
        np.prod(1 + expected_net) - 1
    )


def test_strategy_threshold_can_leave_strategy_flat():
    result = strategy_metrics(
        np.array([0.01, -0.01]),
        np.array([0.0001, -0.0001]),
        threshold_bps=2.0,
        transaction_cost_bps=1.0,
        periods_per_year=1764,
    )

    assert result["active_periods"] == 0
    assert result["net_compounded_return"] == 0.0
    assert result["annualized_net_sharpe"] is None


def test_strategy_metrics_separates_long_and_short_signals():
    actual = np.array([0.01, -0.02, 0.03, -0.04])
    predicted = np.array([0.02, -0.01, -0.01, 0.01])

    long_only = strategy_metrics(
        actual, predicted, 0.0, 0.0, 1764, position_mode="long_only"
    )
    short_only = strategy_metrics(
        actual, predicted, 0.0, 0.0, 1764, position_mode="short_only"
    )

    assert long_only["position_mode"] == "long_only"
    assert long_only["active_periods"] == 2
    assert long_only["directional_accuracy_when_active"] == pytest.approx(0.5)
    assert long_only["gross_compounded_return"] == pytest.approx(
        (1.01 * 0.96) - 1
    )
    assert short_only["position_mode"] == "short_only"
    assert short_only["active_periods"] == 2
    assert short_only["directional_accuracy_when_active"] == pytest.approx(0.5)
    assert short_only["gross_compounded_return"] == pytest.approx(
        (1.02 * 0.97) - 1
    )


def test_strategy_metrics_rejects_unknown_position_mode():
    with pytest.raises(ValueError, match="position_mode"):
        strategy_metrics(
            np.array([0.01]),
            np.array([0.01]),
            0.0,
            0.0,
            1764,
            position_mode="longish",
        )


def test_build_report_includes_baselines_intervals_and_skill():
    actual = np.array([0.01, -0.02, 0.03])
    predicted = np.array([0.008, -0.018, 0.025])

    report = build_report(
        actual,
        predicted,
        predicted - 0.01,
        predicted + 0.01,
        predicted - 0.02,
        predicted + 0.02,
        previous_return=-0.005,
        trainval_mean_return=0.001,
        thresholds_bps=[0.0, 5.0],
        transaction_cost_bps=0.0,
        periods_per_year=1764,
    )

    assert set(report["forecast"]["baselines"]) == {
        "zero_return",
        "trainval_mean_return",
        "lag_one_return",
    }
    assert report["prediction_intervals"]["coverage_90"] == 1.0
    assert set(report["sign_strategy"]["baselines"]) == {
        "always_long",
        "lag_one_return",
    }
    assert len(report["sign_strategy"]["threshold_sweep"]) == 2
    assert len(report["sign_strategy"]["long_only_threshold_sweep"]) == 2
    assert len(report["sign_strategy"]["short_only_threshold_sweep"]) == 2
    assert all(
        row["position_mode"] == "long_only"
        for row in report["sign_strategy"]["long_only_threshold_sweep"]
    )
    assert all(
        row["position_mode"] == "short_only"
        for row in report["sign_strategy"]["short_only_threshold_sweep"]
    )
    assert report["forecast"]["skill_vs_zero_return"]["mae"] > 0


def test_validate_checkpoint_rejects_feature_order_mismatch():
    checkpoint = {
        "state_dict": {},
        "model_options": {},
        "lookback": 24,
        "feature_columns": tuple(reversed(FEATURE_COLUMNS)),
        "scaler_mean": np.zeros(len(FEATURE_COLUMNS)),
        "scaler_scale": np.ones(len(FEATURE_COLUMNS)),
        "interval_offsets": {
            "lo_80": -0.1,
            "hi_80": 0.1,
            "lo_90": -0.2,
            "hi_90": 0.2,
        },
    }

    with pytest.raises(ValueError, match="feature columns/order"):
        validate_checkpoint(checkpoint)


def test_load_prediction_frame_validates_schema(tmp_path):
    import pandas as pd

    path = tmp_path / "predictions.parquet"
    pd.DataFrame({"datetime": ["2024-01-01"], "y": [0.01]}).to_parquet(path)

    with pytest.raises(ValueError, match="missing columns"):
        load_prediction_frame(path)


def test_multi_period_strategy_never_overlaps_trades_and_charges_round_trips():
    actual_timeline = np.array([0.01, 0.01, 0.01, -0.01, -0.01, -0.01])
    actual = np.lib.stride_tricks.sliding_window_view(actual_timeline, 3)
    predicted = np.full_like(actual, 0.01)

    result = multi_period_strategy_metrics(
        actual,
        predicted,
        threshold_bps=0.0,
        transaction_cost_bps=1.0,
        periods_per_year=1764,
        position_mode="long_only",
    )

    assert result["trades"] == 2
    assert result["active_periods"] == 6
    assert result["position_turnover"] == 4.0
    expected = np.array(
        [
            0.01 - 0.0001,
            0.01,
            0.01 - 0.0001,
            -0.01 - 0.0001,
            -0.01,
            -0.01 - 0.0001,
        ]
    )
    assert result["net_compounded_return"] == pytest.approx(
        np.prod(1.0 + expected) - 1.0
    )


def test_multi_period_agreement_filter_rejects_mixed_path():
    actual = np.array([[0.01, 0.01, 0.01], [0.01, 0.01, 0.01]])
    predicted = np.array([[0.02, -0.005, 0.01], [0.01, 0.01, 0.01]])

    result = multi_period_strategy_metrics(
        actual,
        predicted,
        threshold_bps=0.0,
        transaction_cost_bps=0.0,
        periods_per_year=1764,
        position_mode="long_only",
        require_all_steps_agree=True,
    )

    assert result["trades"] == 1
    assert result["signal_rule"] == "cumulative_return_and_all_steps_agree"


def test_build_report_uses_non_overlapping_multi_candle_strategy():
    timeline = np.array([0.01, -0.01, 0.02, 0.01, -0.02])
    actual_steps = np.lib.stride_tricks.sliding_window_view(timeline, 3)
    predicted_steps = actual_steps * 0.8
    actual = np.prod(1.0 + actual_steps, axis=1) - 1.0
    predicted = np.prod(1.0 + predicted_steps, axis=1) - 1.0

    report = build_report(
        actual,
        predicted,
        predicted - 0.01,
        predicted + 0.01,
        predicted - 0.02,
        predicted + 0.02,
        previous_return=-0.005,
        trainval_mean_return=0.001,
        thresholds_bps=[0.0],
        transaction_cost_bps=1.0,
        periods_per_year=1764,
        y_true_steps=actual_steps,
        y_pred_steps=predicted_steps,
    )

    assert report["forecast"]["horizon_periods"] == 3
    assert "sign_strategy" not in report
    assert "multi_candle_strategy" in report
    assert len(
        report["multi_candle_strategy"][
            "persistent_long_cash_threshold_sweep"
        ]
    ) == 1
    assert len(
        report["multi_candle_strategy"]["all_steps_positive_threshold_sweep"]
    ) == 1


def test_horizon_arrays_reads_multi_step_columns():
    import pandas as pd

    frame = pd.DataFrame(
        {
            "y": [0.0],
            "pred": [0.0],
            "y_h1": [0.01],
            "pred_h1": [0.02],
            "y_h2": [-0.01],
            "pred_h2": [-0.02],
        }
    )

    actual, predicted = horizon_arrays(frame)

    np.testing.assert_array_equal(actual, [[0.01, -0.01]])
    np.testing.assert_array_equal(predicted, [[0.02, -0.02]])
