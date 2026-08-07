import numpy as np
import pytest

from scripts.models.evaluate_mamba import (
    build_report,
    load_prediction_frame,
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
