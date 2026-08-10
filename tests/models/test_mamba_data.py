import numpy as np
import pandas as pd

from scripts.models.mamba_data import (
    FEATURE_COLUMNS,
    WindowDataset,
    ZScoreScaler,
    model_feature_frame,
    with_history,
)


def test_scaler_round_trip_target():
    values = np.array([[1.0, 10.0], [3.0, 20.0], [5.0, 30.0]], dtype=np.float32)
    scaler = ZScoreScaler.fit(values)
    scaled = scaler.transform(values)

    np.testing.assert_allclose(scaler.inverse_target(scaled[:, 0]), values[:, 0])


def test_scaler_handles_constant_columns():
    values = np.ones((4, 2), dtype=np.float32)
    scaler = ZScoreScaler.fit(values)

    assert np.isfinite(scaler.transform(values)).all()
    np.testing.assert_array_equal(scaler.scale, np.ones(2, dtype=np.float32))


def test_window_dataset_is_strictly_causal():
    values = np.arange(12, dtype=np.float32).reshape(6, 2)
    dataset = WindowDataset(values, lookback=2, target_index=0)

    window, target = dataset[0]
    np.testing.assert_array_equal(window.numpy(), values[:2])
    assert target.item() == values[2, 0]
    assert len(dataset) == 4


def test_history_rows_are_context_not_labels():
    history = np.arange(12, dtype=np.float32).reshape(6, 2)
    current = np.arange(100, 108, dtype=np.float32).reshape(4, 2)
    dataset = with_history(history, current, lookback=3)

    _, first_target = dataset[0]
    assert len(dataset) == len(current)
    assert first_target.item() == current[0, 0]


def test_window_dataset_returns_complete_multi_step_targets():
    values = np.arange(16, dtype=np.float32).reshape(8, 2)
    dataset = WindowDataset(
        values, lookback=2, target_index=0, forecast_horizon=3
    )

    window, target = dataset[0]

    np.testing.assert_array_equal(window.numpy(), values[:2])
    np.testing.assert_array_equal(target.numpy(), values[2:5, 0])
    assert len(dataset) == 4


def test_history_multi_step_targets_stay_inside_current_split():
    history = np.arange(12, dtype=np.float32).reshape(6, 2)
    current = np.arange(100, 110, dtype=np.float32).reshape(5, 2)
    dataset = with_history(history, current, lookback=3, forecast_horizon=3)

    _, first_target = dataset[0]

    np.testing.assert_array_equal(first_target.numpy(), current[:3, 0])
    assert len(dataset) == 3


def test_model_features_are_invariant_to_price_level_rescaling():
    frame = pd.DataFrame(
        {
            "datetime": pd.date_range("2024-01-02 09:30", periods=3, freq="h"),
            "return_1h": [0.01, -0.005, 0.002],
            "open": [100.0, 101.0, 100.5],
            "high": [102.0, 102.0, 101.5],
            "low": [99.0, 100.0, 100.0],
            "close": [101.0, 100.5, 101.0],
            "volume": [1_000_000, 900_000, 1_100_000],
            "return_4h": [0.01, 0.005, 0.007],
            "return_24h": [0.02, 0.01, 0.015],
            "is_first_bar": [True, False, False],
            "vol_24h": [0.01, 0.01, 0.01],
            "vol_60h": [0.012, 0.012, 0.012],
            "volume_ratio": [1.1, 0.9, 1.2],
            "rsi_14": [55.0, 50.0, 52.0],
            "macd": [1.0, 0.8, 0.9],
            "macd_signal": [0.9, 0.85, 0.88],
            "macd_diff": [0.1, -0.05, 0.02],
            "bb_upper": [105.0, 105.0, 105.5],
            "bb_lower": [95.0, 95.0, 95.5],
            "bb_width": [10.0, 10.0, 10.0],
            "vix_change_1h": [0.01, -0.02, 0.0],
        }
    )
    scaled_prices = frame.copy()
    price_columns = [
        "open",
        "high",
        "low",
        "close",
        "macd",
        "macd_signal",
        "macd_diff",
        "bb_upper",
        "bb_lower",
    ]
    scaled_prices.loc[:, price_columns] *= 5.0

    actual = model_feature_frame(frame)
    rescaled = model_feature_frame(scaled_prices)

    assert tuple(actual.columns) == FEATURE_COLUMNS
    np.testing.assert_allclose(actual, rescaled, rtol=1e-10, atol=1e-10)
    assert np.isfinite(actual.to_numpy()).all()
