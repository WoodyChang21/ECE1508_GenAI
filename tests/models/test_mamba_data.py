import numpy as np

from scripts.models.mamba_data import WindowDataset, ZScoreScaler, with_history


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
