import numpy as np
import pandas as pd
import pytest
from scripts.data.process_utils import (
    filter_market_hours,
    compute_spy_features,
    compute_vix_features,
)


def _make_spy_df(n: int = 200) -> pd.DataFrame:
    np.random.seed(42)
    dates = pd.date_range("2024-01-02 08:00:00", periods=n, freq="1h")
    close = 470 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({
        "date": dates,
        "open": close * (1 + np.random.randn(n) * 0.001),
        "high": close * (1 + np.abs(np.random.randn(n)) * 0.002),
        "low": close * (1 - np.abs(np.random.randn(n)) * 0.002),
        "close": close,
        "volume": np.random.randint(500_000, 5_000_000, n).astype(float),
    })


def _make_vix_df(n: int = 200) -> pd.DataFrame:
    np.random.seed(7)
    dates = pd.date_range("2024-01-02 09:30:00", periods=n, freq="1h")
    return pd.DataFrame({
        "date": dates,
        "close": 15 + np.cumsum(np.random.randn(n) * 0.1),
    })


# --- filter_market_hours ---

def test_filter_market_hours_removes_premarket():
    df = _make_spy_df(50)
    filtered = filter_market_hours(df)
    times = filtered["date"].dt.strftime("%H:%M")
    assert (times >= "09:30").all()
    assert (times <= "16:00").all()


def test_filter_market_hours_preserves_930_open():
    # _make_spy_df uses hourly freq starting 08:00, so 09:30 never appears.
    # Build a small frame that explicitly includes a 09:30 row.
    import pandas as pd
    row = pd.DataFrame({
        "date": [pd.Timestamp("2024-01-02 09:30:00")],
        "open": [470.0], "high": [471.0], "low": [469.0],
        "close": [470.5], "volume": [1_000_000.0],
    })
    df = pd.concat([_make_spy_df(10), row], ignore_index=True)
    filtered = filter_market_hours(df)
    assert "09:30" in filtered["date"].dt.strftime("%H:%M").values


# --- compute_spy_features ---

def test_spy_features_has_required_columns():
    df = _make_spy_df(200)
    df = filter_market_hours(df)
    result = compute_spy_features(df)
    required = [
        "datetime", "open", "high", "low", "close", "volume",
        "return_1h", "return_4h", "return_24h",
        "vol_24h", "vol_60h", "volume_ratio",
        "rsi_14", "macd", "macd_signal", "macd_diff",
        "bb_upper", "bb_lower", "bb_width",
    ]
    for col in required:
        assert col in result.columns, f"Missing column: {col}"


def test_return_1h_is_pct_change():
    df = _make_spy_df(10)
    df = df.copy()
    df["date"] = pd.date_range("2024-01-02 09:30:00", periods=10, freq="1h")
    result = compute_spy_features(df)
    expected = df["close"].pct_change(1).iloc[1]
    assert abs(result["return_1h"].iloc[1] - expected) < 1e-10


def test_spy_features_renames_date_to_datetime():
    df = _make_spy_df(100)
    result = compute_spy_features(df)
    assert "datetime" in result.columns
    assert "date" not in result.columns


# --- compute_vix_features ---

def test_vix_features_returns_datetime_and_vix():
    df = _make_vix_df(50)
    result = compute_vix_features(df)
    assert "datetime" in result.columns
    assert "vix" in result.columns
    assert "vix_change_1h" in result.columns


def test_vix_features_drops_raw_close():
    df = _make_vix_df(50)
    result = compute_vix_features(df)
    assert "close" not in result.columns
