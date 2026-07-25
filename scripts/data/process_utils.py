from pathlib import Path
import json

import numpy as np
import pandas as pd
import ta

MARKET_OPEN = "09:30"
MARKET_CLOSE = "16:00"


def filter_market_hours(df: pd.DataFrame) -> pd.DataFrame:
    t = df["date"].dt.strftime("%H:%M")
    return df[(t >= MARKET_OPEN) & (t <= MARKET_CLOSE)].reset_index(drop=True)


def load_year_jsons(name: str, raw_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted((raw_dir / name).glob(f"{name}_*.json")):
        records = json.loads(path.read_text())
        if records:
            frames.append(pd.DataFrame(records))
    if not frames:
        raise FileNotFoundError(f"No JSON files found in {raw_dir / name}")
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").drop_duplicates("date").reset_index(drop=True)


def compute_spy_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["return_1h"] = df["close"].pct_change(1)
    # Row-count based: after market-hours filter, 4 rows ≈ 4 trading hours;
    # 24 rows ≈ 3-4 trading days (not 24 wall-clock hours).
    df["return_4h"] = df["close"].pct_change(4)
    df["return_24h"] = df["close"].pct_change(24)
    # True for the first bar of each trading date: return_1h includes overnight gap here.
    df["is_first_bar"] = df["date"].dt.date != df["date"].dt.date.shift(1)
    # True for the last bar of each trading date (mirror of is_first_bar; close-of-day effects).
    df["is_last_bar"] = df["date"].dt.date != df["date"].dt.date.shift(-1)

    # --- Calendar / clock features: all knowable in advance, safe as futr_exog ---
    # Cyclical hour-of-day encoding (market hours 09:30-16:00 ET -> 7 hourly bars/day).
    hour_frac = df["date"].dt.hour + df["date"].dt.minute / 60.0
    hours_since_open = hour_frac - 9.5
    day_span_hours = 16.0 - 9.5  # 6.5h trading day
    hour_angle = 2 * np.pi * (hours_since_open / day_span_hours).clip(0, 1)
    df["hour_sin"] = np.sin(hour_angle)
    df["hour_cos"] = np.cos(hour_angle)

    # Cyclical day-of-week encoding (Mon=0 .. Fri=4).
    dow = df["date"].dt.dayofweek
    dow_angle = 2 * np.pi * dow / 5.0
    df["dow_sin"] = np.sin(dow_angle)
    df["dow_cos"] = np.cos(dow_angle)
    df["is_monday"] = dow == 0
    df["is_friday"] = dow == 4

    # Cyclical month-of-year / day-of-month (seasonality; e.g. January effect, month-end flows).
    month_angle = 2 * np.pi * (df["date"].dt.month - 1) / 12.0
    df["month_sin"] = np.sin(month_angle)
    df["month_cos"] = np.cos(month_angle)
    df["is_month_end"] = df["date"].dt.is_month_end
    df["is_month_start"] = df["date"].dt.is_month_start

    # Quarter-end flag (index rebalancing / quarterly options expiry effects).
    df["is_quarter_end"] = df["date"].dt.is_quarter_end

    df["vol_24h"] = df["return_1h"].rolling(24).std()
    df["vol_60h"] = df["return_1h"].rolling(60).std()
    df["volume_ratio"] = df["volume"] / df["volume"].rolling(24).mean()

    df["rsi_14"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()

    macd_ind = ta.trend.MACD(df["close"])
    df["macd"] = macd_ind.macd()
    df["macd_signal"] = macd_ind.macd_signal()
    df["macd_diff"] = macd_ind.macd_diff()

    bb_ind = ta.volatility.BollingerBands(df["close"], window=20)
    df["bb_upper"] = bb_ind.bollinger_hband()
    df["bb_lower"] = bb_ind.bollinger_lband()
    df["bb_width"] = bb_ind.bollinger_wband()

    return df.rename(columns={"date": "datetime"})


def compute_vix_features(df: pd.DataFrame) -> pd.DataFrame:
    # Raw VIXY close is non-stationary due to VIX futures roll cost (~14000x decay 2011-2025).
    # Log-transform reduces scale differences; vix_change_1h is inherently stationary.
    df = df.copy()
    vix_change = df["close"].pct_change(1)
    return pd.DataFrame({
        "datetime": df["date"],
        "vix_log": np.log(df["close"].clip(lower=1e-8)).values,
        "vix_change_1h": vix_change.values,
    })
