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
    df["return_4h"] = df["close"].pct_change(4)
    df["return_24h"] = df["close"].pct_change(24)
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
    df = df.copy()
    vix_change = df["close"].pct_change(1)
    return pd.DataFrame({
        "datetime": df["date"],
        "vix": df["close"].values,
        "vix_change_1h": vix_change.values,
    })
