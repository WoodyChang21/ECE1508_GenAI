"""Process raw year-chunk JSONs into clean Parquet files.

Stages:
  1. Load all spy/*.json -> spy_hourly.parquet (OHLCV + features)
  2. Load all vix/*.json -> vix_hourly.parquet
  3. Merge -> features.parquet (drops warm-up NaN rows)

Usage:
    python scripts/data/02_process.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.data.process_utils import (
    compute_spy_features,
    compute_vix_features,
    filter_market_hours,
    load_year_jsons,
)

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Columns that must be non-NaN in the final feature set.
# These represent the longest-window indicators; everything shorter will also be non-NaN.
KEY_COLS = ["return_1h", "vol_60h", "rsi_14", "macd", "bb_upper"]


def main() -> None:
    # --- SPY ---
    print("Loading SPY raw data...")
    spy_raw = load_year_jsons("spy", RAW_DIR)
    spy_raw = filter_market_hours(spy_raw)
    spy = compute_spy_features(spy_raw)
    spy.to_parquet(PROCESSED_DIR / "spy_hourly.parquet", index=False)
    print(f"  spy_hourly.parquet: {len(spy)} rows, {spy.shape[1]} columns")
    print(f"  Date range: {spy['datetime'].min()} -> {spy['datetime'].max()}")

    # --- VIX ---
    print("\nLoading VIX raw data...")
    vix_raw = load_year_jsons("vix", RAW_DIR)
    vix = compute_vix_features(vix_raw)
    vix.to_parquet(PROCESSED_DIR / "vix_hourly.parquet", index=False)
    print(f"  vix_hourly.parquet: {len(vix)} rows")

    # --- Merge ---
    print("\nMerging SPY + VIX...")
    features = spy.merge(vix, on="datetime", how="left")

    before = len(features)
    features = features.dropna(subset=KEY_COLS).reset_index(drop=True)
    print(f"  Dropped {before - len(features)} warm-up rows (rolling window NaNs)")

    features.to_parquet(PROCESSED_DIR / "features.parquet", index=False)
    print(f"  features.parquet: {len(features)} rows, {features.shape[1]} columns")
    print(f"  Columns: {list(features.columns)}")
    print(f"  Date range: {features['datetime'].min()} -> {features['datetime'].max()}")
    print(f"  NaN check in key cols: {features[KEY_COLS].isna().sum().sum()} NaNs")


if __name__ == "__main__":
    main()
