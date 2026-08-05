"""Ingest raw year-chunk SPY JSON files into a single OHLCV Parquet file.

Reads every steven/data/spy/spy_<year>.json (2010-2025), concatenates,
deduplicates, sorts by datetime ascending, and writes the result to
steven/data/spy_ohlcv_1h.parquet.

Usage:
    python steven/ingest_spy_ohlcv.py
"""

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
RAW_DIR = HERE / "data" / "spy"
OUT_PATH = HERE / "data" / "spy_ohlcv_1h.parquet"

COLUMNS = ["datetime", "open", "high", "low", "close", "volume"]


def load_raw() -> pd.DataFrame:
    files = sorted(RAW_DIR.glob("spy_*.json"))
    if not files:
        raise FileNotFoundError(f"No spy_*.json files found in {RAW_DIR}")

    frames = [pd.read_json(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    df = df.rename(columns={"date": "datetime"})
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df[COLUMNS]


def main() -> None:
    df = load_raw()

    before = len(df)
    df = df.drop_duplicates(subset="datetime").sort_values("datetime").reset_index(drop=True)
    dropped = before - len(df)

    df["volume"] = df["volume"].astype("int64")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)

    print(f"Loaded {before} rows from {RAW_DIR}")
    if dropped:
        print(f"Dropped {dropped} duplicate datetime rows")
    print(f"Wrote {len(df)} rows -> {OUT_PATH}")
    print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")


if __name__ == "__main__":
    main()
