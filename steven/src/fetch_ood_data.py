"""Fetch new SPY hourly bars from FMP (from the end of the existing dataset through
today) and merge them into an extended parquet for out-of-distribution backtesting.

The original `steven/data/spy_ohlcv_1h.parquet` (2010-01 .. 2025-05-30 -- the exact
range every checkpoint on this branch was trained/tested on) is left untouched, so
every existing result stays reproducible. This script writes a separate file,
`steven/data/spy_ohlcv_1h_extended.parquet` (old data + newly fetched bars, deduped and
sorted, same cleaning as `ingest_spy_ohlcv.py`), which `evaluate_revin_hysteresis.py`'s
`--data-path`/`--test-start`/`--test-end` flags can point at to backtest on genuinely
unseen data the model has never seen in training or in the original test split.

Idempotent: re-running only fetches the gap between the existing file's last bar and
--end-date (default: today), so it's safe to rerun periodically as more data becomes
available.

Usage:
    python steven/src/fetch_ood_data.py
    python steven/src/fetch_ood_data.py --end-date 2026-08-01  # override "today"

Requires FMP_API_KEY in the environment (repo-root .env, loaded via python-dotenv).
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))

load_dotenv(REPO_ROOT / ".env")

from scripts.data.fmp_client import fetch_hourly  # noqa: E402

SYMBOL = "SPY"
BASE_PARQUET = HERE.parent / "data" / "spy_ohlcv_1h.parquet"
DEFAULT_OUT = HERE.parent / "data" / "spy_ohlcv_1h_extended.parquet"
COLUMNS = ["datetime", "open", "high", "low", "close", "volume"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--end-date", type=str, default=None,
        help="YYYY-MM-DD, last day to fetch (inclusive). Defaults to today.",
    )
    p.add_argument("--base-parquet", type=str, default=str(BASE_PARQUET))
    p.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    api_key = os.environ["FMP_API_KEY"]

    base = pd.read_parquet(args.base_parquet)
    base["datetime"] = pd.to_datetime(base["datetime"])
    base = base[COLUMNS]
    last_date = base["datetime"].max()
    print(f"Existing data: {base['datetime'].min()} .. {last_date} ({len(base)} rows)")

    fetch_start = last_date.strftime("%Y-%m-%d")
    fetch_end = args.end_date or date.today().strftime("%Y-%m-%d")
    print(f"Fetching {SYMBOL} hourly bars from FMP: {fetch_start} .. {fetch_end} ...")
    records = fetch_hourly(SYMBOL, fetch_start, fetch_end, api_key)
    if not records:
        raise RuntimeError(f"FMP returned no bars for {SYMBOL} {fetch_start}..{fetch_end}")

    new_df = pd.DataFrame(records).rename(columns={"date": "datetime"})
    new_df["datetime"] = pd.to_datetime(new_df["datetime"])
    new_df = new_df[COLUMNS]

    merged = pd.concat([base, new_df], ignore_index=True)
    before = len(merged)
    merged = merged.drop_duplicates(subset="datetime").sort_values("datetime").reset_index(drop=True)
    dropped = before - len(merged)
    merged["volume"] = merged["volume"].astype("int64")

    n_new = len(merged) - len(base)
    if n_new <= 0:
        print(f"No new bars beyond the existing dataset's last date ({last_date.date()}) -- nothing to add.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out_path, index=False)

    print(f"Dropped {dropped} duplicate datetime rows during merge")
    print(f"Wrote {len(merged)} rows -> {out_path}")
    print(f"New range: {merged['datetime'].min()} .. {merged['datetime'].max()}")
    print(f"Added {n_new} new bars beyond the original dataset's last date ({last_date.date()})")


if __name__ == "__main__":
    main()
