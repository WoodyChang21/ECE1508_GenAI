"""Download hourly SPY and VIX from FMP and save as year-chunked JSON files.

Usage:
    python scripts/data/01_collect_raw.py

Skips files that already exist — safe to re-run after interruption.
"""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.data.fmp_client import fetch_hourly

API_KEY = os.environ["FMP_API_KEY"]
RAW_DIR = Path("data/raw")
GLOBAL_START = 2010
GLOBAL_END_YEAR = 2025
GLOBAL_END_DATE = "2025-06-01"

SYMBOLS = {
    "spy": "SPY",
    "vix": "VIXY",  # ^VIX has no hourly data before 2023 on FMP; VIXY (VIX ETF) has full history
}


def collect_symbol(name: str, symbol: str) -> None:
    out_dir = RAW_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    for year in range(GLOBAL_START, GLOBAL_END_YEAR + 1):
        out_file = out_dir / f"{name}_{year}.json"
        if out_file.exists():
            print(f"  [{name}] {year}: already exists, skipping")
            continue

        start = f"{year}-01-01"
        end = f"{year}-12-31" if year < GLOBAL_END_YEAR else GLOBAL_END_DATE

        print(f"  [{name}] Fetching {symbol} {start} to {end} ...", end=" ", flush=True)
        records = fetch_hourly(symbol, start, end, API_KEY)
        out_file.write_text(json.dumps(records, indent=2))
        print(f"{len(records)} bars saved → {out_file}")


if __name__ == "__main__":
    for name, symbol in SYMBOLS.items():
        print(f"\nCollecting {symbol} ({name})...")
        collect_symbol(name, symbol)
    print("\nCollection complete.")
