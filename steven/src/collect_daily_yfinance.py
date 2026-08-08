"""Pulls SPY's full daily OHLCV history via yfinance (inception 1993-01-29 through this
project's fixed test-end date) and caches it to parquet in data_pipeline.py's expected raw
format (RAW_COLUMNS: datetime, open, high, low, close, volume).

Why not just resample the existing hourly parquet (as probe_daily_cvae.py's disposable probe
did)? That only reaches back to 2010 (~3,900 daily bars). daily_signal_probe.md's own
statistical-power argument for testing daily bars at all only holds up with real history behind
it: `sqrt(N)` detection power needs the full ~8,100-bar range, not a ~3,900-bar slice of it, to
have a fair shot at detecting a true signal even 1.5-2x stronger than hourly's.

`auto_adjust=False`: keeps `close` as the actual traded close (matching what open_ret/body_ret/
wick math in compute_raw_features expects -- real intrabar OHLC relationships), not
dividend/split-adjusted values that would distort those relationships around ex-dividend dates.
adj_close is dropped entirely; not used anywhere downstream.

Usage:
    python steven/src/collect_daily_yfinance.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

TICKER = "SPY"
START = "1993-01-29"  # SPY inception
END = "2025-05-30"  # same fixed "current" test-end date used throughout this project
OUT_PATH = Path(__file__).resolve().parents[1] / "data" / "spy_daily_yfinance.parquet"


def collect(start: str = START, end: str = END, out_path: Path = OUT_PATH) -> Path:
    df = yf.download(TICKER, start=start, end=end, interval="1d", progress=False, auto_adjust=False)
    if df.empty:
        raise RuntimeError(f"yfinance returned no rows for {TICKER} {start}..{end}")

    df.columns = [c[0].lower() for c in df.columns]  # flatten yfinance's (field, ticker) MultiIndex
    df = df.reset_index().rename(columns={"Date": "datetime"})
    df = df[["datetime", "open", "high", "low", "close", "volume"]]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    logger.info("wrote %d daily bars (%s to %s) to %s", len(df), df["datetime"].min().date(), df["datetime"].max().date(), out_path)
    return out_path


if __name__ == "__main__":
    collect()
