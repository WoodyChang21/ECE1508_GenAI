"""Pulls VIX's full daily history via yfinance, same range/discipline as
collect_daily_yfinance.py's SPY pull -- a genuine market-derived sentiment/fear proxy (implied
volatility priced into S&P 500 options), not a transform of SPY's own price history the way
ema_cross/trend_position/rsi are (see momentum_pipeline.py's add_vix_feature). Only `close` is
kept: VIX is an index computed from options prices, not something directly traded in the same
OHLC sense, so its own open/high/low/volume aren't meaningful features here.

Usage:
    python steven/src/collect_vix_yfinance.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

TICKER = "^VIX"
START = "1993-01-29"  # matches collect_daily_yfinance.py's SPY range
END = "2025-05-30"
OUT_PATH = Path(__file__).resolve().parents[1] / "data" / "vix_daily_yfinance.parquet"


def collect(start: str = START, end: str = END, out_path: Path = OUT_PATH) -> Path:
    df = yf.download(TICKER, start=start, end=end, interval="1d", progress=False, auto_adjust=False)
    if df.empty:
        raise RuntimeError(f"yfinance returned no rows for {TICKER} {start}..{end}")

    df.columns = [c[0].lower() for c in df.columns]  # flatten yfinance's (field, ticker) MultiIndex
    df = df.reset_index().rename(columns={"Date": "datetime"})
    df = df[["datetime", "close"]].rename(columns={"close": "vix_close"})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    logger.info("wrote %d VIX daily bars (%s to %s) to %s", len(df), df["datetime"].min().date(), df["datetime"].max().date(), out_path)
    return out_path


if __name__ == "__main__":
    collect()
