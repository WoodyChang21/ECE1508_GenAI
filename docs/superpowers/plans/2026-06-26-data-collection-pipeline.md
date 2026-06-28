# Data Collection Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible pipeline that downloads 15 years of hourly SPY OHLCV and VIX data from FMP, computes derived features, and produces train/val/test Parquet splits ready for DeepAR and PatchTST.

**Architecture:** Three independent stages — (1) raw collection: FMP API → JSON files per year, (2) processing: merge + clean + technical indicators → Parquet, (3) splitting: time-based walk-forward → Parquet. Each stage is a standalone script so any stage can be re-run without redoing earlier ones.

**Tech Stack:** Python 3.10+, requests, pandas 2.x, numpy, pyarrow, ta, python-dotenv, pytest

---

## Data Size Estimate

| Asset | Rows (approx) | Raw JSON (15 files) | Processed Parquet |
|---|---|---|---|
| SPY hourly 2010–2025 | ~24,570 | ~4 MB | ~400 KB |
| VIX hourly 2010–2025 | ~24,570 | ~4 MB | ~400 KB |
| features.parquet (~30 cols, merged) | ~24,000* | — | ~2 MB |
| train / val / test splits | ~17k / 2.5k / 4k | — | ~4 MB total |

*Rows drop slightly after rolling-window warm-up (first ~60 hours dropped by NaN filter).

**Total disk usage: < 15 MB.** The entire dataset fits comfortably in RAM on any modern laptop.

### Why not daily or larger data?
- SPY hourly bars × 15 years = ~24 k rows. This is small enough for Parquet; JSON or CSV would be 3–10× larger with no benefit.
- FMP's paid hourly endpoint returns data in ~1-year windows. Year-chunked JSON files make resuming a failed download trivial.

---

## Storage Format Decision

| Format | Pros | Cons | Decision |
|---|---|---|---|
| JSON | Human-readable; direct from API | 10× larger, slow pandas I/O, no dtype preservation | Raw cache only |
| CSV | Simple | No dtypes, 3–5× larger than Parquet | Rejected |
| SQLite | Queryable | Overkill for flat time series; slower pandas I/O | Rejected |
| **Parquet** | Compressed, fast pandas I/O, preserves dtypes, industry-standard for ML | Binary (not human-readable) | **All processed outputs** |

---

## Directory Layout

```
data/                          # excluded from git (added to .gitignore)
  raw/
    spy/
      spy_2010.json            # one file per year, downloaded by 01_collect_raw.py
      spy_2011.json
      ...
      spy_2025.json
    vix/
      vix_2010.json
      ...
      vix_2025.json
  processed/
    spy_hourly.parquet         # clean SPY OHLCV + return columns
    vix_hourly.parquet         # clean VIX close + change
    features.parquet           # merged SPY + VIX + all indicators (model input)
  splits/
    train.parquet              # 2010-01-01 → 2022-12-31
    val.parquet                # 2023-01-01 → 2023-12-31
    test.parquet               # 2024-01-01 → 2025-06-01

scripts/
  __init__.py
  data/
    __init__.py
    fmp_client.py              # FMP API wrapper with year-chunked pagination
    process_utils.py           # pure functions: load, clean, feature engineering
    01_collect_raw.py          # CLI: downloads raw JSON → data/raw/
    02_process.py              # CLI: processes raw → data/processed/
    03_split.py                # CLI: splits features → data/splits/

tests/
  __init__.py
  data/
    __init__.py
    conftest.py                # adds project root to sys.path
    test_fmp_client.py         # mocked API tests
    test_process_utils.py      # unit tests for pure processing functions
    test_split.py              # unit tests for split boundaries

requirements-data.txt
docs/
  superpowers/
    plans/
      2026-06-26-data-collection-pipeline.md   # this file
```

---

## Train / Val / Test Split Boundaries

| Split | Date range | Approx rows | Rationale |
|---|---|---|---|
| Train | 2010-01-01 → 2022-12-31 | ~17,000 | 13 years; multiple full market cycles |
| Val | 2023-01-01 → 2023-12-31 | ~1,650 | Used for lookback window / hyperparameter selection |
| Test | 2024-01-01 → 2025-06-01 | ~2,400 | Held-out; never touched during model development |

Walk-forward only — no shuffling. Future never leaks into past.

---

## Feature Set (output columns of features.parquet)

| Column | Description |
|---|---|
| datetime | Hourly timestamp (UTC, market hours only 09:30–16:00 ET) |
| open, high, low, close, volume | SPY OHLCV |
| return_1h | `close.pct_change(1)` — primary forecast target |
| return_4h | `close.pct_change(4)` |
| return_24h | `close.pct_change(24)` |
| vol_24h | Rolling std of return_1h, window=24 |
| vol_60h | Rolling std of return_1h, window=60 |
| volume_ratio | `volume / volume.rolling(24).mean()` |
| rsi_14 | RSI, window=14 |
| macd, macd_signal, macd_diff | MACD (12, 26, 9) |
| bb_upper, bb_lower, bb_width | Bollinger Bands, window=20 |
| vix | VIX close (merged from vix_hourly.parquet, left-joined) |
| vix_change_1h | `vix.pct_change(1)` |

---

## Task 1: Environment Setup

**Files:**
- Create: `requirements-data.txt`
- Create: `scripts/__init__.py`
- Create: `scripts/data/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/data/__init__.py`
- Create: `tests/data/conftest.py`
- Modify: `.gitignore` (add `data/`)

- [ ] **Step 1: Add `data/` to `.gitignore`**

Append these lines to the bottom of `.gitignore`:

```
# Project data — downloaded, not committed
data/
```

- [ ] **Step 2: Create `requirements-data.txt`**

```
requests>=2.31.0
pandas>=2.0.0
numpy>=1.24.0
pyarrow>=14.0.0
ta>=0.11.0
python-dotenv>=1.0.0
pytest>=7.4.0
```

- [ ] **Step 3: Create `__init__.py` files**

Create empty files at:
- `scripts/__init__.py`
- `scripts/data/__init__.py`
- `tests/__init__.py`
- `tests/data/__init__.py`

- [ ] **Step 4: Create `tests/data/conftest.py`**

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
```

- [ ] **Step 5: Install dependencies**

```bash
pip install -r requirements-data.txt
```

Expected: all packages install without error.

- [ ] **Step 6: Verify imports**

```bash
python -c "import requests, pandas, numpy, pyarrow, ta, dotenv; print('OK')"
```

Expected output: `OK`

- [ ] **Step 7: Commit**

```bash
git add requirements-data.txt scripts/ tests/ .gitignore
git commit -m "feat(data): project structure and dependencies"
```

---

## Task 2: FMP API Client

**Files:**
- Create: `scripts/data/fmp_client.py`
- Create: `tests/data/test_fmp_client.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/data/test_fmp_client.py`:

```python
from unittest.mock import patch, MagicMock
from scripts.data.fmp_client import fetch_hourly


def _bar(date_str):
    return {
        "date": date_str,
        "open": 470.0,
        "high": 471.0,
        "low": 469.0,
        "close": 470.5,
        "volume": 1_000_000,
    }


def _mock_get(data):
    m = MagicMock()
    m.json.return_value = data
    m.raise_for_status = MagicMock()
    return m


def test_single_year_makes_one_api_call():
    with patch("scripts.data.fmp_client.requests.get") as mock_get:
        mock_get.return_value = _mock_get([_bar("2024-01-02 10:00:00")])
        result = fetch_hourly("SPY", "2024-01-01", "2024-12-31", "test_key")
    assert mock_get.call_count == 1
    assert len(result) == 1


def test_three_year_range_makes_three_api_calls():
    with patch("scripts.data.fmp_client.requests.get") as mock_get:
        mock_get.return_value = _mock_get([_bar("2022-06-15 10:00:00")])
        fetch_hourly("SPY", "2022-01-01", "2024-12-31", "test_key")
    assert mock_get.call_count == 3


def test_results_are_flat_list_of_dicts():
    with patch("scripts.data.fmp_client.requests.get") as mock_get:
        mock_get.return_value = _mock_get([_bar("2024-01-02 10:00:00"), _bar("2024-01-02 09:30:00")])
        result = fetch_hourly("SPY", "2024-01-01", "2024-12-31", "test_key")
    assert isinstance(result, list)
    assert len(result) == 2
    assert all("date" in r and "close" in r for r in result)


def test_api_key_passed_in_query_params():
    with patch("scripts.data.fmp_client.requests.get") as mock_get:
        mock_get.return_value = _mock_get([])
        fetch_hourly("SPY", "2024-01-01", "2024-12-31", "my_secret_key")
    call_kwargs = mock_get.call_args
    params = call_kwargs[1]["params"] if "params" in call_kwargs[1] else call_kwargs[0][1]
    assert params["apikey"] == "my_secret_key"


def test_caret_symbol_is_url_encoded():
    """^VIX must be percent-encoded to %5EVIX in the URL path."""
    with patch("scripts.data.fmp_client.requests.get") as mock_get:
        mock_get.return_value = _mock_get([])
        fetch_hourly("^VIX", "2024-01-01", "2024-12-31", "key")
    url_called = mock_get.call_args[0][0]
    assert "%5EVIX" in url_called or "%5evix" in url_called.lower()
```

- [ ] **Step 2: Run tests to confirm failure**

```bash
pytest tests/data/test_fmp_client.py -v
```

Expected: `ImportError` or `ModuleNotFoundError` — `fmp_client` does not exist yet.

- [ ] **Step 3: Write `scripts/data/fmp_client.py`**

```python
import time
import requests
from datetime import datetime, timedelta
from urllib.parse import quote

BASE_URL = "https://financialmodelingprep.com/api/v3"
CHUNK_DAYS = 365


def fetch_hourly(symbol: str, start: str, end: str, api_key: str) -> list[dict]:
    """Download hourly OHLCV from FMP in year-sized chunks. Returns flat list of bar dicts."""
    symbol_encoded = quote(symbol, safe="")
    url = f"{BASE_URL}/historical-chart/1hour/{symbol_encoded}"

    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")

    records: list[dict] = []
    current = start_dt

    while current <= end_dt:
        chunk_end = min(current + timedelta(days=CHUNK_DAYS - 1), end_dt)
        params = {
            "from": current.strftime("%Y-%m-%d"),
            "to": chunk_end.strftime("%Y-%m-%d"),
            "apikey": api_key,
        }
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        chunk = resp.json()
        if isinstance(chunk, list):
            records.extend(chunk)
        current = chunk_end + timedelta(days=1)
        time.sleep(0.3)

    return records
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/data/test_fmp_client.py -v
```

Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add scripts/data/fmp_client.py tests/data/test_fmp_client.py
git commit -m "feat(data): FMP API client with year-chunked pagination"
```

---

## Task 3: Processing Utilities

**Files:**
- Create: `scripts/data/process_utils.py`
- Create: `tests/data/test_process_utils.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/data/test_process_utils.py`:

```python
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
    df = _make_spy_df(50)
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
```

- [ ] **Step 2: Run tests to confirm failure**

```bash
pytest tests/data/test_process_utils.py -v
```

Expected: `ImportError` — `process_utils` does not exist yet.

- [ ] **Step 3: Write `scripts/data/process_utils.py`**

```python
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
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/data/test_process_utils.py -v
```

Expected: `8 passed`

- [ ] **Step 5: Commit**

```bash
git add scripts/data/process_utils.py tests/data/test_process_utils.py
git commit -m "feat(data): processing utilities and feature engineering"
```

---

## Task 4: Raw Data Collection Script

**Files:**
- Create: `scripts/data/01_collect_raw.py`

> Note: This script makes real API calls. No unit test — verified by inspecting output files.

- [ ] **Step 1: Write `scripts/data/01_collect_raw.py`**

```python
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
    "vix": "^VIX",
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

        print(f"  [{name}] Fetching {symbol} {start} → {end} ...", end=" ", flush=True)
        records = fetch_hourly(symbol, start, end, API_KEY)
        out_file.write_text(json.dumps(records, indent=2))
        print(f"{len(records)} bars saved → {out_file}")


if __name__ == "__main__":
    for name, symbol in SYMBOLS.items():
        print(f"\nCollecting {symbol} ({name})...")
        collect_symbol(name, symbol)
    print("\nCollection complete.")
```

- [ ] **Step 2: Run the collection script**

```bash
python scripts/data/01_collect_raw.py
```

Expected: Output like:
```
Collecting SPY (spy)...
  [spy] Fetching SPY 2010-01-01 → 2010-12-31 ... 1623 bars saved → data/raw/spy/spy_2010.json
  [spy] Fetching SPY 2011-01-01 → 2011-12-31 ... 1638 bars saved → data/raw/spy/spy_2011.json
  ...
Collecting ^VIX (vix)...
  ...
Collection complete.
```

- [ ] **Step 3: Verify output structure**

```bash
python -c "
from pathlib import Path
import json

spy_files = sorted(Path('data/raw/spy').glob('spy_*.json'))
print(f'SPY files: {len(spy_files)}')
sample = json.loads(spy_files[-1].read_text())
print(f'Sample record: {sample[0] if sample else \"EMPTY\"}')
print(f'Keys: {list(sample[0].keys()) if sample else \"N/A\"}')
"
```

Expected: 16 SPY files (2010–2025), each record has keys: `date`, `open`, `high`, `low`, `close`, `volume`.

> **VIX fallback:** If `^VIX` hourly data is sparse or missing for early years, check `data/raw/vix/` file sizes. If most files are empty, switch the `vix` symbol to `VIXY` (VIX ETF, has more history on FMP) and re-run. The processing step handles this transparently.

- [ ] **Step 4: Commit**

```bash
git add scripts/data/01_collect_raw.py
git commit -m "feat(data): raw collection script for SPY and VIX"
```

---

## Task 5: Processing Script

**Files:**
- Create: `scripts/data/02_process.py`

- [ ] **Step 1: Write `scripts/data/02_process.py`**

```python
"""Process raw year-chunk JSONs into clean Parquet files.

Stages:
  1. Load all spy/*.json → spy_hourly.parquet (OHLCV + features)
  2. Load all vix/*.json → vix_hourly.parquet
  3. Merge → features.parquet (drops warm-up NaN rows)

Usage:
    python scripts/data/02_process.py
"""

import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()
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
# These are the longest-window indicators; everything shorter will also be non-NaN.
KEY_COLS = ["return_1h", "vol_60h", "rsi_14", "macd", "bb_upper"]


def main() -> None:
    # --- SPY ---
    print("Loading SPY raw data...")
    spy_raw = load_year_jsons("spy", RAW_DIR)
    spy_raw = filter_market_hours(spy_raw)
    spy = compute_spy_features(spy_raw)
    spy.to_parquet(PROCESSED_DIR / "spy_hourly.parquet", index=False)
    print(f"  spy_hourly.parquet: {len(spy)} rows, {spy.shape[1]} columns")
    print(f"  Date range: {spy['datetime'].min()} → {spy['datetime'].max()}")

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
    print(f"  Date range: {features['datetime'].min()} → {features['datetime'].max()}")
    print(f"  NaN check: {features[KEY_COLS].isna().sum().sum()} NaNs in key columns")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the processing script**

```bash
python scripts/data/02_process.py
```

Expected output (approximate):
```
Loading SPY raw data...
  spy_hourly.parquet: 24532 rows, 19 columns
  Date range: 2010-01-04 09:30:00 → 2025-05-30 16:00:00

Loading VIX raw data...
  vix_hourly.parquet: 24510 rows

Merging SPY + VIX...
  Dropped 63 warm-up rows (rolling window NaNs)
  features.parquet: 24469 rows, 21 columns
  Columns: ['datetime', 'open', 'high', 'low', 'close', 'volume', 'return_1h', ...]
  Date range: 2010-01-05 13:30:00 → 2025-05-30 16:00:00
  NaN check: 0 NaNs in key columns
```

- [ ] **Step 3: Spot-check the Parquet**

```bash
python -c "
import pandas as pd
df = pd.read_parquet('data/processed/features.parquet')
print(df.dtypes)
print(df.head(3))
print(df.tail(3))
print('NaN total:', df.isna().sum().sum())
"
```

Expected: All columns have correct dtypes (float64, datetime64). Zero NaNs in core columns.

- [ ] **Step 4: Commit**

```bash
git add scripts/data/02_process.py
git commit -m "feat(data): processing script — cleans, merges, computes features"
```

---

## Task 6: Train / Val / Test Split

**Files:**
- Create: `scripts/data/03_split.py`
- Create: `tests/data/test_split.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/data/test_split.py`:

```python
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.data.split_utils import split_dataframe, TRAIN_END, VAL_END


def _make_features_df() -> pd.DataFrame:
    dates = pd.date_range("2010-01-04 09:30:00", periods=500, freq="1h")
    np.random.seed(0)
    return pd.DataFrame({
        "datetime": dates,
        "close": 470 + np.cumsum(np.random.randn(500) * 0.5),
        "return_1h": np.random.randn(500) * 0.001,
    })


def test_splits_cover_all_rows():
    df = _make_features_df()
    train, val, test = split_dataframe(df)
    assert len(train) + len(val) + len(test) == len(df)


def test_splits_do_not_overlap():
    df = _make_features_df()
    train, val, test = split_dataframe(df)
    assert train["datetime"].max() < val["datetime"].min()
    assert val["datetime"].max() < test["datetime"].min()


def test_train_ends_at_train_end():
    df = _make_features_df()
    train, _, _ = split_dataframe(df)
    if len(train) > 0:
        assert str(train["datetime"].max().date()) <= TRAIN_END


def test_val_within_val_period():
    df = _make_features_df()
    _, val, _ = split_dataframe(df)
    if len(val) > 0:
        assert str(val["datetime"].min().date()) > TRAIN_END
        assert str(val["datetime"].max().date()) <= VAL_END


def test_empty_val_when_data_ends_before_val_period():
    df = _make_features_df()
    early_df = df[df["datetime"] < "2022-01-01"].copy()
    train, val, test = split_dataframe(early_df)
    assert len(val) == 0
    assert len(test) == 0
```

- [ ] **Step 2: Run tests to confirm failure**

```bash
pytest tests/data/test_split.py -v
```

Expected: `ImportError` — `split_utils` does not exist yet.

- [ ] **Step 3: Create `scripts/data/split_utils.py`**

```python
import pandas as pd

TRAIN_END = "2022-12-31"
VAL_END = "2023-12-31"


def split_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dt = pd.to_datetime(df["datetime"])
    train = df[dt <= TRAIN_END].reset_index(drop=True)
    val = df[(dt > TRAIN_END) & (dt <= VAL_END)].reset_index(drop=True)
    test = df[dt > VAL_END].reset_index(drop=True)
    return train, val, test
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/data/test_split.py -v
```

Expected: `5 passed`

- [ ] **Step 5: Write `scripts/data/03_split.py`**

```python
"""Split features.parquet into train/val/test Parquet files.

Split boundaries:
  Train: 2010-01-01 → 2022-12-31
  Val:   2023-01-01 → 2023-12-31
  Test:  2024-01-01 → 2025-06-01

Usage:
    python scripts/data/03_split.py
"""

import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.data.split_utils import split_dataframe

PROCESSED_DIR = Path("data/processed")
SPLITS_DIR = Path("data/splits")
SPLITS_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    df = pd.read_parquet(PROCESSED_DIR / "features.parquet")

    train, val, test = split_dataframe(df)

    train.to_parquet(SPLITS_DIR / "train.parquet", index=False)
    val.to_parquet(SPLITS_DIR / "val.parquet", index=False)
    test.to_parquet(SPLITS_DIR / "test.parquet", index=False)

    for name, split in [("Train", train), ("Val", val), ("Test", test)]:
        if len(split) > 0:
            print(
                f"{name:5s}: {len(split):6d} rows  "
                f"{split['datetime'].min().date()} → {split['datetime'].max().date()}"
            )
        else:
            print(f"{name:5s}:      0 rows  (no data in this date range)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run the split script**

```bash
python scripts/data/03_split.py
```

Expected output:
```
Train: 17043 rows  2010-01-05 → 2022-12-30
Val:    1638 rows  2023-01-03 → 2023-12-29
Test:   3788 rows  2024-01-02 → 2025-05-30
```

- [ ] **Step 7: Run all tests**

```bash
pytest tests/ -v
```

Expected: All tests pass (no failures).

- [ ] **Step 8: Final commit**

```bash
git add scripts/data/split_utils.py scripts/data/03_split.py tests/data/test_split.py
git commit -m "feat(data): train/val/test walk-forward split"
```

---

## Self-Review Checklist

**Spec coverage:**
- [x] SPY hourly OHLCV collected → Task 4
- [x] VIX hourly collected → Task 4
- [x] Returns (1h, 4h, 24h) computed → Task 3 / `compute_spy_features`
- [x] Volatility features computed → Task 3
- [x] Technical indicators (RSI, MACD, BB) computed → Task 3
- [x] Volume feature computed → Task 3
- [x] 2010–2025 date range → Task 4 constants
- [x] Data stored as Parquet (not JSON/CSV/SQLite) → Tasks 5, 6
- [x] Raw JSON kept for reproducibility → Task 4
- [x] `data/` excluded from git → Task 1
- [x] API key loaded from `.env`, never hardcoded → Task 4
- [x] Resumable download (skip existing files) → Task 4
- [x] Walk-forward split, no data leakage → Task 6
- [x] VIX fallback documented → Task 4

**Placeholder scan:** No TBDs, no "implement later", no "similar to Task N" references.

**Type consistency:** `split_dataframe` imported from `split_utils` in both `test_split.py` and `03_split.py`. `TRAIN_END` / `VAL_END` constants defined once in `split_utils.py` and imported where needed.
