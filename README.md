# ECE1508 GenAI — Intraday SPY Forecasting with Mamba

This project forecasts the next three intraday SPY (S&P 500 ETF) returns together with **Mamba**, an attention-free selective state-space model. Raw market columns are converted into 23 stationary, calendar-aware inputs before each hourly row is projected into a continuous embedding; causal Mamba blocks process the lookback window and the final hidden state predicts the three-return path. A one-step compatibility mode remains available with `--forecast-horizon 1`.

---

## Project Structure

```
ECE1508_GenAI/
├── data/                        # Downloaded data — gitignored, not committed
│   ├── raw/
│   │   ├── spy/                 # spy_2010.json ... spy_2025.json
│   │   └── vix/                 # vix_2010.json ... vix_2025.json (VIXY ETF)
│   ├── processed/
│   │   ├── spy_hourly.parquet   # Clean SPY OHLCV + all features
│   │   ├── vix_hourly.parquet   # Clean VIXY log + change features
│   │   └── features.parquet     # Merged SPY + VIX, model-ready, 0 NaNs
│   └── splits/
│       ├── train.parquet        # 2011-01-04 to 2022-12-30
│       ├── val.parquet          # 2023-01-03 to 2023-12-29
│       └── test.parquet         # 2024-01-02 to 2025-05-30
├── scripts/
│   └── data/
│       ├── fmp_client.py        # FMP API wrapper (80-day chunked pagination)
│       ├── process_utils.py     # Pure feature engineering functions
│       ├── split_utils.py       # Walk-forward split logic
│       ├── 01_collect_raw.py    # Stage 1: Download raw JSON from FMP
│       ├── 02_process.py        # Stage 2: Clean, merge, compute features
│       └── 03_split.py          # Stage 3: Create train/val/test splits
│   └── models/
│       ├── mamba_data.py        # Leakage-safe scaling and causal windows
│       ├── mamba_model.py       # Continuous-input Mamba forecaster
│       ├── metrics.py           # Forecast and sign-strategy metrics
│       └── train_mamba.py       # Tune, train, evaluate, save predictions
├── tests/
│   └── data/
│       ├── conftest.py
│       ├── test_fmp_client.py   # 6 tests
│       ├── test_process_utils.py # 9 tests
│       └── test_split.py        # 5 tests
├── docs/
│   └── superpowers/plans/
│       └── 2026-06-26-data-collection-pipeline.md
├── requirements-data.txt
├── requirements-model.txt
└── .env                         # FMP_API_KEY — gitignored
```

---

## Branches

| Branch | Purpose |
|--------|---------|
| `main` | Project root |
| `Data` | Data collection and preprocessing pipeline (this branch) |
| `Model` | Historical DeepAR and PatchTST notebook experiments |

---

## Data Pipeline

The pipeline runs in three independent stages. Each stage can be re-run without redoing earlier stages.

### Stage 1 — Raw Collection (`01_collect_raw.py`)

Downloads hourly OHLCV data from the [Financial Modeling Prep](https://financialmodelingprep.com) (FMP) stable API for two symbols:

| Symbol | Ticker | Notes |
|--------|--------|-------|
| SPY | `SPY` | SPDR S&P 500 ETF Trust |
| VIX proxy | `VIXY` | ProShares VIX Short-Term Futures ETF. `^VIX` hourly data is only available from 2023 on FMP; VIXY has coverage from 2011. |

**Date range:** 2010-01-01 to 2025-06-01 (16 years)

**Output:** One JSON file per year per symbol in `data/raw/{spy,vix}/`. Files are skipped if they already exist, making the script safe to re-run after interruption.

**API design:** The FMP stable endpoint caps responses to approximately the last 90 days from the `to` date regardless of the `from` date. The client (`fmp_client.py`) uses **80-day chunks** to paginate through each year, making ~5 API calls per year-symbol combination (~160 total calls for the full history).

**Raw data counts:**
- SPY: 16 files, ~27,000 bars total
- VIXY: 16 files, ~25,000 bars total (`vix_2010.json` is empty — VIXY launched January 2011)

---

### Stage 2 — Processing (`02_process.py`)

Loads all raw JSON files, cleans, merges, and computes a full feature set.

**Pipeline steps:**
1. Load all year-chunk JSONs → deduplicate on datetime → sort ascending
2. Filter to regular market hours: **09:30–16:00 ET** (removes pre-market and after-hours bars)
3. Compute SPY features (see Feature Reference below)
4. Compute VIX features (log-transform + hourly change)
5. Left-join SPY and VIX on `datetime`
6. Drop warm-up rows where any of `[return_1h, vol_60h, rsi_14, macd, bb_upper, vix_change_1h]` is NaN — these are the first ~60 bars required for rolling indicators to stabilize

**Output files:**

| File | Rows | Columns | Description |
|------|------|---------|-------------|
| `spy_hourly.parquet` | ~27,000 | 20 | Clean SPY OHLCV + all SPY features before VIX merge |
| `vix_hourly.parquet` | ~25,000 | 3 | `datetime`, `vix_log`, `vix_change_1h` |
| `features.parquet` | **25,241** | **22** | Merged dataset, 0 NaNs, model-ready |

---

### Stage 3 — Splitting (`03_split.py`)

Creates a **walk-forward** train/val/test split. No shuffling — future data never leaks into past.

| Split | Date Range | Rows | Purpose |
|-------|-----------|------|---------|
| Train | 2011-01-04 → 2022-12-30 | 21,028 | Model training |
| Val | 2023-01-03 → 2023-12-29 | 1,744 | Lookback window and hyperparameter selection |
| Test | 2024-01-02 → 2025-05-30 | 2,469 | Final held-out evaluation — never touched during development |
| **Total** | | **25,241** | Exact match with `features.parquet` |

The validation set (2023) is used exclusively for selecting the Mamba input lookback window and calibrating prediction intervals. Default candidates are 24, 60, and 120 hourly bars. A 240-bar candidate can still be requested explicitly, but it is omitted by default because it costs substantially more and the existing one-step run selected 24 bars.

---

## Feature Reference

All 22 columns in `features.parquet`:

### Raw OHLCV

| Column | Type | Description |
|--------|------|-------------|
| `datetime` | datetime64 | Hourly timestamp, US Eastern market hours (09:30–16:00) |
| `open` | float64 | SPY opening price for the hour |
| `high` | float64 | SPY high price for the hour |
| `low` | float64 | SPY low price for the hour |
| `close` | float64 | SPY closing price for the hour |
| `volume` | float64 | SPY trading volume for the hour |

### Return Features

| Column | Type | Description | Note |
|--------|------|-------------|------|
| `return_1h` | float64 | `close.pct_change(1)` — 1-bar return | Primary forecast target. For `is_first_bar=True` rows, this includes the overnight gap (not a true intrabar 1-hour return). |
| `return_4h` | float64 | `close.pct_change(4)` — 4-bar return | After market-hours filtering, 4 bars ≈ 4 trading hours within a session. |
| `return_24h` | float64 | `close.pct_change(24)` — 24-bar return | 24 bars ≈ 3–4 trading days across session boundaries, not 24 wall-clock hours. |

### Session Flag

| Column | Type | Description |
|--------|------|-------------|
| `is_first_bar` | bool | `True` for the first hourly bar of each trading date. When `True`, `return_1h` captures the overnight gap return (previous close to current bar), not a pure 1-hour intraday return. Models should use this flag to mask or handle overnight returns appropriately. |

### Volatility & Volume

| Column | Type | Description |
|--------|------|-------------|
| `vol_24h` | float64 | Rolling standard deviation of `return_1h`, window=24 bars (~3 trading days). Measures short-term realized volatility. |
| `vol_60h` | float64 | Rolling standard deviation of `return_1h`, window=60 bars (~8–9 trading days). Measures medium-term realized volatility. |
| `volume_ratio` | float64 | `volume / volume.rolling(24).mean()`. Values > 1 indicate above-average volume; < 1 indicates below-average. Captures abnormal trading activity. |

### Momentum — RSI

| Column | Type | Description |
|--------|------|-------------|
| `rsi_14` | float64 | Relative Strength Index, window=14 bars. Range: 0–100. Values > 70 indicate overbought; < 30 indicate oversold. Computed on hourly close prices. |

### Trend — MACD

The MACD is computed with standard parameters (fast=12, slow=26, signal=9) applied to hourly close prices.

| Column | Type | Description |
|--------|------|-------------|
| `macd` | float64 | MACD line: EMA(12) − EMA(26) of close |
| `macd_signal` | float64 | Signal line: EMA(9) of the MACD line |
| `macd_diff` | float64 | MACD histogram: `macd − macd_signal`. Positive values indicate bullish momentum; negative indicate bearish. |

### Volatility — Bollinger Bands

Computed on hourly close prices with window=20, 2 standard deviations.

| Column | Type | Description |
|--------|------|-------------|
| `bb_upper` | float64 | Upper Bollinger Band: SMA(20) + 2×std(20) |
| `bb_lower` | float64 | Lower Bollinger Band: SMA(20) − 2×std(20) |
| `bb_width` | float64 | Relative band width: `(bb_upper − bb_lower) / SMA(20)`. Higher values indicate higher volatility regime; lower values indicate compression before potential breakout. |

### VIX / Volatility Regime

| Column | Type | Description | Note |
|--------|------|-------------|------|
| `vix_log` | float64 | Natural log of the VIXY hourly close price. Log-transform applied because raw VIXY levels are highly non-stationary — the ETF decays ~14,000× from 2011 to 2025 due to VIX futures roll costs. Log-scale range: approximately 3.7–13.8. | This is **VIXY** (a futures ETF), not the VIX index. It tracks the direction of VIX moves but is not the VIX level itself. |
| `vix_change_1h` | float64 | `VIXY_close.pct_change(1)` — 1-bar percentage change in VIXY. Inherently stationary and captures sudden volatility regime shifts. | NaN for the first VIXY bar (2011-01-04 09:30). Excluded by KEY_COLS NaN drop. |

---

## Design Decisions & Known Caveats

### Overnight Returns in `return_1h`
`pct_change(1)` does not reset at session boundaries. The first bar of each trading day (flagged by `is_first_bar=True`) computes the return from the previous day's final close to the current open — a gap of 16+ hours, not 1 hour. There are approximately **3,622 such rows** in the dataset (~14% of all bars).

This is intentional: overnight gaps are a real market phenomenon and carry information (e.g., gap-up/gap-down at the open). However, models that assume `return_1h` is always a 1-hour return should mask `is_first_bar=True` rows or add the flag as a model input.

### VIXY vs VIX Index
The FMP stable API does not carry `^VIX` hourly data before 2023. VIXY (ProShares VIX Short-Term Futures ETF) is used as a proxy. VIXY tracks VIX futures (not the VIX spot index) and decays structurally over time due to negative roll yield. The raw close is log-transformed in `vix_log` to reduce scale differences; `vix_change_1h` is used for directional information.

### Lookback Window is a Hyperparameter
Mamba's input context length is **not fixed**. Candidate windows of 24, 60, and 120 hourly bars are evaluated on the validation set, and only the best one is used for final training. They represent roughly 3.4, 8.6, and 17 trading days at seven bars per day; they are alternatives, not four simultaneous model inputs. Add `240` to `--lookbacks` only when testing whether roughly 34 trading days adds stable validation value.

### Data Not Committed to Git
All files under `data/` are gitignored. To reproduce the dataset from scratch:

```bash
pip install -r requirements-data.txt
cp .env.example .env          # add your FMP_API_KEY
python scripts/data/01_collect_raw.py
python scripts/data/02_process.py
python scripts/data/03_split.py
```

---

## Running the Pipeline

### Prerequisites

```bash
pip install -r requirements-data.txt
```

Set `FMP_API_KEY` in a `.env` file at the repo root:
```
FMP_API_KEY=your_key_here
```

### Run all three stages

```bash
# Stage 1: Download raw data (~10 minutes, ~200 API calls)
python scripts/data/01_collect_raw.py

# Stage 2: Process and compute features (~30 seconds)
python scripts/data/02_process.py

# Stage 3: Create train/val/test splits (~2 seconds)
python scripts/data/03_split.py
```

Each stage is independent. If interrupted, Stage 1 skips already-downloaded files. Stages 2 and 3 overwrite their outputs.

## Train the Mamba Forecaster

The model consumes the existing Parquet splits, so raw-data collection is not required when those files are already present.

```bash
pip install -r requirements-model.txt
python scripts/models/train_mamba.py
```

The default run predicts three future hourly returns together and tunes lookbacks `[24, 60, 120]`. Raw price levels are converted into relative candle and indicator distances; VIXY's drifting level is omitted in favor of its return; and bar-time/day-of-week cycles are included. The default training loss combines per-step Huber loss and compounded-horizon loss. Direction loss remains available through `--direction-loss-weight`, but defaults to zero after it failed to improve out-of-sample correlation. The regularized default architecture uses two 32-wide Mamba blocks, 0.2 dropout, AdamW at `1e-4`, and a two-epoch warmup followed by cosine decay.

Checkpoint selection no longer treats the single lowest compounded-return MAE as sufficient. For each lookback, epochs within 1% of the best MAE form a quality set; among checkpoints with positive correlation and at least 30 trades, the default selector maximizes net return from one fixed, predeclared 2 bp long/cash all-steps-positive rule. Lookbacks use the same MAE gate and signal checks, with the shorter lookback winning when net returns differ by no more than 0.005. If no checkpoint clears the signal requirements, selection falls back to correlation and then MAE. This can be switched to correlation-only selection with `--checkpoint-selection-metric correlation`.

The first 70% of validation is used for model selection and is included in the final fit. The chronological final 30% remains completely outside model fitting and is used only for residual-interval calibration and selection of one deployment threshold after transaction costs. That locked policy is stored in the checkpoint. The test target is never used for fitting, calibration, or strategy selection.

The fixed 2 bp policy above is only a consistent ruler for comparing checkpoints on the selection segment; it does not choose the deployed threshold. On the untouched calibration tail, threshold selection defaults to annualized net Sharpe. A profit-oriented experiment can instead pass `--strategy-selection-metric net_compounded_return`; the Colab v4 notebook does this while requiring at least 30 calibration trades. This deliberately favors raw return and may increase exposure and drawdown.

The output retains the comparison schema (`datetime`, `y`, `pred`, interval bounds, and `model`), where `y` and `pred` are compounded horizon returns. It also includes `y_h1`...`y_h3` and `pred_h1`...`pred_h3`, plus the horizon end time. Intervals are formed from validation residual quantiles of the compounded return.

For a quick CPU integration check rather than a meaningful experiment:

```bash
python scripts/models/train_mamba.py --lookbacks 24 --epochs 1 \
  --d-model 16 --layers 1 --limit-train 256 --limit-val 64 \
  --limit-test 64 --output data/predictions/mamba_smoke.parquet
```

The full default search is intended for a GPU. Hugging Face Transformers provides a slower sequential Mamba fallback when optimized kernels are unavailable, so CPU execution remains supported but can take substantially longer. The Colab notebook installs the optional `kernels` package to enable a compatible optimized path when available.

### Evaluate a saved Colab checkpoint

Save a self-contained checkpoint during training:

```bash
python scripts/models/train_mamba.py \
  --checkpoint data/checkpoints/mamba.pt
```

After downloading that checkpoint from Colab, evaluate it locally against the held-out test split without retraining:

```bash
python scripts/models/evaluate_mamba.py \
  --checkpoint data/checkpoints/mamba.pt \
  --transaction-cost-bps 1
```

If the Colab run used the default training command and did not save a checkpoint, download its automatically generated `data/predictions/mamba_preds.parquet` instead:

```bash
python scripts/models/evaluate_mamba.py \
  --predictions data/predictions/mamba_preds.parquet \
  --transaction-cost-bps 1
```

Both modes write a full report to `data/predictions/mamba_eval.json`; checkpoint mode also writes fresh forecasts to `data/predictions/mamba_eval.parquet`. For a three-step checkpoint, `locked_strategy` is the primary strategy result because its rule and threshold were selected on the held-out validation calibration tail. Test-set threshold sweeps remain available but are explicitly labeled exploratory. Because the 2024-2025 test results informed the redesign, a later unseen period or walk-forward replay is still required for a genuinely pristine performance claim. Positions are held for exactly three candles, only one position may be open, and entry and exit costs are charged.

The evaluator checks the saved feature schema version and order, model settings, scaler, lookback, forecast horizon, and SHA-256 hashes of the validation/test files. This prevents accidentally evaluating a checkpoint against different data. Feature-schema v1 checkpoints must be evaluated with their original checkout; v2 intentionally rejects them because the model inputs changed.

### Run tests

```bash
pytest tests/ -v
```

---

## Dataset Summary

| Property | Value |
|----------|-------|
| Asset | SPY (SPDR S&P 500 ETF) |
| Frequency | Hourly bars, market hours only (09:30–16:00 ET) |
| Coverage | 2011-01-04 to 2025-05-30 |
| Total bars | 25,241 |
| Features | 22 columns |
| NaN values | 0 (in final splits) |
| File format | Parquet (compressed, dtype-preserving) |
| Total disk usage | < 15 MB |
| Data source | Financial Modeling Prep (FMP) stable API |
| VIX proxy | VIXY (ProShares VIX Short-Term Futures ETF) |
