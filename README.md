# ECE1508 GenAI — Intraday SPY Forecasting: DeepAR vs PatchTST

This project compares two deep learning forecasting paradigms for intraday financial prediction using hourly SPY (S&P 500 ETF) data. The benchmark model is **DeepAR** (probabilistic autoregressive LSTM) and the primary model is **PatchTST** (patch-based Transformer).

**Research question:** Can a Transformer with full historical covariate access (PatchTST) outperform an autoregressive LSTM with limited exogenous input (DeepAR) on one-step-ahead intraday return forecasting?

---

## Branches

| Branch | Purpose |
|--------|---------|
| `main` | Project root — README, shared config |
| `Data` | Data collection and preprocessing pipeline |
| `Model` | DeepAR and PatchTST model implementation ← **you are here** |

---

## Repository Structure

```
ECE1508_GenAI/
├── data/                          # NOT committed to git — regenerate locally
│   ├── raw/
│   │   ├── spy/                   # spy_2010.json … spy_2025.json
│   │   └── vix/                   # vix_2010.json … vix_2025.json (VIXY ETF)
│   ├── processed/
│   │   ├── spy_hourly.parquet     # Clean SPY OHLCV + features (before VIX merge)
│   │   ├── vix_hourly.parquet     # VIXY log + hourly change
│   │   └── features.parquet       # Merged dataset, 25,241 rows, 22 columns, 0 NaNs
│   ├── splits/
│   │   ├── train.parquet          # 2011-01-04 → 2022-12-30  (21,028 rows)
│   │   ├── val.parquet            # 2023-01-03 → 2023-12-29  (1,744 rows)
│   │   └── test.parquet           # 2024-01-02 → 2025-05-30  (2,469 rows)
│   └── predictions/               # Written by model notebooks (gitignored by content)
│       ├── deepar_preds.parquet   # DeepAR test predictions
│       └── patchtst_preds.parquet # PatchTST test predictions
│
├── notebooks/
│   ├── eda.ipynb                  # Exploratory data analysis
│   ├── deepar.ipynb               # DeepAR training, tuning, evaluation  ← run 1st
│   ├── patchtst.ipynb             # PatchTST training, tuning, evaluation ← run 2nd
│   └── comparison.ipynb           # Side-by-side model comparison          ← run 3rd
│
├── scripts/
│   ├── data/                      # Data pipeline scripts (Data branch)
│   │   ├── fmp_client.py          # FMP API wrapper
│   │   ├── process_utils.py       # Feature engineering functions
│   │   ├── split_utils.py         # Walk-forward split logic
│   │   ├── 01_collect_raw.py      # Stage 1: Download raw JSON from FMP API
│   │   ├── 02_process.py          # Stage 2: Clean, merge, compute all features
│   │   └── 03_split.py            # Stage 3: Create train/val/test splits
│   └── models/
│       ├── __init__.py
│       ├── data_loader.py         # Parquet → NeuralForecast DataFrame
│       └── metrics.py             # All 7 evaluation metric functions
│
├── tests/
│   ├── data/                      # Tests for data pipeline (20 tests)
│   └── models/                    # Tests for model utilities (20 tests)
│       ├── conftest.py
│       ├── test_data_loader.py    # 7 tests
│       └── test_metrics.py        # 13 tests
│
├── requirements-data.txt          # Data pipeline dependencies
├── requirements-model.txt         # Model phase dependencies
└── .env                           # FMP_API_KEY — gitignored
```

---

## Prerequisites

### Option A — Google Colab (recommended, GPU available)

No local setup needed. Each notebook has a **Colab Setup cell** at the top that:
1. Installs `neuralforecast` automatically
2. Clones this repo (Model branch) into `/content/ECE1508_GenAI`
3. Prompts you to upload your local `data/splits/` parquet files
4. Reports GPU availability

**Always select a GPU runtime before running training:**
`Runtime → Change runtime type → T4 GPU → Save`

### Option B — Local (CPU only, slow)

```bash
pip install -r requirements-model.txt
```

**Warning:** Training DeepAR and PatchTST locally on CPU takes 2–8 hours per notebook. Colab is strongly recommended.

---

## Step 0 — Reproduce the Data (if you don't have the parquet files)

The `data/` directory is gitignored. If you are starting from scratch:

```bash
# Requires an FMP API key (financialmodelingprep.com — free tier works)
cp .env.example .env   # then add your FMP_API_KEY

pip install -r requirements-data.txt

python scripts/data/01_collect_raw.py   # ~10 min, ~200 API calls
python scripts/data/02_process.py       # ~30 sec
python scripts/data/03_split.py         # ~2 sec
```

This produces `data/splits/train.parquet`, `val.parquet`, and `test.parquet` — the only files the model notebooks need.

---

## Running the Model Notebooks

### Execution order

```
eda.ipynb  →  deepar.ipynb  →  patchtst.ipynb  →  comparison.ipynb
```

`deepar.ipynb` and `patchtst.ipynb` are independent of each other but both must finish before `comparison.ipynb`.

---

### 1. `notebooks/eda.ipynb` — Exploratory Data Analysis

**Purpose:** Understand the distribution, volatility structure, and feature correlations of the SPY dataset before modelling.

**Inputs:** `data/splits/train.parquet`, `val.parquet`, `test.parquet`

**Outputs:** No files written — all outputs are inline plots and print statements.

**What it shows:**
| Cell | Content |
|------|---------|
| 1 | Imports and display settings |
| 2 | Load all three splits, print row counts and NaN check |
| 3 | Histogram + QQ-plot of `return_1h` — shows heavy tails (kurtosis ~39) |
| 4 | Monthly realized volatility 2011–2025 — shows COVID spike and recent regimes |
| 5 | Correlation heatmap of all 22 features — `vix_change_1h` is top correlate (0.80) |
| 6 | Overnight gap bars (`is_first_bar=True`) vs intraday — 2.7× higher std |
| 7 | High-VIX vs low-VIX return distributions + scatter VIX vs realized vol |

**Runtime:** < 1 minute locally or on Colab.

---

### 2. `notebooks/deepar.ipynb` — DeepAR Model

**Purpose:** Train DeepAR (probabilistic LSTM), tune the lookback window on val, evaluate on the 2024–2025 test set.

**Inputs:** `data/splits/train.parquet`, `val.parquet`, `test.parquet`

**Outputs:** `data/predictions/deepar_preds.parquet`

**Architecture:**
- Model: LSTM encoder with probabilistic output (StudentT distribution)
- `h=1` — one-step-ahead forecast only
- Covariates: `is_first_bar` only via `futr_exog_list` (DeepAR cannot use historical-only covariates due to Monte Carlo rollout constraint)
- Lookback candidates: 24, 60, 120, 240 bars — tuned on val MAE
- Evaluation: `cross_validation(test_size=2469, step_size=1, refit=False)` — single fit, slide across test

**What each cell does:**
| Cell | Content |
|------|---------|
| 0 | Colab setup (auto-skipped locally) |
| 1 | Imports: NeuralForecast, DeepAR, DistributionLoss, MQLoss |
| 2 | Load splits via `data_loader.load_nf_dataframe()`, build trainval and full DataFrames |
| 3 | Lookback tuning loop — trains 4 DeepAR models (500 steps each), picks best val MAE |
| 4 | Final model — trains on train+val (1000 steps), runs cross_validation on test |
| 5 | Compute and print all 7 test metrics |
| 6 | Plot: predicted vs actual for first 200 test bars with 80%/90% prediction intervals |
| 7 | Save predictions to `data/predictions/deepar_preds.parquet` |

**Runtime on Colab T4 GPU:** ~15–30 minutes

**Saved prediction schema:**

| Column | Description |
|--------|-------------|
| `ds` | Integer index matching test split |
| `datetime` | Original hourly timestamp |
| `y` | Actual `return_1h` |
| `pred` | DeepAR point prediction (mean) |
| `lo_80` / `hi_80` | 80% prediction interval bounds |
| `lo_90` / `hi_90` | 90% prediction interval bounds |
| `model` | `"DeepAR"` |

---

### 3. `notebooks/patchtst.ipynb` — PatchTST Model

**Purpose:** Train PatchTST (patch-based Transformer) in fully multivariate mode, tune the lookback window on val, evaluate on the 2024–2025 test set.

**Library:** HuggingFace `transformers` (`PatchTSTForPrediction`) — used instead of NeuralForecast because NeuralForecast's PatchTST does not expose multivariate input. HuggingFace implements the original paper's channel-independent architecture where all 21 channels are processed simultaneously.

**Inputs:** `data/splits/train.parquet`, `val.parquet`, `test.parquet`

**Outputs:** `data/predictions/patchtst_preds.parquet`

**Architecture:**
- Model: Patch-based Transformer, channel-independent mode (original paper)
- Input: **21 channels** — `return_1h` (target, channel 0) + all 20 features. PatchTST patches each channel independently across time, applies self-attention within each channel, then predicts all channels at t+1 simultaneously. Only channel 0's prediction is used.
- DeepAR requires future covariate values due to autoregressive rollout — PatchTST does not (direct forecast), so using all 20 historical features is architecturally valid here
- Custom `ZScoreScaler` normalises all 21 channels (fit on train only)
- Prediction intervals: empirical quantiles from val residuals (conformal calibration)
- Lookback candidates: 24, 60, 120, 240 bars

**What each cell does:**
| Cell | Content |
|------|---------|
| 0 | Colab setup — installs `transformers`, clones repo, uploads splits |
| 1 | Imports: torch, HuggingFace PatchTST, constants |
| 2 | Helpers: `ZScoreScaler`, `WindowDataset`, `build_model`, `train_model`, `predict_walkforward` |
| 3 | Load splits, cast bool columns, normalise, print shapes |
| 4 | Lookback tuning — trains 4 models (500 steps each), saves val predictions from best |
| 5 | Final model — trains on train+val (1000 steps) |
| 6 | Walk-forward test predictions, empirical intervals, all 7 metrics |
| 7 | Plot: predicted vs actual for first 200 test bars with 80%/90% intervals |
| 8 | Save predictions to `data/predictions/patchtst_preds.parquet` |

**Runtime on Colab T4 GPU:** ~20–40 minutes

**Saved prediction schema:** Same columns as DeepAR, with `model = "PatchTST"`.

---

### 4. `notebooks/comparison.ipynb` — Model Comparison

**Purpose:** Load both model's saved predictions and produce a complete side-by-side evaluation.

**Inputs:** `data/predictions/deepar_preds.parquet`, `data/predictions/patchtst_preds.parquet`

**Prerequisite:** Run `deepar.ipynb` and `patchtst.ipynb` first.

**What each cell does:**
| Cell | Content |
|------|---------|
| 1 | Imports |
| 2 | Load both parquets, verify lengths match |
| 3 | Side-by-side metrics table (all 7 metrics for both models) |
| 4 | Cumulative return plot: DeepAR long/short vs PatchTST long/short vs Buy & Hold |
| 5 | 90% prediction interval width comparison — measures forecast sharpness |
| 6 | Regime analysis — all 7 metrics split by low-VIX vs high-VIX periods |

**Runtime:** < 1 minute.

---

## Evaluation Metrics

All metrics are computed on the **test set only** (2024-01-02 → 2025-05-30, 2,469 bars).

| Metric | Definition | What it measures |
|--------|-----------|-----------------|
| RMSE | √mean((y_pred − y_true)²) | Point forecast accuracy |
| MAE | mean(|y_pred − y_true|) | Point forecast accuracy (robust to outliers) |
| Directional Acc | % where sign(y_pred) == sign(y_true) | Trading signal quality |
| Coverage 80% | % actuals inside 80% interval | Interval calibration (target: 80%) |
| Coverage 90% | % actuals inside 90% interval | Interval calibration (target: 90%) |
| Sharpe Ratio | mean(sign(pred)×actual) / std × √1680 | Risk-adjusted long/short strategy return |
| Max Drawdown | worst peak-to-trough cumulative loss | Strategy risk |

`1680 = 6.5 bars/day × 252 trading days` (annualization for hourly SPY bars).

---

## Model Utility Code (`scripts/models/`)

These modules are imported by all four notebooks. They are unit-tested — run `pytest tests/models/ -v` to verify (20 tests, all should pass locally without GPU).

### `scripts/models/data_loader.py`

| Symbol | Description |
|--------|-------------|
| `HIST_EXOG_COLS` | List of 20 feature columns used by PatchTST as `hist_exog_list` |
| `FUTR_EXOG_COLS` | `['is_first_bar']` — DeepAR's only allowed exogenous covariate |
| `load_nf_dataframe(path, extra_cols)` | Loads a split parquet and returns `unique_id \| ds \| y \| [extra_cols]` with integer `ds` (not timestamps — avoids market-hours gap issues in PatchTST positional encoding) |
| `build_full_df([df1, df2, ...])` | Concatenates NeuralForecast DataFrames and reassigns a single contiguous `ds` sequence (required before cross_validation) |

### `scripts/models/metrics.py`

| Function | Signature |
|---------|-----------|
| `rmse(y_true, y_pred)` | → float |
| `mae(y_true, y_pred)` | → float |
| `directional_accuracy(y_true, y_pred)` | → float |
| `interval_coverage(y_true, lo, hi)` | → float |
| `sharpe_ratio(y_true, y_pred, periods_per_year=1680)` | → float |
| `max_drawdown(y_true, y_pred)` | → float |
| `compute_all(y_true, y_pred, lo_80, hi_80, lo_90, hi_90)` | → dict with all 7 keys |

---

## Running on Google Colab — Step-by-Step

### First time (new Colab session)

1. Open [Google Colab](https://colab.research.google.com)
2. **File → Open notebook → GitHub** tab → paste repo URL:
   `https://github.com/WoodyChang21/ECE1508_GenAI`
   → select branch `Model` → open `notebooks/deepar.ipynb`
3. **Runtime → Change runtime type → T4 GPU → Save**
4. Run Cell 0 (Colab setup):
   - It installs `neuralforecast` (~2 min)
   - It clones the repo
   - It prompts you to upload your local `data/splits/train.parquet`, `val.parquet`, `test.parquet`
5. Run all remaining cells (Cells 1–7) in order

### After deepar.ipynb completes

- Download `data/predictions/deepar_preds.parquet` from the Colab file browser (left panel → `/content/ECE1508_GenAI/data/predictions/`)
- **Do not start a new runtime session** — open `patchtst.ipynb` in the same session to reuse the uploaded data and installed packages
  - File → Open → navigate to `notebooks/patchtst.ipynb`
  - Cell 0 will detect the splits are already present and skip re-upload
  - Run all cells

### After patchtst.ipynb completes

- Download `data/predictions/patchtst_preds.parquet`
- Open `comparison.ipynb` in the same session
- Run all cells

### Bringing predictions back to local repo

Copy the two parquet files into your local `data/predictions/` folder. They are gitignored by default — if you want to commit them, remove `data/predictions/` from `.gitignore`.

---

## Key Design Decisions

### Why integer `ds` instead of real timestamps?

NeuralForecast uses `ds` to infer frequency and compute positional encodings. Market data has irregular gaps (weekends, public holidays) that cause `pandas.infer_freq` to fail and corrupt PatchTST's patch-positional encoding. Assigning sequential integers sidesteps this entirely. `freq=1` tells NeuralForecast to treat adjacent integers as unit-spaced.

### Why only `is_first_bar` for DeepAR?

DeepAR generates predictions via Monte Carlo rollout: it samples from its learned distribution at each step. Multi-step rollout needs future covariate values. NeuralForecast's DeepAR enforces this at `h=1` too — `hist_exog_list` is simply not supported. Only covariates computable for any future bar (like a calendar flag) can be used as `futr_exog_list`. `is_first_bar` — whether the next bar is the first of a trading day — is deterministic from the timestamp.

### Why is this comparison fair at `h=1`?

At one-step-ahead, DeepAR's autoregressive rollout limitation is minimised (no recursive error accumulation). PatchTST's advantage is access to all 20 historical features. The covariate asymmetry is a genuine architectural difference between the two model families — not a flaw in the experiment — and is documented as a research finding.

### Why `cross_validation(..., refit=False)`?

Training on all of train+val once and sliding the evaluation window across the test set simulates production deployment: a model trained at the start of 2024 making sequential predictions throughout 2024–2025. `refit=True` would retrain at each step, which is computationally prohibitive for this dataset size.

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
| Data source | Financial Modeling Prep (FMP) stable API |
| VIX proxy | VIXY (ProShares VIX Short-Term Futures ETF) |

| Split | Date Range | Rows | Purpose |
|-------|-----------|------|---------|
| Train | 2011-01-04 → 2022-12-30 | 21,028 | Model training |
| Val | 2023-01-03 → 2023-12-29 | 1,744 | Lookback window selection |
| Test | 2024-01-02 → 2025-05-30 | 2,469 | Final held-out evaluation |

---

## Running Tests

```bash
# Data pipeline tests (20 tests — requires no GPU, no data files)
pytest tests/data/ -v

# Model utility tests (20 tests — requires no GPU, no data files)
pytest tests/models/ -v

# All tests
pytest tests/ -v
# Expected: 40 passed
```

---

## Reproducing from Scratch

```bash
# 1. Clone and set up
git clone https://github.com/WoodyChang21/ECE1508_GenAI.git
cd ECE1508_GenAI
git checkout Model

# 2. Install data pipeline dependencies
pip install -r requirements-data.txt

# 3. Add FMP API key
echo "FMP_API_KEY=your_key_here" > .env

# 4. Run data pipeline (local)
python scripts/data/01_collect_raw.py    # ~10 min
python scripts/data/02_process.py        # ~30 sec
python scripts/data/03_split.py          # ~2 sec

# 5. Run unit tests
pytest tests/ -v                         # 40 passed

# 6. Open notebooks in Colab (see "Running on Google Colab" section above)
#    Run in order: eda → deepar → patchtst → comparison
```
