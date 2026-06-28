# Model Implementation Design: DeepAR vs PatchTST

## Goal

Implement and compare DeepAR (probabilistic autoregressive benchmark) and PatchTST (Transformer-based primary model) for one-step-ahead intraday SPY return forecasting using NeuralForecast (Nixtla).

## Architecture

Both models are implemented in self-contained Jupyter notebooks. A small `scripts/models/` utility layer handles data formatting and metric computation shared across notebooks. Each model notebook saves predictions to `data/predictions/` and `comparison.ipynb` loads them for side-by-side evaluation.

## Tech Stack

- **NeuralForecast** (Nixtla) — DeepAR and PatchTST implementations, training loop, cross-validation
- **PyTorch** — backend (installed as NeuralForecast dependency)
- **pandas / pyarrow** — data I/O
- **matplotlib / seaborn** — visualisation in comparison notebook

---

## Project Structure

```
notebooks/
├── eda.ipynb           # Exploratory data analysis
├── deepar.ipynb        # DeepAR training, lookback tuning, test evaluation
├── patchtst.ipynb      # PatchTST training, lookback tuning, test evaluation
└── comparison.ipynb    # Side-by-side metrics, plots, trading simulation

scripts/models/
├── __init__.py
├── data_loader.py      # Parquet → NeuralForecast DataFrame
└── metrics.py          # All evaluation metric functions

data/predictions/       # Written by model notebooks, read by comparison.ipynb
├── deepar_preds.parquet
└── patchtst_preds.parquet

requirements-model.txt
```

---

## Data Format

NeuralForecast requires a long-format DataFrame with columns `unique_id | ds | y | [covariates]`.

**Key design decision — integer `ds`:** Real timestamps are not used because market data has gaps (weekends, holidays) that corrupt PatchTST's positional encoding. `ds` is a sequential integer index instead.

```python
# scripts/models/data_loader.py
def load_nf_dataframe(split_path: str, extra_cols: list[str] = None) -> pd.DataFrame:
    df = pd.read_parquet(split_path)
    nf = df.copy()
    nf = nf.rename(columns={'return_1h': 'y'})
    nf['unique_id'] = 'SPY'
    nf['ds'] = range(len(nf))
    cols = ['unique_id', 'ds', 'y'] + (extra_cols or [])
    return nf[cols]
```

The caller passes `extra_cols` to select covariates appropriate for each model (see below).

**train split:** `data/splits/train.parquet` — 21,028 rows (2011-01-04 to 2022-12-30)
**val split:** `data/splits/val.parquet` — 1,744 rows (2023-01-03 to 2023-12-29)
**test split:** `data/splits/test.parquet` — 2,469 rows (2024-01-02 to 2025-05-30)

`ds` integers are assigned independently per split to avoid leakage. When concatenating train+val for final training, `ds` is reassigned to a single contiguous sequence.

---

## Covariate Strategy

DeepAR and PatchTST have different covariate support due to their architectures.

| Model | Constraint | Covariates used | Rationale |
|---|---|---|---|
| DeepAR | No `hist_exog_list` support (MC rollout requires future values) | `is_first_bar` via `futr_exog_list` | Calendar flag — computable for any future timestep |
| PatchTST | Supports `hist_exog_list` (no recursive rollout) | All 20 non-target features via `hist_exog_list` | Direct forecasting uses only past values |

The 20 PatchTST features are: `open, high, low, close, volume, return_4h, return_24h, is_first_bar, vol_24h, vol_60h, volume_ratio, rsi_14, macd, macd_signal, macd_diff, bb_upper, bb_lower, bb_width, vix_log, vix_change_1h`.

This asymmetry is a research finding, not a flaw. DeepAR's architectural limitation in financial markets (unknown future covariates) is documented as part of the model comparison.

---

## Model Configuration

### DeepAR

```python
from neuralforecast.models import DeepAR
from neuralforecast.losses.pytorch import DistributionLoss, MQLoss

DeepAR(
    h=1,
    input_size=120,                  # tuned on val; candidates: [24, 60, 120, 240]
    lstm_hidden_size=128,
    lstm_n_layers=2,
    lstm_dropout=0.1,
    trajectory_samples=200,
    loss=DistributionLoss(distribution='StudentT', level=[80, 90]),
    valid_loss=MQLoss(level=[80, 90]),
    futr_exog_list=['is_first_bar'],
    max_steps=1000,
    early_stop_patience_steps=50,
    scaler_type='standard',
)
```

Output columns from `predict()`: `DeepAR`, `DeepAR-lo-80`, `DeepAR-hi-80`, `DeepAR-lo-90`, `DeepAR-hi-90`.

### PatchTST

```python
from neuralforecast.models import PatchTST
from neuralforecast.losses.pytorch import MQLoss

PatchTST(
    h=1,
    input_size=120,                  # tuned on val independently; same candidates
    patch_len=16,
    stride=8,
    d_model=128,
    n_heads=16,
    n_layers=3,
    dropout=0.2,
    loss=MQLoss(level=[80, 90]),
    hist_exog_list=[                 # all 20 non-target features
        'open', 'high', 'low', 'close', 'volume',
        'return_4h', 'return_24h', 'is_first_bar',
        'vol_24h', 'vol_60h', 'volume_ratio',
        'rsi_14', 'macd', 'macd_signal', 'macd_diff',
        'bb_upper', 'bb_lower', 'bb_width',
        'vix_log', 'vix_change_1h',
    ],
    max_steps=1000,
    early_stop_patience_steps=50,
    scaler_type='standard',
)
```

Output columns from `predict()`: `PatchTST-median`, `PatchTST-lo-80`, `PatchTST-hi-80`, `PatchTST-lo-90`, `PatchTST-hi-90`.

---

## Lookback Window Tuning

Both models independently tune `input_size` on the val split. Procedure in each model notebook:

1. For each candidate in `[24, 60, 120, 240]`:
   - Instantiate model with that `input_size`
   - `nf.fit(df=train_df, val_size=len(val_df))`
   - `preds = nf.predict()` on val
   - Compute val MAE
2. Select `input_size` with lowest val MAE
3. Re-train on `concat(train_df, val_df)` with the best `input_size`
4. Evaluate on `test_df`

---

## Training Setup (both models)

```python
from neuralforecast import NeuralForecast

nf = NeuralForecast(models=[model], freq=1)
nf.fit(df=train_df, val_size=len(val_df))
preds = nf.predict(futr_df=futr_df)   # futr_df needed for DeepAR (is_first_bar at t+1)
```

For PatchTST, `predict()` is called without `futr_df` since all covariates are historical.

---

## Evaluation Metrics

Defined in `scripts/models/metrics.py`. All metrics computed on the test set only.

### Point Metrics

```python
def rmse(y_true, y_pred): ...
def mae(y_true, y_pred): ...
def directional_accuracy(y_true, y_pred): ...
    # % of predictions where sign(y_pred) == sign(y_true)
```

### Probabilistic Metrics

```python
def interval_coverage(y_true, lo, hi): ...
    # % of actuals that fall within [lo, hi]
    # Expected: ~80% for 80% interval, ~90% for 90% interval
```

### Trading Metrics

```python
def sharpe_ratio(y_true, y_pred, periods_per_year=1680): ...
    # Long if y_pred > 0, short if y_pred < 0
    # Signal * actual return = strategy return per bar
    # Annualized Sharpe: mean / std * sqrt(periods_per_year)
    # 1680 = ~6.5 bars/day * 252 trading days

def max_drawdown(y_true, y_pred): ...
    # Cumulative strategy returns, worst peak-to-trough loss
```

---

## Notebook Responsibilities

### `eda.ipynb`
- Load `data/splits/train.parquet`
- Distribution of `return_1h` (histogram, QQ-plot)
- Rolling volatility over time
- Correlation matrix of all 22 features
- `is_first_bar` vs non-first-bar return comparison (overnight gap effect)
- VIX regime analysis (high vs low vol periods)

### `deepar.ipynb`
- Load train/val/test, format via `data_loader.load_nf_dataframe`
- Lookback tuning loop (val MAE per candidate)
- Final train on train+val with best `input_size`
- Predict on test, compute all metrics via `metrics.py`
- Plot: predicted vs actual returns (first 200 test bars), 90% prediction intervals
- Save `data/predictions/deepar_preds.parquet`

### `patchtst.ipynb`
- Same structure as `deepar.ipynb`
- Uses all 20 historical features
- Save `data/predictions/patchtst_preds.parquet`

### `comparison.ipynb`
- Load both prediction parquets
- Side-by-side metrics table (all 7 metrics)
- Cumulative return plot: DeepAR strategy vs PatchTST strategy vs Buy-and-Hold
- Prediction interval width comparison (sharpness)
- Regime analysis: performance in high-vol vs low-vol periods (split by median `vix_log`)

---

## Saved Prediction Format

Both model notebooks save predictions in the same schema so `comparison.ipynb` can load either identically:

| Column | Description |
|---|---|
| `ds` | Integer index (matches test split) |
| `datetime` | Original timestamp (for plotting) |
| `y` | Actual `return_1h` |
| `pred` | Point prediction (median) |
| `lo_80` | Lower bound of 80% prediction interval |
| `hi_80` | Upper bound of 80% prediction interval |
| `lo_90` | Lower bound of 90% prediction interval |
| `hi_90` | Upper bound of 90% prediction interval |
| `model` | `"DeepAR"` or `"PatchTST"` |

---

## Dependencies

```
# requirements-model.txt
neuralforecast>=2.0.0
torch>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
jupyter>=1.0.0
ipykernel>=6.0.0
```

---

## Known Constraints and Decisions

**DeepAR `hist_exog_list` not supported:** Due to Monte Carlo sampling, DeepAR cannot condition on historical-only covariates during multi-step rollout. At h=1 this is less damaging but the NeuralForecast implementation enforces this constraint regardless. Mitigated by using `is_first_bar` as `futr_exog_list`.

**Integer `ds`:** Market data has irregular gaps (weekends, holidays). Using real timestamps causes `pandas.infer_freq` to fail and corrupts PatchTST's positional encoding. Integer indexing is the recommended workaround for financial data.

**`futr_df` for DeepAR:** At predict time, NeuralForecast requires `futr_df` containing the `futr_exog_list` values for the `h=1` horizon step. `is_first_bar` for the next bar can be computed deterministically from the timestamp.

**Scaler:** Both models use `scaler_type='standard'` (z-score normalization per series). Returns are already near-stationary but the scale varies across regimes; standardization helps training stability.
