# Model Implementation Plan: DeepAR vs PatchTST

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement DeepAR and PatchTST one-step-ahead SPY return forecasting using NeuralForecast, with shared utility code, self-contained Jupyter notebooks per model, and a comparison notebook that loads saved predictions.

**Architecture:** A `scripts/models/` utility layer (data_loader + metrics) is unit-tested with pytest and imported by four Jupyter notebooks (eda, deepar, patchtst, comparison). Both models use integer `ds` indexing to avoid market-hours timestamp gaps. DeepAR uses `is_first_bar` as its only `futr_exog`; PatchTST uses all 20 historical features as `hist_exog_list`. Test evaluation uses NeuralForecast's `cross_validation()` with `refit=False` to simulate walk-forward prediction across the 2024–2025 test set without retraining.

**Tech Stack:** NeuralForecast >= 2.0.0, PyTorch >= 2.0.0, pandas, matplotlib, seaborn, jupyter, pytest

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `requirements-model.txt` | Create | Model-phase pip dependencies |
| `scripts/models/__init__.py` | Create | Package marker |
| `scripts/models/data_loader.py` | Create | Parquet → NeuralForecast DataFrame; exports `HIST_EXOG_COLS`, `FUTR_EXOG_COLS` |
| `scripts/models/metrics.py` | Create | rmse, mae, directional_accuracy, interval_coverage, sharpe_ratio, max_drawdown, compute_all |
| `tests/models/__init__.py` | Create | Test package marker |
| `tests/models/conftest.py` | Create | sys.path insertion so tests can import `scripts.models` |
| `tests/models/test_data_loader.py` | Create | 7 unit tests for data_loader |
| `tests/models/test_metrics.py` | Create | 13 unit tests for metrics |
| `notebooks/eda.ipynb` | Create | EDA: return distribution, rolling vol, correlation, is_first_bar, VIX regime |
| `notebooks/deepar.ipynb` | Create | DeepAR: lookback tuning, cross_validation on test, metrics, plots, save preds |
| `notebooks/patchtst.ipynb` | Create | PatchTST: same pipeline with all 20 hist_exog features |
| `notebooks/comparison.ipynb` | Create | Load both parquets, side-by-side table, cumulative return, interval width, regime |
| `data/predictions/.gitkeep` | Create | Track empty directory in git |

---

### Task 1: Project scaffold

**Files:**
- Create: `requirements-model.txt`
- Create: `scripts/models/__init__.py`
- Create: `tests/models/__init__.py`
- Create: `tests/models/conftest.py`
- Create: `data/predictions/.gitkeep`

- [ ] **Step 1: Create requirements-model.txt**

```
neuralforecast>=2.0.0
torch>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
jupyter>=1.0.0
ipykernel>=6.0.0
scipy>=1.10.0
```

- [ ] **Step 2: Create package markers**

Create `scripts/models/__init__.py` — empty file.

Create `tests/models/__init__.py` — empty file.

- [ ] **Step 3: Create conftest.py**

Create `tests/models/conftest.py`:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
```

- [ ] **Step 4: Create predictions directory**

Create `data/predictions/.gitkeep` — empty file.

- [ ] **Step 5: Install dependencies**

```bash
pip install -r requirements-model.txt
```

Expected: neuralforecast, torch, matplotlib, seaborn, jupyter, scipy all install without error.

- [ ] **Step 6: Commit**

```bash
git add requirements-model.txt scripts/models/__init__.py tests/models/__init__.py tests/models/conftest.py data/predictions/.gitkeep
git commit -m "feat(model): project scaffold — requirements, package markers, predictions dir"
```

---

### Task 2: data_loader.py

**Files:**
- Create: `scripts/models/data_loader.py`
- Create: `tests/models/test_data_loader.py`

- [ ] **Step 1: Write failing tests**

Create `tests/models/test_data_loader.py`:

```python
import pandas as pd
import numpy as np
import pytest

from scripts.models.data_loader import load_nf_dataframe, build_full_df, HIST_EXOG_COLS, FUTR_EXOG_COLS


@pytest.fixture
def sample_parquet(tmp_path):
    df = pd.DataFrame({
        'datetime': pd.date_range('2023-01-03 09:30', periods=10, freq='h'),
        'open': np.ones(10) * 400.0,
        'high': np.ones(10) * 401.0,
        'low': np.ones(10) * 399.0,
        'close': np.ones(10) * 400.0,
        'volume': np.ones(10, dtype=int) * 1_000_000,
        'return_1h': np.linspace(0.001, 0.010, 10),
        'return_4h': np.linspace(0.002, 0.020, 10),
        'return_24h': np.linspace(0.003, 0.030, 10),
        'is_first_bar': [True] + [False] * 9,
        'vol_24h': np.ones(10) * 0.002,
        'vol_60h': np.ones(10) * 0.002,
        'volume_ratio': np.ones(10) * 1.0,
        'rsi_14': np.ones(10) * 50.0,
        'macd': np.zeros(10),
        'macd_signal': np.zeros(10),
        'macd_diff': np.zeros(10),
        'bb_upper': np.ones(10) * 405.0,
        'bb_lower': np.ones(10) * 395.0,
        'bb_width': np.ones(10) * 2.5,
        'vix_log': np.ones(10) * 3.0,
        'vix_change_1h': np.zeros(10),
    })
    path = tmp_path / 'split.parquet'
    df.to_parquet(path)
    return str(path)


def test_required_columns_present(sample_parquet):
    df = load_nf_dataframe(sample_parquet)
    assert {'unique_id', 'ds', 'y'}.issubset(df.columns)


def test_unique_id_is_spy(sample_parquet):
    df = load_nf_dataframe(sample_parquet)
    assert (df['unique_id'] == 'SPY').all()


def test_ds_is_sequential_integers_from_zero(sample_parquet):
    df = load_nf_dataframe(sample_parquet)
    assert list(df['ds']) == list(range(10))


def test_y_equals_return_1h(sample_parquet):
    df = load_nf_dataframe(sample_parquet)
    raw = pd.read_parquet(sample_parquet)
    np.testing.assert_array_almost_equal(df['y'].values, raw['return_1h'].values)


def test_extra_cols_included(sample_parquet):
    df = load_nf_dataframe(sample_parquet, extra_cols=['is_first_bar', 'rsi_14'])
    assert 'is_first_bar' in df.columns
    assert 'rsi_14' in df.columns


def test_no_extra_cols_by_default(sample_parquet):
    df = load_nf_dataframe(sample_parquet)
    assert list(df.columns) == ['unique_id', 'ds', 'y']


def test_build_full_df_reassigns_contiguous_ds(sample_parquet):
    a = load_nf_dataframe(sample_parquet)
    b = load_nf_dataframe(sample_parquet)
    full = build_full_df([a, b])
    assert list(full['ds']) == list(range(20))
    assert len(full) == 20
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/models/test_data_loader.py -v
```

Expected: `ModuleNotFoundError: No module named 'scripts.models.data_loader'`

- [ ] **Step 3: Implement data_loader.py**

Create `scripts/models/data_loader.py`:

```python
import pandas as pd

HIST_EXOG_COLS = [
    'open', 'high', 'low', 'close', 'volume',
    'return_4h', 'return_24h', 'is_first_bar',
    'vol_24h', 'vol_60h', 'volume_ratio',
    'rsi_14', 'macd', 'macd_signal', 'macd_diff',
    'bb_upper', 'bb_lower', 'bb_width',
    'vix_log', 'vix_change_1h',
]

FUTR_EXOG_COLS = ['is_first_bar']


def load_nf_dataframe(split_path: str, extra_cols: list[str] | None = None) -> pd.DataFrame:
    """Load a split parquet and return a NeuralForecast-compatible DataFrame.

    Columns returned: unique_id, ds (sequential int starting at 0), y (= return_1h),
    plus any columns listed in extra_cols.

    Integer ds avoids market-hours timestamp gap issues with PatchTST positional encoding.
    """
    df = pd.read_parquet(split_path)
    nf = df.copy()
    nf['unique_id'] = 'SPY'
    nf['ds'] = range(len(nf))
    nf = nf.rename(columns={'return_1h': 'y'})
    cols = ['unique_id', 'ds', 'y'] + (extra_cols or [])
    return nf[cols].reset_index(drop=True)


def build_full_df(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate NeuralForecast DataFrames and reassign a single contiguous ds sequence."""
    combined = pd.concat(dfs, ignore_index=True)
    combined['ds'] = range(len(combined))
    return combined
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/models/test_data_loader.py -v
```

Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/models/data_loader.py tests/models/test_data_loader.py
git commit -m "feat(model): data_loader — parquet to NeuralForecast format with integer ds"
```

---

### Task 3: metrics.py

**Files:**
- Create: `scripts/models/metrics.py`
- Create: `tests/models/test_metrics.py`

- [ ] **Step 1: Write failing tests**

Create `tests/models/test_metrics.py`:

```python
import numpy as np
import pytest
from scripts.models.metrics import (
    rmse, mae, directional_accuracy,
    interval_coverage, sharpe_ratio, max_drawdown, compute_all,
)


def test_rmse_perfect_forecast():
    y = np.array([1.0, 2.0, 3.0])
    assert rmse(y, y) == pytest.approx(0.0)


def test_rmse_known_value():
    y_true = np.array([0.0, 0.0])
    y_pred = np.array([1.0, 1.0])
    assert rmse(y_true, y_pred) == pytest.approx(1.0)


def test_mae_known_value():
    y_true = np.array([0.0, 0.0, 0.0])
    y_pred = np.array([1.0, -1.0, 2.0])
    assert mae(y_true, y_pred) == pytest.approx(4.0 / 3.0)


def test_directional_accuracy_all_correct():
    y_true = np.array([1.0, -1.0, 1.0])
    y_pred = np.array([0.5, -0.5, 0.3])
    assert directional_accuracy(y_true, y_pred) == pytest.approx(1.0)


def test_directional_accuracy_all_wrong():
    y_true = np.array([1.0, -1.0, 1.0])
    y_pred = np.array([-0.5, 0.5, -0.3])
    assert directional_accuracy(y_true, y_pred) == pytest.approx(0.0)


def test_directional_accuracy_half():
    y_true = np.array([1.0, -1.0])
    y_pred = np.array([1.0, 1.0])
    assert directional_accuracy(y_true, y_pred) == pytest.approx(0.5)


def test_interval_coverage_all_inside():
    y = np.array([0.0, 0.5, -0.5])
    lo = np.array([-1.0, -1.0, -1.0])
    hi = np.array([1.0, 1.0, 1.0])
    assert interval_coverage(y, lo, hi) == pytest.approx(1.0)


def test_interval_coverage_none_inside():
    y = np.array([2.0, 3.0])
    lo = np.array([-1.0, -1.0])
    hi = np.array([1.0, 1.0])
    assert interval_coverage(y, lo, hi) == pytest.approx(0.0)


def test_sharpe_ratio_positive_for_perfect_directional_forecast():
    # Perfect directional forecast: strategy always takes correct side
    y_true = np.array([0.01, -0.005, 0.008, -0.003] * 25)
    y_pred = y_true.copy()  # sign always matches
    assert sharpe_ratio(y_true, y_pred) > 0


def test_sharpe_ratio_negative_for_always_wrong_forecast():
    y_true = np.array([0.01, -0.005, 0.008, -0.003] * 25)
    y_pred = -y_true  # sign always wrong
    assert sharpe_ratio(y_true, y_pred) < 0


def test_max_drawdown_zero_for_monotonic_gains():
    # Strategy always wins: returns are all positive
    y_true = np.array([0.01] * 20)
    y_pred = np.array([0.01] * 20)  # predict positive, actual positive → +0.01 each step
    assert max_drawdown(y_true, y_pred) == pytest.approx(0.0, abs=1e-10)


def test_max_drawdown_is_non_positive():
    y_true = np.array([0.01, -0.02, 0.01, -0.03])
    y_pred = np.array([0.01, 0.02, 0.01, 0.03])  # wrong on negatives
    assert max_drawdown(y_true, y_pred) <= 0.0


def test_compute_all_returns_all_keys():
    y_true = np.array([0.01, -0.005, 0.008] * 10)
    y_pred = np.array([0.01, -0.005, 0.008] * 10)
    lo_80 = y_pred - 0.02
    hi_80 = y_pred + 0.02
    lo_90 = y_pred - 0.03
    hi_90 = y_pred + 0.03
    result = compute_all(y_true, y_pred, lo_80, hi_80, lo_90, hi_90)
    assert set(result.keys()) == {
        'rmse', 'mae', 'dir_acc', 'coverage_80', 'coverage_90', 'sharpe', 'max_drawdown'
    }
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/models/test_metrics.py -v
```

Expected: `ModuleNotFoundError: No module named 'scripts.models.metrics'`

- [ ] **Step 3: Implement metrics.py**

Create `scripts/models/metrics.py`:

```python
import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)))


def interval_coverage(y_true: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    return float(np.mean((y_true >= lo) & (y_true <= hi)))


def sharpe_ratio(y_true: np.ndarray, y_pred: np.ndarray, periods_per_year: int = 1680) -> float:
    """Annualized Sharpe of a long/short strategy driven by predicted sign.

    1680 = ~6.5 bars/day * 252 trading days (hourly SPY market hours).
    """
    strategy_returns = np.sign(y_pred) * y_true
    std = np.std(strategy_returns)
    if std == 0:
        return 0.0
    return float(np.mean(strategy_returns) / std * np.sqrt(periods_per_year))


def max_drawdown(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Worst peak-to-trough cumulative loss of the long/short strategy. Always <= 0."""
    strategy_returns = np.sign(y_pred) * y_true
    cumulative = np.cumsum(strategy_returns)
    running_max = np.maximum.accumulate(cumulative)
    return float(np.min(cumulative - running_max))


def compute_all(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    lo_80: np.ndarray,
    hi_80: np.ndarray,
    lo_90: np.ndarray,
    hi_90: np.ndarray,
) -> dict:
    return {
        'rmse':         rmse(y_true, y_pred),
        'mae':          mae(y_true, y_pred),
        'dir_acc':      directional_accuracy(y_true, y_pred),
        'coverage_80':  interval_coverage(y_true, lo_80, hi_80),
        'coverage_90':  interval_coverage(y_true, lo_90, hi_90),
        'sharpe':       sharpe_ratio(y_true, y_pred),
        'max_drawdown': max_drawdown(y_true, y_pred),
    }
```

- [ ] **Step 4: Run all tests**

```bash
pytest tests/ -v
```

Expected: 40 passed (20 data-pipeline + 7 data_loader + 13 metrics).

- [ ] **Step 5: Commit**

```bash
git add scripts/models/metrics.py tests/models/test_metrics.py
git commit -m "feat(model): evaluation metrics — RMSE, MAE, directional accuracy, interval coverage, Sharpe, drawdown"
```

---

### Task 4: eda.ipynb

**Files:**
- Create: `notebooks/eda.ipynb`

No unit tests — verified by running all cells without error.

- [ ] **Step 1: Create notebooks/eda.ipynb**

Create the notebook with exactly these cells in order. Use Jupyter's File → New Notebook, or create the JSON directly.

**Cell 1 — Setup:**
```python
import sys
sys.path.insert(0, '..')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

sns.set_theme(style='darkgrid')
plt.rcParams['figure.figsize'] = (14, 4)
```

**Cell 2 — Load splits:**
```python
train = pd.read_parquet('../data/splits/train.parquet')
val   = pd.read_parquet('../data/splits/val.parquet')
test  = pd.read_parquet('../data/splits/test.parquet')
full  = pd.concat([train, val, test], ignore_index=True)

print(f"Train: {len(train):,} rows  ({train['datetime'].min().date()} → {train['datetime'].max().date()})")
print(f"Val:   {len(val):,}  rows  ({val['datetime'].min().date()} → {val['datetime'].max().date()})")
print(f"Test:  {len(test):,}  rows  ({test['datetime'].min().date()} → {test['datetime'].max().date()})")
print(f"\nColumns: {list(full.columns)}")
print(f"\nNaN counts:\n{full.isnull().sum()}")
```

**Cell 3 — Return distribution:**
```python
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

train['return_1h'].hist(bins=120, ax=axes[0], edgecolor='none')
axes[0].set_title('return_1h distribution (train)')
axes[0].set_xlabel('return_1h')
axes[0].set_ylabel('count')

stats.probplot(train['return_1h'].dropna(), plot=axes[1])
axes[1].set_title('QQ-plot: return_1h vs Normal')

plt.tight_layout()
plt.show()

desc = train['return_1h'].describe()
print(desc.to_string())
print(f"\nSkewness : {train['return_1h'].skew():.4f}")
print(f"Kurtosis : {train['return_1h'].kurtosis():.4f}")
```

**Cell 4 — Rolling volatility over time:**
```python
monthly_vol = (
    full.set_index('datetime')['return_1h']
    .resample('ME')
    .std()
)

fig, ax = plt.subplots(figsize=(14, 3))
monthly_vol.plot(ax=ax)
ax.set_title('Monthly realized volatility of return_1h (2011–2025)')
ax.set_ylabel('std(return_1h)')
ax.axvline(pd.Timestamp('2023-01-01'), color='red', linestyle='--', alpha=0.5, label='Val start')
ax.axvline(pd.Timestamp('2024-01-01'), color='orange', linestyle='--', alpha=0.5, label='Test start')
ax.legend()
plt.tight_layout()
plt.show()
```

**Cell 5 — Feature correlation matrix:**
```python
num_cols = [c for c in full.columns if c not in ['datetime', 'is_first_bar']]
corr = full[num_cols].corr()

plt.figure(figsize=(14, 11))
sns.heatmap(corr, annot=False, cmap='coolwarm', center=0, vmin=-1, vmax=1,
            linewidths=0.3, square=True)
plt.title('Feature correlation matrix (full dataset)')
plt.tight_layout()
plt.show()

# Top correlates with target
print("Top correlates with return_1h:")
print(corr['return_1h'].drop('return_1h').abs().sort_values(ascending=False).head(10))
```

**Cell 6 — is_first_bar: overnight gap vs intraday:**
```python
first = train[train['is_first_bar']]['return_1h']
other = train[~train['is_first_bar']]['return_1h']

fig, ax = plt.subplots(figsize=(12, 4))
ax.hist(first.clip(-0.05, 0.05), bins=80, alpha=0.6, density=True,
        label=f'is_first_bar=True  n={len(first):,}')
ax.hist(other.clip(-0.05, 0.05), bins=80, alpha=0.6, density=True,
        label=f'is_first_bar=False n={len(other):,}')
ax.legend()
ax.set_title('return_1h: overnight gap bars vs intraday bars (clipped to ±5%)')
ax.set_xlabel('return_1h')
plt.tight_layout()
plt.show()

print(f"is_first_bar=True  — mean={first.mean():.5f}  std={first.std():.5f}")
print(f"is_first_bar=False — mean={other.mean():.5f}  std={other.std():.5f}")
```

**Cell 7 — VIX regime analysis:**
```python
median_vix = train['vix_log'].median()
low_mask  = train['vix_log'] <= median_vix
high_mask = train['vix_log'] > median_vix

low_ret  = train.loc[low_mask,  'return_1h']
high_ret = train.loc[high_mask, 'return_1h']

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

axes[0].hist(low_ret.clip(-0.05, 0.05),  bins=80, alpha=0.6, density=True, label='Low VIX')
axes[0].hist(high_ret.clip(-0.05, 0.05), bins=80, alpha=0.6, density=True, label='High VIX')
axes[0].legend()
axes[0].set_title('return_1h by VIX regime (clipped ±5%)')
axes[0].set_xlabel('return_1h')

axes[1].scatter(train['vix_log'], train['vol_60h'], alpha=0.05, s=1, color='steelblue')
axes[1].set_xlabel('vix_log')
axes[1].set_ylabel('vol_60h')
axes[1].set_title('VIX log vs 60h realized volatility')

plt.tight_layout()
plt.show()

print(f"Low VIX  (vix_log ≤ {median_vix:.2f}) — std(return_1h): {low_ret.std():.5f}")
print(f"High VIX (vix_log > {median_vix:.2f}) — std(return_1h): {high_ret.std():.5f}")
```

- [ ] **Step 2: Run the notebook end to end**

Open Jupyter (`jupyter notebook` from repo root, navigate to `notebooks/eda.ipynb`), then: Kernel → Restart & Run All.

Expected: All 7 cells complete without error. 5 figures render (histogram+QQ, rolling vol, heatmap, is_first_bar comparison, VIX regime).

- [ ] **Step 3: Commit**

```bash
git add notebooks/eda.ipynb
git commit -m "feat(model): EDA notebook — return distribution, volatility, correlation, regime analysis"
```

---

### Task 5: deepar.ipynb

**Files:**
- Create: `notebooks/deepar.ipynb`

- [ ] **Step 1: Create notebooks/deepar.ipynb**

Create the notebook with exactly these cells in order.

**Cell 1 — Imports:**
```python
import sys
sys.path.insert(0, '..')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from neuralforecast import NeuralForecast
from neuralforecast.models import DeepAR
from neuralforecast.losses.pytorch import DistributionLoss, MQLoss

from scripts.models.data_loader import load_nf_dataframe, build_full_df, FUTR_EXOG_COLS
from scripts.models.metrics import compute_all

plt.rcParams['figure.figsize'] = (14, 4)

CANDIDATES = [24, 60, 120, 240]
FREQ = 1   # integer ds — avoids market-hours gap issues
```

**Cell 2 — Load and format data:**
```python
extra = FUTR_EXOG_COLS  # ['is_first_bar']

train_df = load_nf_dataframe('../data/splits/train.parquet', extra_cols=extra)
val_df   = load_nf_dataframe('../data/splits/val.parquet',   extra_cols=extra)
test_df  = load_nf_dataframe('../data/splits/test.parquet',  extra_cols=extra)

# Contiguous ds sequences for cross_validation
trainval_df = build_full_df([train_df, val_df])
full_df     = build_full_df([train_df, val_df, test_df])

print(f"train: {len(train_df):,}  val: {len(val_df):,}  test: {len(test_df):,}")
print(f"trainval: {len(trainval_df):,}  full: {len(full_df):,}")
print(f"futr_exog columns: {FUTR_EXOG_COLS}")
print(full_df.head(3))
```

**Cell 3 — Lookback tuning (val MAE per candidate):**
```python
val_maes = {}

for input_size in CANDIDATES:
    model = DeepAR(
        h=1,
        input_size=input_size,
        lstm_hidden_size=128,
        lstm_n_layers=2,
        lstm_dropout=0.1,
        trajectory_samples=100,          # fewer samples for speed during tuning
        loss=DistributionLoss(distribution='StudentT', level=[80, 90]),
        valid_loss=MQLoss(level=[80, 90]),
        futr_exog_list=FUTR_EXOG_COLS,
        max_steps=500,
        early_stop_patience_steps=30,
        scaler_type='standard',
    )
    nf = NeuralForecast(models=[model], freq=FREQ)
    # cross_validation treats last test_size rows of trainval_df as the "test" (= val here)
    cv = nf.cross_validation(
        df=trainval_df,
        test_size=len(val_df),
        step_size=1,
        refit=False,
    )
    val_mae = float(np.mean(np.abs(cv['y'].values - cv['DeepAR'].values)))
    val_maes[input_size] = val_mae
    print(f"  input_size={input_size:3d}  val MAE={val_mae:.6f}")

best_input_size = min(val_maes, key=val_maes.get)
print(f"\nBest input_size: {best_input_size}  (val MAE={val_maes[best_input_size]:.6f})")
```

**Cell 4 — Final cross_validation on test set:**
```python
# Train on train+val (full_df minus last test_size rows), predict across test
final_model = DeepAR(
    h=1,
    input_size=best_input_size,
    lstm_hidden_size=128,
    lstm_n_layers=2,
    lstm_dropout=0.1,
    trajectory_samples=200,
    loss=DistributionLoss(distribution='StudentT', level=[80, 90]),
    valid_loss=MQLoss(level=[80, 90]),
    futr_exog_list=FUTR_EXOG_COLS,
    max_steps=1000,
    early_stop_patience_steps=50,
    scaler_type='standard',
)
nf_final = NeuralForecast(models=[final_model], freq=FREQ)

cv_test = nf_final.cross_validation(
    df=full_df,
    test_size=len(test_df),
    step_size=1,
    refit=False,
)

print(f"Test predictions: {len(cv_test):,} rows")
print(f"Columns: {list(cv_test.columns)}")
```

**Cell 5 — Compute and print test metrics:**
```python
y_true = cv_test['y'].values
y_pred = cv_test['DeepAR'].values
lo_80  = cv_test['DeepAR-lo-80'].values
hi_80  = cv_test['DeepAR-hi-80'].values
lo_90  = cv_test['DeepAR-lo-90'].values
hi_90  = cv_test['DeepAR-hi-90'].values

results = compute_all(y_true, y_pred, lo_80, hi_80, lo_90, hi_90)

print("=== DeepAR Test Results ===")
print(f"  RMSE              : {results['rmse']:.6f}")
print(f"  MAE               : {results['mae']:.6f}")
print(f"  Directional Acc   : {results['dir_acc']:.4f}")
print(f"  Coverage 80%      : {results['coverage_80']:.4f}  (target: 0.80)")
print(f"  Coverage 90%      : {results['coverage_90']:.4f}  (target: 0.90)")
print(f"  Sharpe Ratio      : {results['sharpe']:.4f}")
print(f"  Max Drawdown      : {results['max_drawdown']:.6f}")
```

**Cell 6 — Plot: predicted vs actual, first 200 test bars:**
```python
n = 200
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(range(n), y_true[:n], label='Actual return_1h', alpha=0.8, linewidth=0.8, color='black')
ax.plot(range(n), y_pred[:n], label='DeepAR (mean)',    alpha=0.8, linewidth=0.8, color='steelblue')
ax.fill_between(range(n), lo_90[:n], hi_90[:n], alpha=0.12, color='steelblue', label='90% interval')
ax.fill_between(range(n), lo_80[:n], hi_80[:n], alpha=0.22, color='steelblue', label='80% interval')
ax.axhline(0, color='gray', linewidth=0.5)
ax.legend(fontsize=9)
ax.set_title('DeepAR: predicted vs actual return_1h — first 200 test bars (2024)')
ax.set_xlabel('Test bar index')
ax.set_ylabel('return_1h')
plt.tight_layout()
plt.show()
```

**Cell 7 — Save predictions:**
```python
test_raw = pd.read_parquet('../data/splits/test.parquet').reset_index(drop=True)

preds_df = pd.DataFrame({
    'ds':       cv_test['ds'].values,
    'datetime': test_raw['datetime'].values[:len(cv_test)],
    'y':        y_true,
    'pred':     y_pred,
    'lo_80':    lo_80,
    'hi_80':    hi_80,
    'lo_90':    lo_90,
    'hi_90':    hi_90,
    'model':    'DeepAR',
})
preds_df.to_parquet('../data/predictions/deepar_preds.parquet', index=False)
print(f"Saved {len(preds_df):,} rows → data/predictions/deepar_preds.parquet")
print(preds_df.head(3))
```

- [ ] **Step 2: Run the notebook end to end**

Kernel → Restart & Run All.

Expected: Tuning loop prints 4 MAE values. Final cross_validation prints row count and column names. Metrics table prints 7 values. Plot renders. Parquet saved.

- [ ] **Step 3: Commit**

```bash
git add notebooks/deepar.ipynb data/predictions/deepar_preds.parquet
git commit -m "feat(model): DeepAR notebook — lookback tuning, test evaluation, predictions saved"
```

---

### Task 6: patchtst.ipynb

**Files:**
- Create: `notebooks/patchtst.ipynb`

- [ ] **Step 1: Create notebooks/patchtst.ipynb**

**Cell 1 — Imports:**
```python
import sys
sys.path.insert(0, '..')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from neuralforecast import NeuralForecast
from neuralforecast.models import PatchTST
from neuralforecast.losses.pytorch import MQLoss

from scripts.models.data_loader import load_nf_dataframe, build_full_df, HIST_EXOG_COLS
from scripts.models.metrics import compute_all

plt.rcParams['figure.figsize'] = (14, 4)

CANDIDATES = [24, 60, 120, 240]
FREQ = 1


def patch_params(input_size: int) -> tuple[int, int]:
    """Return (patch_len, stride) scaled to input_size.

    Ensures at least 4 patches and patch_len <= input_size // 2.
    """
    patch_len = max(4, min(16, input_size // 4))
    stride = max(2, patch_len // 2)
    return patch_len, stride
```

**Cell 2 — Load and format data:**
```python
extra = HIST_EXOG_COLS  # all 20 non-target feature columns

train_df = load_nf_dataframe('../data/splits/train.parquet', extra_cols=extra)
val_df   = load_nf_dataframe('../data/splits/val.parquet',   extra_cols=extra)
test_df  = load_nf_dataframe('../data/splits/test.parquet',  extra_cols=extra)

trainval_df = build_full_df([train_df, val_df])
full_df     = build_full_df([train_df, val_df, test_df])

print(f"train: {len(train_df):,}  val: {len(val_df):,}  test: {len(test_df):,}")
print(f"hist_exog columns ({len(HIST_EXOG_COLS)}): {HIST_EXOG_COLS}")
```

**Cell 3 — Lookback tuning (val MAE per candidate):**
```python
val_maes = {}

for input_size in CANDIDATES:
    patch_len, stride = patch_params(input_size)
    model = PatchTST(
        h=1,
        input_size=input_size,
        patch_len=patch_len,
        stride=stride,
        d_model=128,
        n_heads=8,
        n_layers=3,
        dropout=0.2,
        loss=MQLoss(level=[80, 90]),
        hist_exog_list=HIST_EXOG_COLS,
        max_steps=500,
        early_stop_patience_steps=30,
        scaler_type='standard',
    )
    nf = NeuralForecast(models=[model], freq=FREQ)
    cv = nf.cross_validation(
        df=trainval_df,
        test_size=len(val_df),
        step_size=1,
        refit=False,
    )
    # MQLoss outputs a median column; find it by name
    median_col = [c for c in cv.columns if 'PatchTST' in c and 'median' in c.lower()][0]
    val_mae = float(np.mean(np.abs(cv['y'].values - cv[median_col].values)))
    val_maes[input_size] = val_mae
    print(f"  input_size={input_size:3d}  patch_len={patch_len}  stride={stride}  val MAE={val_mae:.6f}")

best_input_size = min(val_maes, key=val_maes.get)
print(f"\nBest input_size: {best_input_size}  (val MAE={val_maes[best_input_size]:.6f})")
```

**Cell 4 — Final cross_validation on test set:**
```python
patch_len, stride = patch_params(best_input_size)

final_model = PatchTST(
    h=1,
    input_size=best_input_size,
    patch_len=patch_len,
    stride=stride,
    d_model=128,
    n_heads=8,
    n_layers=3,
    dropout=0.2,
    loss=MQLoss(level=[80, 90]),
    hist_exog_list=HIST_EXOG_COLS,
    max_steps=1000,
    early_stop_patience_steps=50,
    scaler_type='standard',
)
nf_final = NeuralForecast(models=[final_model], freq=FREQ)

cv_test = nf_final.cross_validation(
    df=full_df,
    test_size=len(test_df),
    step_size=1,
    refit=False,
)

print(f"Test predictions: {len(cv_test):,} rows")
print(f"Columns: {list(cv_test.columns)}")
```

**Cell 5 — Compute and print test metrics:**
```python
median_col = [c for c in cv_test.columns if 'PatchTST' in c and 'median' in c.lower()][0]
lo80_col   = [c for c in cv_test.columns if 'PatchTST' in c and 'lo-80' in c][0]
hi80_col   = [c for c in cv_test.columns if 'PatchTST' in c and 'hi-80' in c][0]
lo90_col   = [c for c in cv_test.columns if 'PatchTST' in c and 'lo-90' in c][0]
hi90_col   = [c for c in cv_test.columns if 'PatchTST' in c and 'hi-90' in c][0]

y_true = cv_test['y'].values
y_pred = cv_test[median_col].values
lo_80  = cv_test[lo80_col].values
hi_80  = cv_test[hi80_col].values
lo_90  = cv_test[lo90_col].values
hi_90  = cv_test[hi90_col].values

results = compute_all(y_true, y_pred, lo_80, hi_80, lo_90, hi_90)

print("=== PatchTST Test Results ===")
print(f"  RMSE              : {results['rmse']:.6f}")
print(f"  MAE               : {results['mae']:.6f}")
print(f"  Directional Acc   : {results['dir_acc']:.4f}")
print(f"  Coverage 80%      : {results['coverage_80']:.4f}  (target: 0.80)")
print(f"  Coverage 90%      : {results['coverage_90']:.4f}  (target: 0.90)")
print(f"  Sharpe Ratio      : {results['sharpe']:.4f}")
print(f"  Max Drawdown      : {results['max_drawdown']:.6f}")
```

**Cell 6 — Plot: predicted vs actual, first 200 test bars:**
```python
n = 200
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(range(n), y_true[:n], label='Actual return_1h', alpha=0.8, linewidth=0.8, color='black')
ax.plot(range(n), y_pred[:n], label='PatchTST (median)', alpha=0.8, linewidth=0.8, color='darkorange')
ax.fill_between(range(n), lo_90[:n], hi_90[:n], alpha=0.12, color='darkorange', label='90% interval')
ax.fill_between(range(n), lo_80[:n], hi_80[:n], alpha=0.22, color='darkorange', label='80% interval')
ax.axhline(0, color='gray', linewidth=0.5)
ax.legend(fontsize=9)
ax.set_title('PatchTST: predicted vs actual return_1h — first 200 test bars (2024)')
ax.set_xlabel('Test bar index')
ax.set_ylabel('return_1h')
plt.tight_layout()
plt.show()
```

**Cell 7 — Save predictions:**
```python
test_raw = pd.read_parquet('../data/splits/test.parquet').reset_index(drop=True)

preds_df = pd.DataFrame({
    'ds':       cv_test['ds'].values,
    'datetime': test_raw['datetime'].values[:len(cv_test)],
    'y':        y_true,
    'pred':     y_pred,
    'lo_80':    lo_80,
    'hi_80':    hi_80,
    'lo_90':    lo_90,
    'hi_90':    hi_90,
    'model':    'PatchTST',
})
preds_df.to_parquet('../data/predictions/patchtst_preds.parquet', index=False)
print(f"Saved {len(preds_df):,} rows → data/predictions/patchtst_preds.parquet")
print(preds_df.head(3))
```

- [ ] **Step 2: Run the notebook end to end**

Kernel → Restart & Run All.

Expected: Same structure as deepar.ipynb — tuning prints 4 MAE values, final cross_validation runs, metrics print, plot renders, parquet saved.

- [ ] **Step 3: Commit**

```bash
git add notebooks/patchtst.ipynb data/predictions/patchtst_preds.parquet
git commit -m "feat(model): PatchTST notebook — lookback tuning, test evaluation, predictions saved"
```

---

### Task 7: comparison.ipynb

**Files:**
- Create: `notebooks/comparison.ipynb`

Prerequisite: `data/predictions/deepar_preds.parquet` and `data/predictions/patchtst_preds.parquet` must exist (produced by Tasks 5 and 6).

- [ ] **Step 1: Create notebooks/comparison.ipynb**

**Cell 1 — Imports:**
```python
import sys
sys.path.insert(0, '..')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scripts.models.metrics import compute_all

sns.set_theme(style='darkgrid')
plt.rcParams['figure.figsize'] = (14, 5)
```

**Cell 2 — Load predictions:**
```python
deepar   = pd.read_parquet('../data/predictions/deepar_preds.parquet')
patchtst = pd.read_parquet('../data/predictions/patchtst_preds.parquet')

assert len(deepar) == len(patchtst), \
    f"Prediction lengths differ: DeepAR={len(deepar)}, PatchTST={len(patchtst)}"

print(f"Test bars  : {len(deepar):,}")
print(f"Date range : {deepar['datetime'].min()} → {deepar['datetime'].max()}")
print(f"\nDeepAR columns  : {list(deepar.columns)}")
print(f"PatchTST columns: {list(patchtst.columns)}")
```

**Cell 3 — Side-by-side metrics table:**
```python
def get_metrics(df: pd.DataFrame) -> dict:
    return compute_all(
        df['y'].values, df['pred'].values,
        df['lo_80'].values, df['hi_80'].values,
        df['lo_90'].values, df['hi_90'].values,
    )

m_deepar   = get_metrics(deepar)
m_patchtst = get_metrics(patchtst)

summary = pd.DataFrame({'DeepAR': m_deepar, 'PatchTST': m_patchtst}).T
summary.columns = ['RMSE', 'MAE', 'Dir Acc', 'Coverage 80%', 'Coverage 90%', 'Sharpe', 'Max DD']

print("=== Test Set Comparison ===")
print(summary.round(4).to_string())
summary.round(4)
```

**Cell 4 — Cumulative return: both strategies vs Buy & Hold:**
```python
def strategy_cumret(df: pd.DataFrame) -> np.ndarray:
    return np.cumsum(np.sign(df['pred'].values) * df['y'].values)

def buyhold_cumret(df: pd.DataFrame) -> np.ndarray:
    return np.cumsum(df['y'].values)

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(strategy_cumret(deepar),   label='DeepAR long/short',   linewidth=1.3, color='steelblue')
ax.plot(strategy_cumret(patchtst), label='PatchTST long/short', linewidth=1.3, color='darkorange')
ax.plot(buyhold_cumret(deepar),    label='Buy & Hold SPY',      linewidth=1.0, linestyle='--',
        alpha=0.6, color='gray')
ax.axhline(0, color='black', linewidth=0.4)
ax.set_title('Cumulative return: long/short strategy vs Buy & Hold (test set 2024–2025)')
ax.set_xlabel('Test bar index')
ax.set_ylabel('Cumulative return_1h')
ax.legend()
plt.tight_layout()
plt.show()
```

**Cell 5 — Prediction interval width (sharpness):**
```python
deepar['width_90']   = deepar['hi_90']   - deepar['lo_90']
patchtst['width_90'] = patchtst['hi_90'] - patchtst['lo_90']

n = 500
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(deepar['width_90'].values[:n],   label='DeepAR 90% width',   alpha=0.8, linewidth=0.8, color='steelblue')
ax.plot(patchtst['width_90'].values[:n], label='PatchTST 90% width', alpha=0.8, linewidth=0.8, color='darkorange')
ax.set_title('90% prediction interval width — first 500 test bars (lower = sharper)')
ax.set_xlabel('Test bar index')
ax.set_ylabel('hi_90 − lo_90')
ax.legend()
plt.tight_layout()
plt.show()

print(f"DeepAR   mean 90% interval width : {deepar['width_90'].mean():.6f}")
print(f"PatchTST mean 90% interval width : {patchtst['width_90'].mean():.6f}")
```

**Cell 6 — Regime analysis: low vs high VIX:**
```python
test_raw  = pd.read_parquet('../data/splits/test.parquet').reset_index(drop=True)
vix_vals  = test_raw['vix_log'].values[:len(deepar)]
med_vix   = np.median(vix_vals)

low_mask  = vix_vals <= med_vix
high_mask = vix_vals > med_vix

def regime_metrics(df: pd.DataFrame, mask: np.ndarray) -> dict:
    sub = df[mask].copy()
    return compute_all(
        sub['y'].values, sub['pred'].values,
        sub['lo_80'].values, sub['hi_80'].values,
        sub['lo_90'].values, sub['hi_90'].values,
    )

regime = pd.DataFrame({
    'DeepAR  — Low VIX':   regime_metrics(deepar,   low_mask),
    'DeepAR  — High VIX':  regime_metrics(deepar,   high_mask),
    'PatchTST — Low VIX':  regime_metrics(patchtst, low_mask),
    'PatchTST — High VIX': regime_metrics(patchtst, high_mask),
}).T

regime.columns = ['RMSE', 'MAE', 'Dir Acc', 'Coverage 80%', 'Coverage 90%', 'Sharpe', 'Max DD']

print(f"=== Regime Analysis  (median vix_log split = {med_vix:.3f}) ===")
print(regime.round(4).to_string())
regime.round(4)
```

- [ ] **Step 2: Run the notebook end to end**

Kernel → Restart & Run All.

Expected: All 6 cells execute. Length assertion passes. Metrics table, cumulative return plot, interval width plot, and regime table all render.

- [ ] **Step 3: Run full test suite one final time**

```bash
pytest tests/ -v
```

Expected: 40 passed.

- [ ] **Step 4: Commit**

```bash
git add notebooks/comparison.ipynb
git commit -m "feat(model): comparison notebook — metrics table, cumulative returns, interval width, VIX regime"
```

- [ ] **Step 5: Push Model branch**

```bash
git push origin Model
```

---

## Self-Review

**Spec coverage:**
- ✅ `requirements-model.txt` — Task 1
- ✅ `scripts/models/__init__.py`, `data_loader.py`, `metrics.py` — Tasks 1–3
- ✅ Integer `ds` with `load_nf_dataframe` and `build_full_df` — Task 2
- ✅ `HIST_EXOG_COLS` (20 features) and `FUTR_EXOG_COLS` (`['is_first_bar']`) exported from data_loader — Task 2
- ✅ DeepAR: `DistributionLoss(StudentT)`, `futr_exog_list`, `trajectory_samples=200`, lookback tuning — Task 5
- ✅ PatchTST: `MQLoss`, `hist_exog_list=HIST_EXOG_COLS`, `patch_params()` scaling, lookback tuning — Task 6
- ✅ `cross_validation(test_size=len(test_df), step_size=1, refit=False)` for test evaluation — Tasks 5–6
- ✅ All 7 metrics: rmse, mae, dir_acc, coverage_80, coverage_90, sharpe, max_drawdown — Task 3
- ✅ `compute_all` signature consistent across metrics.py, model notebooks, comparison notebook — Tasks 3, 5, 6, 7
- ✅ Prediction save schema: ds, datetime, y, pred, lo_80, hi_80, lo_90, hi_90, model — Tasks 5–6
- ✅ EDA: distribution, rolling vol, correlation, is_first_bar, VIX regime — Task 4
- ✅ Comparison: side-by-side table, cumulative return, interval width, regime analysis — Task 7

**Placeholder scan:** No TBDs or incomplete steps found.

**Type consistency:**
- `load_nf_dataframe(split_path, extra_cols)` → same signature used in Tasks 2, 5, 6 ✅
- `build_full_df(list[pd.DataFrame])` → same signature in Tasks 2, 5, 6 ✅
- `compute_all(y_true, y_pred, lo_80, hi_80, lo_90, hi_90)` → same signature in Tasks 3, 5, 6, 7 ✅
- `HIST_EXOG_COLS`, `FUTR_EXOG_COLS` imported from `data_loader` in Tasks 5, 6 ✅
- `patch_params(input_size) -> tuple[int, int]` defined in Cell 1 of patchtst.ipynb, used in Cells 3 and 4 ✅
