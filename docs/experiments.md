# Experiment Log

One row per completed run of a model notebook. This is the source of truth for
comparing configs/data versions over time — don't fork notebooks into
`_v1`/`_v2` copies; rerun the same notebook, log the result here, and let git
history hold the code diff.

How to add an entry after a run:
1. Note the git commit the run was executed against (`git rev-parse --short HEAD`).
2. Copy the metrics block printed by the notebook's results cell.
3. Copy the per-`input_size` val MAE printed during the tuning loop.
4. Add a row to the relevant model's table below, plus a couple sentences on
   what changed and why.

---

## DeepAR

| # | Date | Commit | Data features | Config changes | RMSE | MAE | Dir Acc | Cov 80% | Cov 90% | Sharpe | Max DD |
|---|------|--------|----------------|-----------------|------|-----|---------|---------|---------|--------|--------|
| 1 | 2026-07-24 | `f21d62e` | `futr_exog = [is_first_bar]` (1 feature) | Baseline: `input_size` tuned over {24,60,120,240} → 120; `lstm_hidden_size=128`, `lstm_n_layers=2`, `trajectory_samples=100/200`, `StudentT` loss, `scaler_type=standard` | 0.004169 | 0.002354 | 0.5103 | 0.7898 | 0.8854 | -0.4188 | -0.3525 |

**Run 1 — `input_size` tuning (val MAE):**

| input_size | 24 | 60 | 120 (selected) | 240 |
|---|---|---|---|---|
| val MAE | 0.002169 | 0.002155 | **0.002146** | 0.002150 |

**Run 1 notes:** Baseline. `input_size` tuning was flat across candidates (val MAE range 0.002146–0.002169, ~1% relative) — not a meaningful lever. Directional accuracy barely above chance and negative Sharpe suggest the model is close to predicting the unconditional mean. 90% interval undercovers (0.885 vs 0.90 target).

| # | Date | Commit | Data features | Config changes | RMSE | MAE | Dir Acc | Cov 80% | Cov 90% | Sharpe | Max DD |
|---|------|--------|----------------|-----------------|------|-----|---------|---------|---------|--------|--------|
| 2 | 2026-07-25 | `f9a6169` | `futr_exog` = 13 features: `is_first_bar`, `is_last_bar`, `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`, `is_monday`, `is_friday`, `month_sin`, `month_cos`, `is_month_end`, `is_month_start`, `is_quarter_end` | Data-only change: same model config as Run 1 (`input_size` tuned over {24,60,120,240} → 60; `lstm_hidden_size=128`, `lstm_n_layers=2`, `trajectory_samples=100/200`, `StudentT` loss, `scaler_type=standard`) | 0.004170 | 0.002373 | 0.5180 | 0.7752 | 0.8728 | 0.4825 | -0.2391 |

**Run 2 — `input_size` tuning (val MAE):**

| input_size | 24 | 60 (selected) | 120 | 240 |
|---|---|---|---|---|
| val MAE | 0.002175 | **0.002150** | 0.002150 | 0.002154 |

**Run 2 notes (data-only optimization — calendar `futr_exog` features added, model config unchanged):**
- **Sharpe flipped from -0.4188 to +0.4825**, and max drawdown improved from -35.3% to -23.9% — the largest change of any metric. Directional accuracy also nudged up (0.5103 → 0.5180).
- RMSE/MAE are essentially flat to very slightly worse (0.002354 → 0.002373 MAE) — the calendar features didn't reduce raw point-forecast error.
- Both interval coverages moved slightly further from nominal (80% → 0.7752, 90% → 0.8728, both now undercovering more than Run 1) — the added covariates shifted the learned StudentT scale without a corresponding recalibration; still supports adding conformal calibration as a follow-up.
- `input_size` selection shifted from 120 → 60 and the tuning curve is still nearly flat (0.002150–0.002175) — lookback length remains a low-leverage axis.
- Interpretation: the calendar features didn't improve raw error metrics, but meaningfully changed the *shape* of the model's predictions in a way that helped the derived trading-strategy metrics (Sharpe, drawdown) and directional accuracy — worth digging into which specific feature(s) drove this before attributing it to the full set. Next: model-wise optimizations (conformal intervals, capacity/tuning-grid changes) discussed separately.

| # | Date | Commit | Data features | Config changes | RMSE | MAE | Dir Acc | Cov 80% | Cov 90% | Sharpe | Max DD |
|---|------|--------|----------------|-----------------|------|-----|---------|---------|---------|--------|--------|
| 3 | 2026-07-25 | `59a1415` | Same as Run 2: `futr_exog` = 13 calendar features (unchanged) | Model-only change: tuning grid widened to `input_size` x `lstm_hidden_size` x `scaler_type` (16 configs) → best = `input_size=240, lstm_hidden_size=64, scaler_type=robust`; conformal intervals (calibrated from validation residuals) replace parametric StudentT intervals as the reported metric; `trajectory_samples` 200→500 | 0.004152 | 0.002350 | 0.5213 | 0.7971 | 0.8870 | 0.1536 | -0.3148 |

**Run 3 — tuning grid (val MAE, 16 configs = `input_size` x `lstm_hidden_size` x `scaler_type`):**

| input_size | hidden=64, standard | hidden=64, robust | hidden=128, standard | hidden=128, robust |
|---|---|---|---|---|
| 24 | 0.002172 | 0.002180 | 0.002175 | 0.002182 |
| 60 | 0.002155 | 0.002166 | 0.002150 | 0.002165 |
| 120 | 0.002151 | 0.002153 | 0.002150 | 0.002152 |
| 240 | 0.002151 | **0.002142** (selected) | 0.002154 | 0.002153 |

**Run 3 notes (model-only optimization — tuning grid widened, conformal intervals, more trajectory samples; data config unchanged from Run 2):**
- **Conformal calibration worked as intended**: coverage 80% moved from 0.7752 (Run 2, parametric) to 0.7971 (target 0.80) and coverage 90% moved from 0.8728 to 0.8870 (target 0.90) — both now much closer to nominal. The notebook's side-by-side parametric comparison on this same run (0.7918 / 0.8846) confirms conformal calibration is the better-calibrated of the two, though the gap is smaller than expected — the parametric intervals were already close.
- **RMSE/MAE improved slightly** (0.004170→0.004152 RMSE, 0.002373→0.002350 MAE) — small but in the right direction, plausibly from the wider/better-fitting `input_size=240, robust` config rather than a large real gain.
- **Directional accuracy edged up again** (0.5180→0.5213), continuing the (still statistically weak, ~1pp-scale) upward trend across all three runs.
- **Sharpe collapsed relative to Run 2** (0.4825→0.1536) and max drawdown got worse again (-23.9%→-31.5%), even though every other metric held steady or improved. This is the clearest evidence yet that Sharpe/Max DD are dominated by incidental path effects rather than tracking real forecasting skill — the point-forecast accuracy barely moved, but the derived trading metric swung by more than half its Run 2 value.
- The winning config (`input_size=240`, `hidden=64`, `robust`) suggests smaller LSTM capacity paired with outlier-robust scaling squeezes out a little more val MAE than the Run 1/2 default (`hidden=128`, `standard`) — consistent with the "model was probably oversized" hypothesis — but the whole grid is still bunched within ~0.002142–0.002182, i.e. still a flat surface overall.

---

## PatchTST

| # | Date | Commit | Data features | Config changes | RMSE | MAE | Dir Acc | Cov 80% | Cov 90% | Sharpe | Max DD |
|---|------|--------|----------------|-----------------|------|-----|---------|---------|---------|--------|--------|
| 1 | | | `hist_exog` = all 20 features (fully multivariate, channel-independent) | | | | | | | | |

<!-- Fill in from the last executed patchtst.ipynb run once available -->
