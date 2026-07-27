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

### Overall conclusion (Runs 1–3)

Three runs — a data-only pass (calendar `futr_exog` features) and a model-only pass (wider tuning grid, conformal intervals, more MC samples) — moved DeepAR only modestly, and mostly on metrics that don't reflect real forecasting skill:

- **Point-forecast accuracy (RMSE/MAE) barely moved across all three runs** (RMSE 0.004169 → 0.004170 → 0.004152; MAE 0.002354 → 0.002373 → 0.002350). Neither the data nor the model changes meaningfully improved the model's ability to predict the actual return value.
- **Directional accuracy crept up slightly and monotonically** (0.5103 → 0.5180 → 0.5213) but stayed within ~1–2pp of chance the whole way — on a ~2,469-bar test set that's within the statistical noise floor (~1pp standard error), not confirmed signal.
- **Interval calibration is the one unambiguous win**, and it came from the model-wise pass specifically: conformal calibration (Run 3) pulled both 80%/90% coverage close to their targets, fixing a real, measured defect from Run 1/2's parametric StudentT intervals.
- **Sharpe and Max Drawdown are not reliable indicators here** — they swung the most of any metric (-0.42 → +0.48 → +0.15; -35.3% → -23.9% → -31.5%) while RMSE/MAE/DirAcc moved smoothly and only slightly. Since the trading-strategy metrics are a path-dependent function of point predictions on a single, short, heavy-tailed test window (kurtosis ~39), these swings look like incidental noise rather than genuine gains or losses in skill. Treat any future Sharpe/Max DD change with the same skepticism until it's checked across multiple random seeds.
- **Neither the 13-feature calendar set nor the widened model grid overcame the fundamental difficulty of the problem**: forecasting one-step-ahead hourly SPY returns is close to market-efficient noise, and DeepAR at `h=1` seems to be near its practical ceiling for this data — directional accuracy stuck barely above chance is the clearest evidence of that.

**What would move the needle further, if pursued:** feature ablation (isolate which, if any, of the 13 calendar features carry real signal — `hour_sin/cos` is the most plausible candidate given the EDA's overnight-gap/intraday-vol findings), and multi-seed repetition (to establish noise floors for Sharpe/DirAcc before trusting any future delta). Absent those, the honest takeaway is: DeepAR's performance on this task is essentially flat across all three optimization attempts, with calibration being the only concretely fixed problem.

---

## PatchTST

| # | Date | Commit | Data features | Config changes | RMSE | MAE | Dir Acc | Cov 80% | Cov 90% | Sharpe | Max DD |
|---|------|--------|----------------|-----------------|------|-----|---------|---------|---------|--------|--------|
| 1 | 2026-07-26 | `31448f0` | `hist_exog` = all 20 features (fully multivariate, `channel_attention=False`) | Fair-baseline pass: switched `loss='mse'` → `loss='nll'` with `distribution_output='student_t'`, matching DeepAR's `DistributionLoss`. Point forecast = analytic StudentT mean (`loc`); 80%/90% intervals = analytic quantiles (`scipy.stats.t.ppf`), replacing conformal calibration. Required bypassing a `transformers` library bug: `PatchTSTPredictionHead.forward()`'s final tuple-transpose assumes `prediction_length > 1` and crashes with `IndexError` at `prediction_length=1` when a distribution head is used — both training (manual NLL via `torch.distributions.StudentT.log_prob`) and inference call `model.model(...)` + `model.head.projection(...)` directly to stop short of the broken line. `input_size` tuned over {24,60,120,240} → 240 | 0.004365 | 0.002544 | 0.5006 | 0.7663 | 0.8615 | -0.4484 | -0.2182 |

**Run 1 — `input_size` tuning (val MAE):**

| input_size | 24 | 60 | 120 | 240 (selected) |
|---|---|---|---|---|
| val MAE | 0.594528 | 0.590923 | 0.576269 | **0.564193** |

**Run 1 notes (fair-baseline pass — distributional loss, data config unchanged):**
- **For context, the prior (never formally logged) point-wise `loss='mse'` + conformal-calibration run** on this same 21-channel setup scored: RMSE 0.004175, MAE 0.002373, Dir Acc 0.5006, Cov 80% 0.7967, Cov 90% 0.8801, Sharpe -0.0123, Max DD -18.4%. Comparing to that:
- **RMSE/MAE got slightly worse** (0.004175→0.004365, 0.002373→0.002544) rather than better — switching to NLL did not improve raw point-forecast accuracy here, contrary to what you might hope from "a more principled objective."
- **Interval calibration also got worse**, not better (Cov 80%: 0.7967→0.7663; Cov 90%: 0.8801→0.8615, both moving further from nominal) — surprising, since the whole point of a native distribution was to get calibration "for free" instead of needing conformal correction. The old conformal-calibrated intervals were, empirically, better calibrated than these new analytic ones.
- **Directional accuracy is essentially identical** (0.5006 vs 0.5006) — expected, since this axis doesn't depend on the loss family, and confirms (again) the model has ~no directional edge regardless of point-vs-distributional training.
- **Sharpe got much worse** (-0.0123 → -0.4484) — consistent with the now-familiar pattern that this metric is highly noisy/path-dependent and shouldn't be read as a skill signal on its own.
- **Plausible explanations for the worse fit, worth investigating before concluding NLL "doesn't work" here**: (a) NLL optimizes mean *and* scale/tail-shape jointly, which can genuinely trade off some point-forecast accuracy against calibration, especially under a fixed, possibly-too-short training budget (same `max_steps=500/1000` as the MSE run); (b) the manual bypass training loop (replicating the library's intended NLL computation by hand, since the library's own path crashes at `prediction_length=1`) could have subtly different numerics than the untested "intended" path — worth a sanity check that the manual loss matches what the library would compute if it worked; (c) StudentT NLL is known to be a harder loss surface to optimize than plain MSE, and may need more steps or a lower/warmed-up learning rate to converge as well.
- **Bottom line**: the fairness motivation for this change stands (comparable probabilistic modeling to DeepAR), but it did not yet deliver better numbers.
- **⚠️ Superseded by Run 2 below.** A follow-up lag-correlation diagnostic on this run's predictions found `corr(pred[t], y[t]) ≈ 0` but `corr(pred[t], y[t-1]) = 0.81` — the model was echoing a damped version of the *previous* actual return rather than predicting the next one. Root cause: this run's training loss was averaged **equally across all 21 channels** (not just `return_1h`), unlike DeepAR, whose loss is inherently computed only on its single univariate target. Run 2 fixes this and should be treated as the actual fair baseline; this run is kept here for the record, not as a valid comparison point.

| # | Date | Commit | Data features | Config changes | RMSE | MAE | Dir Acc | Cov 80% | Cov 90% | Sharpe | Max DD |
|---|------|--------|----------------|-----------------|------|-----|---------|---------|---------|--------|--------|
| 2 | 2026-07-26 | `8083e60` | `hist_exog` = all 20 features (fully multivariate, `channel_attention=False`) | **True fair-baseline pass.** Same as Run 1, plus: training loss now computed **only on channel 0 (`return_1h`)** instead of averaged across all 21 channels — matching DeepAR, whose loss has only ever been computed on its single univariate target. Model still sees all 21 channels as context (channel-independent patching/encoding unaffected); only the training gradient signal changed. `input_size` tuned over {24,60,120,240} → 240 | 0.004158 | 0.002350 | 0.5152 | 0.8068 | 0.9000 | -0.0673 | -0.2345 |

**Run 2 — `input_size` tuning (val MAE):**

| input_size | 24 | 60 | 120 | 240 (selected) |
|---|---|---|---|---|
| val MAE | 0.547853 | 0.541291 | 0.541689 | **0.537668** |

**Run 2 — lag diagnostic (correlation with y[t-1], normalised space):**

| | Run 1 (21-channel loss) | Run 2 (channel-0 loss) |
|---|---|---|
| corr(loc_raw, y[t-1]) — learned weights' own signal | 0.7604 | **0.1609** |
| corr(win_loc, y[t-1]) — non-learned instance norm | 0.0667 | 0.0667 |
| corr(final pred, y[t-1]) | 0.8073 | **0.1201** |

**Run 2 notes (true fair-baseline pass — channel-0-only loss):**
- **The lag-echo effect collapsed**: `corr(loc_raw, y[t-1])` dropped from 0.76 to 0.16 — confirms the diagnosis was correct. Restricting the loss to `return_1h` stopped the shared weights from learning a persistence strategy borrowed from the other 20 (genuinely autocorrelated) channels. `win_loc`'s correlation is unchanged (0.0667, as expected — that path was never learned, just mechanical per-window normalization, and was already ruled out as the cause).
- **Every accuracy metric improved over Run 1**: RMSE 0.004365→0.004158, MAE 0.002544→0.002350, Dir Acc 0.5006→0.5152.
- **This is now also the best MAE across every PatchTST run to date**, including the original point-wise `loss='mse'` + conformal baseline (MAE 0.002373) — the channel-0-only distributional loss beats both prior approaches on raw point-forecast accuracy.
- **Interval calibration is now excellent**: Cov 80% = 0.8068 (target 0.80) and Cov 90% = 0.9000 (target 0.90, hit almost exactly) — a large improvement over Run 1's 0.7663/0.8615, and better than the original conformal-calibrated run's 0.7967/0.8801 too.
- **Sharpe improved substantially** (-0.4484 → -0.0673) though still negative; Max Drawdown got slightly worse (-21.8%→-23.4%). Per the running theme with DeepAR, treat these two as noisy/path-dependent rather than a clean verdict.
- **Directional accuracy (0.5152) is still close to chance**, but is now PatchTST's best result on this axis across all runs, and slightly ahead of DeepAR's best (0.5213 from DeepAR Run 3 — nearly tied).
- **Bottom line**: this is the actual fair, methodologically-sound baseline to compare against DeepAR and to optimize further from. `channel_attention=True` — letting `return_1h` actually use information from the other 20 channels, which it structurally still cannot do here — remains the next planned pass and is now motivated by cleaner evidence, since the channel-weighting confound from Run 1 has been removed.
