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
- **⚠️ Important caveat, easy to miss: this run's PatchTST is functionally univariate, not multivariate.** PatchTST's `channel_attention=False` mode treats each of the 21 channels as a fully independent time series (channels only get folded into the batch dimension for computation — they never attend to or mix with each other in the forward pass). Combined with restricting the loss to channel 0 only, this means the other 20 channels are computed but **discarded**: they contribute zero information to `return_1h`'s forecast and receive zero gradient (nothing in the computation graph connects them to the loss). So despite being fed 21 channels, this run's model is *mathematically equivalent* to training on `return_1h` alone — architecturally different from DeepAR (Transformer vs. LSTM) but informationally no more advantaged. "Fair baseline" here only means *the training objective* became comparable to DeepAR's (loss on the single target, not diluted across 21 channels) — it does **not** mean this run tested the project's actual premise of giving PatchTST genuine covariate access. That test only begins at Run 3 (`channel_attention=True`).

| # | Date | Commit | Data features | Config changes | RMSE | MAE | Dir Acc | Cov 80% | Cov 90% | Sharpe | Max DD |
|---|------|--------|----------------|-----------------|------|-----|---------|---------|---------|--------|--------|
| 3 | 2026-07-26 | `4714067` | `hist_exog` = all 20 features (fully multivariate) | Same as Run 2, plus: `channel_attention=True` — adds a per-layer cross-channel attention sublayer so `return_1h`'s representation can actually attend to the other 20 channels' patch representations (previously `False`: fully channel-independent, no cross-channel information flow at all). `input_size` tuned over {24,60,120,240} → 240 | 0.004159 | 0.002358 | **0.5362** | 0.7890 | 0.8850 | 0.5254 | -0.2718 |

**Run 3 — `input_size` tuning (val MAE):**

| input_size | 24 | 60 | 120 | 240 (selected) |
|---|---|---|---|---|
| val MAE | 0.539819 | 0.538239 | 0.538333 | **0.538231** |

**Run 3 — lag diagnostic (correlation with y[t-1], normalised space):**

| | Run 1 (21-ch loss) | Run 2 (ch-0 loss) | Run 3 (+ channel_attention) |
|---|---|---|---|
| corr(loc_raw, y[t-1]) | 0.7604 | 0.1609 | **0.0074** |
| corr(win_loc, y[t-1]) | 0.0667 | 0.0667 | 0.0667 |
| corr(final pred, y[t-1]) | 0.8073 | 0.1201 | **0.0721** |

**Run 3 notes (`channel_attention=True` — return_1h can now use the other 20 channels):**
- **⚠️ The directional-accuracy gain is debunked — it is not genuine skill.** A follow-up check (base rate + prediction-sign-rate + accuracy-by-confidence-quartile) found: the model predicts "up" on **96.7%** of all test bars (vs. only 53.4% of bars actually being up), i.e. it has collapsed to a near-constant, tiny positive output (`std(pred)` ≈ 6% of `std(y)`). The test period's true base rate is 53.4% up — a trivial "always predict up" strategy scores 53.4% dir_acc for free. The model's actual 53.62% is only 0.2pp above that trivial baseline, and accuracy does **not** improve with prediction confidence (quartile dir_acc: 0.506, 0.566, 0.535, 0.539 — flat/noisy, not monotonic, as genuine skill would show). **Conclusion: the model learned the test period's overall bullish drift, not per-bar directional signal.**
- **The "lag-echo fixed" framing below also needs a caveat, given the above.** `corr(loc_raw, y[t-1])` dropping to 0.0074 is consistent with a genuine fix, but is *also* exactly what you'd expect from a near-constant series with almost no variance — correlation with anything trends toward zero when there's little signal to correlate in the first place. This doesn't cleanly separate "the persistence heuristic was replaced with something better" from "the model just collapsed to a safer, blander constant." Treat the lag-diagnostic improvement as suggestive, not confirmed.
- **RMSE/MAE are essentially unchanged** (0.004158→0.004159, 0.002350→0.002358) — no measurable improvement in point-forecast accuracy from this change.
- **Interval calibration regressed**: Cov 80% 0.8068→0.7890 and Cov 90% 0.9000→0.8850 — both moved from Run 2's near-exact calibration back toward undercoverage. This is the one clear, directly-attributable cost of this change.
- **Sharpe improved a lot** (-0.0673→0.5254) but Max Drawdown got worse (-23.4%→-27.2%) — per the established pattern with DeepAR, treat both as noisy/path-dependent, not confirmation of anything.
- **Net honest read**: given RMSE/MAE flat, calibration worse, and the dir_acc "win" debunked, Run 3 does not yet demonstrate a measurable improvement over Run 2 on any metric. It's being kept (not reverted to `channel_attention=False`) for a structural reason, not a performance one: given the channel-0-only training loss (Run 2's fix), `channel_attention=False` means the other 20 channels receive **zero gradient and contribute zero information** to `return_1h`'s forecast — they're computed and then completely discarded. `channel_attention=True` is the only mechanism by which those channels can matter at all under this loss setup, so it's a structural prerequisite for testing "do covariates help," not an optimization that's paid off yet on its own.

### Biggest remaining problem, and the next optimization to try

**Two candidate explanations are now on the table for why this run underdelivered, and they point to different next steps:**
1. **Undertraining**: `channel_attention=True` roughly doubles the attention computation per layer with no increase in training budget (`max_steps` was still 500/1000, same as every prior run) or capacity (`d_model=128`, `num_hidden_layers=3` unchanged). This could explain both the calibration regression *and* the collapse to a near-constant output (a "safe," low-loss default a model falls back on when it hasn't had enough training to learn something better).
2. **Genuine ceiling**: consistent with every other result in this project (DeepAR's `loc` near zero, near-chance directional accuracy everywhere), it's also possible `return_1h` genuinely has too little one-step-ahead signal for more training to unlock much more.

**Next step (already in progress as of this note): `max_steps` doubled** (tuning 500→1000, final 1000→2000), same architecture and channel config otherwise. If predictions become more varied and genuinely track the true series (not just a bigger constant nudge), that confirms undertraining was the bottleneck. If it still collapses to a near-constant output even with double the training, that's real evidence of a genuine ceiling — a much better-earned conclusion than assuming it without testing.

| # | Date | Commit | Data features | Config changes | RMSE | MAE | Dir Acc | Cov 80% | Cov 90% | Sharpe | Max DD |
|---|------|--------|----------------|-----------------|------|-----|---------|---------|---------|--------|--------|
| 4 | 2026-07-27 | `40268ed` | `hist_exog` = all 20 features (fully multivariate) | Same as Run 3, plus: `max_steps` doubled (tuning 500→1000, final 1000→2000) to test the undertraining hypothesis. `input_size` tuned over {24,60,120,240} → **60** (was 240 in Run 3 — the flat tuning curve shifted which candidate wins, another sign this surface is noisy) | 0.004161 | 0.002356 | 0.4970 | 0.7618 | 0.8615 | 0.0031 | -0.2217 |

**Run 4 — `input_size` tuning (val MAE):**

| input_size | 24 | 60 (selected) | 120 | 240 |
|---|---|---|---|---|
| val MAE | 0.543702 | **0.536928** | 0.538877 | 0.538860 |

**Run 4 — diagnostic comparison against Run 3 (same architecture, 2x training steps):**

| | Run 3 (500/1000 steps) | Run 4 (1000/2000 steps) |
|---|---|---|
| Fraction of predictions that are "up" | 96.7% | **55.6%** (true base rate: 53.4%) |
| mean(pred) | positive, clearly biased | **-0.000008** (essentially zero) |
| std(pred) / std(y) | 0.06 | 0.08 (still tiny, but no longer collapsed to near-constant) |
| corr(loc_raw, y[t-1]) | 0.0074 | -0.0289 |
| Dir Acc | 0.5362 (debunked — see above) | **0.4970** |
| Cov 80% / 90% | 0.7890 / 0.8850 | 0.7618 / 0.8615 (worse) |
| Sharpe | 0.5254 (debunked — see above) | **0.0031** |

**Run 4 notes (`max_steps` doubled — testing the undertraining hypothesis):**
- **Part 1 of the hypothesis confirmed: the near-constant "always predict up" collapse was genuinely an undertraining artifact.** With double the steps, the prediction-positive rate dropped from a wildly-biased 96.7% to 55.6% — close to the test set's actual 53.4% up-rate, and `mean(pred)` moved from clearly-positive to essentially zero. This is a clean, well-earned confirmation: Run 3's model hadn't been trained long enough to move past a degenerate "predict positive most of the time" shortcut.
- **Part 2: fixing that artifact did not reveal hidden skill underneath.** Once the bias is gone, directional accuracy reverts to **0.4970 — at or slightly below pure chance**, and both Sharpe (0.5254→0.0031) and the earlier "improvement" essentially vanish along with it. This directly confirms the debunking analysis from Run 3: that run's impressive-looking Dir Acc and Sharpe were artifacts of the bias, not real signal, and removing the bias exposes there was nothing underneath it.
- **RMSE/MAE remain completely flat** (0.004159→0.004161, 0.002358→0.002356) — across every single PatchTST run logged in this document (Runs 1–4, spanning point-wise MSE, distributional NLL, channel-weighting fixes, cross-channel attention, and now 2x training budget), MAE has stayed in a narrow 0.00235–0.00254 band. Nothing tried so far has moved this metric meaningfully.
- **Calibration got worse, not better, with more training** (Cov 80%: 0.7890→0.7618; Cov 90%: 0.8850→0.8615) — this refutes the other half of the undertraining hypothesis. More steps didn't fix calibration; if anything it's now the worst of any post-Run-1 result. The calibration regression from `channel_attention=True` is evidently not simply a training-budget issue.
- **Bottom line**: this was a well-designed, well-earned test, and its answer is now clear rather than assumed. `max_steps` was a real bottleneck for the specific degenerate collapse, but not for the model's actual forecasting ability or its interval calibration — both look like genuine ceilings of this architecture/data combination at `h=1`, not artifacts of insufficient training. This is now consistent with everything else found across both models in this entire project: one-step-ahead hourly SPY return prediction sits very close to the noise floor, and no optimization attempted so far (data-wise or model-wise, for either DeepAR or PatchTST) has moved point-forecast accuracy or directional accuracy meaningfully past that floor.

### Where this leaves PatchTST optimization

Increasing training budget is now a dead end — ruled out with direct evidence rather than assumed. Remaining untried levers, in rough order of promise:
1. **Feature curation** (raised earlier, now more clearly motivated): trim redundant/heavily-lagged indicators (`bb_upper/lower/width`, `MACD`, `RSI` are all derived from the same underlying price series and highly collinear with each other) and consider adding features that more directly capture very-recent price dynamics, now that `channel_attention=True` means such changes could actually reach `return_1h`'s forecast.
2. **Recalibration**: since more training made calibration worse rather than better, a conformal-calibration layer on top of the analytic quantiles (removed after Run 2, since analytic intervals were briefly excellent) may need to come back as a permanent fix rather than a fallback.
3. **Accept the ceiling**: given the consistency of this finding across both models and every optimization attempted, it may be more productive to shift focus to final DeepAR-vs-PatchTST comparison/synthesis than to continue chasing marginal PatchTST tuning.

**Two concrete next steps, cheapest first:**
1. **Increase `max_steps`** (e.g. 1000/2000 instead of 500/1000) — cheapest possible test of the "undertrained for the added complexity" hypothesis, no architecture change.
2. **Reintroduce a conformal recalibration layer** on top of the analytic quantiles (the same technique already used for DeepAR, and for PatchTST's original point-wise run) — this would directly guarantee correct coverage regardless of whether the underlying `scale` parameter is well-calibrated, at the cost of losing the "fully analytic, no post-hoc correction" property Run 2/3 currently have.
