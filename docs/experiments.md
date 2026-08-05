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
| 1 | 2026-08-04 | [`4369142`](https://github.com/WoodyChang21/ECE1508_GenAI/commit/4369142) | `futr_exog = [is_first_bar]` (1 feature) | Baseline: `input_size` tuned over {24,60,120,240} → 120; `lstm_hidden_size=128`, `lstm_n_layers=2`, `trajectory_samples=100/200`, `StudentT` loss, `scaler_type=standard` | 0.004164 | 0.002351 | 0.5156 | 0.8019 | 0.8991 | -0.2255 | -0.3020 |

<sup>`input_size` val MAE: 24→0.002169, 60→0.002155, **120→0.002146 (selected)**, 240→0.002148 — flat across candidates, not a meaningful lever.</sup>

Baseline. Dir Acc near chance, negative Sharpe — model close to predicting the unconditional mean.

| # | Date | Commit | Data features | Config changes | RMSE | MAE | Dir Acc | Cov 80% | Cov 90% | Sharpe | Max DD |
|---|------|--------|----------------|-----------------|------|-----|---------|---------|---------|--------|--------|
| 2 | 2026-08-04 | [`08c549c`](https://github.com/WoodyChang21/ECE1508_GenAI/commit/08c549c) | `futr_exog` = 13 features (adds calendar: `is_last_bar`, `hour_sin/cos`, `dow_sin/cos`, `is_monday`, `is_friday`, `month_sin/cos`, `is_month_end/start`, `is_quarter_end`) | Data-only change, model config unchanged from Run 1 | 0.004175 | 0.002373 | 0.5132 | 0.7910 | 0.8882 | -0.1319 | -0.2907 |

<sup>`input_size` val MAE: 24→0.002175, 60→0.002150, **120→0.002150 (selected, tie)**, 240→0.002154 — still flat.</sup>

Vs. Run 1: RMSE/MAE roughly flat (data change didn't reduce point-forecast error), Sharpe/MaxDD less negative but noisy — treat as inconclusive without a multi-seed check.

| # | Date | Commit | Data features | Config changes | RMSE | MAE | Dir Acc | Cov 80% | Cov 90% | Sharpe | Max DD |
|---|------|--------|----------------|-----------------|------|-----|---------|---------|---------|--------|--------|
| 3 | 2026-08-04 | [`30c26d6`](https://github.com/WoodyChang21/ECE1508_GenAI/commit/30c26d6) | Same as Run 2 (13 calendar features) | Model-only: tuning grid widened to `input_size` × `lstm_hidden_size` × `scaler_type` (16 configs); conformal intervals replace parametric; `trajectory_samples` 200→500 | 0.004151 | 0.002351 | 0.5209 | 0.7967 | 0.8870 | 0.1549 | -0.2926 |

<sup>Best of 16 configs: `input_size=240, hidden=64, scaler=robust` (val MAE 0.002142) — grid stayed flat overall (0.002142–0.002183).</sup>

Closest reproduction of the three DeepAR runs (see commit message for full grid). Conformal Cov 80/90% (0.797/0.887) isn't clearly better than Run 1's parametric coverage (0.802/0.899) — the "conformal fixes calibration" claim from the original log doesn't hold up cleanly on rerun.

### DeepAR summary (Runs 1–3, reproduced)

RMSE/MAE are flat across all three (~0.0041–0.0042 / ~0.00235–0.00237) — neither the calendar features nor the wider tuning grid moved point-forecast accuracy. Dir Acc stays within 1–2pp of chance the whole way (no clean trend after reproduction). Sharpe/Max DD swing the most of any metric across runs and don't track the other metrics — treat as noise, not skill, until checked across seeds. Net: DeepAR is essentially flat across all three configs tried; nothing here clears the noise floor for hourly SPY `h=1` prediction.

---

## PatchTST

| # | Date | Commit | Data features | Config changes | RMSE | MAE | Dir Acc | Cov 80% | Cov 90% | Sharpe | Max DD |
|---|------|--------|----------------|-----------------|------|-----|---------|---------|---------|--------|--------|
| 1 | 2026-08-04 | [`4024a14`](https://github.com/WoodyChang21/ECE1508_GenAI/commit/4024a14) | `hist_exog` = all 20 features (fully multivariate, `channel_attention=False`) | Fair-baseline pass: `loss='mse'` → `loss='nll'` + `distribution_output='student_t'` (matches DeepAR). Analytic StudentT mean/quantiles replace conformal. `input_size` tuned over {24,60,120,240} → 240 | 0.004401 | 0.002545 | 0.5071 | 0.7582 | 0.8550 | -0.1937 | -0.2218 |

<sup>`input_size` val MAE: 24→0.607802, 60→0.567296, 120→0.583586, **240→0.565678 (selected)**.</sup>

**⚠️ Superseded by Run 2.** Training loss here was averaged across all 21 channels (not just `return_1h`), causing the model to echo a damped version of the *previous* actual return (`corr(pred[t], y[t-1])` ≈ 0.8 in the original diagnostic) rather than predict the next one. Run 2 fixes this and is the real fair baseline — this run is kept for the record only, not as a valid comparison point.

| # | Date | Commit | Data features | Config changes | RMSE | MAE | Dir Acc | Cov 80% | Cov 90% | Sharpe | Max DD |
|---|------|--------|----------------|-----------------|------|-----|---------|---------|---------|--------|--------|
| 2 | 2026-08-05 | [`47cfca1`](https://github.com/WoodyChang21/ECE1508_GenAI/commit/47cfca1) | Same as Run 1 (21 channels, `channel_attention=False`) | **True fair-baseline**, plus a config fix: `PatchTSTConfig`'s real param is `patch_stride`, not `stride` — the original code's `stride=stride` was silently absorbed as an unused kwarg (HF configs don't error on unrecognized kwargs), so every prior run likely used the library's default patch stride, not the intended one. Training loss computed only on channel 0 (`return_1h`), matching DeepAR. `input_size` tuned over {24,60,120,240} → 240 | 0.004161 | 0.002359 | 0.4966 | 0.7655 | 0.8607 | -0.6301 | -0.3090 |

<sup>`input_size` val MAE: 24→0.550806, 60→0.540597, 120→0.539879, **240→0.539000 (selected)**.</sup>

**Lag diagnostic** (corr with y[t-1]): `loc_raw` (learned) -0.0205, `win_loc` (non-learned, deterministic) 0.0667 — lag-echo stays fixed. RMSE/MAE/Dir Acc are within this project's established rerun-noise band vs. the pre-fix version; coverage and Sharpe/Max DD moved more, but a single comparison can't isolate whether `patch_stride` specifically caused that — open question, not a confirmed effect.

**⚠️ Structural caveat that still applies**: with `channel_attention=False` + channel-0-only loss, the other 20 channels are computed but discarded — zero gradient, zero information reaching `return_1h`'s forecast. This run is mathematically equivalent to training on `return_1h` alone. "Fair baseline" means the *training objective* matches DeepAR's, not that PatchTST's multivariate premise was tested — that starts at Run 3 (`channel_attention=True`).

**📌 Forward pointer**: `channel_attention=True` was tested twice (Runs 3–4) and did not beat this run on any non-debunked metric — **Run 2 is the best, most defensible PatchTST result**, and the baseline Run 2a/2b build from.

| # | Date | Commit | Data features | Config changes | RMSE | MAE | Dir Acc | Cov 80% | Cov 90% | Sharpe | Max DD |
|---|------|--------|----------------|-----------------|------|-----|---------|---------|---------|--------|--------|
| 3 | 2026-08-05 | [`28518ec`](https://github.com/WoodyChang21/ECE1508_GenAI/commit/28518ec) | `hist_exog` = all 20 features (fully multivariate) | Same as Run 2, plus `channel_attention=True` — `return_1h` can now attend to the other 20 channels' patch representations (previously `False`: fully channel-independent). `input_size` tuned over {24,60,120,240} → 240 | 0.004153 | 0.002344 | 0.5322 | 0.7955 | 0.8951 | 0.3311 | -0.2463 |

<sup>`input_size` val MAE: 24→0.541789, 60→0.541271, 120→0.538401, **240→0.537351 (selected)**.</sup>

**⚠️ Debunked — Dir Acc/Sharpe are a bias artifact, not skill.** Model predicts "up" on 92.7% of test bars (true base rate 53.4%); overall Dir Acc (0.5322) is only ~0.1pp above that trivial baseline; `std(pred)/std(y)` ≈ 0.05 (near-collapsed); accuracy-by-confidence-quartile is flat/noisy (0.503/0.554/0.535/0.536), not monotonic as real skill would show. RMSE/MAE unchanged from Run 2; calibration slightly regressed. **Kept anyway for a structural reason**: with Run 2's channel-0-only loss, `channel_attention=False` means the other 20 channels get zero gradient — `channel_attention=True` is the only way they can matter at all, making this a required step to even test "do covariates help," independent of whether it paid off yet.

| # | Date | Commit | Data features | Config changes | RMSE | MAE | Dir Acc | Cov 80% | Cov 90% | Sharpe | Max DD |
|---|------|--------|----------------|-----------------|------|-----|---------|---------|---------|--------|--------|
| 4 | 2026-08-05 | [`c24a537`](https://github.com/WoodyChang21/ECE1508_GenAI/commit/c24a537) | `hist_exog` = all 20 features (fully multivariate) | Same as Run 3, plus `max_steps` doubled (tuning 500→1000, final 1000→2000) to test the undertraining hypothesis. `input_size` tuned over {24,60,120,240} → 240 | 0.004149 | 0.002337 | 0.5367 | 0.7999 | 0.8919 | 0.6373 | -0.2485 |

<sup>`input_size` val MAE: 24→0.542263, 60→0.540281, 120→0.538330, **240→0.537122 (selected)** — note the winner flipped again vs. the original run's `60`, another data point that this axis is flat/noisy.</sup>

**⚠️ Correction on rerun: the original "undertraining hypothesis confirmed" claim does not reproduce.** The original run found doubling `max_steps` fixed Run 3's bias collapse (predict-up rate 96.7%→55.6%, close to the 53.4% true base rate). This rerun, same doubled-training protocol, still shows strong bias: predict-up rate **88.3%**, Dir Acc only 0.2pp above base rate, confidence-quartile accuracy flat (0.506/0.548/0.543/0.549). Most likely explanation: the different `input_size` winner (240 here vs. 60 originally) and/or ordinary training variance — not a deterministic fix. Net conclusion is unchanged (Dir Acc/Sharpe are still artifacts, not skill), but "more training fixes the collapse" should be treated as an open question, not a settled finding.

**📌 Direction change after Run 4**: `channel_attention=True` underperformed Run 2 on every non-debunked metric across both attempts (Run 3, Run 4) — `patchtst.ipynb` was reverted to Run 2's config (`channel_attention=False`, channel-0-only loss, `max_steps=500/1000`). All further optimization below builds from **Run 2 specifically** — see the "Run 2-series" sub-experiments that follow.

---

## Run 2-series: architecture sub-experiments on the Run 2 baseline

Everything in this section starts from **Run 2's exact configuration** (`channel_attention=False`, channel-0-only loss, `max_steps=500/1000`, `input_size` tuned over {24,60,120,240}) and changes exactly one thing at a time. Motivation: Run 2 has a residual, unexplained lag-echo (`corr(loc_raw, y[t-1]) = 0.1609`). Since `channel_attention=False` makes `return_1h`'s forecast depend *only* on its own patch history (proven: the other 20 channels receive zero gradient and contribute zero information under this config), this residual correlation can only be coming from how `return_1h`'s own history gets processed — not from feature curation of the other channels, which would be structurally inert here. Two architecture-level (not data-level) hypotheses are being tested, one at a time:

- **Run 2a — pooling mechanism.** Mean-pooling across all patches gives the oldest and newest patch equal weight, which may dilute whatever recency signal exists. Tests switching to "use only the most recent patch's representation" instead of averaging all patches. Code: `POOLING_MODE = 'last_patch'` in `_distribution_params` (`notebooks/patchtst.ipynb`, commit `9b2f337`). Nothing else changed from Run 2.
- **Run 2b — internal instance normalization (planned, not yet implemented).** PatchTST's own per-window RevIN-style scaler (`win_loc`/`win_scale`) explicitly centers each window on its own local mean — a reasonable inductive bias for trending price levels, but not obviously appropriate for `return_1h`, which has no meaningful "local level" to track (true lag-1 autocorrelation ≈ 0.01). Tests `scaling=None` in `PatchTSTConfig` to remove this internal re-centering entirely.

Each is tested in isolation against Run 2, so any change in `corr(loc_raw, y[t-1])` or the headline metrics can be attributed to that one specific change.

### Run 2a — last-patch pooling

| # | Date | Commit | Base | Change from Run 2 | RMSE | MAE | Dir Acc | Cov 80% | Cov 90% | Sharpe | Max DD | corr(loc_raw, y[t-1]) |
|---|------|--------|------|--------------------|------|-----|---------|---------|---------|--------|--------|------------------------|
| 2a | 2026-07-27 | `04af727` | Run 2 (`8083e60`) | Pooling: mean-pool across all patches → use only the most recent patch's representation (`POOLING_MODE='last_patch'`) | 0.004151 | 0.002343 | 0.5342 | 0.8469 | 0.9255 | 0.8638 | -0.1591 | 0.0076 |

**Run 2a — `input_size` tuning (val MAE):**

| input_size | 24 | 60 | 120 | 240 (selected) |
|---|---|---|---|---|
| val MAE | 0.550978 | 0.542312 | 0.542438 | **0.537826** |

**Run 2a — verification diagnostic (same checks that debunked Run 3's gains):**

| | Run 2 | Run 2a | Run 3 (debunked, for comparison) |
|---|---|---|---|
| Fraction of predictions "up" | *(not checked)* | **84.97%** | 96.72% |
| True base rate "up" | 53.42% | 53.42% | 53.42% |
| Dir Acc | 0.5152 | 0.5342 | 0.5362 |
| Dir Acc by |pred| quartile (low→high conf.) | *(not checked)* | 0.551, 0.524, 0.517, 0.545 (flat/noisy) | 0.506, 0.566, 0.535, 0.539 (flat/noisy) |
| std(pred)/std(y) | *(not checked)* | 0.080 | 0.060 |

**Run 2a notes (last-patch pooling — mixed result, not a clean win):**
- **⚠️ The directional-accuracy and Sharpe gains closely resemble Run 3's debunked pattern, just milder.** The model predicts "up" on 85% of bars (vs. a 53.4% true base rate), and directional accuracy (0.5342) matches the true base rate almost to the decimal — the same signature that turned out to be a positive-bias artifact, not genuine skill, in Run 3. The confidence-quartile check confirms this: accuracy does not climb with prediction confidence (0.551 → 0.524 → 0.517 → 0.545, flat/noisy), exactly as a real signal would not look. **Treat Dir Acc 0.5342 and Sharpe 0.8638 as unconfirmed, likely substantially artifact — not a verified improvement.**
- **RMSE/MAE improved slightly and genuinely** (0.004158→0.004151, 0.002350→0.002343) — small, but this particular pair of metrics is bias-agnostic (unlike Dir Acc/Sharpe), so this is a real, if modest, point-forecast improvement from the pooling change.
- **Interval calibration regressed** — and in a new direction: Run 2 was almost exactly calibrated (0.8068/0.9000 vs 0.80/0.90 targets); Run 2a now *over*-covers (0.8469/0.9255). Over-covering is a milder problem than under-covering (conservative rather than overconfident), but it's still a real regression from Run 2's excellent calibration.
- **`corr(loc_raw, y[t-1])` dropped sharply (0.1609→0.0076), but interpret this cautiously**, per the same caveat that applied to Run 3: `std(pred)/std(y)` is only 0.080 — a near-collapsed, low-variance prediction will show low correlation with almost anything by construction, not necessarily because the recency-dilution hypothesis was confirmed. This metric alone can't distinguish "genuinely fixed" from "collapsed to something blander."
- **Bottom line**: last-patch pooling gave a small, real MAE/RMSE improvement, but reintroduced a milder version of the same positive-bias artifact seen in Run 3, and made calibration worse. Not a clean win over Run 2 — the honest read is "mixed, and the headline-looking numbers (Dir Acc, Sharpe) shouldn't be trusted without this same verification applied every time." Run 2b (`scaling=None`, testing the internal-instance-normalization hypothesis) is still worth running as the other isolated candidate explanation for Run 2's residual lag-echo.

### Run 2b — disable internal instance normalization (`scaling=None`)

Isolated against **Run 2 directly** (`POOLING_MODE='mean'`, i.e. Run 2a's pooling change is **not** stacked here — only the scaling change is being tested).

| # | Date | Commit | Base | Change from Run 2 | RMSE | MAE | Dir Acc | Cov 80% | Cov 90% | Sharpe | Max DD | corr(loc_raw, y[t-1]) |
|---|------|--------|------|--------------------|------|-----|---------|---------|---------|--------|--------|------------------------|
| 2b | 2026-07-27 | `3a1f3f2` | Run 2 (`8083e60`) | `scaling=None` in `PatchTSTConfig` — disables PatchTST's internal per-window RevIN-style instance normalization (`win_loc` fixed at 0, `win_scale` fixed at 1, identity) | **0.004142** | 0.002344 | 0.5075 | 0.8109 | 0.8935 | 0.8466 | -0.1244 | 0.0023 |

**Run 2b — `input_size` tuning (val MAE):**

| input_size | 24 | 60 (selected) | 120 | 240 |
|---|---|---|---|---|
| val MAE | 0.537746 | **0.536904** | 0.539477 | 0.537979 |

**Run 2b — verification diagnostic (same checks applied to Run 2a/Run 3):**

| | Run 2 | Run 2a | Run 2b |
|---|---|---|---|
| Fraction of predictions "up" | *(not checked)* | 84.97% | **72.58%** |
| True base rate "up" | 53.42% | 53.42% | 53.42% |
| Dir Acc | 0.5152 | 0.5362 | **0.5075** |
| Dir Acc by \|pred\| quartile (low→high conf.) | *(not checked)* | 0.551, 0.524, 0.517, 0.545 (flat/noisy) | **0.462, 0.507, 0.520, 0.541 (monotonic increasing)** |
| std(pred)/std(y) | *(not checked)* | 0.080 | 0.036 |
| corr(win_loc, y[t-1]) | 0.0667 | 0.0667 | **nan** (win_loc constant by construction — proves it) |

**⚠️ Correction after visual inspection of the predicted-vs-actual plot**: the plot shows the predicted line essentially flat at zero across all 200 plotted bars — visually indistinguishable from "no prediction at all." The notes originally below called this "the most promising result in the Run 2-series," which overclaimed based on the numeric diagnostics without weighing this. Corrected assessment follows.

**Run 2b notes (disabling internal instance normalization — a narrow, real finding about loss-minimizing shrinkage, not confirmed evidence of a usable directional signal):**
- **`corr(loc_raw, y[t-1])` dropped from Run 2's 0.1609 to 0.0023 — the largest reduction of any Run 2-series experiment, with cleaner attribution than Run 2a.** `win_loc` is *provably* constant here (`corr = nan`, fixed at 0 by the config change), so it could not have contributed to Run 2's 0.1609 correlation — the drop is attributable to the learned weights' own behavior, not the pooling-style collapse ambiguity Run 2a had. This part of the finding stands.
- **RMSE is the best of any PatchTST run to date: 0.004142.** But this needs to be read alongside the plot, not in isolation: `mean|pred|` in even the *highest*-confidence quartile is only 0.000273 (0.027%) — the model achieves this RMSE by shrinking its forecast to something close to zero on almost every bar, which is the mathematically correct response when true one-step signal is very weak (consistent with everything else found in this project), not evidence that it found new information. A model that always predicts exactly 0 would already do quite well on RMSE/MAE against a series this close to a random walk — Run 2b's improvement over Run 2 is a matter of *degree of shrinkage*, not of forecasting skill.
- **The confidence-quartile pattern (0.462 → 0.507 → 0.520 → 0.541, monotonic) is suggestive but should not be oversold.** The actual prediction magnitudes underlying even the "high confidence" quartile are still tiny (0.000024 to 0.000273) — a monotonic trend built from numbers this small, with only ~617 samples per quartile and a top-to-bottom gap of ~2–3 standard errors, is exactly the kind of pattern that can emerge from noise, especially after searching across as many configurations as this project has tried (a real multiple-comparisons risk: some result was likely to look good by chance eventually). This is not strong enough evidence to call it confirmed skill.
- **Overall Dir Acc (0.5075) is barely above chance**, and a real, if milder, positive bias remains (72.6% "predict up" vs. 53.4% true base rate) — consistent with a model still leaning slightly toward the test period's bullish drift, just less aggressively than Run 2a/Run 3.
- **Calibration is close to on-target** (Cov 80%/90% = 0.8109/0.8935) and **Sharpe (0.8466) is the best of any run**, but per the established pattern in this project, Sharpe shouldn't be trusted without multi-seed verification, and calibration alone doesn't offset the "the actual predictions are flat" finding above.
- **Bottom line, corrected**: Run 2b is a real, narrow finding — internal instance normalization does appear to be *part* of the mechanism behind Run 2's residual lag-echo, and disabling it lets the model shrink its forecasts more consistently toward zero, which happens to minimize RMSE/MAE. But visually and substantively, this is still a model producing an essentially flat, uninformative prediction — same conclusion as every other PatchTST configuration tried in this project, just achieved with slightly better loss-function numbers via more aggressive, better-calibrated shrinkage. It should not be treated as "the best PatchTST result" in any sense beyond "best RMSE/MAE," and the quartile pattern should be treated as an open question, not a finding, pending a seed-repeat check.
