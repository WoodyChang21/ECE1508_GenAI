# Experiment Log

One row per completed run. Don't fork notebooks into `_v1`/`_v2` — rerun the
same notebook, log the result here, let git history hold the code diff.

**Commit column** = a commit whose notebook contains the *actual executed
output* for that row, not just the code (verified/reproduced 2026-08-04–05).
Reruns use a pinned `TARGET_COMMIT` in the notebook's setup cell + `git reset
--hard`/`clean -fd` inside the Colab clone (not plain `checkout` — the clone
persists across cell reruns in one session and a prior run's local changes
make plain `checkout` fail). From PatchTST Run 2 onward, reruns also fix
`PatchTSTConfig`'s `stride=` kwarg to the real param name `patch_stride=`
(the old name was silently absorbed as an unused kwarg by HF's
`PretrainedConfig`, so every original run likely used the library's default
stride instead of the intended one).

**Reading the noise**: across every reproduced run, RMSE/MAE/Dir Acc reproduce
closely (small decimal drift from stochastic training). Sharpe/Max Drawdown/
coverage swing much more — sometimes a lot — even with *zero code change*
between reruns (same seed logged, different hardware/run). Treat any Sharpe
or Max DD delta as noise unless checked across multiple seeds. Several
headline-looking Dir Acc/Sharpe "wins" in this log turned out to be the model
collapsing to a near-constant, mildly-positive output that free-rides the
test period's 53.4% bullish base rate — always sanity-check with: fraction of
predictions that are "up" vs. true base rate, and accuracy by
confidence-quartile (should climb with confidence if it's real skill; flat/
noisy means it isn't).

---

## DeepAR

| # | Commit | Data (`futr_exog`) | Config | RMSE | MAE | Dir Acc | Cov 80% | Cov 90% | Sharpe | Max DD |
|---|--------|---------------------|--------|------|-----|---------|---------|---------|--------|--------|
| 1 | [`4369142`](https://github.com/WoodyChang21/ECE1508_GenAI/commit/4369142) | `[is_first_bar]` (1 feat.) | Baseline. `lstm_hidden_size=128`, StudentT loss, `scaler_type=standard`, parametric intervals | 0.004164 | 0.002351 | 0.5156 | 0.8019 | 0.8991 | -0.2255 | -0.3020 |
| 2 | [`08c549c`](https://github.com/WoodyChang21/ECE1508_GenAI/commit/08c549c) | 13 feat. (+calendar: hour/dow/month cyclical, is_monday/friday, month/quarter boundaries) | Data-only vs. Run 1, model unchanged | 0.004175 | 0.002373 | 0.5132 | 0.7910 | 0.8882 | -0.1319 | -0.2907 |
| 3 | [`30c26d6`](https://github.com/WoodyChang21/ECE1508_GenAI/commit/30c26d6) | Same as Run 2 | Model-only: grid over `input_size`×`hidden_size`×`scaler_type` (16 configs) → `240/64/robust`; conformal intervals; `trajectory_samples` 200→500 | 0.004151 | 0.002351 | 0.5209 | 0.7967 | 0.8870 | 0.1549 | -0.2926 |

- `input_size` tuning is flat/noisy across all 3 runs (candidates within ~1% of each other; winner shifts 120→60(tie)→240 with no real signal).
- Conformal calibration (Run 3) is **not** a clear win on rerun — its coverage (0.797/0.887) isn't clearly better than Run 1's parametric coverage (0.802/0.899).
- **Summary**: RMSE/MAE flat across all 3 configs (~0.0041–0.0042 / ~0.00235–0.00237); Dir Acc stays within 1–2pp of chance; Sharpe/Max DD swing the most of any metric and don't track anything else — treat as noise. Neither the calendar features nor the wider tuning grid moved point-forecast accuracy. DeepAR is flat across everything tried here.

---

## PatchTST

| # | Commit | Config | RMSE | MAE | Dir Acc | Cov 80% | Cov 90% | Sharpe | Max DD |
|---|--------|--------|------|-----|---------|---------|---------|--------|--------|
| 1 | [`4024a14`](https://github.com/WoodyChang21/ECE1508_GenAI/commit/4024a14) | ⚠️ superseded (see below). `loss='nll'`+StudentT, 21 channels, `channel_attention=False`, analytic intervals | 0.004401 | 0.002545 | 0.5071 | 0.7582 | 0.8550 | -0.1937 | -0.2218 |
| 2 | [`47cfca1`](https://github.com/WoodyChang21/ECE1508_GenAI/commit/47cfca1) | **True fair baseline.** Loss restricted to channel 0 (`return_1h`) only; `patch_stride` fix applied | 0.004161 | 0.002359 | 0.4966 | 0.7655 | 0.8607 | -0.6301 | -0.3090 |
| 3 | [`28518ec`](https://github.com/WoodyChang21/ECE1508_GenAI/commit/28518ec) | ⚠️ debunked (see below). Same as Run 2 + `channel_attention=True` | 0.004153 | 0.002344 | 0.5322 | 0.7955 | 0.8951 | 0.3311 | -0.2463 |
| 4 | [`c24a537`](https://github.com/WoodyChang21/ECE1508_GenAI/commit/c24a537) | ⚠️ debunked (see below). Same as Run 3 + `max_steps` doubled (500/1000→1000/2000) | 0.004149 | 0.002337 | 0.5367 | 0.7999 | 0.8919 | 0.6373 | -0.2485 |

All 4 runs select `input_size=240` from {24,60,120,240} except Run 4's *original* pre-rerun pass, which picked 60 — the tuning surface is flat/noisy throughout (candidates within ~2% of each other), the winner isn't a meaningful signal.

- **Run 1 — superseded.** Training loss was averaged across all 21 channels instead of just `return_1h`, so the model learned a persistence/lag-echo strategy borrowed from the other (genuinely autocorrelated) channels rather than predicting `return_1h` itself. Not a valid comparison point — kept for the record only.
- **Run 2 — the real baseline everything else compares to.** Fixing the loss to channel-0-only collapses the lag-echo (`corr(loc_raw, y[t-1])`: 0.76→-0.02) and gives the best-calibrated, most defensible result in the project. **Caveat**: with `channel_attention=False`, the other 20 channels are computed but discarded — zero gradient, zero information reaches `return_1h`. This is architecturally equivalent to training on `return_1h` alone; "fair baseline" only means the *loss* is now comparable to DeepAR's, not that PatchTST's multivariate premise was actually tested.
- **Run 3 — debunked.** `channel_attention=True` lets the other channels matter for the first time (structurally necessary to test covariates at all), but its Dir Acc/Sharpe gain is an artifact: predicts "up" 92.7% of bars (true base rate 53.4%), Dir Acc only ~0.1pp above that baseline, confidence-quartile accuracy flat. RMSE/MAE unchanged from Run 2; calibration slightly worse.
- **Run 4 — debunked, and the "fix" doesn't reproduce.** Doubling `max_steps` was meant to test whether Run 3's bias was an undertraining artifact. The *original* run found it mostly resolved (55.6% predict-up, near the 53.4% true rate). **This does not reproduce**: rerun still shows 88.3% predict-up, same flat quartile pattern. Net verdict (artifact, not skill) is unchanged, but "more training fixes it" is an open question, not a confirmed finding — most likely explained by the flat/noisy `input_size` tuning surface picking a different winner, or plain training variance.
- **Net result**: `channel_attention=True` never beat Run 2 on any non-debunked metric across two attempts → reverted to Run 2's config for all further work below.

---

## PatchTST Run 2-series: architecture tests on the Run 2 baseline

All three below build from **Run 2's exact config**, changing one thing at a time, to isolate Run 2's residual lag-echo (`corr(loc_raw, y[t-1]) = 0.16`). Since `channel_attention=False` makes `return_1h` depend only on its own patch history, the echo can only come from how that history is pooled/normalized — not from the other channels (proven inert under this config).

**Run 2b not rerun** — its commit already contains real matching output from its original run (verified 2026-08-05), and given how consistently this project's noise pattern has held, a further rerun was judged unlikely to change the conclusion enough to justify the GPU time. Treat it as slightly less rigorously verified than the runs with commit hashes from 2026-08-05/06.

| # | Commit | Change from Run 2 | RMSE | MAE | Dir Acc | Cov 80% | Cov 90% | Sharpe | Max DD | corr(loc_raw, y[t-1]) |
|---|--------|--------------------|------|-----|---------|---------|---------|--------|--------|------------------------|
| 2a | [`ae66645`](https://github.com/WoodyChang21/ECE1508_GenAI/commit/ae66645) | Pooling: mean-pool all patches → last-patch only; `patch_stride` fix applied | 0.004149 | 0.002348 | 0.5176 | 0.8254 | 0.9101 | 0.3361 | -0.1517 | -0.0965 |
| 2b | [`3a1f3f2`](https://github.com/WoodyChang21/ECE1508_GenAI/commit/3a1f3f2) | `scaling=None` — disables PatchTST's internal per-window RevIN normalization | **0.004142** (best of any run) | 0.002344 | 0.5075 | 0.8109 | 0.8935 | 0.8466 | -0.1244 | 0.0023 |
| 2c | [`0ec84f1`](https://github.com/WoodyChang21/ECE1508_GenAI/commit/0ec84f1) | `pooling_type=None` — paper's actual flatten-all-patches head (see note below) | 0.004177 | 0.002391 | 0.5188 | 0.8186 | 0.9121 | **1.1165** (highest of any run) | -0.2162 | -0.0188 |

<sup>Run 2a required a rerun: the first attempt's tuning loop silently stopped after 3 of 4 `input_size` candidates with no "Best input_size" summary line, while later cells still produced results identical to the original run to every decimal — a stale-kernel-state artifact (same class of bug as the earlier DeepAR module-caching issue). The runtime was fully restarted for the result logged above.</sup>

- **Run 2a — still an artifact, milder than before.** Dir Acc/Sharpe repeat Run 3's bias signature, though less severe on rerun: predicts "up" on 70.0% of bars (orig 84.97%; true base rate 53.4%); Dir Acc (0.5176) doesn't even clear the trivial "always predict up" baseline this time; confidence-quartile accuracy stays flat/noisy (0.515/0.489/0.540/0.527). RMSE/MAE reproduce closely against the original run (0.004149/0.002348 vs. 0.004151/0.002343) — genuine, bias-agnostic confirmation these two metrics are stable. `corr(loc_raw, y[t-1])` flipped sign (0.008→-0.097) but stayed small either way — consistent with a near-collapsed prediction correlating weakly with anything, not a clean "fixed" signal.
- **Run 2b — best RMSE in the project, but a shrinkage effect, not skill.** `win_loc` is provably constant here (`corr=nan`), so its lag-echo drop (0.16→0.002) is cleanly attributable to the learned weights, unlike Run 2a's ambiguous case. But the predicted-vs-actual plot shows an essentially flat line at zero — the model wins on RMSE/MAE by shrinking predictions toward zero more aggressively, the mathematically correct response to a near-random-walk target, not new information. Confidence-quartile accuracy is monotonic (0.46→0.51→0.52→0.54) but built from tiny magnitudes (~617 samples/quartile, few-standard-error gaps) — suggestive, not confirmed, especially given how many configs this project has searched (multiple-comparisons risk). Best RMSE/MAE of any run; not the best result in any broader sense.
- **Run 2c — the paper's actual pooling design, and still no confirmed skill, but for a new reason.** Neither Run 2's mean-pooling nor Run 2a's `last_patch` matches the original PatchTST paper's supervised-forecasting head, which flattens every patch representation (`vec(Z) ∈ R^{D×N}`) into the linear head instead of pooling. Verified against the `transformers` source (`PatchTSTPredictionHead`): `pooling_type=None` + `use_cls_token=False` keeps all patches unpooled and sizes the head as `d_model × num_patches` — exactly the paper's design. Testing it: predictions are the **least biased and least collapsed of any PatchTST run** (58.3% predict-up vs. the 53.4% true rate — every other run ranged 70–97%; `std(pred)/std(y)=0.135`, the highest/least-degenerate of any run). But the confidence-quartile accuracy is **monotonically decreasing** (0.539→0.522→0.511→0.504) — the opposite of what real skill looks like, and a stronger red flag than the flat/noisy patterns seen elsewhere. The standout Sharpe (1.1165, highest in the project) should not be read as confirmed skill given this.

---

## PatchTST feature importance analysis

Dedicated notebook: `notebooks/patch_tst_feature_importance.ipynb`. Trains its own `channel_attention=True` baseline (channel-0-only loss, `patch_stride` fix, mean pooling, `input_size=240` fixed, `max_steps=1000`) — required since `channel_attention=False` makes the other 20 channels structurally inert (see Run 2's caveat above), so feature importance is only answerable under this config. Adds two things missing from every other notebook in this project: a **training loss curve** (previously only post-hoc financial metrics existed, which can't distinguish "didn't converge" from "no signal") and **grouped/individual permutation importance** (shuffle a feature across test windows, measure the increase in test MAE) plus **channel-attention weight extraction**.

| Run | Commit | Baseline RMSE / MAE / Dir Acc / Sharpe |
|---|---|---|
| No seed control (first run) | [`b7d06e5`](https://github.com/WoodyChang21/ECE1508_GenAI/commit/b7d06e5) | 0.004146 / 0.002344 / 0.5164 / 0.5959 |
| `SEED=0` (explicit seed control added) | [`de8f191`](https://github.com/WoodyChang21/ECE1508_GenAI/commit/de8f191) | 0.004147 / 0.002342 / 0.5269 / 0.8229 |

Both consistent with Run 3's numbers above, within the established noise band.

- **Training loss is flat/noisy for the entire 1000 steps** — no real downward trend, oscillates in roughly the same band from step 1 to step 1000. Direct evidence (not inferred from downstream metrics) that the model reaches something close to its loss floor almost immediately and never meaningfully improves.
- **Grouped permutation importance replicates cleanly across both runs** (same code, two different training draws): `return_1h`'s own history and price (OHLC) are clearly helpful (ΔMAE positive, outside the shuffle-noise band both times); **momentum (RSI/MACD family) and lagged returns are actively harmful** — ΔMAE negative and outside noise both times, meaning the model performs *better* with them scrambled, not just indifferent to them. Volatility/Bollinger and calendar (`is_first_bar`) are mildly positive; volume and VIX are within noise.
- **Channel-attention weights are nearly uniform across all 21 channels** (0.047–0.050, ≈1/21 = no preference) — including for `return_1h` itself, despite permutation importance showing it's by far the most relied-upon channel. A clean, concrete illustration that attention weight and causal contribution disagree here; don't trust attention as an importance signal on its own.
- **Bug found and fixed while building this**: `PatchTSTEncoder.forward()`'s `output_attentions=True` only exposes time-attention (sublayer 1) — it silently discards channel-attention (sublayer 2), confirmed against source (only `layer_outputs[1]` is ever appended to `all_attentions`). Required replicating the encoder's forward pass layer-by-layer to capture the real channel-attention weights. Separately, the trained model's default `sdpa` attention backend doesn't support `output_attentions=True` at all (returns `None` silently) — fixed by building an `attn_implementation='eager'` copy of the trained model for that one analysis cell only.
- **Not yet done**: only 2 runs so far (momentum/lagged-returns-harmful should be confirmed with at least one more distinct seed before treating it as settled), and no full leave-one-out retraining to confirm the permutation-importance ranking causally.

### Follow-up: retraining on only the "helpful" features doesn't help

Dedicated notebook: `notebooks/patch_tst_reduced_features.ipynb`. Same setup as the feature-importance baseline above, but `ALL_COLS` restricted to just the two groups permutation importance called clearly helpful: `return_1h` + OHLC (5 channels total, dropping momentum/lagged-returns/volatility/volume/VIX/calendar). Tests whether the permutation-importance signal survives an actual retrain, not just re-evaluation of the original model.

| Run | Commit | RMSE / MAE / Dir Acc / Sharpe |
|---|---|---|
| 21-channel baseline (`SEED=0`, for reference) | [`de8f191`](https://github.com/WoodyChang21/ECE1508_GenAI/commit/de8f191) | 0.004147 / 0.002342 / 0.5269 / 0.8229 |
| 5-channel reduced (`SEED=0`) | [`eaaa35e`](https://github.com/WoodyChang21/ECE1508_GenAI/commit/eaaa35e) | 0.004148 / 0.002342 / 0.5318 / 0.8098 |

- **No improvement.** Every metric lands within this project's established noise band of the 21-channel baseline — RMSE/MAE are essentially identical, Dir Acc/Sharpe move by less than this project's typical run-to-run noise. Dropping the momentum/lagged-returns channels that permutation importance flagged as *actively harmful* did not recover any accuracy on retrain.
- **Training loss curve is flat/noisy across all 1000 steps**, same shape as the 21-channel run — direct evidence the model hits its loss floor almost immediately regardless of which channels are available, not an artifact of evaluation.
- **Reading**: permutation importance measures the *trained* model's reliance on a feature, not a causal "removing it improves the model" guarantee — a feature can look harmful to a fixed model's predictions while a model retrained without it lands at the same optimum. Combined with Run 2's finding that even `channel_attention=False` (all other channels structurally inert) reproduces the same RMSE/MAE band, the more likely explanation is that `return_1h`'s own history already carries essentially all the exploitable signal at this information ceiling, and no combination of these covariates — present, absent, attended-to, or not — moves the needle.
- One seed only; not yet rerun with a different `SEED`.

---

## Where this leaves the project

Across both models and every optimization attempted (data features, tuning grids, loss functions, pooling, normalization, channel attention, training budget), **RMSE/MAE never moved meaningfully and Dir Acc never cleared chance by more than noise-level margins**. Sharpe/Max Drawdown are unreliable on their own — they swing by more than the headline "wins" they're sometimes used to support. Several apparent directional-accuracy gains were traced to the model collapsing toward a near-constant, mildly-positive prediction that free-rides the test period's bullish drift, not genuine per-bar skill.

**Feature ablation was followed up and came back negative.** Permutation importance flagged momentum/lagged-returns as actively harmful and `return_1h`/OHLC as the clearest genuine contributors, but retraining on just the "helpful" 5 channels reproduced the 21-channel baseline almost exactly (RMSE/MAE identical to noise-level, Dir Acc/Sharpe within the usual swing) — permutation importance on the original model didn't translate into a better model on retrain. Combined with Run 2's finding that even fully discarding the other 20 channels (`channel_attention=False`) reproduces the same RMSE/MAE band, the more defensible conclusion is that `return_1h`'s own history already sits at this setup's information ceiling — no combination of these covariates tried so far, in or out, moves point-forecast accuracy. The honest takeaway is that one-step-ahead hourly SPY return prediction sits very close to the noise floor for both architectures tried here, and further gains likely require different data (e.g. higher-frequency signals, order-book/microstructure features) rather than more configurations of what's already in this dataset.
