# Experiment Log

Human-readable summary of results already stored under `steven/comparison_metrics/*.json`
(source of truth) and `steven/outputs/metrics.json` (Steven's baseline). This file just
makes them easy to scan/compare -- see the JSON files for full detail (backtest included,
for the baseline).

## hf_patchtst_fused_head

**What**: HF `PatchTSTModel` backbone (real channel-independent patching + self-attention)
vs. Steven's hand-rolled early-channel-fusion PatchTST, `channel_attention` swept True/False.
Same fused output head, same anchored log-return target + volume, same loss
(`weighted_mse_loss`, `w_price=1.0`/`w_vol=0.5`) -- a parity check on architecture alone,
before trying volume-drop / raw-price-target variants next.
Notebook: `steven/train_patchtst_hf_channel_attention.ipynb`.

| Model | Windows | OHLC RMSE ($) | Volume RMSE | Dir Acc (bar 1 / 2 / 3) |
|---|---|---|---|---|
| Steven's baseline (long bucket, ctx 56-70) | 999 | 13.76 | 6,398,097 | 0.508 / 0.529 / 0.541 |
| HF PatchTST, `channel_attention=False` | 2397 | 4.93 | **3,686,648** | 0.527 / 0.544 / **0.555** |
| HF PatchTST, `channel_attention=True` | 2397 | **4.84** | 4,007,731 | **0.530** / **0.541** / 0.545 |

**Conclusion**: both HF variants beat Steven's baseline decisively (~2.8x tighter OHLC RMSE,
~1.6-1.7x tighter volume RMSE, directional accuracy up on every bar) -- the HF backbone is
the clear win over the hand-rolled model. Between the two `channel_attention` settings it's
not a clean win either way: `True` is marginally better on OHLC RMSE and dir acc bars 1-2,
`False` is meaningfully better on volume RMSE and dir acc bar 3. Given the split, no strong
reason yet to prefer one setting over the other from this run alone.

**Caveats**: one seed each; MAE not recorded this round (notebook reports RMSE only), so the
volume MAE/RMSE heavy-tail check from earlier runs on this branch couldn't be repeated here;
no backtest yet (deferred until a target/volume variant is picked).

## hf_patchtst_fused_head_no_volume

**What**: Step 2 of the ladder -- same setup as `hf_patchtst_fused_head` (fused head,
anchored log-return target, `channel_attention` swept), but volume dropped entirely (not a
context input, not a training target). Motivated by step 1's training logs showing
`price_loss` (~7e-5) swamped by `vol_loss` (~0.18-0.27) despite `w_vol=0.5` -- the loss is
now plain MSE over the 4 price components only (`price_only_mse_loss`).
Notebook: `steven/train_patchtst_hf_channel_attention.ipynb`.

| Model | Windows | OHLC RMSE ($) | Dir Acc (bar 1 / 2 / 3) |
|---|---|---|---|
| `channel_attention=False` | 2397 | 3.73 | 0.463 / 0.458 / **0.552** |
| `channel_attention=True` | 2397 | **3.45** | 0.463 / **0.541** / 0.448 |

**Conclusion: this is the same collapse already documented once on this branch, now
confirmed under plain (unweighted) MSE too -- do not read the improved OHLC RMSE as a win.**
Both settings beat step 1 on point error (3.45-3.73 vs. 4.84-4.93) and `best_val_loss` is
near-zero (~2e-5), but **4 of the 6 directional-accuracy numbers are at or below 0.50** --
worse than a coin flip. Removing volume removed the one loss term with real gradient
variance to chase on a near-random-walk return target, so MSE again finds "predict ~near-0"
as a cheap shortcut: tiny point error, no real directional signal. This is the exact failure
mode flagged in the notebook's intro (the earlier return-target + no-volume run on this
branch produced zero backtest trades at every threshold) -- reproducing it here with a
cleaner single-term loss rules out "it was the specific 1.0/0.5 weighting" as the cause; the
real driver is the return-based target's near-zero mean/variance once volume's gradient
signal is gone.

**Implication for the ladder**: dropping volume is not viable with the anchored-return
target, regardless of loss weighting. Step 3 (raw-price target + volume) and step 4
(raw-price target, no volume) are the more informative next runs -- RevIN's absolute-price
target already fixed this exact collapse once before on this branch (0/2397 -> 95-98%
coherence), so the open question is whether that fix holds with volume back in the loss
(step 3), not whether volume-dropping itself is salvageable under a return target.

**Caveats**: one seed each; no backtest run (the near/below-chance directional accuracy
already rules this out without needing one); no MAE recorded (RMSE only, same as step 1).

## hf_patchtst_revin_raw_price_with_volume

**What**: Step 3 of the ladder -- switches from the anchored log-return decomposition to a
real RevIN (Kim et al. 2021) raw-price target, volume back in. RevIN was already proven on
this branch to fix step 2's exact collapse (0/2397 -> 95-98% coherence), but that earlier
proof dropped volume entirely -- this run adds it back to test whether the fix holds.
**This run applies RevIN uniformly to all 5 channels (OHLC + log-volume)** -- the followup
`hf_patchtst_revin_ohlc_global_volume` run (RevIN for OHLC only, volume kept on
`data_pipeline.py`'s global `log_volume_norm` scale instead) is queued next to test whether
volume's normalization treatment is what's driving the result below.
Notebook: `steven/train_patchtst_hf_channel_attention.ipynb`.

| Model | Windows | OHLC MAE/RMSE ($) | Volume MAE/RMSE | Dir Acc (bar 1/2/3) | Coherence |
|---|---|---|---|---|---|
| `channel_attention=False` | 2397 | 1.88 / 3.19 | 2.23M / 3.98M | 0.470 / 0.467 / 0.467 | 0.878 |
| `channel_attention=True` | 2397 | 2.08 / 3.29 | 2.28M / 4.20M | 0.474 / 0.469 / 0.460 | **0.941** |

**Conclusion: coherence is high (RevIN's fix for step 2's collapse does hold with volume
back in), but directional accuracy is now below chance on all 6 bar/setting combinations
(0.46-0.47, vs. 0.50 for a coin flip) -- worse than every other variant tried on this
branch.** This is a different, more subtle failure than step 2's: the model isn't
collapsing to "predict near-zero" (coherence is 0.88-0.94, i.e. it's confidently picking a
direction), it's confidently picking the **wrong** direction more often than not. OHLC point
error is comparable to the volume-free RevIN run from earlier on this branch (MAE/RMSE
1.81/3.05 and 1.90/3.11 for False/True) -- only slightly worse -- so the price channels
alone aren't badly damaged. The likely culprit is volume's normalization: RevIN's per-window
mean/std, applied to a bursty signal like volume over only 70 samples, can produce noisy
per-window scale estimates that this run doesn't isolate from the price channels (see
`hf_patchtst_revin_ohlc_global_volume`, queued next, which excludes volume from RevIN and
gives it a stable global scale instead).

**Caveats**: one seed each; below-chance directional accuracy makes a backtest not worth
running on this variant as-is; the followup run (global volume normalization) is needed
before concluding whether raw-price + volume is viable at all, or whether this specific
uniform-RevIN treatment is just a bad fit for volume.

## hf_patchtst_revin_ohlc_global_volume

**What**: Follow-up to `hf_patchtst_revin_raw_price_with_volume` above, testing the
hypothesis that RevIN's per-window normalization is a bad fit for volume specifically
(bursty signal, noisy std estimate over only 70 samples). Same raw-price RevIN target, but
volume excluded from RevIN and instead kept on `data_pipeline.py`'s existing global
`log_volume_norm` scale (the same normalization every return-based notebook on this branch
already uses for volume) -- OHLC still RevIN-normalized. Everything else identical
(architecture, loss, hyperparameters, `channel_attention` sweep).
Notebook: `steven/train_patchtst_hf_channel_attention.ipynb`.

| Model | Windows | OHLC MAE/RMSE ($) | Volume MAE/RMSE | Dir Acc (bar 1/2/3) | Coherence |
|---|---|---|---|---|---|
| `channel_attention=False` | 2397 | 1.92 / 3.12 | 2.19M / 3.79M | 0.478 / 0.475 / 0.480 | 0.811 |
| `channel_attention=True` | 2397 | 1.99 / 3.24 | 2.22M / 3.77M | 0.475 / 0.477 / 0.464 | 0.809 |

**Conclusion: the hypothesis was wrong -- volume's normalization scheme is not the driver.**
Directional accuracy is essentially unchanged from the uniform-RevIN run (0.46-0.48 either
way, vs. 0.50 chance) despite volume now getting a stable global scale instead of RevIN's
per-window one. Coherence actually dropped slightly (0.81 here vs. 0.88-0.94 uniform-RevIN)
rather than improving. OHLC MAE/RMSE and volume MAE/RMSE both improved marginally, so the
global-volume-scale change wasn't harmful -- it just isn't what's causing the below-chance
directional accuracy.

**Revised picture, comparing all three raw-price RevIN variants tried on this branch:**

| Variant | Volume | Dir Acc range | 
|---|---|---|
| Original RevIN (this branch's earlier work, before this ladder) | dropped entirely | 0.528-0.552 (above chance) |
| `hf_patchtst_revin_raw_price_with_volume` | RevIN (uniform) | 0.460-0.474 (below chance) |
| `hf_patchtst_revin_ohlc_global_volume` | global log-standardize | 0.464-0.480 (below chance) |

The pattern points at volume's mere **presence** as a joint prediction target, not its
normalization method -- every raw-price RevIN variant with volume included lands
below-chance, while the one without it was solidly above chance. Plausible mechanism: with
a shared fused head projecting from all `N_CHANNELS` pooled representations at once, adding
a 5th (volume) output competes with the 4 price outputs for the same limited head capacity
and gradient budget, degrading price-direction learning specifically -- regardless of how
volume itself is scaled going in.

**Caveats**: one seed each; the "original RevIN, no volume" row is from earlier work on this
branch (different point in the code's history, not literally the same notebook run) --
worth a same-notebook rerun with volume dropped (this ladder's step 4) for a truly
apples-to-apples confirmation before treating this conclusion as settled; below-chance
directional accuracy on both volume variants means no backtest run on either.

## hf_patchtst_revin_no_volume

**What**: Step 4 of the ladder -- the same-notebook, same-infra confirmation the caveat
above called for. Identical architecture/loss/hyperparameters to
`hf_patchtst_revin_ohlc_global_volume`, volume removed entirely (not a context input, not
a training target) -- `N_CHANNELS=4` (OHLC only), single `RevIN` over all four channels,
no split logic needed.
Notebook: `steven/train_patchtst_hf_channel_attention.ipynb`.

| Model | Windows | OHLC MAE/RMSE ($) | Dir Acc (bar 1/2/3) | Coherence |
|---|---|---|---|---|
| `channel_attention=False` | 2397 | 1.92 / 3.11 | 0.525 / 0.543 / 0.543 | 0.953 |
| `channel_attention=True` | 2397 | 2.15 / 3.30 | 0.529 / **0.544** / **0.549** | **0.987** |

**Conclusion: hypothesis confirmed.** Dropping volume restores above-chance directional
accuracy on every bar for both settings (0.525-0.549, vs. 0.46-0.48 with volume included
either way it was normalized) -- and coherence is the highest seen on this branch
(0.95-0.99, vs. 0.81-0.94 with volume, vs. 0/2397 for step 2's return-based collapse). OHLC
MAE/RMSE is essentially unchanged from the volume-included RevIN runs (3.11-3.30 here vs.
3.12-3.29 there) -- so volume's presence wasn't buying any price-accuracy benefit either,
it was purely costing directional accuracy. This closes the loop from
`hf_patchtst_revin_raw_price_with_volume`'s caveat: the mechanism really is volume
competing with price for the shared fused head's capacity/gradient budget, not a
normalization artifact.

**This is the best result on the branch to date, on the metrics that matter for trading**:
best coherence, above-chance directional accuracy on every bar, and OHLC RMSE competitive
with (not meaningfully worse than) every other variant tried. `channel_attention=True`
edges out `False` here on dir acc (bars 2-3) and coherence, at a small cost in OHLC RMSE
(3.30 vs 3.11) -- similar to the split seen in `hf_patchtst_fused_head` (step 1), where
`True` also traded a bit of point-error for slightly better directional/coherence numbers.

**Implication for the ladder**: raw-price RevIN target + volume dropped is the strongest
target/volume combination found so far -- first real candidate for an actual backtest
(deferred on every other variant due to at/below-chance directional accuracy). Step 5
(head comparison) should build on this combination rather than the anchored-return one.

**Caveats**: one seed each; still no backtest run (this notebook family computes forecasting
metrics only, backtest wiring deferred until a combination is picked -- this result is the
strongest case yet for actually doing that wiring); `channel_attention=True` vs `False` here
is a smaller, closer call than the volume question was, not yet a settled pick.
