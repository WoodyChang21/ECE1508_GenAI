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

## hf_patchtst_revin_ohlc_global_volume_dirloss

**What**: Tests whether an explicit directional loss term can correct
`hf_patchtst_revin_ohlc_global_volume`'s below-chance directional accuracy without having
to drop volume. Identical code/architecture/data to that run (checked out from its exact
commit, `f1ad007`), with one addition: `loss = mse + DIR_LOSS_WEIGHT * dir_loss`, where
`dir_loss = -log(sigmoid(DIR_LOSS_SCALE * pred_ret * true_ret))` and `pred_ret`/`true_ret`
are the predicted/true close-price change relative to `close_0` -- the same anchor
`directional_accuracy` is measured against. `DIR_LOSS_WEIGHT=0.1`, `DIR_LOSS_SCALE=1.0`
(untuned defaults, not swept).
Notebook: `steven/train_patchtst_hf_channel_attention.ipynb`.

| Model | Windows | OHLC MAE/RMSE ($) | Dir Acc (bar 1/2/3) | Coherence |
|---|---|---|---|---|
| `channel_attention=False` | 2397 | 1.87 / 3.10 | 0.482 / 0.479 / 0.470 | 0.720 |
| `channel_attention=True` | 2397 | 1.97 / 3.25 | 0.467 / 0.475 / 0.468 | 0.812 |

**Conclusion: essentially a null result at these hyperparameter values -- do not read this
as "directional loss doesn't work," read it as "this weight/scale didn't move the needle."**
Directional accuracy is unchanged within noise from the no-directional-loss run (0.467-0.482
here vs. 0.464-0.480 there, both still solidly below chance). OHLC RMSE is essentially
identical too (3.10-3.25 vs. 3.12-3.24). The one real change is `channel_attention=False`'s
coherence, which *dropped* (0.720 vs. 0.811) -- the directional term perturbed training
somewhere without fixing the thing it was meant to fix.

**Most likely explanation: `DIR_LOSS_WEIGHT=0.1` is too small to meaningfully compete with
the MSE term**, which given this branch's own numbers (`vol_loss` alone was ~0.18-0.27 in
step 1's unrelated but comparably-scaled MSE runs) is plausibly still the dominant term in
the combined loss at a 10x-smaller weight. This isn't evidence the mechanism is wrong, just
that this first, untuned pass didn't apply enough pressure to change what the model
optimizes toward.

**Recommended next step, not yet done**: sweep `DIR_LOSS_WEIGHT` upward (e.g. 0.5, 1.0, 3.0)
before concluding whether a directional loss can work here -- this run only rules out 0.1,
not the approach itself. If a higher weight still doesn't move directional accuracy while
visibly degrading OHLC RMSE, that would be stronger evidence the volume-competition
mechanism isn't fixable by reweighting the loss and `hf_patchtst_revin_no_volume` (no
directional loss needed) remains the better path forward.

**Caveats**: one seed each; `DIR_LOSS_SCALE=1.0` also untuned and not isolated from
`DIR_LOSS_WEIGHT` in this single run -- a 2D sweep (not just weight) may be needed; no
backtest run (still below-chance direction).

## hf_patchtst_revin_ohlc_global_volume_dirloss -- weight sweep

**What**: The recommended follow-up above -- `DIR_LOSS_WEIGHT` swept over `[0.5, 1.0, 3.0]`
(30x the original 0.1), crossed with both `channel_attention` settings, 6 runs total.
`DIR_LOSS_SCALE` held fixed at `1.0`. Same code/architecture otherwise.
Notebook: `steven/train_patchtst_hf_channel_attention.ipynb`.

| Weight | `channel_attention` | OHLC RMSE ($) | Dir Acc (bar 1/2/3) | Coherence |
|---|---|---|---|---|
| 0.5 | False | 3.21 | 0.472 / 0.468 / 0.473 | 0.809 |
| 0.5 | True | 4.01 | 0.464 / 0.461 / 0.451 | **0.951** |
| 1.0 | False | 3.38 | 0.466 / 0.458 / 0.457 | 0.867 |
| 1.0 | True | 4.05 | 0.462 / 0.461 / 0.454 | 0.950 |
| 3.0 | False | 3.34 | 0.469 / 0.470 / 0.456 | 0.881 |
| 3.0 | True | 3.71 | 0.464 / 0.457 / 0.462 | 0.882 |

**Conclusion: decisive negative result -- the directional-loss mechanism, as implemented,
does not work here at any weight tried.** Directional accuracy never crosses back above
0.50 across all 6 runs (0.451-0.473, no upward trend as weight increases 30x) -- this rules
out "weight=0.1 was just too small," not merely fails to confirm it. Worse, OHLC RMSE gets
meaningfully *worse* at every swept weight compared to the no-directional-loss baseline
(3.12-3.24): up to 3.21-4.05 depending on weight/setting, non-monotonically. So increasing
the directional term's weight bought nothing on direction while actively costing point
accuracy -- a strictly worse tradeoff at higher weights, not a partial win.

**Most striking finding: `channel_attention=True` at weight 0.5-1.0 is simultaneously the
*most* coherent result on this branch (0.95, tied with `hf_patchtst_revin_no_volume`'s 0.99)
and confidently wrong (dir acc 0.45-0.46, the lowest on the branch)** -- the directional
loss term, instead of correcting the wrong-sign bias, appears to have made the model even
more consistently committed to it. This is a clean illustration of exactly the "confidently
wrong" failure mode discussed earlier: coherence measures self-consistency, not
correctness, and this run pushed self-consistency up while correctness stayed down.

**Implication: this specific mechanism (soft sign-agreement loss added to RevIN MSE, on
the volume-included variant) is not the fix.** Two paths remain, in priority order: (1)
`DIR_LOSS_SCALE` sweep, per the earlier caveat -- untested, and a poorly-scaled return
product sitting in the sigmoid's flat tails would produce exactly this kind of
weight-insensitive null result regardless of `DIR_LOSS_WEIGHT`; but (2) given
`hf_patchtst_revin_no_volume` already achieves above-chance direction with zero loss
engineering, continuing to chase the volume-included variant has a shrinking case for
priority -- the practical recommendation is to move on to backtesting step 4 rather than
running a further scale sweep on this path, unless there's a specific reason volume must
stay in the model.

**Caveats**: one seed per combination (not per-weight noise-checked); `DIR_LOSS_SCALE`
still fixed/untested; no backtest run (still below-chance direction on every combination).

## hf_patchtst_revin_no_volume_dirloss -- weight sweep (step 4 + directional loss)

**What**: Closes the gap the caveat above left open -- directional loss had only ever been
tried on step 3b, a volume-included variant that was *already* below-chance before the loss
term was added, so that result only showed the mechanism doesn't rescue a broken model, not
that it can't help a working one. This run ports the same `directional_loss()` mechanism
(`loss = mse + DIR_LOSS_WEIGHT * dir_loss`, `dir_loss = -log(sigmoid(DIR_LOSS_SCALE *
pred_ret * true_ret))`) onto `hf_patchtst_revin_no_volume` (step 4) itself -- the model
that's actually being walk-forward backtested, and the best forecasting result on this
branch (dir acc 0.52-0.55, coherence 0.97-0.99). `DIR_LOSS_WEIGHT` swept over
`[0.1, 0.5, 1.0, 3.0]`, crossed with both `channel_attention` settings, `DIR_LOSS_SCALE`
held fixed at `1.0`. 8 runs total, identical architecture/data/RevIN otherwise.
Notebook: `steven/train_patchtst_hf_channel_attention.ipynb`.

| Weight | `channel_attention` | OHLC RMSE ($) | Dir Acc (bar 1/2/3) | Coherence |
|---|---|---|---|---|
| -- (baseline) | False | 3.1484 | 0.5240 / 0.5403 / 0.5524 | 0.9741 |
| -- (baseline) | True | 3.2878 | 0.5344 / 0.5440 / 0.5519 | 0.9875 |
| 0.1 | False | 3.1253 | 0.5236 / 0.5382 / 0.5499 | 0.9412 |
| 0.1 | True | 3.4010 | 0.5265 / 0.5419 / 0.5524 | 0.9808 |
| 0.5 | False | 3.1253 | 0.5065 / 0.5219 / 0.5273 | 0.7939 |
| 0.5 | True | 3.2380 | 0.5198 / 0.5382 / 0.5490 | 0.9712 |
| 1.0 | False | 3.1285 | 0.5140 / 0.5186 / 0.5365 | 0.8798 |
| 1.0 | True | 3.0628 | 0.5173 / 0.5223 / 0.5336 | 0.7972 |
| 3.0 | False | 3.1027 | **0.4827 / 0.4764 / 0.4902** | 0.7764 |
| 3.0 | True | 3.7407 | 0.5035 / 0.4827 / 0.4856 | 0.9145 |

**Conclusion: decisive negative result, and a stronger one than step 3b's.** Directional
loss doesn't just fail to help step 4 -- it actively erodes the two properties that made
step 4 worth backtesting in the first place. Directional accuracy never improves over the
no-dirloss baseline at any weight; it degrades as weight increases, and at `weight=3.0`
(`channel_attention=False`) drops to **0.4827-0.4902 -- below chance**, undoing exactly the
above-chance signal step 4 was built to demonstrate. Coherence takes the bigger, more
consistent hit: every single weight/setting combination scores below its matching
baseline, sometimes drastically (0.9741 -> 0.7939 at weight=0.5/False; 0.9875 -> 0.7972 at
weight=1.0/True) -- there is no weight where coherence stays intact. OHLC RMSE is a mixed
bag (flat-to-slightly-better for `False`, worse for `True` at weight 0.1 and 3.0,
peaking at 3.74) -- not the main story either way.

**This is a stronger negative result than `hf_patchtst_revin_ohlc_global_volume_dirloss`'s
weight sweep**: there, directional accuracy stayed flat/null (0.451-0.473, no trend) --
the mechanism simply didn't move the needle on an already-broken model. Here, applied to a
model that already had real, above-chance signal, the same mechanism actively destroys it
as weight increases. Taken together, directional loss (as implemented -- a soft
sign-agreement term added to RevIN MSE) has now been tested on both a broken and a working
base model and failed decisively both times. The lowest weight tried (0.1) comes closest
to neutral (dir acc within noise of baseline, RMSE roughly flat) but still costs 3-6 points
of coherence for no measurable directional benefit -- there is no weight in this grid worth
trading for.

**Implication for the training-pipeline review**: this closes out the loss-engineering
lever entirely on this branch. The recommendation going forward is to leave
`hf_patchtst_revin_no_volume`'s existing (no-dirloss) checkpoints as the one being
backtested, and move to the other pipeline levers identified in the training-pipeline
review instead -- an LR schedule and/or overlapping patches (`patch_length > patch_stride`)
-- rather than any further directional-loss weight/scale tuning.

**Caveats**: one seed per combination, same as every other run on this branch;
`DIR_LOSS_SCALE=1.0` still fixed/untested (though given directional accuracy actively
*worsens* with weight here, a scale sweep looks unlikely to rescue this mechanism the way
it might have for step 3b's flat-null result); no backtest run on any of these 8
checkpoints (dropping in-sample directional accuracy on 7/8 combinations, and below-chance
on 1/8, makes a backtest not worth running here -- the existing step 4 checkpoints remain
the ones to backtest).

## hf_patchtst_revin_no_volume_lrschedule (step 4 + cosine LR schedule)

**What**: Tests the optimization-side lever from the training-pipeline review, on the same
model directional loss was already ruled out on. Step 4's baseline training curve was
inspected directly (not just its final metrics): train loss was still dropping ~4-5% per
epoch through epoch 20 with no flattening, and best val loss landed at the final epoch
(`False`) or second-to-last epoch (`True`) rather than plateauing earlier -- evidence the
flat-`LR=6e-4`-for-20-epochs baseline hadn't converged. This run adds
`torch.optim.lr_scheduler.CosineAnnealingLR(T_max=20, eta_min=6e-6)`, stepped once per
epoch, no warmup -- otherwise identical architecture/data/loss (plain RevIN MSE, no
directional term) to step 4. Notebook: `steven/train_patchtst_hf_channel_attention.ipynb`.

| Setting | OHLC RMSE ($) | Dir Acc (bar 1/2/3) | Coherence | Best val loss |
|---|---|---|---|---|
| baseline (flat LR), False | 3.1484 | 0.5240 / 0.5403 / 0.5524 | 0.9741 | 0.1331 |
| baseline (flat LR), True | 3.2878 | 0.5344 / 0.5440 / 0.5519 | 0.9875 | 0.1316 |
| cosine LR, False | **3.0547** | 0.5006 / 0.5252 / 0.5219 | 0.9107 | 0.1756 |
| cosine LR, True | **3.0033** | **0.4948** / 0.5027 / 0.5048 | 0.8882 | 0.1731 |

**Conclusion: RMSE improves, but at a real cost to the two metrics that actually matter for
trading -- another negative result for the goal of this branch.** OHLC RMSE drops
meaningfully for both settings (3.15 -> 3.05 for `False`, 3.29 -> 3.00 for `True`, the best
RMSE seen on this branch), confirming the schedule does let the optimizer fit the price
level more precisely, as hypothesized from the unconverged training curve. But directional
accuracy falls on every bar for both settings, and `channel_attention=True`'s bar-1
accuracy drops to **0.4948 -- below chance**, something step 4's baseline never showed on
any bar. Coherence takes the bigger hit: 0.9741 -> 0.9107 (`False`) and 0.9875 -> 0.8882
(`True`), the largest coherence drop seen from a "successful" (RMSE-improving) change on
this branch. Best val loss also rose for both settings (0.133 -> 0.176, 0.132 -> 0.173)
despite the RMSE improvement -- val loss and test-set RMSE are measured on different splits
(val vs. test) so this isn't strictly contradictory, but it does mean the schedule's
selected checkpoint fits its own validation set less tightly than the flat-LR baseline did.

**This is the same RMSE/directional-accuracy divergence the whole branch has already
established, now demonstrated from the opposite direction**: earlier experiments showed
RMSE improvements don't reliably track directional accuracy; this one shows a change that
tightens RMSE can actively *degrade* directional accuracy and coherence. A schedule that
lets training converge more precisely on point-level price fit appears to push the model
further toward a lower-error-but-less-committed prediction (right in line with the
"MSE-optimal forecast on a near-random-walk series looks like persistence" theory from the
training-pipeline review) -- exactly the failure mode the coherence/direction metrics were
designed to catch.

**Implication**: extending training precision on the current RevIN-MSE objective is not
the fix -- it trades away the one thing (coherence + above-chance direction) that made step
4 worth backtesting in the first place, for a metric (RMSE) that's never once tracked
trading-relevant quality on this branch. Combined with `hf_patchtst_revin_no_volume_dirloss`,
both the loss-engineering and (this) optimization-schedule levers have now failed on step
4's exact model. The unconverged-training-curve finding remains worth acting on, but via a
lever that doesn't also over-fit price level -- e.g. more epochs *without* a decay-to-zero
schedule (so the model gets more time but isn't pushed as hard toward the RMSE-minimizing
regime), which is the next experiment queued.

**Caveats**: one seed per setting; `LR_ETA_MIN_FRAC=0.01` and no-warmup are both untuned --
a shallower decay floor might recover some of the coherence loss while keeping part of the
RMSE gain, untested; no backtest run (directional accuracy dropped, in one case below
chance, so not worth backtesting over the existing step 4 checkpoints).
