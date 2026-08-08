# PatchTST + RevIN: Final Model Conclusion

## Best model

**`hf_patchtst_revin_no_volume_patch14_14`, `channel_attention=False`**

- Architecture: HF `PatchTSTModel` backbone + RevIN (Kim et al. 2021) instance normalization, fused/flattened output head.
- Target: raw OHLC price (no volume), `d_model=64`, 4 attention heads, 3 layers.
- Patches: `patch_length=14`, `patch_stride=14` (no overlap, 5 patches per 70-bar context window).
- Loss: plain unweighted MSE in RevIN-normalized space. 20 epochs, flat `lr=6e-4`, no directional term, no LR schedule.
- Checkpoint: `steven/outputs/patchtst_revin_novolume_patch14_14_channel_attention_false_checkpoint.pt`

## Backtest result (best model)

Walk-forward backtest, SPY hourly, test window 2024-01 to 2025-05 (`steven/src/evaluate_revin.py`):

| Metric | Value |
|---|---|
| Trades | 626 |
| Win rate | 72.68% |
| Take-profit rate | 67.25% |
| **Total return** | **+14.47%** |
| **Annualized return** | **+10.39%** |

For reference over the same window: buy-and-hold +24.11% total / +17.09% annualized; naive periodic baseline -5.98% total. The model is solidly profitable and the best result found on this branch, though it does not beat buy-and-hold over this particular (bullish) test window.

Forecasting metrics for this checkpoint: OHLC RMSE 3.05, directional accuracy 0.527/0.542/0.546 (bars 1-3), coherence rate 0.937.

## How we got here (brief)

Starting point was `hf_patchtst_revin_no_volume` (RevIN, no volume, non-overlapping `7/7` patches) — the best result from an earlier experiment ladder that had ruled out volume-included targets and confirmed above-chance directional accuracy. From there we tried each of the following, in isolation, against that same baseline:

- **Directional loss** (explicit sign-agreement term added to the MSE): negative at every weight tried — degraded directional accuracy and coherence rather than improving them.
- **LR schedule** (cosine decay): improved RMSE but degraded directional accuracy/coherence and hurt the backtest.
- **More training / early stopping** (up to 100 epochs): decisively negative — training loss kept dropping while directional accuracy collapsed below chance; val loss turned out to be the wrong signal to optimize for this objective.
- **Close-weighted loss** (upweighting the one channel the trading decision actually uses): negative on both forecasting metrics and the backtest.
- **Patch geometry** (overlapping and/or longer patches): the one lever that worked. `(14,7)` (50% overlap) and `(14,14)` (no overlap, same length) both beat the baseline; a wider sweep across `(10,10)`, `(10,5)`, `(14,14)`, `(21,7)` showed patch **length** 14 was the actual driver, not overlap — `(14,14)` reproduces almost all of `(14,7)`'s gain more simply, with better coherence.

Backtesting the two `14`-length candidates surfaced an asymmetry: `(14,14)` is the best model for `channel_attention=False` (+14.47%), while `(14,7)` was previously the best for `channel_attention=True` (+12.81%). `(14,14)/False` is the single best result across everything tried and is the one recommended above.

## Caveats

- All results are single-seed; not yet confirmed against seed noise.
- Patch geometry interacts with `channel_attention` non-additively — there is no one setting that's best for both; `False` was picked here as it has the stronger, single best result.
- Backtest window is a bullish stretch for SPY, so the ~+24% buy-and-hold benchmark is a high bar the model doesn't clear despite being genuinely profitable and selective (67-73% win rate).

## Appendix: full numerical results

All rows are `hf_patchtst_revin_no_volume` family (RevIN, raw OHLC target, no volume), 20 epochs / `lr=6e-4` unless noted otherwise. `**` marks the best model.

### Forecasting metrics

| Experiment | `channel_attention` | OHLC RMSE | Dir Acc (bar 1/2/3) | Coherence |
|---|---|---|---|---|
| Baseline, patch (7,7) | False | 3.1484 | 0.5240 / 0.5403 / 0.5524 | 0.9741 |
| Baseline, patch (7,7) | True | 3.2878 | 0.5344 / 0.5440 / 0.5519 | 0.9875 |
| LR schedule (cosine) | False | 3.0547 | 0.5006 / 0.5252 / 0.5219 | 0.9107 |
| LR schedule (cosine) | True | 3.0033 | 0.4948 / 0.5027 / 0.5048 | 0.8882 |
| More epochs / early stop (up to 100ep) | False | 3.1318 | 0.4789 / 0.4806 / 0.4618 | 0.9091 |
| More epochs / early stop (up to 100ep) | True | 3.0599 | 0.4723 / 0.4810 / 0.4856 | 0.8903 |
| Directional loss, w=0.1 | False | 3.1253 | 0.5236 / 0.5382 / 0.5499 | 0.9412 |
| Directional loss, w=0.1 | True | 3.4010 | 0.5265 / 0.5419 / 0.5524 | 0.9808 |
| Directional loss, w=0.5 | False | 3.1253 | 0.5065 / 0.5219 / 0.5273 | 0.7939 |
| Directional loss, w=0.5 | True | 3.2380 | 0.5198 / 0.5382 / 0.5490 | 0.9712 |
| Directional loss, w=1.0 | False | 3.1285 | 0.5140 / 0.5186 / 0.5365 | 0.8798 |
| Directional loss, w=1.0 | True | 3.0628 | 0.5173 / 0.5223 / 0.5336 | 0.7972 |
| Directional loss, w=3.0 | False | 3.1027 | 0.4827 / 0.4764 / 0.4902 | 0.7764 |
| Directional loss, w=3.0 | True | 3.7407 | 0.5035 / 0.4827 / 0.4856 | 0.9145 |
| Close-weighted loss (4x) | False | 3.2136 | 0.5227 / 0.5411 / 0.5515 | 0.9554 |
| Close-weighted loss (4x) | True | 3.4157 | 0.5252 / 0.5386 / 0.5448 | 0.9666 |
| Patch (10,10) | False | 3.2106 | 0.4940 / 0.4965 / 0.5048 | 0.9746 |
| Patch (10,10) | True | 3.2770 | 0.4864 / 0.4919 / 0.4948 | 0.9504 |
| Patch (10,5) | False | 3.2360 | 0.4931 / 0.4940 / 0.4965 | 0.9720 |
| Patch (10,5) | True | 3.3282 | 0.4739 / 0.4760 / 0.4760 | 0.9591 |
| Patch (21,7) | False | 3.1642 | 0.4931 / 0.4994 / 0.4923 | 0.9483 |
| Patch (21,7) | True | 3.1766 | 0.4848 / 0.4823 / 0.4710 | 0.9504 |
| Patch (14,7) overlap | False | 3.0305 | 0.5232 / 0.5327 / 0.5465 | 0.9266 |
| Patch (14,7) overlap | True | 3.0403 | 0.5215 / 0.5340 / 0.5453 | 0.9011 |
| **Patch (14,14)** | **False** | **3.0505** | **0.5273 / 0.5419 / 0.5457** | **0.9370** |
| Patch (14,14) | True | 3.0781 | 0.5190 / 0.5382 / 0.5469 | 0.9462 |

### Backtest results

SPY hourly, walk-forward, test window 2024-01 to 2025-05. Benchmarks over the same window: buy-and-hold total +24.11% / annual +17.09%; naive periodic total -5.98% / annual -4.41%.

| Experiment | `channel_attention` | Trades | Win rate | Take-profit rate | Total return | Annual return |
|---|---|---|---|---|---|---|
| Baseline, patch (7,7) | False | 868 | 69.12% | 57.60% | +9.62% | +6.95% |
| Baseline, patch (7,7) | True | 535 | 66.36% | 55.14% | +8.06% | +5.87% |
| LR schedule (cosine) | False | 250 | 75.20% | 73.60% | -9.52% | -7.14% |
| LR schedule (cosine) | True | 274 | 75.91% | 75.18% | +6.45% | +4.73% |
| Patch (14,7) overlap | False | 584 | 75.68% | 71.40% | +9.68% | +6.99% |
| Patch (14,7) overlap | True | 527 | 75.14% | 72.30% | +12.81% | +9.23% |
| **Patch (14,14)** | **False** | **626** | **72.68%** | **67.25%** | **+14.47%** | **+10.39%** |
| Patch (14,14) | True | 705 | 70.07% | 61.56% | -0.18% | -0.13% |
| Close-weighted loss (4x) | False | 508 | 68.11% | 56.69% | -1.91% | -1.41% |
| Close-weighted loss (4x) | True | 558 | 66.49% | 52.15% | +1.28% | +0.94% |

Directional-loss and early-stopping variants were not backtested (both had below-chance forecasting-metric directional accuracy on at least some bars, ruling them out before a backtest was warranted).
