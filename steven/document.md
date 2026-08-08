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
