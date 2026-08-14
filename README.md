# ECE1508 GenAI — PatchTST + RevIN (final branch)

This branch holds the final PatchTST model for short-horizon (1-3 hour) SPY forecasting: a HuggingFace `PatchTSTModel` backbone combined with Reversible Instance Normalization (RevIN), predicting the next three hourly OHLC bars directly from raw price (no volume, no log-return reparametrization). See `steven/experiments.md` for the full ablation log and `steven/colab_train.ipynb` for the executed training/evaluation run this branch's results are drawn from.

## Result

Best checkpoint: patch length 14 (no overlap), `channel_attention=False`, `d_model=64`, 3 layers, 4 heads (253,076 trainable parameters).

Backtested two ways on the same checkpoint:

| Strategy | Trades | Win Rate | Total Return | Annualized |
|---|---|---|---|---|
| Take-profit order | 626 | 72.7% | +14.47% | +10.39% |
| **Hysteresis entry/exit (enter=4.5bps/exit=-1.0bps, 1bp cost each way)** | **158** | **57.6%** | **+27.27%** | **+19.30%** |
| Buy-and-hold (benchmark) | -- | -- | +24.11% | +17.09% |

The hysteresis strategy — re-forecast every hour, enter long once the predicted next-bar return clears the entry threshold, exit once it drops below the exit threshold, no fixed holding horizon — is the only configuration on this branch that beats buy-and-hold outright, and is this branch's headline result.

## Key files

- `steven/colab_train.ipynb` — executed notebook with the final checkpoint's training, forecasting metrics, and both backtests above.
- `steven/experiments.md` — full experiment log (loss-term/schedule/patch-geometry ablations, what was tried and rejected, and why).
- `steven/src/` — model (`models/patchtst_hf.py`), data pipeline, and evaluation scripts (`evaluate_revin.py`, `evaluate_revin_hysteresis.py`).
- `steven/outputs/` — saved checkpoints and per-run backtest metrics JSON.
