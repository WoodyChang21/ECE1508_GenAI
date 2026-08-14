# ECE1508 GenAI — Intraday SPY Forecasting

This project evaluates three deep learning architectures for short-horizon (1-3 hour) SPY forecasting on hourly OHLCV data: a patch-based transformer (PatchTST + RevIN), a conditional variational autoencoder (CVAE), and a state-space sequence model (Mamba). Each model was developed independently with its own data representation, training pipeline, and backtest, so `main` holds no model code — each branch below is self-contained; check out the branch for the model you want to run.

## Branches

| Branch | Contents |
|---|---|
| `patchtst_ohlcv_mse` | **Final PatchTST model.** RevIN + HuggingFace `PatchTSTModel` backbone predicting raw OHLC directly. Includes the full training-pipeline ablation sweep (loss terms, LR schedule, patch geometry), the take-profit-order walk-forward backtest, and the hysteresis entry/exit backtest (the model's best result). See `steven/experiments.md` and `steven/colab_train.ipynb`. |
| `mamba-model` | **Final Mamba model.** Attention-free selective state-space model predicting SPY returns from 20+ engineered per-bar features (returns, volatility, RSI, MACD, Bollinger, VIX). See `scripts/models/`. |
| `steven4` (remote-only — `git checkout -t origin/steven4`) | **Final CVAE model.** Conditional VAE framing forecasting as candle inpainting, with a rolling hour-by-hour backtest. See `steven/colab_train.ipynb` and `steven/rolling_hour_backtest.md`. |
| `Data` | Shared data collection and preprocessing pipeline (FMP ingestion, feature engineering, train/val/test splitting) that the model branches were originally forked from. |

Older branches (`Model`, `PatchTST_OLCV`, `steven`, `steven2`, `steven3`) are earlier iterations superseded by the branches above and are kept only for history.
