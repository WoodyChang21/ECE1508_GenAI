# steven4 — CVAE

Conditional VAE that generates SPY candles via image-inpainting-style reconstruction:
block out the next candle and train a model to reconstruct it from what came before.

## Data

- SPY hourly OHLCV, reparametrized into log-return features: `open_ret`, `body_ret`,
  `upper_wick`, `lower_wick`, `log_volume_norm`.
- Optional momentum features (enabled in the current config): EMA9/EMA21 crossover,
  RSI-14, VIX (previous day's close).

## Model

- Conditional VAE — `src/models/cvae_inpainting.py`.
- Training config — `configs/cvae_h1.yaml`.
- Predicts one candle at a time (`HORIZON=1`).

## Outputs

- `outputs/cvae_checkpoint_h1.pt` — trained checkpoint.
- `outputs/generative_metrics_h1.json` + `outputs/generative_plots_h1/` — diversity/
  calibration diagnostics for the checkpoint.

## How to train

Open `colab_train_h1.ipynb` on a Colab GPU runtime and run the cells top to bottom.

## File tree (CVAE-relevant, under `steven/`)

```
steven/
├── colab_train_h1.ipynb
├── configs/
│   └── cvae_h1.yaml
├── data/
│   └── spy_ohlcv_1h.parquet
├── outputs/
│   ├── cvae_checkpoint_h1.pt
│   ├── generative_metrics_h1.json
│   └── generative_plots_h1/
└── src/
    ├── data_pipeline.py
    ├── momentum_pipeline.py
    ├── train_cvae.py
    └── models/
        └── cvae_inpainting.py
```
