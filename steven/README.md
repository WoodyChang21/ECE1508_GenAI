# steven4

CVAE-based candle generation for SPY, framed as image inpainting: block out the next
candle and train a model to reconstruct it from what came before. This branch is a
pivot from `steven3` — see below for what changed.

## What's different from steven3

- **Predicts one candle at a time (`HORIZON=1`), not three.** The model, loss, and data
  pipeline all changed shape accordingly.
- **New backtest**: instead of a fixed bracket order (buy, take-profit/stop-loss, hold
  exactly 3 bars), the strategy now re-predicts every real hour and holds open-endedly —
  buy on an "uptrend" call, sell on a real-price loss, a "downtrend" call, or a hard 1%
  stop-loss.
- Old `steven3` checkpoints (`cvae_checkpoint.pt`, etc.) **no longer load** on this
  branch — the model's shapes changed. Use `steven3` if you need those.

## Key files

- `src/models/cvae_inpainting.py` — the CVAE architecture.
- `src/rolling_backtest.py` — the new backtest's decision logic.
- `configs/cvae_h1.yaml` — training config for the current model.
- `colab_train_h1.ipynb` — train on Colab, then run the backtest, then sync results back
  to GitHub. This is the notebook to run.
- `rolling_hour_backtest.md` — the full writeup: exact backtest mechanics, what's been
  verified vs. not, and a caveat worth reading before trusting any result (the trade
  signal re-tests something already found harmful in an earlier version of this project).

## Quick start

Open `colab_train_h1.ipynb` on a Colab GPU runtime and run the cells top to bottom —
clones this branch, trains the CVAE, runs the backtest, prints the headline numbers.

## Tests

`pytest steven/tests` — should be green before trusting anything above.
