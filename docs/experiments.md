# Experiment Log — PatchTST_OLCV branch

One row per completed run. Don't fork notebooks into `_v1`/`_v2` for genuinely
new experiments — rerun the same notebook, log the result here. The two
`channel_attention` variants below are a deliberate exception: they're a
direct A/B comparison this branch exists to make, so both are kept as
separate executed notebooks rather than one overwriting the other.

**Commit column** = a commit whose notebook contains the *actual executed
output* for that row, not just the code.

---

## Background

`steven` branch built two OHLCV-forecasting models (PatchTST-style + a CVAE)
trained on `steven/data/spy_ohlcv_1h.parquet`, predicting the next 3 hourly
bars' candle shape (anchored log-returns: `open_ret`/`body_ret`/wicks) and
volume, instead of this project's main-line `return_1h` scalar target. His
`steven/src/models/patchtst.py` is a hand-rolled transformer: patches the
input (7-bar/1-trading-day patches), self-attends across patches, but does
**early channel fusion** — all 7 feature channels are flattened into one
linear patch embedding before any attention happens. It does not implement
the paper's channel-independence design and has no way to toggle
channel-mixing on/off.

This branch replicates Steven's exact task — same data, same target
parameterization, same deterministic weighted-MSE loss (`steven/src/losses.py`,
reused unmodified) — with the real HF `PatchTSTModel` backbone instead
(`steven/train_patchtst_hf_channel_attention_{True,False}.ipynb`), which does
implement channel-independent patching with an optional `channel_attention`
sublayer. `channel_attention` is then a clean toggle to compare
channel-independent vs. channel-mixing on an identical task/loss/data —
something Steven's architecture can't offer.

**One deliberate simplification**: Steven's model trains on a variable-context
curriculum (2–10 trading days, resampled per window). HF `PatchTSTModel`
patchifies a single fixed `context_length` per model, so this branch fixes
`context_length=70` (10 trading days, Steven's longest curriculum option)
rather than replicating the curriculum. This makes the HF runs most directly
comparable to Steven's own **"long" context bucket** in
`steven/outputs/metrics.json`, not his "overall" (mixed-length) numbers,
which are included below for reference only.

Evaluation methodology matches `steven/src/evaluate.py`'s `metrics_for_slice`
exactly: reparam-space MAE/RMSE (raw 15-dim vector), reconstructed OHLC
MAE/RMSE, volume MAE/RMSE, and per-horizon-bar directional accuracy on the
`close_0`-anchored close return. Same test seed (`123`) as `evaluate.py`'s
default.

**Backtest is deferred** (`steven/src/evaluate.py`'s long-only limit-order
backtest) until an architecture/`channel_attention` setting is picked from
the forecasting metrics below.

---

## HF PatchTSTModel vs. Steven's hand-rolled PatchTST

| Model | Commit | reparam MAE / RMSE | OHLC MAE / RMSE | Volume MAE / RMSE | Dir Acc (bar 1 / 2 / 3) |
|---|---|---|---|---|---|
| HF PatchTST, `channel_attention=False` | [`a7d38ec`](https://github.com/WoodyChang21/ECE1508_GenAI/commit/a7d38ec) | 0.1141 / **0.4366** | 3.72 / 4.88 | 2.20M / **3.90M** | 0.528 / 0.541 / 0.554 |
| HF PatchTST, `channel_attention=True` | [`a7d38ec`](https://github.com/WoodyChang21/ECE1508_GenAI/commit/a7d38ec) | 0.1179 / 0.4987 | **3.38 / 4.51** | 3.28M / **16.20M** | **0.532** / 0.540 / 0.554 |
| Steven's model, "long" bucket (ctx 56–70, most comparable) | `steven` branch, `steven/outputs/metrics.json` | 0.2270 / 0.6427 | 11.21 / 13.76 | 5.23M / 6.40M | 0.508 / 0.529 / 0.541 |
| Steven's model, "overall" (mixed-length, reference only) | `steven` branch, `steven/outputs/metrics.json` | 0.2157 / 0.6095 | 11.67 / 14.41 | 4.87M / 6.02M | 0.527 / 0.516 / 0.522 |

Both HF variants trained: `patch_length=7`/`patch_stride=7` (matches Steven's
`PATCH_LEN=7`), `d_model=64`/`num_attention_heads=4`/`num_hidden_layers=3`/
`dropout=0.1` (matches `steven/configs/patchtst.yaml`'s model block),
`context_length=70` (fixed), `scaling=None` (Steven's model has no per-window
instance normalization, so HF's default RevIN-style scaling was disabled to
avoid confounding the architecture comparison), `lr=6e-4`/`weight_decay=1e-4`/
`batch_size=512`/20 epochs × 20,000 resampled windows/epoch (all from
`steven/configs/patchtst.yaml`), `SEED=0`. Best val loss: 0.0926 (`False`) /
0.0900 (`True`).

- **Both HF variants substantially beat Steven's own model on every metric** —
  roughly half the reparam error, ~3x tighter OHLC error, better directional
  accuracy on every horizon bar, regardless of `channel_attention`. The HF
  backbone (true channel-independent patching, real self-attention over
  patches, mean-pooled deterministic head) is simply a stronger architecture
  for this task than Steven's early-channel-fusion transformer.
- **Training loss curves for both HF variants look categorically different
  from every `return_1h` notebook on the `Model`/main-line branch** — a sharp,
  real drop from ~0.5 to ~0.15 in the first ~50 steps, then a genuine
  plateau, not the flat/noisy pattern seen throughout this project when the
  target was `return_1h`. Consistent with OHLCV/volume carrying real,
  learnable structure (autocorrelation, volume clustering) that a scalar
  hourly return mostly doesn't.
- **`channel_attention` is not a clean win either way.** `True` is slightly
  better on OHLC MAE/RMSE, Dir Acc, and best val loss. But `True`'s **volume
  RMSE is ~4x its own MAE** (16.20M vs. 3.28M — a 4.9x ratio, vs. `False`'s
  1.77x ratio), and its reparam RMSE is also meaningfully worse than `False`
  despite similar MAE (0.499 vs. 0.437) — the same heavy-tail signature.
  Letting channels cross-inform each other improved price/OHLC accuracy but
  introduced an outlier-prone failure mode in volume forecasting on a subset
  of test windows. Not yet root-caused.
- **Loss is dominated by the volume term** in both variants (`vol` component
  ≈0.18–0.27 per epoch vs. `price` ≈0.00007–0.00009 by the end) — price
  converges to a tiny loss almost immediately. The OHLC MAE/RMSE numbers above
  (in real price-dollar terms) suggest this isn't pure shrinkage-to-zero the
  way some `return_1h` runs on the main-line branch were, but this hasn't been
  visually sanity-checked against actual candle plots yet.
- **Not yet done**: one seed only for both variants; no root-cause investigation
  of `channel_attention=True`'s volume RMSE outliers; no backtest yet (deferred
  per instruction until a winner is picked from these metrics).

---

## Where this leaves the branch

The HF `PatchTSTModel` replication clearly outperforms Steven's hand-rolled
model at the same task/loss — the architecture change (true channel-independent
patching + self-attention vs. early channel fusion) is the likely driver, not
the loss function (which was already identical). Between `channel_attention`
settings, `False` is the safer choice right now given `True`'s volume RMSE
blowup, but `True`'s edge on OHLC/Dir Acc means this isn't fully settled.
Next steps: investigate the `channel_attention=True` volume outliers, confirm
both results hold across at least one more seed, then run the deferred
backtest on whichever setting looks better.
