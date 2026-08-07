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
