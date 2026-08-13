# steven4: HORIZON=1 + rolling hour-by-hour backtest

Formalizes the steven4 pivot: predict one candle at a time instead of three, and evaluate
with an open-ended rolling hold instead of a fixed-width bracket order. This doc covers
what changed, how to retrain, how to run the new backtest, and what's still open.

## Why

The pre-pivot strategy (`steven`/`steven3`) generated 3 candles jointly and placed a fixed
bracket order (buy, take-profit + stop-loss, walk exactly 3 real bars). The new strategy
instead: predicts one candle ahead, re-predicts every real hour as new data arrives, buys
on an "uptrend" consensus call, holds open-endedly, and sells on a real-price divergence
loss, a "downtrend" consensus call, or a hard 1% stop-loss. See the mechanics spec in
`src/rolling_backtest.py`'s module docstring for the exact per-step decision order.

## What changed: HORIZON 3 -> 1

A full repo audit (before making any change) found every place that assumed 3 horizon bars
-- some correctly derived from the `HORIZON` constant (auto-adapted), several hardcoded
literals that didn't (real bugs, silently corrupting or crashing once `HORIZON` changed).

**Fixed to derive from `HORIZON` instead of a hardcoded literal:**
- `src/data_pipeline.py`: `HORIZON = 1`; removed `build_window`'s bar-1/2 anchor-correction
  loop (only meaningful when there's more than one horizon bar to chain across -- bar 0 is
  already correctly anchored).
- `src/losses.py::unpack_y`: price/volume split width now `HORIZON*4`/`HORIZON`, not a
  hardcoded 12/15.
- `src/models/cvae_inpainting.py`: decoder output width now `2*(HORIZON*4+HORIZON)`
  (=10 at HORIZON=1), not a hardcoded 30; `decode()`'s internal split likewise.
- `src/models/patchtst.py`: same pattern, head width and `forward()`'s split.
- `src/diagnose_cvae_direction.py`, `src/evaluate_generative.py`: hardcoded `3`/`range(3)`/
  `[:12]`/`[12:15]` occurrences fixed to derive from `HORIZON`.
- `src/evaluate.py` (the pre-pivot bracket-order file, kept for `v1.md`/history -- see its
  own PRE-PIVOT module note): `naive_periodic_benchmark`'s trade-block width,
  `run_walk_forward`'s `y`-unpack, and `make_plots`'/`make_random_sample_plots`' plotted-
  region width were all fixed too, even though the bracket-order *strategy* itself wasn't
  redesigned -- these were mechanical shape/cadence bugs, and leaving them broken would
  make the file uselessly crash-prone rather than just "legacy." `naive_periodic_benchmark`
  in particular degenerates to a 0-return-per-block benchmark at HORIZON=1 (buy and sell at
  the same bar's close) -- an honest reflection of "same cadence as a 1-bar strategy," not
  a bug.

**Deliberately not redesigned** (the bracket-order *strategy*, as opposed to its shape
bugs): `bracket_exit`, `run_walk_forward`, `classify_walk_forward_decision`, and
`src/walk_forward_trend_gate.py` still implement "buy, bracket order, walk N horizon bars"
-- just correctly at N=1 now. The actual strategy redesign (open-ended rolling hold) lives
in `src/rolling_backtest.py` instead.

**All existing tests updated** to HORIZON-derived shapes (`tests/test_data_pipeline.py`,
`test_cvae_inpainting.py`, `test_losses.py`, `test_evaluate.py`) -- 3-bar-sequential
bracket_exit scenarios that are no longer reachable through the module-level `HORIZON`
constant were rewritten as single-bar scenarios rather than deleted. New:
`tests/test_rolling_backtest.py` for the new state machine. Full suite: `pytest steven`.

**Consequence worth knowing:** `CVAEInpainting`'s decoder width is derived from the
*global* `HORIZON` constant at construction time, not saved per-checkpoint. This means
every checkpoint trained *before* this migration (`cvae_checkpoint.pt`,
`cvae_checkpoint_generative.pt`, `cvae_checkpoint_generative_mse_baseline.pt`,
`cvae_checkpoint_pre_fix_repro.pt`) can no longer be loaded by any script in this
codebase -- `load_state_dict` will raise a size mismatch (decoder width 30 in the saved
weights vs. 10 in the freshly-constructed model), the same failure mode as a wrong
`in_channels`. This isn't fixable without either retraining every old checkpoint or making
`HORIZON` an instance parameter instead of a module constant (out of scope here). To
re-run anything against those older checkpoints, check out `steven3` (or the commit on
this branch right before this migration).

## Retraining

Not done as part of this migration -- a real 30-epoch run is compute-heavy and this
project's convention is to train on Colab (`colab_train_generative.ipynb`), not locally.
What *was* verified locally: a 2-epoch, 200-window smoke run
(`python steven/src/train_cvae.py --config steven/configs/cvae_h1.yaml --max-epochs 2
--train-windows-per-epoch 200 --windows-per-eval-set 60 --device cpu`) completes with no
shape errors, produces a checkpoint whose decoder output width is confirmed 10 (not 30),
and that checkpoint loads and runs correctly end-to-end through
`rolling_trend_backtest.py` (201 trades, structurally sane `outcome_breakdown`). That
smoke checkpoint was deleted afterward -- it's a plumbing check, not a real model, and
keeping it around risked being mistaken for one.

**To actually retrain**: `configs/cvae_h1.yaml` (and `configs/patchtst_h1.yaml` for the
benchmark) are ready -- identical recipes to `cvae_generative.yaml`/
`patchtst_hourly_momentum.yaml`, differing only in output checkpoint path (model
architecture now adapts to `HORIZON` automatically, no config field changes needed).
Either point `colab_train_generative.ipynb` at `cvae_h1.yaml` instead of
`cvae_generative.yaml`, or run `train_cvae.py --config steven/configs/cvae_h1.yaml`
directly on a GPU runtime.

## Running the new backtest

```
python steven/src/rolling_trend_backtest.py \
    --cvae-checkpoint steven/outputs/cvae_checkpoint_h1.pt
```

Key flags: `--consensus-up-threshold`/`--consensus-down-threshold` (default 0.6/0.4 --
fraction of k sampled draws predicting a positive close return, see `classify_signal`),
`--stop-loss-pct` (default 0.01), `--num-samples` (defaults to the checkpoint's own
`config['inference']['num_samples']`). Writes a metrics JSON with `rolling_backtest`
(n_trades, total_return, win_rate, outcome_breakdown, ...) and a `buy_and_hold` comparison.

## The consensus-gate risk -- read results skeptically

The trade signal here is a **consensus fraction across k sampled draws** (`frac_up` in
`classify_signal`), not a deterministic point estimate. This was a deliberate choice
discussed with the user, made with a specific precedent on the table: the old bracket-order
strategy already tried a structurally similar consensus gate for CVAE and killed it after
direct A/B testing found it *actively harmful*, not just unreliable -- the unanimous-
agreement bucket was the **worst**-performing one (`cvae_direction_collapse.md`,
`backlog.md`'s "trade confidence" entry).

The case for trying it again here: it's a genuinely different usage pattern. The old gate
was a one-shot filter on a single fixed-3-bar bet; here it's evaluated every hour, driving
continuous entry *and* hold *and* exit decisions on an open-ended position. It's plausible
that changes the picture. It is not guaranteed to. Before trusting any positive result from
this backtest:
- Compare against a version using a deterministic point estimate (decode `mu_p`, no
  sampling) for the same entry/exit logic -- if consensus doesn't clearly beat a point
  estimate, the added sampling cost isn't earning its keep.
- Check across multiple training seeds before trusting any single run's numbers --
  `cvae_direction_collapse.md` already found one similarly-sized effect (a daily-momentum
  correlation of +0.202) that turned out to be training-seed noise (5-seed range -0.17 to
  +0.29, mean +0.093) once checked properly.
- Compare against `buy_and_hold` for the same period, not just against 0% -- a positive
  `total_return` that's still far below buy-and-hold is "less bad," not "working."

## Open follow-ups

- Run the real HORIZON=1 retrain (CVAE + PatchTST) on Colab, then the actual rolling
  backtest against it -- everything above is plumbing-verified, not yet run for real.
- The point-estimate-vs-consensus comparison described above.
- Multi-seed check on whatever the first real result shows.
- `src/update_report.py::sample_section`'s 3-candle report table (for `v1.md`) still
  assumes the pre-pivot format -- needs a real redesign once the rolling strategy has its
  own reportable shape decided, not attempted here.
