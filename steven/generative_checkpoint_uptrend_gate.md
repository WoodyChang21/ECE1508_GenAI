# Generative-checkpoint sample grids + trend-gated walk-forward — how to reproduce

Two small scripts, plus this doc, formalizing an exploratory check run against
`cvae_checkpoint_generative.pt` (the NLL-reconstruction "generative pivot" checkpoint, see
`cvae_direction_collapse.md`): (1) does a single context's own k sampled candle completions
look collapsed or diverse, and (2) does gating CVAE's walk-forward trades to only fire when
the recent trend already looks like an uptrend change the backtest result. Kept as real,
reusable scripts (not one-off notebook cells) since we may want to re-run either check
against a future retrain.

## Context: two checkpoints, two eval paths

- `outputs/cvae_checkpoint.pt` — plain MSE reconstruction. This is what `evaluate.py`'s
  walk-forward backtest (`run_walk_forward`/`make_cvae_predict_fn`/`walk_forward_stats`)
  defaults to, and is the lineage `cvae_direction_collapse.md`'s whole investigation
  diagnosed as collapsed/under-dispersed.
- `outputs/cvae_checkpoint_generative.pt` — NLL reconstruction (Laplace on open_ret/body_ret,
  Gaussian on wicks/volume), trained separately, specifically to fix that by rewarding
  calibrated spread directly through the loss instead of leaving all anti-collapse pressure
  on the KL term (see `src/models/cvae_inpainting.py`'s module docstring).
- **`evaluate.py`'s `main()` currently only renders scenario charts** — the walk-forward
  backtest code is still in the file (`run_walk_forward` etc.) but unused by `main()` since
  the "replace the backtest with a chart-only scenario comparison" commit. Both scripts
  below call that code directly instead of going through `evaluate.py`'s CLI.

## Script 1: `src/sample_candle_grid.py`

Ground truth + k sampled candle completions, side by side, for one example window per trend
label (uptrend/downtrend/choppy) — a single-context view of sample diversity in the same
visual language (candlesticks) as the presentation's main diagram, as opposed to
`build_diversity_fan`'s k=32 semi-transparent close-price lines.

```
python steven/src/sample_candle_grid.py \
    --cvae-checkpoint steven/outputs/cvae_checkpoint_generative.pt
```

Writes `steven/outputs/generative_plots/sample_candles_{uptrend,downtrend,choppy}.png`.
`--k`, `--n-examples`, `--ctx-bars`, `--trend-lookback`, `--step`, `--out-dir`, `--seed` are
all overridable — see `--help`.

**What we saw on the first run** (k=5, one example per label): not a full collapse — the 5
samples per window differ in body size, color, and the size of the final candle, not
identical clones. Diversity was clearest in the downtrend example (one sample notably bigger
than the other 4) and weakest in the choppy example (all 5 fairly similar). Comparing real
vs. sampled final close: downtrend and choppy both had the real outcome fall inside the
5-sample spread; uptrend didn't (real close 558.76 vs. a sampled range of only 552.98–556.17)
— consistent with `cvae_direction_collapse.md`'s finding that this model's predicted spread
is still only ~6% of real market variability, i.e. improved but still under-dispersed, not a
clean pass/fail.

**Caveat**: a single context's own k samples cannot, by themselves, distinguish "the model
is context-sensitive" from "the model injects realistic-looking noise around a
context-blind prediction" — both look like visual diversity here. That distinction requires
comparing across many different contexts (variance-ratio / correlation-with-trend style
diagnostics, e.g. `evaluate_generative.py`'s `variance_ratio`/`crps`), not a handful of
single-window grids.

## Script 2: `src/walk_forward_trend_gate.py`

Runs the walk-forward backtest twice against the same checkpoint and the same seeded random
draws — once unrestricted, once with CVAE's trade gated off whenever the trailing 20 bars
aren't classified "uptrend" by `trend_z_score`/`classify_trend` (the same classifier behind
the sample-grid labels above and `evaluate.py`'s scenario charts). The gate is a
`predict_fn` wrapper that forces `passes_quality_gate=False` on disallowed-label windows —
no changes to `run_walk_forward` itself.

```
python steven/src/walk_forward_trend_gate.py \
    --cvae-checkpoint steven/outputs/cvae_checkpoint_generative.pt \
    --allowed-labels uptrend
```

Writes `steven/outputs/walk_forward_trend_gate_metrics.json`. `--allowed-labels` accepts a
comma-separated subset of `{uptrend,downtrend,choppy}`; other flags (`--num-samples`,
`--cvae-sell-quantile`, `--cvae-min-return-threshold`, `--stop-loss-pct`, `--trend-lookback`,
`--seed`) mirror `evaluate.py`'s old walk-forward CLI defaults — see `--help`.

**Result, test period 2024-01 to 2025-05 (~1.37 years)**:

| | baseline (every eligible window) | uptrend-only |
|---|---|---|
| trades | 730 | 294 |
| win_rate | 80.1% | 81.6% |
| **total_return** | **-1.78%** | **+2.07%** |
| annual_return | -1.31% | +1.51% |
| buy & hold (same period) | +24.11% | +24.11% |

Gating to uptrend-only flips this checkpoint from a small loss to a small gain — real, but
worth reading narrowly:

- **`trend_z_score` is a realized-momentum filter on the past, not a model prediction.**
  "Uptrend-only" means "only trade when the market was already trending up going in," not
  "only trade when CVAE is confident about an upward move."
- **`n_decisions` differs between the two runs (937 vs. 1809) and isn't directly
  comparable.** `run_walk_forward`'s clock advances 1 bar on no-trade vs. `HORIZON` (3) bars
  on a trade — a less-active strategy visits more decision points over the same calendar
  span. `total_return`/equity accounting is unaffected (compounds over trades taken, not
  decision count); `outcome_breakdown` fractions are the part not to over-read. See
  `cvae_direction_collapse.md`'s still-open to-do about unifying this advance rule.
- **Single seed, single checkpoint, one test period that itself drifted up ~24%.**
  `cvae_direction_collapse.md` already found one similarly-sized single-run correlation
  (+0.202 on a daily-momentum model) that turned out to be training-seed noise once checked
  across 5 seeds (range -0.17 to +0.29, mean +0.093). This result hasn't had that check yet
  — it could easily be mostly "a positive-entry filter riding a bull-trending test period,"
  not a CVAE-specific edge.
- **Still far below buy-and-hold** (+2.07% vs. +24.11%) — read this as "less bad than the
  unrestricted baseline," not "a working strategy."

## Open follow-ups (not done yet)

- Re-run `walk_forward_trend_gate.py` across multiple `--seed` values to check whether
  +2.07% is stable or, like the daily-momentum finding, mostly seed noise.
- A trivial control: trade every uptrend-classified window at a fixed target return,
  ignoring CVAE's sampled price entirely, to check whether CVAE's take-profit target adds
  anything beyond the trend filter itself.
- Same two checks against `cvae_checkpoint.pt` (the MSE checkpoint), to see whether the
  trend gate helps a known-collapsed model too, or whether this is specific to the
  generative-pivot checkpoint.
