# CVAE direction collapse — modeling problem to work through

## The observation that started this

All 5 sample plots in `v1.md`'s Results section show CVAE-generated candles with the same
basic shape: three thin, roughly flat-bodied candles with modest wicks — regardless of
whether the real market during that window was trending up, trending down, or chopping.
Visually, CVAE's "generated future" looks almost the same every time.

## Quantified: this is real, not a plotting artifact

Ran two diagnostics against the current checkpoint (`steven/outputs/cvae_checkpoint.pt`,
`z_dim=8, ctx_dim=64, decoder_hidden=128, ctx_dropout=0.3`).

**1. Across-window vs. within-window variation** (40 different real test windows, k=5
samples each, bar 0 components):

| component | across-window std (this window's own mean) | within-window std (across k samples) | ratio |
|---|---|---|---|
| `open_ret` | 0.00011 | 0.00004 | 2.7 |
| `body_ret` | 0.00013 | 0.00024 | **0.56** |
| `upper_wick` | 0.00519 | 0.00204 | 2.5 |
| `lower_wick` | 0.00515 | 0.00203 | 2.5 |

`body_ret` — the component that decides whether the candle is red or green, and by how
much — varies *less* across 40 completely different real market windows than it does
across the model's own k=5 samples for the *same* window. Context explains less variance
than the model's own sampling noise does, for the single most important directional
component.

**2. Correlation with real market direction**: predicted `body_ret` (mean over k) vs. the
real preceding 10-bar trend, across 40 windows: **r = -0.004**. Essentially zero.

**3. Scale vs. real data** (100 different test windows vs. 2000 real training-data
windows):

| | real train data std | CVAE predicted std (mean-over-k, across windows) | ratio |
|---|---|---|---|
| `open_ret` | 0.00246 | 0.00012 | 0.049 |
| `body_ret` | 0.00293 | 0.00013 | 0.046 |

The model's predictions vary **~22x less** across genuinely different real contexts than
real candles actually do. It has converged to outputting a near-constant, tiny, roughly
symmetric `open_ret`/`body_ret` regardless of what's actually happening in the market —
while the **wick** components (`upper_wick`/`lower_wick`) retain meaningfully more
across-window spread (ratio ~2.5, not <1).

## Why this went unnoticed until now

The earlier posterior-collapse fix (`z_dim` 16→8, `free_bits` 0.05→0.15, cyclical KL
annealing, `ctx_dropout=0.3` — see `backlog.md`) was judged successful because CVAE's
*sample-consensus fraction* (fraction of k draws agreeing on an upward move) gained real
spread post-fix (0.2–1.0 across percentiles, no longer pinned near 1.0). That diagnostic
is computed from `exit_price_from_components`, which averages all 3 bars' open/close
prices — a statistic that mixes in wick-driven exit-price movement alongside
open_ret/body_ret. The consensus spread genuinely exists, but it looks like it's coming
substantially from wick noise, not from the model learning genuinely different directional
outcomes for genuinely different contexts. That's consistent with two things already
found and documented but not yet connected to this:
- CVAE's consensus gate turned out to be *actively harmful* when tested directly — the
  unanimous-agreement bucket was the *worst*-performing, and winners/losers had
  statistically indistinguishable consensus (see `backlog.md`'s "trade confidence" entry).
- CVAE's predicted edges run roughly an order of magnitude smaller than PatchTST's (median
  ~0.04% vs. ~0.26%).

Both are exactly what you'd expect if the "signal" being thresholded was mostly noise.

## Open questions to work through

1. **Is this collapse specific to `open_ret`/`body_ret`, or does the decoder route z
   almost entirely into the wick channels and starve the direction channels?** The
   `ctx_dropout` fix targeted decoder reliance on context *in general* — it may have
   pushed the decoder to lean on `z` for wicks (an easier, lower-stakes target to get
   "some" gradient signal from) while direction stayed collapsed to the marginal mean,
   which is a much harder, noisier regression target (predicting the sign of a next-hour
   candle is close to a coin flip in real markets).
2. **Is `body_ret`/`open_ret` fundamentally too hard to learn from this context window at
   all** (i.e. is near-zero achievable correlation actually close to the ceiling for
   1-hour-ahead direction from 70 bars of OHLCV, and any model would land near the
   marginal mean here), **or is this a training/architecture deficiency specific to this
   CVAE** (e.g. loss weighting between direction and wick terms in `cvae_loss`, capacity,
   optimization)? Worth checking whether PatchTST's own predicted `body_ret` shows more
   real across-window variation on the same test windows — if PatchTST *also* can't beat
   nearly this, direction may just be this hard.
3. **Does the recognition network (trained with real horizon access) actually encode
   direction into z**, even if the *prior* network can't recover it from context alone?
   If yes, the bottleneck is prior-conditioning capacity/training, not "direction is
   unlearnable" — worth checking `mu_q`/`logvar_q` from `encode_recognition` against real
   `body_ret` for correlation, as a ceiling check.
4. **Should the loss (`cvae_loss` in `src/losses.py`) weight `open_ret`/`body_ret`
   reconstruction more heavily relative to the wick terms and volume?** If direction is
   a small fraction of the total reconstruction loss, gradient descent may have simply
   found it cheaper to nail wicks/volume and give up on direction.
5. **What does this mean for the project's central thesis** — that CVAE's generative
   uncertainty (multiple plausible futures) is a real advantage over PatchTST's single
   point forecast? If the direction component has collapsed to the marginal mean, CVAE's
   "different futures" are currently mostly different *wick* realizations of the same
   near-flat body, not genuinely different directional scenarios. That's a much weaker
   form of the thesis than intended, and separate from (arguably more fundamental than)
   the consensus-gate finding already in `backlog.md`.

## Root cause found: volume dominates the reconstruction loss by ~1800:1

Ran diagnostics for questions 2, 3, and 4 against the same checkpoint, on 200 test-set
windows (bar 0 components; scratch script, not committed — see chat).

**Q4 first, because it explains the other two.** `apply_normalize` in `data_pipeline.py`
z-scores `log_volume_norm` to unit variance on the training set, but `open_ret`/
`body_ret`/`upper_wick`/`lower_wick` are left as raw log-returns — i.e. variance
~1e-6-1e-5. `weighted_mse_loss` (`src/losses.py`) then combines them with
`w_price=1.0, w_vol=0.5`, a weighting that assumes the two loss terms start out on
comparable scales. They don't:

| component | target variance | this batch's MSE | % of total weighted recon loss |
|---|---|---|---|
| `open_ret` | 0.0000054 | 0.000006 | — |
| `body_ret` | 0.0000066 | 0.000007 | — |
| `upper_wick` | 0.0000010 | 0.000153 | — |
| `lower_wick` | 0.0000020 | 0.000154 | — |
| **all 12 price terms combined** | | 0.000083 | **0.055%** |
| **volume** (z-scored, var≈1) | 0.69 | 0.30 (0.15 after ×0.5) | **99.945%** |

Every gradient step's reconstruction signal is ~99.95% "get volume right," ~0.05% "get
any price component right" — direction *and* wicks both. `w_price=1.0` vs. `w_vol=0.5`
is a 2x correction against a real scale mismatch of ~5 orders of magnitude; it does
essentially nothing. This is almost certainly the dominant reason price reconstruction
in general is starved, and it fully explains open question 4 above (loss weighting) —
the imbalance isn't direction-vs-wick, it's price-vs-volume, and it dwarfs any
within-price imbalance.

**Q3 (recognition ceiling check)**: decoding directly from `mu_q` (recognition
network's posterior mean, computed with real, unmasked horizon access — the best-case
z the model ever has) gives `body_ret` std 0.00026 (vs. real 0.00256, ~10x too tight)
and correlation with real `body_ret` of **r=0.12** — weak, but not zero, and clearly
better than the prior's r≈0 (below). So *some* direction signal reaches z when the
recognition net can cheat and look at the true horizon directly, but even that
best-case pathway is heavily squeezed — consistent with the volume-dominated loss
leaving z_dim=8's capacity mostly spent elsewhere, compounded by the KL free-bits floor
constraining how much of that weak signal survives into the sampled prior.

**Prior at inference** (`mu_p`, context-only, no sampling noise): `body_ret` std
0.00004, r=-0.09 vs. real — collapsed, consistent with the doc's r=-0.004 (computed
with k=5 sampling noise added on top, which only dilutes it further).

**Q2 (PatchTST on the same 200 windows)**: predicted `body_ret` std 0.00089, r=0.045
vs. real. Also weak — so direction may genuinely be close to the ceiling for either
architecture from this context (partial support for the "fundamentally hard" branch of
question 2) — but PatchTST's spread is still ~3.4x wider than CVAE's prior-mean spread
(0.00089 vs. 0.00004) and closer to CVAE's own *hindsight* ceiling (0.00026) despite
using the identical `w_price=1.0/w_vol=0.5` weights (`configs/patchtst.yaml`). That
loss imbalance hits both models equally, but only CVAE has the additional z_dim=8 +
KL-free-bits bottleneck squeezing whatever direction signal the shared encoder does
pick up — consistent with question 1: it's not that the decoder routes z into wicks
specifically at direction's expense, it's that the whole shared representation is
optimized almost entirely for volume, and the VAE's information bottleneck is an
*extra* tax on top of that for price generally, direction most of all (lowest
raw-loss-reduction-per-capacity of the four price terms, per the variance table above).

## Not yet done / ruled out

- Haven't checked whether this holds on the *training* set too (would distinguish "model
  never learned it" from "model learned it but doesn't generalize to test period").
- Haven't tried a retrain with volume/price put on comparable loss scales (e.g. drop
  `w_vol` by ~3 orders of magnitude, or z-score the price components for the loss
  computation the way volume already is) to see how much of the direction collapse is
  fixable vs. a genuine ceiling — the root-cause section above makes a strong case this
  is worth trying before concluding direction is unlearnable.

## Re-checked against the retrained checkpoint (post `price_scale` fix) — partial improvement, not a fix

The "not yet done" item above (retrain with price/volume on comparable loss scales) happened —
the retrain whose log shows `price_scale (open_ret, body_ret, upper_wick, lower_wick): [0.00268, ...]`
being computed and passed into the loss is exactly this fix actually running, and
`outputs/cvae_checkpoint.pt` now reflects it. Re-ran this file's own diagnostics (100 test windows,
k=5, same methodology as above; script not committed, same precedent as the original) to check
whether it worked:

**Table 1 (across-window vs. within-window variation) — the headline number moved in the right
direction**:

| component | across-window std | within-window std | ratio | old ratio |
|---|---|---|---|---|
| `open_ret` | 0.00011 | 0.00012 | 0.92 | 2.7 |
| `body_ret` | 0.00018 | 0.00017 | **1.08** | **0.56** |
| `upper_wick` | 0.00010 | 0.00002 | 4.05 | 2.5 |
| `lower_wick` | 0.00011 | 0.00003 | 4.05 | 2.5 |

`body_ret`'s ratio flipped from 0.56 (context explains *less* variance than the model's own sampling
noise — the literal signature of collapse) to 1.08 (context now explains *slightly more*). That's a
real, qualitative change in the specific pathology this file opened with, not nothing.

**But table 3 (scale vs. real data) shows the absolute magnitude barely moved**:

| component | real train std | CVAE pred std | ratio | old ratio | PatchTST pred std | PatchTST ratio |
|---|---|---|---|---|---|---|
| `open_ret` | 0.00268 | 0.00011 | 0.041 | 0.049 | 0.00133 | 0.496 |
| `body_ret` | 0.00289 | 0.00018 | 0.063 | 0.046 | 0.00121 | 0.417 |

CVAE's `body_ret` spread is still only ~6% of real market variability (was ~4.6%) — better, but nowhere
near closed. PatchTST, unaffected by any of this (no VAE bottleneck, not retrained), sits at ~42–50% of
real variability for the same components — confirming again that CVAE's ceiling problem is real and
specific to it, not just "direction is unlearnable from this context" in general.

**Correlation with a real preceding-10-bar `body_ret` trend (a proxy for "does the model track
momentum," not the same measurement construction as the original `r=-0.004`, so not directly
comparable — but answering the same question)**: CVAE prior (mean-over-k) `r=-0.027`; the recognition
network's best-case ceiling (decoding `mu_q`, which sees the real horizon) `r=-0.021`; **PatchTST itself
`r=-0.137`** — also indistinguishable from zero at this sample size (N=100, so |r| needs to clear
roughly 0.20 to be a confident nonzero effect). Since even PatchTST — fully deterministic, unaffected by
any VAE-specific pathology — shows no usable correlation with this particular simple trend proxy either,
this particular test doesn't cleanly separate "CVAE-specific defect" from "this simple trend signal
isn't real for either model at this context length." Don't read the recognition network's `r=-0.021` as
a regression from the original's `r=0.12` — both are within one standard error of zero at these sample
sizes (N=40 originally, N=100 here), i.e. statistically indistinguishable from each other and from "no
signal."

**Net read**: the fix produced a genuine, measurable improvement on the exact symptom that motivated it
(the across/within-window variance ratio for `body_ret`, the clearest single-number signature of
collapse, flipped from <1 to slightly >1) — this isn't nothing, and it's a cleaner, more direct
"did the fix do anything" signal than the consensus-fraction evidence in the posterior-collapse section
above (which turned out to be uninformative noise once A/B tested). But the absolute scale of captured
variation is still small (~6% of real), and there's still no demonstrated linear correlation with a
simple trend signal. Read together with `backlog.md`'s posterior-collapse entry: **two independent
interventions (KL-budget/annealing/dropout, and now loss-scale correction) have each moved one
diagnostic number in the right direction without producing a demonstrably *useful* CVAE direction
signal yet.** The most direct next check, per the original open question 2 in this file: is `body_ret`
fundamentally this hard from a 70-bar context regardless of architecture (plausible, given PatchTST's
own weak trend-correlation here), or is there still a fixable CVAE-specific bottleneck on top of a hard
problem both models share? The scale comparison (CVAE still at ~15% of PatchTST's own predicted std for
`body_ret`, despite an identical `w_price`/`w_vol` loss weighting now) argues there's still a
CVAE-specific gap beyond whatever the shared difficulty is — the VAE information bottleneck
(`z_dim=8` + KL free-bits) is still squeezing genuinely harder than PatchTST's architecture does.

## To-do

- [ ] **Change the walk-forward advance rule: always skip 3 candles (HORIZON), whether a
  trade was taken or not.** Currently (`run_walk_forward` in `src/evaluate.py`), a no-trade
  decision advances the clock by 1 bar and a trade advances it by `HORIZON` (3 bars) — see
  *Walk-forward backtest* point 3 in `v1.md`. Change this so every decision point, traded
  or not, advances by 3 bars uniformly. To discuss: why we want this for testing (keeps
  the model's decision cadence identical/comparable regardless of trade/no-trade, avoids
  the two models drifting onto different real-time clocks per *Walk-forward backtest*
  point 3 in `v1.md`), and what it changes about the backtest (far fewer decision points
  overall, changes `n_decisions` denominators in `outcome_breakdown`, and changes what
  "consecutive" windows look like for any future diagnostic that walks decisions in order).
- [x] ~~Try dropping `w_vol` by ~2-3 orders of magnitude...~~ Started (`w_vol` 0.5 → 0.001,
  commit `d657937`), but reverted before a retrain finished (commit `3bc2989`) — the KL number
  checked partway through a run was pinned at the same free-bits floor seen in every prior
  retrain, which isn't actually diagnostic for this experiment (see `backlog.md`), but the call
  was made to go straight to the architectural lever below instead of waiting the loss-reweighting
  experiment out. Not a negative result — genuinely untested to completion.
- [x] **Architectural: bottleneck the decoder's own view of context (`decoder_ctx_dim`,
  `src/models/cvae_inpainting.py`)** — tried, checked against a real retrain, **did not fix
  collapse.** Rechecked with the same three signals used for the `price_scale` fix, against
  checkpoint `72147bb`:
  - Across/within-window `body_ret` variance ratio: **0.91** — actually below 1 (collapsed),
    slightly worse than the `price_scale`-only checkpoint's 1.08.
  - Correlation with a 10-bar trend signal: **r=+0.10** (N=100) — up from -0.03, but still
    within noise range at this sample size.
  - Practical effect on the walk-forward backtest was dramatic, though: predicted return
    (`take_profit` vs. `close_0`) across 300 fresh test windows was **negative on 100% of
    them** (mean -0.09%, tightly clustered -0.12% to -0.06%), vs. the previous checkpoint's
    near-universal *positive* bias. CVAE went from trading on ~99.9% of decisions to 0%.
  - Read together: the model still emits an almost context-independent constant either way —
    this retrain just landed on a **different-signed** constant (mean `body_ret` -0.00135, or
    ~47% of real train std, a large systematic offset, not small noise) than the previous run
    did. That's consistent with "a different arbitrary collapsed fixed point from a fresh
    training run," not "the bottleneck reduced collapse." Left in place (doesn't hurt,
    combines with whatever fixes collapse for real) but not the fix on its own.
  - Side effect worth remembering: this is also why `n_decisions` jumped from ~800 to 2397 in
    this run without the test period changing — see the walk-forward advance-rule item above.
- [x] **Auxiliary direction loss (`w_direction`/`direction_temperature` in `configs/cvae.yaml`,
  `losses.py`)** — tried next, since neither loss-scale correction nor an architectural
  bottleneck gave plain MSE enough signal to stop `body_ret` from collapsing to *some*
  constant. Adds a binary cross-entropy term on the sign of `per_bar_close_return`
  (`open_ret + body_ret` — the same quantity PatchTST's own quality gate checks the sign of,
  and exactly what `exit_price_from_components`'s take-profit target is built from), a much
  more direct classification-style signal than an MSE term ever gives for a binary-ish
  property like direction. **Checked against a real retrain (checkpoint `ceb78d7`) — did not
  fix collapse, and made the practical symptom worse:**
  - The training log itself is the clearest signal: `dir=` (the direction BCE term) settled
    at **0.70-0.72** by the end of training — barely above `ln(2)≈0.693`, the loss value for
    a classifier that always predicts 50/50 regardless of input. It never meaningfully
    dropped across any of the three annealing cycles.
  - Rechecked with the same three signals as before, against real weights: across/within
    `body_ret` variance ratio **0.928** (still <1, collapsed), correlation with a 10-bar
    trend **r=+0.146** (N=100, still noise-level), predicted return vs. `close_0` across 300
    fresh windows **negative on 100% of them again** — mean **-0.14%** (vs. -0.09% on the
    `decoder_ctx_dim`-only checkpoint), i.e. *more* negatively biased, not less.
  - Read together with the two entries above: **three independent interventions in a row**
    (loss-scale correction, architectural context bottleneck, direct classification loss on
    direction) have each converged to the same qualitative outcome — a small, nearly
    context-independent constant, whose sign flips arbitrarily between training runs, with
    correlation-to-trend bouncing around zero within noise every time (-0.03, -0.03, +0.10,
    +0.15 across four checkpoints now measured this way). None has produced a directional
    signal that's demonstrably better than the model's own best constant guess. This is
    starting to look less like "the CVAE side has a fixable bug" and more like a genuine
    ceiling: PatchTST's own trend correlation was also weak (r=-0.137), consistent with
    3-bar-ahead direction from a 70-bar hourly SPY context being close to unpredictable, and
    any MSE/BCE-trained model rationally converging toward its best constant for a
    mostly-noise target regardless of which knob gets turned.

## Revisiting the pre-collapse-chasing checkpoint

Before any of the three interventions above, commit `2c4ad99` ("Add a shared 2% stop-loss")
had the best walk-forward result CVAE has produced under the current (comparable,
apples-to-apples) walk-forward methodology: **total_return +6.79%, annual_return +4.91%,
win_rate 88.3%, 684/1030 decisions traded (66.4% — genuinely selective, not the ~100%-or-0%
extremes every checkpoint since has landed on)**. Its architecture was `z_dim=8, ctx_dim=64,
decoder_hidden=128, ctx_dropout=0.3` (the KL-budget-concentration fix already in place) with
**no `price_scale`, no `decoder_ctx_dim`, no `w_direction`** — i.e. it predates the discovery
that volume was dominating the reconstruction loss ~1800:1. Every fix attempted since has
made the loss *more correct* by that diagnosis but hasn't reproduced (or exceeded) this
result — worth being honest that this could mean the loss-scale bug wasn't actually the
thing standing between CVAE and a working strategy, or it could mean `2c4ad99`'s selectivity
pattern was a favorable roll against this one 1.37-year test path rather than a real edge
(no correlation-with-trend check exists for this specific checkpoint, since that diagnostic
didn't exist yet when it was current — everything measured *since* has been indistinguishable
from noise). See discussion with the user before deciding whether to reproduce this config
as a baseline to compare against, or to treat it as a data point rather than a target.

## Momentum-feature enrichment (EMA9/EMA21) — also didn't fix it

`daily_signal_probe.md` flagged richer conditioning inputs (VIX/RSI/MACD/Bollinger Bands from
`data/processed/features.parquet`) as the safer alternative to changing sampling frequency —
more informative context, without touching what the model is asked to reconstruct. Tested the
smallest version of that idea first: `src/momentum_pipeline.py` adds two causal, z-scored
features computed from the hourly close series — `ema_cross_norm` (`log(ema9/ema21)`, the
classic golden-cross/death-cross signal) and `trend_position_norm` (`log(close/ema21)`,
where price sits relative to the slower trend line). Both are functions of the close price
CVAE is predicting, so both get masked to 0 at horizon positions exactly like the existing
price/volume channels — verified directly (`masked_tensor`'s horizon rows read exactly `0.0`
for both new columns; `full_tensor`'s read the true values) before trusting any training run
against them. Stayed on the hourly pipeline (single-variable test — only the input features
changed, not sampling frequency, config, or context length). `CVAEInpainting` gained an
`in_channels` param (default preserves the existing 9-channel behavior) so the encoder could
actually consume the 2 extra channels (11 total with padding/target masks).

Full 30-epoch retrain (`probe_momentum_cvae.py`, `configs/cvae_momentum.yaml` — otherwise
identical recipe to `configs/cvae.yaml`) against real weights:

- Direction BCE (`dir=`) settled at **0.729** — again barely above `ln(2)≈0.693` chance level,
  the same as the `w_direction`-only run.
- `body_ret` variance ratio **0.606** (still <1, collapsed) — same signature as every
  previous run.
- Correlation with 10-bar trend: **r=-0.122** (N=300) — still squarely inside the noise band
  every prior checkpoint has landed in (-0.03, -0.03, +0.10, +0.146, and now this).
- Walk-forward: 17/2363 decisions traded, **total_return -0.32%**, 82.4% win rate on those 17
  — collapsed to a near-zero/slightly-negative constant (mean predicted return -0.0055%, only
  33.3% of test windows even eligible), the same "which side of the trading threshold an
  arbitrary constant happens to land on" pattern as the daily-bars cross-run check above, not
  a working strategy.

This is now **six** independent interventions (loss-scale correction, architectural context
bottleneck, auxiliary direction classification loss, daily-bars resampling, a cross-run
stability check, and now momentum-feature enrichment) that have each converged on the same
outcome: CVAE settles on a small, nearly context-independent constant for `body_ret`/
`open_ret`, and neither changing the loss, the architecture, the sampling frequency, nor the
input features has produced a correlation with real market direction distinguishable from
noise.

## Adding RSI-14 broke the collapse pattern, but not in a useful way

Discussed with the user: MACD was deliberately skipped as the next feature to add, since it's
itself an EMA-difference construction (`EMA12-EMA26`) -- the same category of signal as
`ema_cross` above, which already tested null, so it would mostly re-test something already
disproven rather than add real information. RSI-14 (Wilder's formulation, z-scored, added as
a third momentum channel alongside the existing two, same leakage-masking treatment verified
directly again before trusting a run against it) is a genuinely different construction --
average-gain/average-loss ratio, not a trend slope -- so it was added instead.

Full retrain (same config, only the extra `rsi_norm` channel changed) produced a real
qualitative shift, not just another null result:

| | EMA-only (previous) | + RSI-14 |
|---|---|---|
| `body_ret` variance ratio | 0.606 (<1, collapsed) | **1.156 (>1, for the first time)** |
| correlation with 10-bar trend | -0.122 | -0.014 |
| predicted-return spread (p5 to p95) | -0.025% to +0.012% (tight) | **-0.009% to +0.024% (real spread)** |
| walk-forward trades | 17/2363 | **184/2029** |
| walk-forward total_return | -0.32% | **-4.29%** |
| win_rate | 82.4% | **90.8%** |

For the first time across every intervention tried, the model stopped emitting a
near-constant `body_ret` -- across-window variance now genuinely exceeds the model's own
sampling noise, and it's trading 10x more often. But this is **not** the fix it might look
like at first glance: correlation with real trend is still statistically indistinguishable
from zero (-0.014, same noise band as always), and the backtest result got *worse*, not
better. A 90.8% win rate combined with a *negative* total return is the signature of the
asymmetric-payoff problem already documented in `evaluate.py`'s own module notes (CVAE's
average loss running far larger than its average win, payoff ratio ~0.14) -- RSI gave the
model more confidence to trade on what still isn't real directional information, and the
existing loss/win size asymmetry punished the extra activity. In short: RSI changed *how
often* and *how variably* CVAE trades, not *whether it's right* when it does.

## Two more variables at once: rolling-window sampling + a real 1993-2025 daily model

Discussed with the user next: switch training from `WindowSampler`'s random draw across 9
context lengths (14-70 bars) to a new `RollingWindowSampler` (`data_pipeline.py`) -- ONE
fixed context length, sliding by 1 bar, covering every valid window exactly once per epoch --
and build a real daily-bars model (SPY's full 1993-2025 history via `yfinance`, not the
~2010-2025 resample the earlier disposable probe used) alongside a matching hourly model, both
enriched with the same EMA9/EMA21 + RSI-14 features. Explicit expectation set going in:
random-vs-rolling window *selection* doesn't change the underlying context->direction
relationship being learned, so this isn't expected to fix collapse on its own -- it's motivated
by two separate real problems: (1) training's context length finally matches
`WALK_FORWARD_CTX_BARS` exactly (today's random sampler trains across 9 lengths but evaluation
only ever tests at 70), and (2) it removes an oversampling risk the daily-bars probe had and
never fixed (`train_windows_per_epoch=20000`, a number picked for hourly's ~21k-row pool,
overshot the old daily probe's ~3k-row pool by ~7x, ~199x exposure per row -- rolling makes
"one epoch" mean "one real pass," at whatever pool size actually exists). Both models use a
matched 10-trading-day context (hourly: 70 bars; daily: 10 bars) for a fair frequency
comparison, not the hourly model's raw bar count. `RollingWindowSampler.pairs()` returns the
same `(start_idx, ctx_bars)` shape `WindowSampler.draw` does, so `WindowDataset` needed no
changes. Full 30-epoch retrains, `probe_momentum_rolling_cvae.py --frequency {hourly,daily}`:

| | hourly (rolling, ctx=70) | daily (rolling, ctx=10, 1993-2025) |
|---|---|---|
| `body_ret` variance ratio | 0.620 (<1, collapsed) | 0.588 (<1, collapsed) |
| correlation with 10-bar trend | **-0.004** | **+0.202** |
| predicted-return mean / pct eligible | +0.027% / 91.7% | +0.046% / 83.3% |
| walk-forward trades | 633/1131 | 174/243 |
| win_rate | 89.9% | 86.2% |
| total_return (stop_loss=0.02, hourly-calibrated) | +0.95% | -13.53% |
| total_return (stop_loss=daily-calibrated p99, 0.0515) | n/a (hourly already uses 0.02 correctly) | **+1.80%** |
| buy&hold / naive_periodic (same period) | +24.11% / -5.98% | +48.34% / +30.68% |

**Hourly**: still collapsed by every measure (variance ratio <1, correlation ≈0) -- the
positive +0.95% total_return is almost certainly a favorable constant-bias roll (a
near-context-independent small positive `predicted_return` happened to land this training
run), not evidence of skill, exactly like the earlier "different-signed collapsed constant"
pattern documented above. Rolling-window sampling alone doesn't change hourly's story.

**Daily is the most interesting result in this whole document**: r=+0.202 is meaningfully
outside the noise band every other measurement across six prior interventions has landed in
(-0.03 to +0.15). Caveat, and an important one: this is measured on 300 *randomly drawn,
overlapping* windows from a test period of only ~590 valid daily starts -- the effective
independent sample size behind that correlation is smaller than N=300 suggests, exactly the
"stable across resamples, not a single lucky number" concern `daily_signal_probe.md` flagged
in its own proposed significance bar. Worth re-checking on a non-overlapping or walk-forward
resample before trusting it as a real edge, not just noting it and moving on.

**Found and fixed a real bug while investigating why an 86% win rate produced a *negative*
total_return**: `stop_loss_pct=0.02` (evaluate.py's shared default) was calibrated
specifically against *hourly* 3-bar move sizes ("just outside the training data's own
empirical p99 |anchored log return|, ~1.9%" -- see evaluate.py's module note). Daily 3-bar
(3-day) moves are naturally much larger, so reusing 0.02 there was tightening the stop far
past what daily volatility actually looks like -- converting ordinary daily-scale noise into
forced stop-outs on trades that would otherwise have been fine. Recomputing the same
`train_exit_return_bound` p99 diagnostic already used to calibrate the take-profit shrink
(0.0515 for this run, not 0.02) and using *that* as the daily stop-loss instead flipped total
return from -4.90% to **+1.80%** on the identical trained checkpoint and otherwise-identical
trades. `probe_momentum_rolling_cvae.py` now derives stop_loss_pct from `sell_bound` for the
daily frequency rather than reusing the hourly-tuned constant. Real methodological lesson
independent of the collapse question: any shared-across-frequencies constant calibrated once
against one frequency's data (`sell_bound` was already handled correctly; the stop-loss
wasn't) needs re-deriving, not reusing, when the underlying bar scale changes.

Even after that fix, +1.80% total_return is still far below buy&hold's +48.34% over the same
daily test period -- this isn't a working strategy yet, just no longer an artificially
sabotaged one. Whether the r=+0.202 correlation is a real, exploitable signal or a fortunate
roll on ~590 overlapping test windows is the open question to resolve next, before investing
further in the daily-momentum direction.

## Adding VIX (CBOE put/call ratio checked and ruled out)

Investigated CBOE's total/equity/index put/call ratio as a second sentiment source alongside
VIX -- real, freely downloadable CSVs exist (`cdn.cboe.com/resources/options/
volume_and_call_put_ratios/{totalpc,equitypc,indexpcarchive}.csv`), but they stop at
**2019-10-04** (CBOE discontinued the free feed and moved current data behind their paid
DataShop product). That's before every test period this project uses (2023-2025 daily,
2024-2025 hourly) -- unusable for backtesting here regardless of effort spent scraping it.
Ruled out; not pursued further.

VIX itself (`src/collect_vix_yfinance.py`, full 1993-2025 history, same source/discipline as
the SPY daily pull) was added as a third momentum-adjacent feature (`vix_norm` in
`src/momentum_pipeline.py`) -- unlike EMA/RSI, a genuine market-derived sentiment/fear proxy
external to SPY's own price history, not a transform of it. Shifted by one trading day before
merging (`add_vix_feature`, `merge_asof(direction="backward")`) so each row gets the *prior*
day's VIX close -- literally "before today's open," not the same-day value. Verified directly
before trusting a run: no NaNs, correct prior-day alignment (spot-checked against the raw VIX
file), and horizon masking confirmed (`masked_tensor` reads exactly `0.0` at horizon
positions, `full_tensor` reads the true shifted value).

Full retrain, both frequencies, VIX added on top of EMA9/EMA21+RSI-14 (otherwise identical to
the rolling-window runs above):

| | daily, EMA+RSI only | **daily, +VIX** | hourly, EMA+RSI only | **hourly, +VIX** |
|---|---|---|---|---|
| correlation with trend | +0.202 | **-0.104** | -0.004 | +0.010 |
| pct eligible | 83.3% | 99.7% | 91.7% | 100.0% |
| trades | 174/243 | **197/197** | 633/1131 | **799/799** |
| total_return | +1.80% (stop-loss fixed) | -14.17% | +0.95% | +7.64% |

**VIX regressed the one promising result in this document.** The daily model's r=+0.202
dropped back to -0.104 (squarely back in the noise band) once VIX was added, and both models
collapsed into trading *every single decision* with an extremely tight, always-positive
predicted-return band -- a different failure signature than anything seen before (partial
eligibility, at least some selectivity). Read together with VIX tending to run low during
both models' training years (a mostly bull-trending stretch): this looks like the model
learned "VIX low → predict the market's own unconditional positive drift almost always,"
i.e. re-deriving the equity risk premium from a regime proxy, not learning anything
conditional about direction. Hourly's total_return jumping to +7.64% likely the same
mechanism as the earlier "+0.95%" and "+7.64%"-adjacent constant-bias rolls already
documented -- a favorable bias that happens to match this test period's own upward drift, not
evidence of skill (correlation stayed at noise level, +0.010).

**Decision (discussed with the user): keep VIX in the pipeline anyway, revisit later.** Not
because this result argues for it -- it doesn't -- but because a single retrain isn't enough
to distinguish "VIX genuinely hurts" from "the daily correlation number was never stable to
begin with." The bigger, still-open finding this surfaces: **r=+0.202 → -0.104 from one added
feature means that number was volatile, not a stable signal** -- reinforcing rather than
resolving the "is this real or a lucky draw on ~590 overlapping windows" question flagged
above. Next step, before adding anything else (a Kalman-filtered trend feature was discussed
and deferred for the same reason -- see chat): re-run the EMA+RSI-only daily config across
multiple seeds/resamples to see whether +0.202 reproduces at all, or was itself already noise.

## Resolved: r=+0.202 was training-seed noise, not a real signal

`robustness_check_momentum.py` separates the two distinct sources of variance the question
above conflated, both against the pre-VIX daily config (`include_vix=False`, EMA9/EMA21+
RSI-14 only -- momentum_pipeline.py gained an `include_vix` toggle, defaulting to `True` per
the decision above, purely to make this controlled comparison possible):

1. **Training-seed variance**: same config, 5 different training seeds (42-46), each
   evaluated with the same fixed `eval_seed=123` -- isolates how much the correlation moves
   from training randomness alone, holding the evaluation sample fixed.
2. **Evaluation-sampling variance**: the *same* trained model (seed=42), re-evaluated with 5
   different `eval_seed`s -- isolates how much the correlation moves just from which ~300 (of
   ~590 available, heavily overlapping) test windows get drawn, holding the model fixed.

| | training-seed variance (5 seeds) | evaluation-sampling variance (1 model, 5 reseeds) |
|---|---|---|
| correlations | +0.150, +0.206, -0.010, +0.290, -0.170 | +0.150, +0.206, +0.147, +0.116, +0.144 |
| mean | +0.093 | +0.153 |
| std | **0.164** | 0.030 |
| range | **-0.170 to +0.290** | +0.116 to +0.206 |

**Verdict: the correlation number is dominated by training-seed randomness, not evaluation
noise.** Holding the model fixed, the correlation estimate is fairly stable (std=0.03) --
the "effective N smaller than 300" overlapping-windows concern turns out to be a comparatively
minor contributor. But across 5 different training seeds of the *identical* config, the
correlation ranges from -0.170 to +0.290 -- swinging from clearly negative to clearly
positive depending purely on which random seed the training run happened to use. A std of
0.164 comfortably covers every single-run correlation number reported anywhere in this
document for the daily model (+0.202 pre-VIX, -0.104 post-VIX, and everything in between).
**+0.202 was not a real, reproducible signal -- it was one favorable draw from a
training-seed distribution centered near zero (mean +0.093, well within noise of 0).** VIX's
apparent "regression" to -0.104 needs no separate explanation beyond this: it's just another
draw from the same noisy distribution, not evidence VIX specifically hurt anything.

**Methodological lesson for everything else in this document, not just this one number**:
every single-run correlation reported above (hourly and daily alike, across all seven+
interventions) carries this same training-seed-variance risk and was never checked against
it. This doesn't retroactively invalidate the qualitative pattern (all of them landing in a
similar noise-centered range is itself consistent with "no real signal," which is the
conclusion the whole document has been converging toward anyway) -- but it does mean no
*individual* number in this document should be treated as more precise than it is. Any future
claim of the form "this change moved the correlation from X to Y" needs multiple seeds to
mean anything; a single before/after comparison cannot distinguish a real effect from
training noise with a spread this wide.
