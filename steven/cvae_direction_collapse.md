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
