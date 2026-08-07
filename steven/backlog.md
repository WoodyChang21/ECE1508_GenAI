# Backlog

Future experiments, not built in v1.

## PatchTST wick-based take-profit target

CVAE's take-profit is the 70th percentile of its *k* sampled exit prices (see `v1.md`'s Trading criteria
section) — it has a real distribution to pick a target from. PatchTST is a single deterministic point
forecast with no such distribution; its target moved once already (open/close average → max of its 3
predicted closes, via `max_close_from_components`), but that's now aggressive enough that its take-profit
order essentially never resolves before expiry (well under 1% take-profit rate at every confidence
threshold, per the current backtest). Worth trying instead: set PatchTST's take-profit
near its own predicted candle *highs* (e.g. the max predicted high across the 3 horizon bars, minus a
small safety margin) rather than its predicted *closes* — this would actually use the wick predictions,
which are currently generated but thrown away by both `exit_price_from_components` and
`max_close_from_components`. Not obviously better or worse than the current max-close target without
comparing them on the same backtest — could land anywhere between "still too aggressive" and "a genuine
improvement," since predicted highs run above predicted closes but real wicks may not extend as far as
the model's guess either. Deliberately held off this pass since PatchTST is meant to stay a simple
benchmark, not necessarily mirror CVAE's approach.

## CVAE posterior collapse (KL pinned at the free-bits floor) -- fix attempted, mixed evidence

CVAE's reported KL loss used to settle to exactly `z_dim * free_bits` (0.8 = 16 * 0.05) and stay there for
the rest of training, on every run so far (see `v1.md`'s "A training diagnostic" note under Loss) — every
latent dimension sitting exactly at the free-bits floor, meaning the true, unclamped KL was at or below it
everywhere. That's posterior collapse: the recognition network wasn't encoding meaningfully more than the
context-only prior already had, so `z` likely wasn't carrying much real per-example signal -- consistent
with the walk-forward backtest showing CVAE trading on ~99.6% of its decisions (its own confidence rarely
disagrees with itself across the *k* samples).

Likely root cause, specific to this architecture: `decode()` concatenates `z` with the *full* context
representation (`ctx_repr`) and reconstruction loss is plain MSE. Since SPY 1h log-returns are close to
unpredictable noise beyond context, encoding real target-specific information into `z` buys very little
extra MSE reduction relative to its KL cost -- the optimizer can rationally let `z` collapse to whatever
the free-bits floor forces and get "diversity" for free by decoding noise around a context-only
prediction, rather than genuinely different, data-driven hypotheses.

**Three changes made together** (`steven/configs/cvae.yaml`, `steven/src/train_cvae.py`,
`steven/src/models/cvae_inpainting.py`), not yet validated against a real training run:
1. **Concentrate the KL budget**: `z_dim` 16 → 8, `free_bits` 0.05 → 0.15 (total floor 0.8 → 1.2, spread
   over half as many dimensions -- forces more real signal per dimension instead of a thin trickle across
   16).
2. **Cyclical KL annealing** (`kl_beta_schedule` in `train_cvae.py`, replacing the old single 0→1-then-flat
   ramp): 3 cycles across training instead of 1, each ramping 0→1 over its first half then holding --
   once a dimension collapses under a flat, maxed-out penalty the gradient to "wake it back up" is weak;
   periodically relaxing the penalty gives the decoder repeated chances to learn `z` is worth using.
3. **`ctx_dropout=0.3`** on `ctx_repr` inside `decode()` only (never before the prior's own `prior_head`) --
   weakens the decoder's always-available context-only bypass during training, the direct analogue of the
   classic "word dropout" fix from the original VAE-collapse literature.

Next step: retrain via `colab_train.ipynb` and check whether the reported KL loss actually settles above
the new floor (1.2) rather than exactly at it -- if it's still pinned exactly at `z_dim * free_bits`,
these changes weren't enough and the next lever to pull is architectural (e.g. decouple the decoder's
context input from the prior's, or reduce decoder capacity so it can't reconstruct well from context
alone -- see the root-cause paragraph above).

**First retrain result (2026-08-07)**: KL settled at `1.2000-1.2029` in most epochs (occasionally
`1.2125` right after a cyclical-annealing reset) -- no longer pinned to 4 decimal places with zero
deviation the way the old `0.8000` was, and it visibly reacts to each beta reset (rises, then gets pulled
back down as beta climbs back to 1.0), so *something* changed. But the excess over the floor is small
(roughly 0.2-1%), so this reads as "still substantially collapsed, marginally less totally so" rather than
a clear fix -- not enough on its own to conclude collapse is resolved.

**Also found and fixed a real bug in the same run, unrelated to whether the model-side fix worked**:
`train_cvae.py` was selecting the "best" checkpoint by raw `val_loss` (`recon_loss + beta * kl_loss`).
Since cyclical annealing makes `beta` cycle between its ramp-fraction low point and 1.0, that raw total is
only comparable *within* the same phase of a cycle -- comparing across phases will always favor whichever
epoch has the lowest beta, independent of model quality. Confirmed in the log: every "saved best
checkpoint" event landed on exactly the first epoch of a cycle (epoch 1, 11, 21 -- each right after a beta
reset), never anywhere else. The checkpoint that got kept was essentially an arbitrary early snapshot, not
the best-trained one. Fixed by selecting on reconstruction loss alone (`recon_loss`, unweighted by beta,
now logged every epoch as `train_recon`/`val_recon`) instead of the raw total -- `losses.py`'s `cvae_loss`
now also returns `recon_loss` in its parts dict for this purpose.

**Second retrain, with the checkpoint-selection fix actually in effect (2026-08-07)**: `val_recon` fell
from `0.20` (epoch 1) to `0.069` (epoch 28, the checkpoint actually kept) -- a genuine, mostly-monotonic
~65% reduction held across all three beta cycles, so this is a real, well-converged model, not an
artifact. KL itself still tells the same marginal story as the first retrain: `1.2000-1.2029` in most
hold-phases, brief spikes to `1.2052-1.2123` right after each reset -- roughly 0.2-1% above the new floor,
same relative magnitude as before. On the KL number alone, still reads as "substantially collapsed."

**But the walk-forward backtest tells a more encouraging story than the KL number does.** Direct
inspection of this checkpoint's confidence/edge-size distributions (not just the aggregate walk-forward
outcome) found:
- CVAE's sample-consensus fraction now spans a real range -- 5th/25th/50th/75th/95th percentiles
  `0.2 / 0.4 / 0.6 / 0.8 / 1.0`, with 67.7% clearing 0.5. Before this fix, this was pinned near 1.0
  almost everywhere (consistent with the ~99.6% trade rate the old checkpoint produced). This is real,
  qualitatively different behavior, and it's evidence the fix accomplished something meaningful even
  though the KL number barely moved relative to its (higher) floor.
- CVAE's predicted edge sizes shrank dramatically in the same run -- among eligible decisions, predicted
  return ran `0.006% / 0.023% / 0.038% / 0.052% / 0.072%` (5th-95th percentile), entirely below the old
  shared `min_return_threshold=0.1%`. That threshold was tuned against the old (collapsed) checkpoint's
  edge scale and, combined with the new checkpoint's much smaller edges, was silently filtering out
  100% of CVAE's trades regardless of confidence -- not a sign the model broke, but a sign the fixed
  absolute threshold no longer matched CVAE's shifted output scale. See the "trade confidence" entry
  below for the fix (per-model thresholds).

Net read: the fix most likely *did* increase genuine sample diversity (the confidence-spread evidence is
fairly direct), but the KL metric itself is a weak/noisy read on that in this case -- the practical,
walk-forward-level signal (confidence distribution, trade selectivity) turned out to be more informative
than staring at the aggregate KL number. Not fully validated as "collapse solved," but no longer reads as
"the fix clearly did nothing" either. If a future retrain still shows CVAE's confidence pinned near 1.0
(not the spread seen here), that's the point to escalate to the architectural lever from the root-cause
paragraph above.

## "Trade confidence" redesign (alias: **trade confidence**) -- implemented (2026-08-07)

**Was contingent on the CVAE posterior-collapse fix above; proceeded once the walk-forward evidence (not
the KL number itself) showed CVAE's sample consensus had genuinely gained real spread** (see the
posterior-collapse entry above) -- confirming the underlying signal was worth preserving as its own gate
rather than something to drop.

**Implementing this surfaced a second, unrelated-to-collapse bug**: the shared `min_return_threshold=0.1%`
(same absolute value for both models) had been tuned against the old, collapsed CVAE checkpoint's edge
scale. The new checkpoint's predicted edges shrank to `0.006%-0.072%` (5th-95th percentile) -- entirely
below that threshold -- so CVAE was trading on exactly 0 of 2397 decisions even though its confidence
signal looked healthy. PatchTST's edges, by contrast, run `0.046%-0.51%` -- roughly 7x larger. A single
shared absolute threshold was never going to fit two models with different (and apparently driftable)
output scales. Fixed by giving each model its own `--patchtst-min-return-threshold` (kept at 0.001) and
`--cvae-min-return-threshold` (new default 0.0002, chosen near CVAE's own 25th percentile on this
checkpoint -- re-check after any retrain that might shift the scale again). With both fixes together
(quality-gate redesign below + per-model return thresholds), the same checkpoint went from 0 trades to a
sane, non-degenerate 663/1072 (61.8%, 86.7% win rate, 86.4% take-profit rate) -- PatchTST similarly moved
to 460/1477 (31.1%, 63.5% win rate), since dropping its old online-ranking gate (see below) also made it
noticeably more permissive than before.

**The problem being addressed**: the walk-forward backtest used to gate each trade on two things --
`predicted_return >= min_return_threshold` (mechanical, same formula for both models), and a "confidence"
score that is defined completely differently per model and squeezed onto the same 0-1 scale as if they
were comparable (PatchTST: a rank of predicted-move magnitude against an online, expanding history of its
own prior decisions; CVAE: fraction of its *k* sampled draws that agree on an upward move). A "0.7" from
one model isn't really the same kind of number as a "0.7" from the other, and PatchTST's version is also
noisier early in each walk (thin/empty reference to rank against).

**Replacement implemented**: the shared "confidence" concept is gone. The return-size gate is now
per-model too (see above), and the fuzzy shared score is now two separately-named, separately-thresholded
"quality gates," one per model, with no claim that they're comparable to each other:
- **PatchTST**: a plain boolean (`make_patchtst_predict_fn`) -- do all 3 predicted horizon bars agree on
  the "up" direction? No score, no threshold, no dependence on walk history. This *deleted* the old
  online-ranking mechanism (`patchtst_walk_forward_confidence` and its mutable, growing reference list)
  entirely -- a real simplification of the code independent of anything about CVAE, and it removes the
  "noisier early in the walk" asymmetry the two gates used to have.
- **CVAE**: kept the existing sample-consensus fraction (`cvae_confidence_scores`, unchanged), thresholded
  by its own dedicated knob (`--cvae-consensus-threshold`, default 0.5) instead of a shared
  `--confidence-threshold` pretending to mean the same thing for both models.

A trade requires `eligible AND meets_return_threshold AND` (coherent, for PatchTST | consensus clears its
own threshold, for CVAE) -- see `classify_walk_forward_decision`'s `passes_quality_gate` parameter. The
5-case taxonomy (`no_trade`/`skipped`/3 trade outcomes) didn't change -- `skipped` is now concretely
"cleared the return bar but failed its own model's quality gate," defined per model, instead of
"confidence too low" against a shared scale.

**Why not just drop the model-specific gate for both and go purely mechanical** (return-size threshold
only, no secondary gate at all): that was considered and rejected for now, specifically because CVAE's
sample agreement is real information a single-point forecaster like PatchTST structurally cannot produce,
and it's the whole thesis this project argues for CVAE over PatchTST in the first place (see Motivation in
`v1.md`). Dropping it entirely would turn CVAE into "just another quantile forecaster" for trading
purposes, even though it would still nominally generate *k* samples. Keeping a model-specific gate (even
at the cost of the two models no longer being directly comparable on one shared "confidence" axis) keeps
that signal load-bearing in the backtest, not just visible in the diagnostic spread table.

**Known tradeoff to accept, not a blocker**: this makes the two models' selectivity asymmetric in a new
way -- CVAE keeps a continuous, tunable threshold; PatchTST's gate is a fixed binary with nothing to dial
up or down beyond "on." A single "how selective is each model's own gate" headline comparison becomes
awkward as a result. Considered acceptable since the two gates were never really comparable under the old
design either -- this redesign is honest about that instead of implying otherwise via a shared 0-1 scale.

**Update (2026-08-07): CVAE's gate turned out not just unreliable but actively harmful, and was removed
entirely.** The paragraph above ("why not drop the model-specific gate entirely") argued CVAE's sample
agreement was too valuable to give up without testing -- so it got tested directly, and the case for
keeping it didn't survive contact with the data:
- Bucketing CVAE's real trades by consensus level found *unanimous* (5/5 samples agree) was the
  **worst**-performing bucket (84.8% win rate, avg return -0.018%), not the best -- backwards from what
  "confidence" is supposed to mean. `[0.6, 0.8, 1.0]` were the only achievable levels at `k=5` (consensus
  is a coarse 6-value statistic at this sample count), and none of them showed a clean "higher = better"
  relationship.
- Winners and losers had statistically indistinguishable consensus distributions (mean 0.746 vs. 0.750,
  identical median 0.8; 10% of losing trades happened at full unanimous agreement). Consensus could not
  tell, in advance, which trades were about to go bad.
- Directly A/B tested on the same checkpoint: removing the gate entirely (keep only eligibility + the
  per-model return threshold) took CVAE from 663 trades / 86.7% win rate / **-1.67%** total return to 697
  trades / **89.0%** win rate / **+8.60%** total return. Win rate went *up*, not down, when the gate was
  removed -- the ~34 additional decisions it had been excluding (0/5, 1/5, or 2/5 agreement) were good
  trades on average, not bad ones.

CVAE's quality gate (`passes_quality_gate` in `make_cvae_predict_fn`) is now always `True` --
`cvae_confidence_scores` was deleted along with `--cvae-consensus-threshold`. Only eligibility and
`--cvae-min-return-threshold` decide whether a CVAE window trades. PatchTST keeps its boolean coherence
gate -- that one hasn't been tested for the same failure mode, and is a different kind of check (a
hard consistency constraint on the prediction itself, not a soft agreement-based confidence score), so
there's less a priori reason to suspect it has the same problem, but it's untested, not proven safe.

**This doesn't necessarily kill the "generative uncertainty as an edge" thesis, just this specific
implementation of it.** `k=5` produces only 6 distinct consensus values -- a very coarse instrument to
expect real discriminating power out of. Before concluding CVAE's sample-level uncertainty carries no
usable signal at all, worth trying a much larger `--num-samples` (e.g. 15-25) to get a finer-grained
consensus score and re-running the same bucket/A-B analysis -- if it's still flat/inverted at higher *k*,
that's much stronger evidence the signal genuinely isn't there (rather than just being too coarse to see).

**Also tested and NOT adopted**: swapping CVAE's take-profit formula from `exit_price_from_components`
(mean of the 3 bars' 6 open/close values) to `max_close_from_components` (PatchTST's more aggressive
max-of-3-closes) -- bigger individual wins (+0.114% vs. +0.075% avg) but a lower take-profit rate (80.8%
vs. 86.4%) pushes more trades to expiry, netting worse overall (+3.03% with the gate / +6.95% without it,
both below the mean-target-no-gate result above). The mean-based target plus no gate is the best
configuration found so far. Caveat that applies to every number in this section: all tested against one
checkpoint over one fixed test period -- real risk of overfitting to this specific window given how many
configurations have now been tried against it.

## Fixed, volatility-calibrated stop-loss -- explored ad hoc, not implemented

**Motivation**: after finding CVAE's win rate and take-profit rate were both high but total return was
still negative (663 trades, 86.7% win rate, 86.4% take-profit rate, -1.67% total return -- see the "trade
confidence" entry above for the baseline this was measured against, before the gate was removed), the
working theory was a small number of large losing trades dragging the average down, not the win/loss split
itself. A hard stop-loss was tried as a mitigation, distinct from the mirrored 1:1 stop-loss already
covered in `v1.md`'s "why there's no stop-loss" section (that one was sized *off the take-profit target*
and found to backfire by triggering on ordinary intrabar noise; this one is sized as a fixed level, off the
instrument's own volatility, independent of whatever the take-profit happens to be).

**Calibration**: rather than pick a stop level arbitrarily, it was checked against `train_exit_return_bound`
-- the empirical p99 of `|anchored log return|` over the training data (~1.9%) -- as a sanity bound for
what a "big" 3-bar move actually looks like for hourly SPY. A 2% stop sits just outside that p99, i.e. it
should trigger on genuine tail moves, not on everyday noise.

**Result (tested against the then-current gated 663-trade CVAE baseline, pessimistic same-bar tie-break --
stop wins over take-profit on same-bar overlap)**:
- 2% stop-loss: total return improved from -1.67% to -0.37%. Directionally consistent with the "a few big
  losses are dragging the average down" theory.
- 1% and 0.5% stops: both backfired badly (-5.37% and -15.41% respectively) -- tight enough to trigger
  routinely on ordinary intrabar noise rather than only on genuine adverse moves, the same failure mode
  that killed the mirrored 1:1 version.

**Status: not implemented in the shipped backtest, and not re-tested since.** This was run via an ad hoc
diagnostic script, before CVAE's quality gate was found net-harmful and removed (see the "trade confidence"
entry above). Removing that gate alone already flipped CVAE's total return from -1.67% to +8.60% without
any stop-loss at all, which somewhat reduces the urgency here -- but the underlying tail-risk concern (some
single trade losing several percent) is orthogonal to the gate question and could still matter in a
different sample or period. Before adopting a 2% (or any) fixed stop as part of the real strategy: re-run
this same 2%/1%/0.5% sweep against the current no-gate baseline to confirm the finding still holds, then
implement it as an actual bracket order in `take_profit_exit()` (or a sibling function) rather than a
one-off script, with its own test coverage.

## Middle-masking ablation

Deployment always masks the rightmost 3 bars, and v1 trains that way too. Open question
worth a future experiment: train a second sub-model where the 3-bar mask is placed at a
random interior position during training instead of always at the end, and compare it
against the rightmost-only sub-model on the same rightmost-masked test set. Interesting
result either way — if interior-masking training doesn't help or hurts, that's informative
about whether the model is learning general context representations or just memorizing
"the end is always missing."
