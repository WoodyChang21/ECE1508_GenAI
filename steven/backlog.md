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

## CVAE posterior collapse (KL pinned at the free-bits floor) -- fix attempted, needs a retrain to verify

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
now also returns `recon_loss` in its parts dict for this purpose. **The KL numbers above are from *before*
this fix**, so they should be treated as provisional -- worth retraining again with the corrected
selection criterion before drawing a final conclusion either way.

## "Trade confidence" redesign (alias: **trade confidence**) -- gated on the posterior-collapse retrain above

**Contingent on the CVAE posterior-collapse fix directly above.** Only worth pursuing once that retrain
confirms the reported KL actually settles above the new floor (`8 × 0.15 = 1.2`) instead of pinned exactly
at it. If collapse persists, CVAE's sample-agreement gate (see below) will still fire true on nearly every
decision no matter what threshold is picked, and this redesign wouldn't fix that on its own -- it isn't a
substitute for the model fix, it's the next thing to reconsider once the model fix is validated.

**The problem being addressed**: today's walk-forward backtest gates each trade on two things --
`predicted_return >= min_return_threshold` (mechanical, same formula for both models), and a "confidence"
score that is defined completely differently per model and squeezed onto the same 0-1 scale as if they
were comparable (PatchTST: a rank of predicted-move magnitude against an online, expanding history of its
own prior decisions; CVAE: fraction of its *k* sampled draws that agree on an upward move). A "0.7" from
one model isn't really the same kind of number as a "0.7" from the other, and PatchTST's version is also
noisier early in each walk (thin/empty reference to rank against).

**Proposed replacement**: drop the shared "confidence" concept entirely. Keep the mechanical return-size
gate as-is (shared formula, both models), and replace the fuzzy shared score with two separately-named,
separately-thresholded "quality gates," one per model, with no claim that they're comparable to each
other:
- **PatchTST**: a plain boolean -- do all 3 predicted horizon bars agree on the "up" direction? No score,
  no threshold, no dependence on walk history. This *deletes* the current online-ranking mechanism
  (`patchtst_walk_forward_confidence` and its mutable, growing reference list) entirely -- a real
  simplification of the code independent of anything about CVAE, and it removes the "noisier early in the
  walk" asymmetry between the two models' gates.
- **CVAE**: keep the existing sample-consensus fraction (`cvae_confidence_scores`, unchanged), thresholded
  by its own dedicated knob (e.g. `--cvae-consensus-threshold`, not a shared `--confidence-threshold`
  pretending to mean the same thing for both models).

A trade requires `eligible AND meets_return_threshold AND` (coherent, for PatchTST | consensus clears its
own threshold, for CVAE). The 5-case taxonomy (`no_trade`/`skipped`/3 trade outcomes) doesn't change --
`skipped` just becomes concretely "cleared the return bar but failed its own model's quality gate,"
defined per model, instead of "confidence too low" against a shared scale.

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

## Middle-masking ablation

Deployment always masks the rightmost 3 bars, and v1 trains that way too. Open question
worth a future experiment: train a second sub-model where the 3-bar mask is placed at a
random interior position during training instead of always at the end, and compare it
against the rightmost-only sub-model on the same rightmost-masked test set. Interesting
result either way — if interior-masking training doesn't help or hurts, that's informative
about whether the model is learning general context representations or just memorizing
"the end is always missing."
