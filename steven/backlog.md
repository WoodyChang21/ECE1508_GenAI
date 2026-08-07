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

## CVAE posterior collapse (KL pinned at the free-bits floor)

CVAE's reported KL loss settles to exactly `z_dim * free_bits` (0.8 = 16 * 0.05) and stays there for the
rest of training, on every run so far (see `v1.md`'s "A training diagnostic" note under Loss) — every
latent dimension sitting exactly at the free-bits floor, which means the true, unclamped KL is at or
below it everywhere. That's posterior collapse: the recognition network isn't encoding meaningfully more
than the context-only prior already has, so `z` likely isn't carrying much real per-example signal.
Candidate next steps, roughly cheapest-first: (1) check whether reconstruction loss keeps improving
across epochs despite the collapsed KL -- if it does, the context-only path is still learning real
structure, just not via `z`, which narrows down where the problem is; (2) lower `free_bits` (currently
0.05) to see whether the floor itself is set too generously, letting the model coast without ever being
forced to use `z`; (3) slow down `kl_anneal_epochs` (currently 5 of 30) -- beta reaching 1.0 that early
may be pressuring the model to minimize KL before it's learned to rely on `z` for anything; (4) check
decoder capacity/context-conditioning strength -- if the decoder can already reconstruct well from
context alone, there's no loss pressure to ever use `z` regardless of the KL schedule.

## Middle-masking ablation

Deployment always masks the rightmost 3 bars, and v1 trains that way too. Open question
worth a future experiment: train a second sub-model where the 3-bar mask is placed at a
random interior position during training instead of always at the end, and compare it
against the rightmost-only sub-model on the same rightmost-masked test set. Interesting
result either way — if interior-masking training doesn't help or hurts, that's informative
about whether the model is learning general context representations or just memorizing
"the end is always missing."
