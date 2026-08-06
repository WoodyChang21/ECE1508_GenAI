# Backlog

Future experiments, not built in v1.

## PatchTST wick-based take-profit target

CVAE's take-profit is the 70th percentile of its *k* sampled exit prices (see `v1.md`'s Trading criteria
section) — it has a real distribution to pick a target from. PatchTST is a single deterministic point
forecast with no such distribution; its target moved once already (open/close average → max of its 3
predicted closes, via `max_close_from_components`), but that's now aggressive enough that its bracket
order essentially never resolves before expiry (well under 1% combined take-profit + stop-loss rate at
every confidence threshold, per the current backtest). Worth trying instead: set PatchTST's take-profit
near its own predicted candle *highs* (e.g. the max predicted high across the 3 horizon bars, minus a
small safety margin) rather than its predicted *closes* — this would actually use the wick predictions,
which are currently generated but thrown away by both `exit_price_from_components` and
`max_close_from_components`. Not obviously better or worse than the current max-close target without
comparing them on the same backtest — could land anywhere between "still too aggressive" and "a genuine
improvement," since predicted highs run above predicted closes but real wicks may not extend as far as
the model's guess either. Deliberately held off this pass since PatchTST is meant to stay a simple
benchmark, not necessarily mirror CVAE's approach.

## Middle-masking ablation

Deployment always masks the rightmost 3 bars, and v1 trains that way too. Open question
worth a future experiment: train a second sub-model where the 3-bar mask is placed at a
random interior position during training instead of always at the end, and compare it
against the rightmost-only sub-model on the same rightmost-masked test set. Interesting
result either way — if interior-masking training doesn't help or hurts, that's informative
about whether the model is learning general context representations or just memorizing
"the end is always missing."
