# Backlog

Future experiments, not built in v1.

## Middle-masking ablation

Deployment always masks the rightmost 3 bars, and v1 trains that way too. Open question
worth a future experiment: train a second sub-model where the 3-bar mask is placed at a
random interior position during training instead of always at the end, and compare it
against the rightmost-only sub-model on the same rightmost-masked test set. Interesting
result either way — if interior-masking training doesn't help or hurts, that's informative
about whether the model is learning general context representations or just memorizing
"the end is always missing."
