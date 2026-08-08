# Proposal: is CVAE's collapse a modeling problem, or a data/frequency problem?

*Status: `steven/probe_daily_cvae.py` implements the experiment described in "What's actually
being tested" below. The rest of this doc (yfinance/1993, intraday-aggregated daily features,
the classical-baseline design) remains a proposal, not yet built.*

## The question

Four independent fixes in a row — the `price_scale` loss-scale correction, the
`decoder_ctx_dim` architectural bottleneck, and the auxiliary `w_direction` classification
loss (plus the pre-fix `2c4ad99` baseline, once tested the same way) — have all converged on
the same finding: CVAE's predicted direction correlates with a simple trend signal at a level
indistinguishable from noise. PatchTST's own trend correlation was similarly weak
(r=-0.137). Every fix so far targeted the *model* (loss scale, architecture, loss shape).
None questioned the *input*: is 3-bar-ahead direction from a 70-bar hourly SPY context simply
close to unpredictable at this sampling frequency, no matter what the model does with it?

This doesn't contradict "SPY is a reliable investment." Weak-form market efficiency
specifically predicts that short-horizon price *changes* are close to unpredictable from past
price/volume history alone, while the same asset can still have a reliably positive long-run
drift (the equity risk premium). "Reliable" and "short-horizon-predictable-from-its-own-price-history"
are different claims, and there's real market-microstructure literature suggesting shorter
sampling intervals have a *worse* signal-to-noise ratio (bid-ask bounce and temporary
liquidity effects make up a bigger share of an hourly return than of a daily one) — which
lines up with everything measured so far.

## What's already sitting unused

Before proposing anything, I checked what's actually available in the repo right now:

- **`data/processed/features.parquet`** (repo root, committed, not gitignored): 25,241 rows,
  2011–2025, hourly, with a much richer feature set than `steven/src/data_pipeline.py`
  currently uses — VIX (via the VIXY ETF proxy), RSI-14, MACD, Bollinger Bands, rolling
  realized volatility, volume ratio. Built by the project's "Data" branch with a proper
  walk-forward split discipline. `steven/`'s CVAE/PatchTST only sees 4 raw price components +
  volume today — none of this is wired in.
  - This is the safer version of "should we denoise the data": literally smoothing the
    price/return *target* is risky (most smoothing filters are non-causal unless carefully
    restricted, and a smoothed value used as input or label at time *t* can leak future
    information into a backtest in a way that looks like skill but evaporates in production).
    Feeding *already-causal* volatility/momentum/VIX-regime context is the safe version of
    the same idea — more informative conditioning, without touching what the model is asked
    to reconstruct.
- **Daily bars don't require new data collection to test at the current 2010–2025 range** —
  `steven/data/spy_ohlcv_1h.parquet` can be resampled to daily OHLCV directly. But testing the
  frequency hypothesis *properly* benefits from more history than that range gives (see below).

## How much daily data can we actually get, and is it enough?

Computed with `pd.bdate_range` + a 252.5-trading-day/year estimate:

| Range | Years | Est. trading days |
|---|---|---|
| SPY inception (1993-01-29) to 2025-05-30, via yfinance | 32.3 | **~8,164** |
| Current hourly range (2010-2025) resampled to daily | 15.4 | ~3,889 |
| **Current hourly, actual (what CVAE/PatchTST train on today)** | 15.4 | **27,007** |

Applying the project's existing split proportions (~83% train / ~7% val / ~10% test) to the
1993–2025 daily range gives roughly **6,200–6,800 train / 560–1,060 val / ~790–800 test**
daily bars — a **~3.3x reduction** in raw training rows versus today.

**This is a real trade-off, not a free upgrade, and it complicates daily bars as a clean test
of the frequency hypothesis:**

- `train_cvae.py`'s fixed regimen (20,000 sampled windows/epoch × 30 epochs) against a
  ~21,000-row hourly train pool means each row is seen **~29 times** over a training run.
  Against a ~6,500-row daily pool with the *same* regimen, each row would be seen **~90
  times** — a real jump in repeated-exposure/memorization risk that would need deliberate
  retuning (fewer epochs, more regularization), not a drop-in data swap.
- The window-sampler's "14–70 bars" context range would mean 3 weeks-to-14 weeks of calendar
  time at daily granularity, not the current 2 hours-to-10 days — a different context scope
  that needs redesigning, not just resampling.
- ~560–1,060 daily val bars (~2.2–4.2 years) is thin for the kind of val-based
  hyperparameter selection (context/horizon) the broader project's methodology relies on.
- This matches a known real tension in quant ML: higher-frequency data is often *preferred*
  for deep models specifically because they're data-hungry, even though each observation is
  noisier. Daily-only, single-asset SPY history is nowhere near "big data" scale for a neural
  generative model, inception-to-now or not.
- **Practical note**: `yfinance` is free and does reach back to SPY's 1993 inception, but it's
  an unofficial Yahoo Finance scraper (no key, no official support) with a history of breaking
  when Yahoo changes its endpoints. During this analysis, a direct probe of Yahoo's endpoint
  from this environment returned HTTP 429 (rate-limited but reachable) — internet access
  works, but `yfinance` itself wasn't yet installed or test-run, so an actual successful pull
  still needs to be confirmed as a first implementation step.

**Net honest read**: switching straight to single-asset daily SPY bars introduces a *second*
confound (much less training data) on top of the frequency change we actually want to
isolate. If a full daily-bar CVAE/PatchTST port doesn't improve results, we won't cleanly know
whether that's because daily direction is *still* unpredictable, or because the model is now
data-starved.

## Row count isn't the whole story — statistical power is

Comparing row counts alone doesn't answer "is it enough," because it doesn't say enough *for
what*. The right frame is **statistical power to detect the effect we're actually looking
for**. A correlation estimate's standard error shrinks roughly as `1/sqrt(N)`, so the ability
to detect a given true signal scales with `true_correlation × sqrt(N)`. Going from hourly to
daily costs ~3.3x in N (today's ~21,000 hourly train rows vs. ~6,500 daily), which costs
`sqrt(3.3) ≈ 1.8x` in detection power. So daily bars only come out *ahead* on power if the
true underlying correlation is more than **~1.8x stronger** at daily granularity than
hourly — plausible given the microstructure-noise story, but not something we know in advance,
and the hourly→daily frequency jump here is much smaller than what the microstructure
literature usually studies (tick/minute data vs. daily). This is an empirical question, not
one more arithmetic can resolve — which is the actual argument for building something and
looking, rather than reasoning about it further.

Two consequences for how any daily experiment (classical probe or the real model) should be
evaluated, given this:

1. **Maximize pooled out-of-sample observations, not a single held-out tail.** An
   expanding-window walk-forward (train through year N, evaluate year N+1 out-of-sample,
   extend training, repeat) pools far more out-of-sample observations than a single fixed test
   tail — meaningfully more power for the same underlying data.
2. **Treat a null result as ambiguous, not conclusive.** A null result is consistent with "no
   real signal" *or* "signal exists but there isn't enough power to see it at this N" — worth
   stating explicitly rather than reading a null as a clean "daily doesn't work either."

## What's actually being tested (built): the real CVAE, not a classical proxy

Training this project's CVAE is cheap — full 30-epoch runs complete in under two minutes on an
A100 (see `colab_train.ipynb`'s training logs). That changes the calculus versus the classical
LogisticRegression probe designed above: rather than building a separate simplified model as a
cheap stand-in, it's cheaper to just test the real thing directly. `steven/probe_daily_cvae.py`
does this:

- **Data**: resamples the *existing* `steven/data/spy_ohlcv_1h.parquet` (2010–2025, already on
  disk, no new collection) to daily OHLCV — plain typical aggregation (open=first, high=max,
  low=min, close=last, volume=sum). This is the smaller, ~3,900-bar daily range from the table
  above, not the full 1993–2025 yfinance pull — that longer-history option remains a documented
  but unbuilt next step if this test finds something worth pursuing further.
- **Model**: CVAE only, using `configs/cvae.yaml`'s exact current recipe (the
  `price_scale`/`decoder_ctx_dim`/`w_direction` combination), so this is a single-variable
  test — only the data/context length changes, not the modeling approach already investigated
  at length in `cvae_direction_collapse.md`. PatchTST is excluded: its architecture
  (`src/models/patchtst.py`) hardcodes `N_PATCHES`/`PATCH_LEN`/`MAX_CONTEXT` via
  `from src.data_pipeline import ...` at import time, which would need real code changes
  (parametrized constructor args) to run at a different context length — out of scope for this
  probe.
- **Context sweep**: 5/10/15/20 daily bars (1–4 trading weeks), trained as one checkpoint
  (exposed to all four lengths during training) and evaluated separately at each fixed length,
  mirroring how `WALK_FORWARD_CTX_BARS` already works independently of training-time window
  sampling for the hourly pipeline.
- **Test period**: 2023-01-01 to 2025-05-30 (one more year than the hourly project's
  2024–2025), val shrunk to 2022 alone to make room, train unchanged through 2021.
- **Benchmark**: buy-and-hold and the naive-periodic benchmark, computed fresh per context
  length with the same entry-anchoring convention `evaluate.py` already uses (anchored to the
  walk-forward's own first feasible decision point, not the test split's literal first bar) —
  so every number for a given context length covers an identical calendar span.
- **Known caveat, reported rather than silently corrected**: this reuses `configs/cvae.yaml`'s
  training regimen (20,000 sampled windows/epoch × 30 epochs) unchanged against a much smaller
  daily train pool (~2,900 rows for 2010–2021 vs. ~21,000 hourly) — a real jump in
  repeated-exposure/memorization risk (see the row-count section above). Deliberately not
  rebalanced here to keep this a single-variable test (only the data/context differs); the
  script reports the actual exposure ratio so results can be read with that in mind.

## Recommendation: a cheap probe before a full pipeline port

Rather than committing to a full CVAE/PatchTST port to daily granularity (a genuine multi-file
rebuild: new data collection, redesigned context/horizon semantics, retuned training regimen)
on a guess, test for a real daily-direction signal first with a small, fast, disposable
script. If a classical model this simple can't find a significant edge, a small CVAE/PatchTST
almost certainly won't either — that's real evidence the problem isn't sampling frequency, and
effort should redirect to the VIX/technical-context enrichment path instead. If it *does* find
something statistically significant and stable, that justifies the bigger daily-bar
engineering effort (likely combined with multi-asset training, given the row-count concern
above).

### Proposed script (not built): `steven/probe_daily_signal.py`

Superseded for now by `steven/probe_daily_cvae.py` (see "What's actually being tested" above)
since testing the real model turned out to be cheaper than building a classical proxy for it.
Left here as a documented fallback: if the real-CVAE test is inconclusive or its
smaller-daily-pool caveat makes results hard to trust, this simpler, faster, and much
higher-powered (via yfinance's full 1993–2025 history) classical baseline is the next thing to
try instead.

A single, self-contained, disposable analysis script — not a new production pipeline, not
integrated with `data_pipeline.py`'s CVAE/PatchTST-specific windowing machinery (that would be
over-engineering for a go/no-go probe).

1. **Data**: pull SPY daily OHLCV via `yfinance` from 1993-01-01 through 2025-05-30 (the same
   fixed "current" test-end date used throughout this project), cache to
   `steven/data/spy_daily_probe.parquet` so repeated runs don't re-hit the API. First
   implementation step: confirm an actual `yfinance` pull succeeds (its bulk
   `download()`/`Ticker.history()` calls may not hit the same rate limit as the raw endpoint
   probe above), with basic retry/backoff if it does.
2. **Features**: reuse the `ta` library the same way root `scripts/data/process_utils.py`
   already does for the hourly pipeline, just with daily-appropriate windows instead of
   porting that hourly-specific code directly — lagged daily returns (1/2/3/5/10-day), rolling
   realized vol (10d/20d), `ta.momentum.RSIIndicator` (14-day — this is actually the
   *conventional* RSI-14 definition; the existing hourly pipeline is the unusual adaptation),
   `ta.trend.MACD`, `ta.volatility.BollingerBands`.
3. **Target**: sign of the 3-trading-day-ahead close-to-close return — matching `HORIZON=3`
   used throughout `data_pipeline.py`/`evaluate.py`, so this is apples-to-apples with what
   CVAE/PatchTST already predict, just at daily granularity.
4. **Model**: `sklearn.linear_model.LogisticRegression` — simplest, fastest classical
   baseline.
5. **Evaluation**: expanding-window walk-forward with ~5 sequential yearly test folds (no
   shuffling, matching this project's split discipline elsewhere) — retrain on all prior data
   before each fold. Report, per fold and pooled:
   - accuracy vs. majority-class baseline
   - a formal significance test against chance (binomial test vs. 50%, and/or a
     correlation significance threshold: roughly |r| > 0.11–0.20 depending on fold size)
   - explicit stability check: does any effect hold across multiple folds, or only one —
     applying a "stable across resamples, not a single lucky number" bar, since every
     correlation measured on CVAE/PatchTST so far (-0.03, -0.03, +0.10, +0.146, +0.15) has
     failed to clear this same kind of significance threshold.
6. Print a clear pass/fail summary against that bar so the go/no-go decision on porting
   CVAE/PatchTST to daily bars isn't re-litigated after the fact.

**New dependency**: `steven/requirements-probe.txt` (`yfinance`, `scikit-learn`, `ta`) — kept
separate from `requirements-model.txt` since the main CVAE/PatchTST pipeline doesn't need any
of these.

### Verification

- Run `python steven/probe_daily_signal.py` end-to-end; confirm it downloads, caches, trains,
  and prints a report without manual intervention.
- Sanity-check the report's numbers by hand for at least one fold (recompute accuracy/binomial
  p-value independently) before trusting the printed verdict.
- This is a probe, not shipped code — no changes to `configs/`, `src/`, or `tests/` unless the
  result justifies moving forward with a real daily-bar pipeline, which would be a separate,
  later proposal.

## Options considered but not chosen (for now)

1. **Multi-asset daily training** (SPY + similarly-behaved large index ETFs) to restore a
   comparable row count — bigger scope, adds a cross-asset-generalization assumption. Worth
   revisiting if the probe finds real signal but row count is still limiting.
2. **Commit to daily bars anyway**, redesigning the training regimen for a small-data regime
   directly — skips the cheap validation step this proposal recommends.
3. **Drop daily bars, do VIX/technical-context enrichment of the existing hourly pipeline
   instead** — sidesteps the data-scarcity risk entirely, but doesn't test the frequency
   hypothesis at all. Still worth doing in parallel or afterward regardless of the probe's
   outcome, since it's low-risk and uses data that already exists.
