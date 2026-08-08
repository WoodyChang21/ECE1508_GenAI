"""Adds causal momentum/trend signals to the hourly SPY pipeline, as extra input channels --
a much smaller, more targeted version of the VIX/RSI/MACD/Bollinger-Band enrichment
daily_signal_probe.md proposed (see that doc's "What's already sitting unused"), after five
independent modeling-side fixes (price_scale, decoder_ctx_dim, w_direction, daily-bars
resampling, and a cross-run stability check) all converged on CVAE finding no learnable
direction signal in the raw 5 price/volume features alone.

Hourly, not daily: this stays on the same 70-bar/3-bar-horizon hourly pipeline the rest of
the project (and v1.md's results) is built on -- the daily-bars probe tested a different
hypothesis (sampling frequency) and adding momentum features there would confound two
untested variables at once. Everything below is computed on the *hourly* close series, so
there's no daily-vs-intraday question to resolve.

EMA9/EMA21 (added first, see cvae_direction_collapse.md's "Momentum-feature enrichment"
section -- also didn't fix the collapse, but left in place: harmless, and RSI below is a
genuinely different signal, not more of the same one) -- two log-ratios, on the same small
scale as the existing open_ret/body_ret log-return features, both z-scored the same way
apply_normalize already treats log_volume_norm:
- `ema_cross_norm`: log(ema9/ema21) -- the classic golden-cross/death-cross signal. Sign is
  trend direction, magnitude is how far the fast/slow EMAs have diverged.
- `trend_position_norm`: log(close/ema21) -- where price currently sits relative to the
  slower trend line, which the crossover alone doesn't capture (price can be above a
  still-rising ema21 well after the actual cross happened).

RSI-14 (added second, deliberately skipping MACD -- MACD is itself an EMA-difference
construction, essentially the same category of signal as ema_cross above, which already
tested null; RSI is a genuinely different one, built from the ratio of average gains to
average losses rather than a trend slope, so it's a real new piece of information rather
than a redundant one):
- `rsi_norm`: Wilder's RSI-14 (the conventional definition), z-scored like the others. RSI's
  raw range (0-100) would otherwise dominate the ConvEncoder's input scale relative to the
  ~0.001-0.01-scale log-return channels -- the same category of scale-mismatch bug that
  caused the original volume/price loss imbalance, just on the input side this time, so it
  gets the same z-scoring treatment rather than being fed in raw.

VIX (added third, see add_vix_feature) -- unlike everything above, this is a genuine
market-derived sentiment/fear proxy (implied volatility priced into S&P 500 options by other
market participants), not a transform of SPY's own price history. Pulled separately via
collect_vix_yfinance.py (VIX isn't a column in the SPY OHLCV parquet) and merged in by date.
Shifted by one trading day before merging, so each row gets the PRIOR day's VIX close --
literally "what was happening before today's market open," matching the user's own framing,
not today's own VIX (which settles alongside today's SPY session and wouldn't be knowable
before today's open).

Leakage note: every derived feature above is a function of the close price, which is exactly
what CVAE is being asked to predict at the 3 horizon bars -- so all of them MUST be zeroed at
horizon positions in masked_tensor, the same way open_ret/body_ret/wicks/volume already are
(see build_window's PRICE_VOL_IDX masking). This module extends that masked index set
accordingly (MOMENTUM_PRICE_VOL_IDX below) rather than leaving the new columns unmasked,
which would smuggle a hint about the future close straight into the model's "masked" input.

Column layout: the new columns are inserted *after* log_volume_norm (index 4) and *before*
time_gap_norm/day_bar_index_norm (calendar features, never masked, always known regardless
of price outcome). This preserves every existing hardcoded index into the first 5 columns
elsewhere in the codebase -- build_window's `y` construction (`horizon_feats_true[:, :4]` for
price, `[:, 4]` for volume) and train_cvae.py's price_scale (`feat[train_lo:train_hi, :4]`)
both only look at indices 0-4, untouched by anything appended after them.

Implementation: reuses data_pipeline.py's load_and_validate/compute_raw_features/
chronological_bounds/fit_normalize/apply_normalize UNCHANGED, then monkey-patches its
module-level FEATURE_COLS/N_FEATURE_CHANNELS/N_CHANNELS/PRICE_VOL_IDX constants -- safe here
for the same reason probe_daily_cvae.py's context-length patching is safe: every downstream
function this touches (extract_arrays, build_window, WindowSampler, WindowDataset) reads
those names as free variables inside its own body at call time, not as default-argument
values bound at def time. This is CVAE-only, like the daily-bars probe: src/models/
patchtst.py imports N_PATCHES/PATCH_LEN from data_pipeline at import time (an independent
copy patching afterward can't reach), and its nn.Parameter shapes are fixed at construction
time regardless.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import src.data_pipeline as dp

EMA_FAST = 9
EMA_SLOW = 21
RSI_PERIOD = 14

_BASE_FEATURE_COLS = [
    "open_ret", "body_ret", "upper_wick", "lower_wick", "log_volume_norm",
    "ema_cross_norm", "trend_position_norm", "rsi_norm",
]
_CALENDAR_COLS = ["time_gap_norm", "day_bar_index_norm"]  # never masked -- always known

# Default (VIX included) -- the committed-going-forward layout (see cvae_direction_collapse.md's
# "Adding VIX" section). MOMENTUM_FEATURE_COLS_NO_VIX exists ONLY for the controlled
# with/without-VIX comparison that section's own findings called for (does VIX's regression
# of the daily correlation number replicate, or was that number never stable to begin with?)
# -- not a general-purpose toggle for future feature work.
MOMENTUM_FEATURE_COLS = _BASE_FEATURE_COLS + ["vix_norm"] + _CALENDAR_COLS
MOMENTUM_FEATURE_COLS_NO_VIX = _BASE_FEATURE_COLS + _CALENDAR_COLS

MOMENTUM_N_FEATURE_CHANNELS = len(MOMENTUM_FEATURE_COLS)  # 11
MOMENTUM_N_CHANNELS = MOMENTUM_N_FEATURE_CHANNELS + 2  # + padding_mask, target_mask = 13
# Everything except the trailing two calendar features gets masked at horizon positions --
# see module docstring's leakage note. Original PRICE_VOL_IDX was [0,1,2,3,4]; ema_cross/
# trend_position/rsi/vix land at [5,6,7,8] given the column layout above, so they join the
# list -- see add_vix_feature's docstring for why VIX joins this group despite being external.
MOMENTUM_PRICE_VOL_IDX = [0, 1, 2, 3, 4, 5, 6, 7, 8]


@dataclass
class MomentumStats:
    ema_cross_mean: float
    ema_cross_std: float
    trend_position_mean: float
    trend_position_std: float
    rsi_mean: float
    rsi_std: float
    vix_mean: float | None = None
    vix_std: float | None = None


def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds raw (unnormalized) ema_cross/trend_position/rsi columns. `.ewm(adjust=False)` is
    inherently causal -- each row's value depends only on itself and prior rows, so computing
    these over the whole chronological series before any train/val/test split can't leak
    future information across the split boundary (same reasoning compute_raw_features already
    relies on for open_ret/body_ret/wicks).

    RSI is Wilder's original formulation: average gain/loss smoothed with alpha=1/RSI_PERIOD
    (equivalent to Wilder's smoothing method), not a plain SMA -- the conventional definition.
    `avg_loss == 0` (a strict, unbroken uptrend over the whole lookback) would otherwise divide
    by zero; RSI's correct limiting value there is 100 (no losses at all), handled explicitly
    rather than left to produce inf/NaN.

    Burn-in note: the very first ~RSI_PERIOD/EMA_SLOW rows have not-yet-converged values
    (`ewm` starts from the first observation with full weight on it, and the first `diff()` is
    NaN by construction). That's <0.1% of the ~27,000-row series and lands at its very start,
    years before the earliest train window this project actually samples from with any real
    frequency -- not worth a special-cased warmup drop, except backfilling RSI's single
    leading NaN (from the first `diff()`) with the neutral midpoint so it doesn't propagate
    NaN through every downstream normalize/window/train step."""
    df = df.copy()
    ema9 = df["close"].ewm(span=EMA_FAST, adjust=False).mean()
    ema21 = df["close"].ewm(span=EMA_SLOW, adjust=False).mean()
    df["ema_cross"] = np.log(ema9 / ema21)
    df["trend_position"] = np.log(df["close"] / ema21)

    delta = df["close"].diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / RSI_PERIOD, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / RSI_PERIOD, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    df["rsi"] = rsi.where(avg_loss != 0.0, 100.0).fillna(50.0)
    return df


def add_vix_feature(df: pd.DataFrame, vix_path: Path | str) -> pd.DataFrame:
    """Merges in VIX (see module docstring) via merge_asof(direction="backward") rather than
    an exact-date join, so any SPY/VIX calendar mismatch (a handful of holiday/data-gap edge
    cases) falls back to the most recent known VIX value instead of producing a missing row --
    the two trade on the same exchange calendar, so this is a robustness measure, not
    something expected to fire often. `df` must already be sorted by `datetime` (true of
    anything that's gone through load_and_validate) since merge_asof requires its join key
    sorted.

    Horizon masking: even though VIX is external, not derived from SPY's own close, it still
    joins the same masked-at-horizon group (MOMENTUM_PRICE_VOL_IDX) as every other momentum
    feature. The horizon's first day's shifted VIX value happens to already be visible in the
    last context row (it's yesterday's close relative to a still-future day), but days 2/3's
    would depend on VIX closes that haven't happened yet at decision time -- simpler and safer
    to mask the whole horizon uniformly, matching every other feature here, than special-case
    day 1.

    Burn-in: `.shift(1)` leaves the VIX series' own first row NaN (no prior day to shift from)
    -- backfilled from the next valid value, same treatment as RSI's single leading NaN,
    before merging (so it can never surface as a NaN feature value downstream)."""
    vix = pd.read_parquet(vix_path)[["datetime", "vix_close"]].sort_values("datetime").reset_index(drop=True)
    vix["vix_close"] = vix["vix_close"].shift(1).bfill()
    # astype match: the two parquet files' datetime columns can round-trip to different
    # sub-second units (us vs ms) depending on how each was written, which merge_asof treats
    # as genuinely incompatible dtypes rather than silently coercing.
    vix["date"] = vix["datetime"].dt.normalize().astype("datetime64[ns]")

    df = df.copy()
    df["date"] = df["datetime"].dt.normalize().astype("datetime64[ns]")
    merged = pd.merge_asof(df, vix[["date", "vix_close"]], on="date", direction="backward")
    return merged.drop(columns="date")


def fit_momentum_stats(df: pd.DataFrame, train_bounds: tuple[int, int], include_vix: bool = True) -> MomentumStats:
    lo, hi = train_bounds
    train_slice = df.iloc[lo:hi]
    return MomentumStats(
        ema_cross_mean=float(train_slice["ema_cross"].mean()),
        ema_cross_std=float(train_slice["ema_cross"].std()),
        trend_position_mean=float(train_slice["trend_position"].mean()),
        trend_position_std=float(train_slice["trend_position"].std()),
        rsi_mean=float(train_slice["rsi"].mean()),
        rsi_std=float(train_slice["rsi"].std()),
        vix_mean=float(train_slice["vix_close"].mean()) if include_vix else None,
        vix_std=float(train_slice["vix_close"].std()) if include_vix else None,
    )


def apply_momentum_normalize(df: pd.DataFrame, stats: MomentumStats, include_vix: bool = True) -> pd.DataFrame:
    df = df.copy()
    df["ema_cross_norm"] = (df["ema_cross"] - stats.ema_cross_mean) / stats.ema_cross_std
    df["trend_position_norm"] = (df["trend_position"] - stats.trend_position_mean) / stats.trend_position_std
    df["rsi_norm"] = (df["rsi"] - stats.rsi_mean) / stats.rsi_std
    if include_vix:
        df["vix_norm"] = (df["vix_close"] - stats.vix_mean) / stats.vix_std
    return df


def patch_momentum_constants(include_vix: bool = True) -> None:
    """Monkey-patches data_pipeline's module-level constants -- see module docstring for why
    this is safe for CVAE (and would NOT be for PatchTST)."""
    cols = MOMENTUM_FEATURE_COLS if include_vix else MOMENTUM_FEATURE_COLS_NO_VIX
    dp.FEATURE_COLS = cols
    dp.N_FEATURE_CHANNELS = len(cols)
    dp.N_CHANNELS = len(cols) + 2
    # mask everything except the trailing _CALENDAR_COLS, regardless of include_vix
    dp.PRICE_VOL_IDX = list(range(len(cols) - len(_CALENDAR_COLS)))


def build_momentum_dataset(
    path: Path | str, vix_path: Path | str | None = None, include_vix: bool = True,
) -> tuple[pd.DataFrame, dict[str, tuple[int, int]], dp.NormStats, MomentumStats]:
    """Momentum-enriched equivalent of data_pipeline.build_dataset. Patches data_pipeline's
    module constants as a side effect (see patch_momentum_constants) -- call this before any
    other data_pipeline function that reads FEATURE_COLS/N_FEATURE_CHANNELS/N_CHANNELS/
    PRICE_VOL_IDX (extract_arrays, build_window, WindowDataset, CVAEInpainting via
    train_cvae.py's normal import path all qualify). `vix_path` is the same
    vix_daily_yfinance.parquet regardless of whether `path` is hourly or daily SPY data --
    merge_asof handles the date alignment either way (see add_vix_feature). `include_vix`
    exists only for the controlled with/without-VIX comparison in
    cvae_direction_collapse.md's "Adding VIX" section -- VIX stays the default (True) going
    forward per that section's discussion; vix_path is required when True."""
    if include_vix and vix_path is None:
        raise ValueError("vix_path is required when include_vix=True")

    df = dp.load_and_validate(path)
    df = dp.compute_raw_features(df)
    df = add_momentum_features(df)
    if include_vix:
        df = add_vix_feature(df, vix_path)
    bounds = dp.chronological_bounds(df)
    stats = dp.fit_normalize(df, bounds["train"])
    df = dp.apply_normalize(df, stats)
    momentum_stats = fit_momentum_stats(df, bounds["train"], include_vix=include_vix)
    df = apply_momentum_normalize(df, momentum_stats, include_vix=include_vix)
    patch_momentum_constants(include_vix=include_vix)
    return df, bounds, stats, momentum_stats
