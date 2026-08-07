"""Data pipeline for SPY 1H candle-completion.

Load + validate raw OHLCV, reparametrize into anchor-corrected log-return features,
chronological train/val/test splits, and a day-aligned window sampler that produces
fixed-length (73-slot) padded tensors for both the PatchTST benchmark and the CVAE
inpainting model. See docs/../plan for the full derivation of the feature formulas.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

RAW_COLUMNS = ["datetime", "open", "high", "low", "close", "volume"]

# Order matters: PRICE_VOL_IDX below assumes the first 5 are price/volume components.
FEATURE_COLS = [
    "open_ret",
    "body_ret",
    "upper_wick",
    "lower_wick",
    "log_volume_norm",
    "time_gap_norm",
    "day_bar_index_norm",
]
N_FEATURE_CHANNELS = len(FEATURE_COLS)  # 7
N_CHANNELS = N_FEATURE_CHANNELS + 2  # + padding_mask, target_mask = 9
PRICE_VOL_IDX = [0, 1, 2, 3, 4]  # zeroed at horizon positions in the masked tensor

CONTEXT_LENGTHS = [14, 21, 28, 35, 42, 49, 56, 63, 70]  # 2..10 trading days, step 7 (1 day = 7 bars)
MAX_CONTEXT = 70
HORIZON = 3

# Hard cap on predicted per-bar log-return components (open_ret, body_ret, wicks).
# Training data's most extreme single hourly bar ever seen is open_ret=0.116 (p99.9 is
# ~0.02-0.03); 0.15 is generous headroom above that, chosen to make runaway predictions
# (e.g. an undertrained head outputting |x| ~ 1-2, which exp() turns into 3-10x price
# moves) architecturally impossible rather than relying on training to avoid them.
MAX_LOG_RETURN = 0.15
TOTAL_LEN = MAX_CONTEXT + HORIZON  # 73
PATCH_LEN = 7
N_PATCHES = MAX_CONTEXT // PATCH_LEN  # 10

TRAIN_END = "2021-12-31"
VAL_START = "2022-01-01"
VAL_END = "2023-12-31"
TEST_START = "2024-01-01"
TEST_END = "2025-05-30"

DEFAULT_SEED = 42


# ---------------------------------------------------------------------------
# Load + validate
# ---------------------------------------------------------------------------


def load_and_validate(path: Path | str) -> pd.DataFrame:
    """Load the raw OHLCV parquet, sort, and validate structure. Never drops or fills
    anything here -- only reports what it finds."""
    df = pd.read_parquet(path)

    missing = set(RAW_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"missing expected columns: {missing}")
    df = df[RAW_COLUMNS].sort_values("datetime").reset_index(drop=True)

    if not np.issubdtype(df["datetime"].dtype, np.datetime64):
        raise TypeError(f"datetime column has dtype {df['datetime'].dtype}, expected datetime64")

    n_dupes = int(df["datetime"].duplicated().sum())
    if n_dupes:
        raise ValueError(f"{n_dupes} duplicate datetime rows found")
    if not df["datetime"].is_monotonic_increasing:
        raise ValueError("datetime column is not sorted increasing after sort_values")

    expected_times = {pd.Timestamp(f"2000-01-01 {h:02d}:30:00").time() for h in range(9, 16)}
    observed_times = set(df["datetime"].dt.time)
    if observed_times != expected_times:
        logger.warning(
            "Unexpected bar times found: %s (expected %s)",
            sorted(observed_times), sorted(expected_times),
        )

    gap_report = build_gap_report(df)
    logger.info(
        "Gap report: %d fully missing weekdays, %d short sessions (<7 bars)",
        gap_report["n_missing_days"], gap_report["n_short_days"],
    )
    return df


def build_gap_report(df: pd.DataFrame) -> dict:
    """Compare observed bars/day against the theoretical Mon-Fri calendar between the
    dataset's min and max date. Reports only -- never drops or fills."""
    dates = df["datetime"].dt.normalize()
    bars_per_day = df.groupby(dates).size()

    all_weekdays = pd.bdate_range(df["datetime"].min().normalize(), df["datetime"].max().normalize())
    bars_per_day = bars_per_day.reindex(all_weekdays, fill_value=0)

    missing_days = bars_per_day[bars_per_day == 0]
    short_days = bars_per_day[(bars_per_day > 0) & (bars_per_day < 7)]

    return {
        "n_missing_days": int(len(missing_days)),
        "n_short_days": int(len(short_days)),
        "missing_days": [str(d.date()) for d in missing_days.index],
        "short_days": {str(d.date()): int(c) for d, c in short_days.items()},
    }


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


def compute_raw_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds chained log-return features. Drops exactly one row: the very first bar of
    the whole series, which has no previous close to compute a return from."""
    df = df.copy()

    prev_close = df["close"].shift(1)
    time_gap_minutes = df["datetime"].diff().dt.total_seconds() / 60.0

    dropped_ts = df["datetime"].iloc[0]
    logger.info("Dropping first row (%s): no previous close to compute a return from", dropped_ts)

    df = df.iloc[1:].reset_index(drop=True)
    prev_close = prev_close.iloc[1:].reset_index(drop=True)
    time_gap_minutes = time_gap_minutes.iloc[1:].reset_index(drop=True)

    r_open = np.log(df["open"] / prev_close)
    r_high = np.log(df["high"] / prev_close)
    r_low = np.log(df["low"] / prev_close)
    r_close = np.log(df["close"] / prev_close)

    df["open_ret"] = r_open
    df["body_ret"] = r_close - r_open
    df["upper_wick"] = r_high - np.maximum(r_open, r_close)
    df["lower_wick"] = np.minimum(r_open, r_close) - r_low

    n_neg_wick = int((df["upper_wick"] < 0).sum() + (df["lower_wick"] < 0).sum())
    if n_neg_wick:
        logger.warning("%d negative wick values found (OHLC invariant violation in raw data)", n_neg_wick)

    df["log_volume"] = np.log1p(df["volume"].astype(np.float64))
    df["log_time_gap"] = np.log1p(time_gap_minutes)

    day_bar_index = df.groupby(df["datetime"].dt.normalize()).cumcount()
    df["day_bar_index_norm"] = day_bar_index / 6.0

    return df


@dataclass
class NormStats:
    log_volume_mean: float
    log_volume_std: float
    log_time_gap_mean: float
    log_time_gap_std: float


def chronological_bounds(df: pd.DataFrame) -> dict[str, tuple[int, int]]:
    """Contiguous positional [lo, hi) row ranges per split, cut on real calendar dates."""
    dt = df["datetime"].values
    train_hi = int(np.searchsorted(dt, np.datetime64(TRAIN_END) + np.timedelta64(1, "D")))
    val_lo = int(np.searchsorted(dt, np.datetime64(VAL_START)))
    val_hi = int(np.searchsorted(dt, np.datetime64(VAL_END) + np.timedelta64(1, "D")))
    test_lo = int(np.searchsorted(dt, np.datetime64(TEST_START)))
    test_hi = int(np.searchsorted(dt, np.datetime64(TEST_END) + np.timedelta64(1, "D")))
    return {"train": (0, train_hi), "val": (val_lo, val_hi), "test": (test_lo, test_hi)}


def fit_normalize(df: pd.DataFrame, train_bounds: tuple[int, int]) -> NormStats:
    lo, hi = train_bounds
    train_slice = df.iloc[lo:hi]
    return NormStats(
        log_volume_mean=float(train_slice["log_volume"].mean()),
        log_volume_std=float(train_slice["log_volume"].std()),
        log_time_gap_mean=float(train_slice["log_time_gap"].mean()),
        log_time_gap_std=float(train_slice["log_time_gap"].std()),
    )


def apply_normalize(df: pd.DataFrame, stats: NormStats) -> pd.DataFrame:
    df = df.copy()
    df["log_volume_norm"] = (df["log_volume"] - stats.log_volume_mean) / stats.log_volume_std
    df["time_gap_norm"] = (df["log_time_gap"] - stats.log_time_gap_mean) / stats.log_time_gap_std
    return df


def build_dataset(path: Path | str) -> tuple[pd.DataFrame, dict[str, tuple[int, int]], NormStats]:
    """Full pipeline: load, validate, feature-engineer, split, normalize (train-only stats)."""
    df = load_and_validate(path)
    df = compute_raw_features(df)
    bounds = chronological_bounds(df)
    stats = fit_normalize(df, bounds["train"])
    df = apply_normalize(df, stats)
    return df, bounds, stats


def extract_arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flat numpy views used by the window builder -- avoids repeated pandas .iloc overhead."""
    feat = df[FEATURE_COLS].to_numpy(dtype=np.float32)
    opens = df["open"].to_numpy(dtype=np.float64)
    closes = df["close"].to_numpy(dtype=np.float64)
    return feat, opens, closes


def context_bucket(ctx_bars: int) -> str:
    """Buckets by lookback-window size -- named narrow/moderate/wide (not
    short/medium/long) specifically to avoid colliding with "long"/"short" as trading
    directions elsewhere in this project (long-only strategy, no short-selling)."""
    if ctx_bars <= 28:
        return "narrow"
    if ctx_bars <= 49:
        return "moderate"
    return "wide"


# ---------------------------------------------------------------------------
# Window sampling
# ---------------------------------------------------------------------------


class WindowSampler:
    """Enumerates valid (start_idx, ctx_bars) pairs within a split's row range and draws
    unique combinations (per context-length bucket) without replacement."""

    def __init__(self, lo: int, hi: int):
        self.lo = lo
        self.hi = hi

    def valid_starts(self, ctx_bars: int) -> np.ndarray:
        last_start = self.hi - ctx_bars - HORIZON  # inclusive
        if last_start < self.lo:
            return np.empty(0, dtype=np.int64)
        return np.arange(self.lo, last_start + 1, dtype=np.int64)

    def draw(self, n: int, rng: np.random.Generator) -> list[tuple[int, int]]:
        base, rem = divmod(n, len(CONTEXT_LENGTHS))
        pairs: list[tuple[int, int]] = []
        for i, ctx_bars in enumerate(CONTEXT_LENGTHS):
            k = base + (1 if i < rem else 0)
            starts = self.valid_starts(ctx_bars)
            if len(starts) < k:
                logger.warning(
                    "Only %d valid starts for ctx_bars=%d (wanted %d); using all available",
                    len(starts), ctx_bars, k,
                )
                k = len(starts)
            chosen = rng.choice(starts, size=k, replace=False) if k > 0 else np.empty(0, dtype=np.int64)
            pairs.extend((int(s), ctx_bars) for s in chosen)
        return pairs


def build_window(
    feat: np.ndarray, opens: np.ndarray, closes: np.ndarray, start_idx: int, ctx_bars: int
) -> dict:
    """Materializes one 73-slot x 9-channel padded window.

    feat: (N, 7) chained feature matrix in FEATURE_COLS order.
    opens/closes: (N,) raw prices, used for the close_0 anchor correction.
    """
    ctx_end = start_idx + ctx_bars
    hz_end = ctx_end + HORIZON

    context_feats = feat[start_idx:ctx_end]
    horizon_feats_true = feat[ctx_end:hz_end].copy()

    close_0 = float(closes[ctx_end - 1])

    # Anchor correction: horizon bars 2 & 3 (0-indexed 1, 2) need open_ret relative to
    # close_0, not chained to the previous horizon bar's (model-predicted, at inference)
    # close. body_ret/upper_wick/lower_wick are anchor-invariant and need no correction.
    for i in (1, 2):
        horizon_feats_true[i, 0] = np.log(opens[ctx_end + i] / close_0)

    pad_len = MAX_CONTEXT - ctx_bars
    padded_context = np.zeros((MAX_CONTEXT, N_FEATURE_CHANNELS), dtype=np.float32)
    padded_context[pad_len:] = context_feats

    horizon_feats_masked = horizon_feats_true.copy()
    horizon_feats_masked[:, PRICE_VOL_IDX] = 0.0

    full_feats = np.concatenate([padded_context, horizon_feats_true], axis=0).astype(np.float32)
    masked_feats = np.concatenate([padded_context, horizon_feats_masked], axis=0).astype(np.float32)

    padding_mask = np.zeros(TOTAL_LEN, dtype=np.float32)
    padding_mask[pad_len:] = 1.0  # real bars: context + horizon (horizon value masked, slot itself real)
    target_mask = np.zeros(TOTAL_LEN, dtype=np.float32)
    target_mask[MAX_CONTEXT:] = 1.0

    full_tensor = np.concatenate([full_feats, padding_mask[:, None], target_mask[:, None]], axis=1)
    masked_tensor = np.concatenate([masked_feats, padding_mask[:, None], target_mask[:, None]], axis=1)

    y = np.concatenate(
        [horizon_feats_true[:, :4].reshape(-1), horizon_feats_true[:, 4]]
    ).astype(np.float32)  # 12 price components (3 bars x 4) + 3 volume values

    return {
        "masked_tensor": masked_tensor.astype(np.float32),
        "full_tensor": full_tensor.astype(np.float32),
        "y": y,
        "close_0": close_0,
        "ctx_bars": ctx_bars,
        "start_idx": start_idx,
    }


def to_patchtst_input(masked_tensor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Context-only slice for PatchTST: (70,7) features + a per-patch padding boolean
    (True = fully-padded patch, to ignore in attention)."""
    context = masked_tensor[:MAX_CONTEXT, :N_FEATURE_CHANNELS]
    padding_mask = masked_tensor[:MAX_CONTEXT, N_FEATURE_CHANNELS]
    patch_padding = padding_mask.reshape(N_PATCHES, PATCH_LEN)
    patch_key_padding_mask = ~(patch_padding.any(axis=1))
    return context, patch_key_padding_mask


# ---------------------------------------------------------------------------
# Reconstruction (eval / plotting)
# ---------------------------------------------------------------------------


def reconstruct_prices(components: np.ndarray, close_0) -> np.ndarray:
    """components: (..., 3, 4) [open_ret, body_ret, upper_wick, lower_wick] per horizon
    bar (wick terms already non-negative for model predictions, see MAX_LOG_RETURN).
    close_0: scalar, or (...) matching components' leading batch dims.
    Returns (..., 3, 4) absolute prices [open, high, low, close]."""
    open_ret, body_ret, upper_wick, lower_wick = (components[..., i] for i in range(4))
    r_open = open_ret
    r_close = open_ret + body_ret
    r_high = np.maximum(r_open, r_close) + upper_wick
    r_low = np.minimum(r_open, r_close) - lower_wick
    stacked = np.stack([r_open, r_high, r_low, r_close], axis=-1)  # (..., 3, 4)

    if np.isscalar(close_0):
        return close_0 * np.exp(stacked)
    close_0 = np.asarray(close_0)[..., None, None]  # broadcast over (3, 4)
    return close_0 * np.exp(stacked)


def reconstruct_volume(log_volume_norm: np.ndarray, stats: NormStats) -> np.ndarray:
    log_volume = log_volume_norm * stats.log_volume_std + stats.log_volume_mean
    return np.expm1(log_volume)


# ---------------------------------------------------------------------------
# Long-only backtest helpers
# ---------------------------------------------------------------------------


def exit_price_from_components(components: np.ndarray, close_0) -> np.ndarray:
    """components: (..., 3, 4) [open_ret, body_ret, upper_wick, lower_wick] per horizon
    bar. close_0: scalar or (...) matching components' leading batch dims.

    Exit price = mean of the 3 bars' open+close absolute prices (6 values / 6) -- a
    simple, deterministic stand-in for "some reasonable fill within the 3-bar window"
    that avoids picking a specific exit-timing rule. Returns (...,)."""
    ohlc = reconstruct_prices(components, close_0)  # (..., 3, 4): open,high,low,close
    return ohlc[..., [0, 3]].mean(axis=(-2, -1))  # mean of open(idx0) & close(idx3), all 3 bars


def max_close_from_components(components: np.ndarray, close_0) -> np.ndarray:
    """components: (..., 3, 4) [open_ret, body_ret, upper_wick, lower_wick] per horizon
    bar. close_0: scalar or (...) matching components' leading batch dims.

    Exit price = the highest of the 3 bars' own predicted close prices -- a simpler,
    more aggressive stand-in than exit_price_from_components's 6-value average; used as
    PatchTST's take-profit target (a single point forecast, so "aim near the best close
    it predicted" is as far as a benchmark rule can reasonably go without a distribution
    to draw a quantile from). Returns (...,)."""
    ohlc = reconstruct_prices(components, close_0)  # (..., 3, 4): open,high,low,close
    return ohlc[..., 3].max(axis=-1)  # max close (idx3) across the 3 bars


def per_bar_close_return(components: np.ndarray) -> np.ndarray:
    """components: (..., 3, 4). Returns (..., 3): close_0-anchored close return
    (open_ret + body_ret) per horizon bar -- used to check directional coherence."""
    return components[..., 0] + components[..., 1]


def train_exit_return_bound(
    opens: np.ndarray, closes: np.ndarray, train_bounds: tuple[int, int], percentile: float = 99.0
) -> float:
    """Empirical p`percentile` of |close_0-anchored log return| across every real
    (open, close) in the 3-bar horizon, pooled over every possible anchor point in the
    train split -- independent of ctx_bars, since the horizon's placement doesn't depend
    on context length (see build_window). This is the actual multi-candle move the
    model's predicted exit price implies, unlike MAX_LOG_RETURN (see above), which was
    calibrated from single-bar extremes and ends up looser than intended once anchored
    open/close returns are pooled 3 bars out. Used to recalibrate predictions post-hoc
    via shrink_components without retraining."""
    lo, hi = train_bounds
    anchors = np.arange(lo + 1, hi - HORIZON + 1)  # ctx_end candidates; need lo<=anchor-1 and anchor+HORIZON-1<hi
    if len(anchors) == 0:
        raise ValueError(f"train_bounds {train_bounds} too narrow for HORIZON={HORIZON}")
    close_0 = closes[anchors - 1]
    horizon_idx = anchors[:, None] + np.arange(HORIZON)[None, :]  # (M, 3)
    open_ret = np.log(opens[horizon_idx] / close_0[:, None])  # (M, 3)
    close_ret = np.log(closes[horizon_idx] / close_0[:, None])  # (M, 3)
    pooled = np.abs(np.concatenate([open_ret, close_ret], axis=1))  # (M, 6)
    return float(np.percentile(pooled, percentile))


def shrink_components(components: np.ndarray, bound: float) -> np.ndarray:
    """components: (..., 3, 4) [open_ret, body_ret, upper_wick, lower_wick], already
    bounded by the model's own MAX_LOG_RETURN. Applies a second, tighter squash -- same
    tanh shape, tighter scale -- calibrated from train_exit_return_bound instead of
    single-bar extremes, so a prediction that's technically within MAX_LOG_RETURN can
    still be shrunk toward a trader-plausible multi-candle move. Not a clip: values well
    inside `bound` pass through close to unchanged (tanh(x/b)*b ~= x for |x| << b), only
    the tail gets compressed."""
    ret, wick = components[..., :2], components[..., 2:]
    return np.concatenate([bound * np.tanh(ret / bound), bound * np.tanh(wick / bound)], axis=-1)


# ---------------------------------------------------------------------------
# torch Dataset wrappers
# ---------------------------------------------------------------------------


class WindowDataset(Dataset):
    """Materializes a fixed list of (start_idx, ctx_bars) pairs into tensors. Callers
    resample a fresh pair list each epoch for train, and reuse one fixed list for val/test."""

    def __init__(
        self, feat: np.ndarray, opens: np.ndarray, closes: np.ndarray, pairs: list[tuple[int, int]]
    ):
        self.feat = feat
        self.opens = opens
        self.closes = closes
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        start_idx, ctx_bars = self.pairs[idx]
        w = build_window(self.feat, self.opens, self.closes, start_idx, ctx_bars)
        return {
            "masked_tensor": torch.from_numpy(w["masked_tensor"]),
            "full_tensor": torch.from_numpy(w["full_tensor"]),
            "y": torch.from_numpy(w["y"]),
            "close_0": w["close_0"],
            "ctx_bars": w["ctx_bars"],
        }
