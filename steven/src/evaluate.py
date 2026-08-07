"""Evaluate PatchTST and the CVAE on the same fixed test window set.

Usage:
    python steven/src/evaluate.py \\
        --patchtst-checkpoint steven/outputs/patchtst_checkpoint.pt \\
        --cvae-checkpoint steven/outputs/cvae_checkpoint.pt \\
        --n-test-windows 300 --num-plot-windows-per-bucket 2
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mplfinance as mpf

from src.data_pipeline import (
    HORIZON,
    WindowDataset,
    WindowSampler,
    build_dataset,
    context_bucket,
    exit_price_from_components,
    extract_arrays,
    max_close_from_components,
    per_bar_close_return,
    reconstruct_prices,
    reconstruct_volume,
    shrink_components,
    to_patchtst_input,
    train_exit_return_bound,
)
from src.models.cvae_inpainting import CVAEInpainting
from src.models.patchtst import PatchTST

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

BUCKETS = ["narrow", "moderate", "wide"]  # lookback-window size -- see context_bucket
BACKTEST_THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--patchtst-checkpoint", type=str, default="steven/outputs/patchtst_checkpoint.pt")
    p.add_argument("--cvae-checkpoint", type=str, default="steven/outputs/cvae_checkpoint.pt")
    p.add_argument("--n-test-windows", type=int, default=3000)
    p.add_argument("--num-samples", type=int, default=5)
    p.add_argument("--num-plot-samples", type=int, default=5)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--metrics-out", type=str, default="steven/outputs/metrics.json")
    p.add_argument("--plots-dir", type=str, default="steven/outputs/sample_plots")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument(
        "--sell-bound-percentile", type=float, default=99.0,
        help="Percentile (over train data) of |close_0-anchored log return| used to shrink predicted "
        "price components -- a tighter, empirically-calibrated bound on top of the model's own "
        "MAX_LOG_RETURN, applied to metrics/backtest/plots alike. See train_exit_return_bound.",
    )
    p.add_argument(
        "--cvae-sell-quantile", type=float, default=70.0,
        help="Percentile (over the K sampled draws' own predicted exit prices) used as CVAE's "
        "take-profit target, instead of their mean -- higher = a more aggressive target (wider "
        "profit gap, lower expected chance of filling before the 3-candle order expires). CVAE "
        "actually has a sampled distribution to pick a target from; PatchTST doesn't (single point "
        "forecast), so it uses the max of its 3 predicted closes instead (see "
        "max_close_from_components).",
    )
    p.add_argument("--batch-size", type=int, default=256)
    # default 0: benchmarked in steven/configs/patchtst.yaml -- DataLoader multiprocessing
    # is slower here, not faster, since build_window() is cheap numpy per sample.
    p.add_argument("--num-workers", type=int, default=0)
    return p.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def collate(batch: list[dict]) -> dict:
    context, patch_pad = zip(*(to_patchtst_input(b["masked_tensor"].numpy()) for b in batch))
    return {
        "masked_tensor": torch.stack([b["masked_tensor"] for b in batch]),
        "context": torch.stack([torch.from_numpy(c) for c in context]),
        "patch_key_padding_mask": torch.stack([torch.from_numpy(m) for m in patch_pad]),
        "y": torch.stack([b["y"] for b in batch]),
        "close_0": torch.tensor([b["close_0"] for b in batch], dtype=torch.float64),
        "ctx_bars": torch.tensor([b["ctx_bars"] for b in batch], dtype=torch.int64),
    }


def mae_rmse(pred: np.ndarray, true: np.ndarray) -> tuple[float, float]:
    diff = pred - true
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff**2)))
    return mae, rmse


def directional_accuracy(pred_close_ret: np.ndarray, true_close_ret: np.ndarray) -> np.ndarray:
    """pred/true_close_ret: (N, 3). Returns (3,) accuracy per horizon step."""
    return (np.sign(pred_close_ret) == np.sign(true_close_ret)).mean(axis=0)


def collect_predictions(patchtst, cvae, loader, device, num_samples):
    """Runs both models over the fixed test loader, returns flat numpy arrays."""
    all_true_y, all_pt_y = [], []
    all_cvae_y = []  # (K, N, 15)
    all_close0, all_ctx = [], []

    for batch in loader:
        context = batch["context"].to(device)
        patch_pad = batch["patch_key_padding_mask"].to(device)
        masked_tensor = batch["masked_tensor"].to(device)
        y = batch["y"]

        with torch.no_grad():
            pt_price, pt_vol = patchtst(context, patch_pad)
            pt_y = torch.cat([pt_price.reshape(pt_price.shape[0], -1), pt_vol], dim=1)

            cvae_price, cvae_vol = cvae.sample(masked_tensor, k=num_samples)  # (K,B,3,4),(K,B,3)
            K, B = cvae_price.shape[0], cvae_price.shape[1]
            cvae_y = torch.cat(
                [cvae_price.reshape(K, B, -1), cvae_vol], dim=-1
            )  # (K, B, 15)

        all_true_y.append(y.numpy())
        all_pt_y.append(pt_y.cpu().numpy())
        all_cvae_y.append(cvae_y.cpu().numpy())
        all_close0.append(batch["close_0"].numpy())
        all_ctx.append(batch["ctx_bars"].numpy())

    true_y = np.concatenate(all_true_y, axis=0)
    pt_y = np.concatenate(all_pt_y, axis=0)
    cvae_y = np.concatenate(all_cvae_y, axis=1)  # (K, N, 15)
    close_0 = np.concatenate(all_close0, axis=0)
    ctx_bars = np.concatenate(all_ctx, axis=0)
    return true_y, pt_y, cvae_y, close_0, ctx_bars


def metrics_for_slice(true_y, pt_y, cvae_y, close_0, stats) -> dict:
    """true_y/pt_y: (N,15). cvae_y: (K,N,15). close_0: (N,). Returns a metrics dict."""
    n = len(true_y)
    if n == 0:
        return {"n_windows": 0}

    true_price = true_y[:, :12].reshape(n, 3, 4)
    true_vol_norm = true_y[:, 12:15]
    pt_price = pt_y[:, :12].reshape(n, 3, 4)
    pt_vol_norm = pt_y[:, 12:15]
    cvae_price_mean = cvae_y[..., :12].reshape(cvae_y.shape[0], n, 3, 4).mean(axis=0)
    cvae_vol_mean = cvae_y[..., 12:15].mean(axis=0)

    out = {"n_windows": n}

    # reparam-target MAE/RMSE (15-dim vector directly)
    out["patchtst_reparam_mae_rmse"] = mae_rmse(pt_y, true_y)
    out["cvae_reparam_mae_rmse"] = mae_rmse(cvae_y.mean(axis=0), true_y)

    # reconstructed OHLCV MAE/RMSE
    true_ohlc = reconstruct_prices(true_price, close_0)
    pt_ohlc = reconstruct_prices(pt_price, close_0)
    cvae_ohlc = reconstruct_prices(cvae_price_mean, close_0)
    out["patchtst_ohlc_mae_rmse"] = mae_rmse(pt_ohlc, true_ohlc)
    out["cvae_ohlc_mae_rmse"] = mae_rmse(cvae_ohlc, true_ohlc)

    true_vol = reconstruct_volume(true_vol_norm, stats)
    pt_vol = reconstruct_volume(pt_vol_norm, stats)
    cvae_vol = reconstruct_volume(cvae_vol_mean, stats)
    out["patchtst_volume_mae_rmse"] = mae_rmse(pt_vol, true_vol)
    out["cvae_volume_mae_rmse"] = mae_rmse(cvae_vol, true_vol)

    # directional accuracy on close_0-anchored close return, per horizon step
    true_close_ret = true_price[:, :, 0] + true_price[:, :, 1]  # open_ret + body_ret
    pt_close_ret = pt_price[:, :, 0] + pt_price[:, :, 1]
    cvae_close_ret = cvae_price_mean[:, :, 0] + cvae_price_mean[:, :, 1]
    out["patchtst_directional_accuracy"] = directional_accuracy(pt_close_ret, true_close_ret).tolist()
    out["cvae_directional_accuracy"] = directional_accuracy(cvae_close_ret, true_close_ret).tolist()

    # CVAE sample spread vs realized variance of true close price, per horizon step
    cvae_price_samples = cvae_y[..., :12].reshape(cvae_y.shape[0], n, 3, 4)
    cvae_close_price_samples = reconstruct_prices(cvae_price_samples, close_0)[..., 3]  # (K, N, 3)
    sample_var_per_window = cvae_close_price_samples.var(axis=0)  # (N, 3)
    avg_sample_var = sample_var_per_window.mean(axis=0)  # (3,)
    realized_var = true_ohlc[..., 3].var(axis=0)  # (3,) cross-window variance of true close
    out["cvae_avg_sample_variance"] = avg_sample_var.tolist()
    out["realized_close_variance"] = realized_var.tolist()
    out["cvae_spread_to_realized_ratio"] = (avg_sample_var / np.maximum(realized_var, 1e-12)).tolist()

    return out


# ---------------------------------------------------------------------------
# Long-only selective backtest
#
# Entry is always close_0 (real, known). Exit is a take-profit-only limit order (see
# take_profit_exit): the 3 real horizon bars are walked in order, and if any bar's real
# [low,high] range reaches the take-profit price, the order fills there; otherwise the
# position is force-closed at the 3rd bar's real close (expiry). An earlier pass added a
# mirrored 1:1 stop-loss (a "bracket order") on top of this, but it was removed after the
# backtest showed CVAE's stop-loss rate exceeding its take-profit rate at low thresholds
# (32.3% vs 29.0% at 0.5) -- once the recalibrated take-profit sits close enough to entry
# to be realistic, an equally-close mirrored stop is also close enough for ordinary
# intrabar noise to trigger it, and the pessimistic same-bar tie-break (stop wins on
# overlap) then converts many of those into guaranteed losses rather than gains. See
# backlog.md for the bracket-order variant if it's worth revisiting with an asymmetric
# risk:reward instead of 1:1. Each model's own confidence score decides which windows are
# worth trading at all; the realized return used for PnL always comes from ground truth,
# never predictions.
# ---------------------------------------------------------------------------


def percentile_rank(values: np.ndarray) -> np.ndarray:
    """0..1 rank (0 = smallest). Used to put PatchTST's point-forecast magnitude on the
    same [0,1] threshold scale as CVAE's sample-agreement fraction."""
    n = len(values)
    if n <= 1:
        return np.zeros(n)
    order = np.argsort(values)
    ranks = np.empty(n)
    ranks[order] = np.arange(n)
    return ranks / (n - 1)


def cvae_confidence_scores(cvae_price_samples: np.ndarray, close_0: np.ndarray) -> np.ndarray:
    """cvae_price_samples: (K,N,3,4). Returns (N,): fraction of the K sampled paths
    whose predicted exit price comes out above close_0. ~1 = strong sample consensus on
    an upward exit, ~0.5 = samples disagree (chop), ~0 = consensus downtrend -- one
    score captures all three of the "don't trade this" cases at once."""
    if cvae_price_samples.shape[1] == 0:
        return np.empty(0)
    exit_prices = exit_price_from_components(cvae_price_samples, close_0)  # (K, N)
    sample_return = exit_prices / close_0[None, :] - 1.0
    return (sample_return > 0).mean(axis=0)


def patchtst_confidence_scores(pt_price: np.ndarray, close_0: np.ndarray) -> np.ndarray:
    """pt_price: (N,3,4). PatchTST has no sample distribution, so: (1) require all 3
    predicted bar-close-returns to agree on the up direction (else auto-dismissed as
    choppy/down -- confidence 0), (2) rank surviving windows by predicted exit-return
    magnitude and convert to a [0,1] percentile, matching CVAE's threshold scale."""
    n = len(pt_price)
    if n == 0:
        return np.empty(0)
    coherent_up = (per_bar_close_return(pt_price) > 0).all(axis=1)
    exit_price = exit_price_from_components(pt_price, close_0)
    predicted_exit_return = exit_price / close_0 - 1.0

    confidence = np.zeros(n)
    if coherent_up.any():
        confidence[coherent_up] = percentile_rank(predicted_exit_return[coherent_up])
    return confidence


def take_profit_exit(
    true_ohlc: np.ndarray, take_profit: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Simulates a take-profit-only limit order (no stop-loss -- see the module note
    above for why) placed at the same time as the close_0 buy. Walks the 3 REAL horizon
    bars in order; the first bar whose real [low,high] range reaches take_profit fills
    the order there. If it's never reached across all 3 bars, the position is
    force-closed at the 3rd bar's real close instead (order expiry). true_ohlc: (N,3,4)
    real [open,high,low,close]. take_profit: (N,). Returns (realized_sell_price (N,),
    hit_take_profit (N,) bool)."""
    n = true_ohlc.shape[0]
    low = true_ohlc[:, :, 2]
    high = true_ohlc[:, :, 1]

    sell_price = true_ohlc[:, HORIZON - 1, 3].copy()  # default: last horizon bar's real close
    hit_take_profit = np.zeros(n, dtype=bool)
    resolved = np.zeros(n, dtype=bool)

    for bar in range(HORIZON):
        tp_touch = (~resolved) & (low[:, bar] <= take_profit) & (take_profit <= high[:, bar])
        sell_price[tp_touch] = take_profit[tp_touch]
        hit_take_profit[tp_touch] = True
        resolved |= tp_touch

    return sell_price, hit_take_profit


CASE_LABELS = ["win_take_profit", "win_expiry", "no_trade", "lose_expiry"]
CASE_TITLES = {
    "win_take_profit": "Win -- take-profit hit",
    "win_expiry": "Win -- expiry (gain)",
    "no_trade": "No trade (target <= buy)",
    "lose_expiry": "Lose -- expiry (loss)",
}


def classify_outcomes(
    eligible: np.ndarray, hit_take_profit: np.ndarray, trade_return: np.ndarray
) -> np.ndarray:
    """Sorts every window into exactly one of CASE_LABELS, independent of any confidence
    threshold: 'win_take_profit' (target reached), 'win_expiry' (target missed but the
    forced 3rd-candle close still came in above buy), 'no_trade' (this model's own
    target never cleared the buy price, so no trade would ever be placed regardless of
    confidence), 'lose_expiry' (target missed and the forced close came in at/below buy).
    eligible/hit_take_profit: (N,) bool. trade_return: (N,) fraction, meaningful only
    where eligible (see take_profit_exit -- it still runs unconditionally, but a
    not-eligible window's return is never actually realized as a trade). Returns (N,)
    array of dtype=object strings."""
    labels = np.full(eligible.shape, "no_trade", dtype=object)
    entered = eligible
    labels[entered & hit_take_profit] = "win_take_profit"
    expired = entered & ~hit_take_profit
    labels[expired & (trade_return > 0)] = "win_expiry"
    labels[expired & (trade_return <= 0)] = "lose_expiry"
    return labels


def outcome_breakdown(labels: np.ndarray) -> dict:
    """labels: (N,) from classify_outcomes. Returns {case: fraction of all N windows in
    that case} -- rows sum to 1.0 (or empty dict if N=0). Reported once per model,
    independent of confidence threshold: a 'no_trade' window is excluded from every
    threshold's selection by construction (see sweep_thresholds' eligible mask), so it
    can never appear as a row inside the threshold-swept table -- this is the only place
    it's visible at all."""
    n = len(labels)
    if n == 0:
        return {}
    return {case: float((labels == case).mean()) for case in CASE_LABELS}


def model_trade_outcomes(true_ohlc: np.ndarray, take_profit: np.ndarray, close_0: np.ndarray) -> dict:
    """Bundles one model's full per-window take-profit-exit outcome against real price
    action -- shared by run_backtest (for the threshold sweep + case breakdown) and
    make_plots (to pick illustrative examples of each case). true_ohlc: (N,3,4) real
    [open,high,low,close]. take_profit, close_0: (N,). Returns a dict of (N,)-shaped
    arrays: sell_price, hit_take_profit, trade_return, eligible, label."""
    sell_price, hit_take_profit = take_profit_exit(true_ohlc, take_profit)
    trade_return = sell_price / close_0 - 1.0
    eligible = take_profit > close_0
    label = classify_outcomes(eligible, hit_take_profit, trade_return)
    return {
        "sell_price": sell_price,
        "hit_take_profit": hit_take_profit,
        "trade_return": trade_return,
        "eligible": eligible,
        "label": label,
    }


def nearest_draw_index(draw_exit_prices: np.ndarray, target: float) -> int:
    """draw_exit_prices: (K,) each of CVAE's K sampled draws' own exit_price_from_components
    value for one window. target: the take-profit price actually used for that window (a
    percentile across all K draws -- see main()/run_backtest -- not tied to any single
    draw). Returns the index of whichever draw's own exit price is closest to target --
    used by make_plots to pick one illustrative draw to render as candles that's visually
    consistent with a target that's really a statistic over the whole ensemble."""
    return int(np.argmin(np.abs(draw_exit_prices - target)))


def sweep_thresholds(
    confidence: np.ndarray, trade_return: np.ndarray, thresholds: list[float],
    hit_take_profit: np.ndarray | None = None,
    eligible: np.ndarray | None = None,
) -> list[dict]:
    """eligible, if given, is an (N,) bool mask ANDed into every threshold's selection --
    used to hard-exclude windows whose own predicted take-profit doesn't even clear the
    buy price (see run_backtest), regardless of how high their confidence score is."""
    n = len(confidence)
    rows = []
    for th in thresholds:
        mask = confidence >= th
        if eligible is not None:
            mask = mask & eligible
        k = int(mask.sum())
        row = {"threshold": th, "n_trades": k, "selectivity": (k / n) if n else 0.0}
        if k > 0:
            rets = trade_return[mask]
            row["win_rate"] = float((rets > 0).mean())
            row["avg_return"] = float(rets.mean())
            row["total_return"] = float(rets.sum())
            row["take_profit_rate"] = float(hit_take_profit[mask].mean()) if hit_take_profit is not None else None
        else:
            row["win_rate"] = None
            row["avg_return"] = None
            row["total_return"] = None
            row["take_profit_rate"] = None
        rows.append(row)
    return rows


def buy_and_hold_benchmark(df: pd.DataFrame, bounds: dict) -> dict:
    """Naive baseline for the backtest below: buy SPY at the test period's first close,
    hold to its last close -- no model, no confidence threshold, no trade selectivity.
    Something to compare the model-driven total/annualized returns against."""
    test_lo, test_hi = bounds["test"]
    entry, exit_ = df.iloc[test_lo], df.iloc[test_hi - 1]
    entry_price, exit_price = float(entry["close"]), float(exit_["close"])
    total_return = exit_price / entry_price - 1.0
    elapsed_years = (exit_["datetime"] - entry["datetime"]).days / 365.25
    annual_return = (1.0 + total_return) ** (1.0 / elapsed_years) - 1.0
    return {
        "entry_date": str(entry["datetime"].date()),
        "entry_price": entry_price,
        "exit_date": str(exit_["datetime"].date()),
        "exit_price": exit_price,
        "elapsed_years": elapsed_years,
        "total_return": total_return,
        "annual_return": annual_return,
    }


def run_backtest(
    true_y: np.ndarray, pt_y: np.ndarray, cvae_y: np.ndarray, close_0: np.ndarray,
    cvae_sell_quantile: float = 70.0,
) -> dict:
    """true_y/pt_y: (N,15). cvae_y: (K,N,15). close_0: (N,)."""
    n = len(true_y)
    note = (
        "long-only take-profit backtest (no stop-loss -- an earlier 1:1 mirrored bracket "
        "was removed after it was found to lose more often than it won once the "
        "recalibrated take-profit sat close enough to entry for ordinary intrabar noise "
        "to trigger the mirrored stop; see the module comment above run_backtest): "
        "entry=close_0 (real). Each model's own predicted take-profit price is placed as "
        "a limit sell order at the same time as entry. PatchTST's take-profit is the max "
        "of its predicted 3 horizon bars' own closes (single point forecast, no "
        f"distribution to draw a quantile from -- see max_close_from_components); CVAE's "
        f"is the p{cvae_sell_quantile:.0f} percentile of its K sampled draws' own "
        "predicted exit prices, since CVAE actually has a sampled distribution to pick a "
        "target from. A window is only ever eligible to trade if its own take-profit "
        "clears the buy price -- confidence alone can't override this. The 3 REAL "
        "horizon bars are then walked in order (see take_profit_exit): if any bar's real "
        "[low,high] range reaches the take-profit, the order fills there; if it's never "
        "reached, the position is force-closed at the 3rd REAL bar's close (order "
        "expiry). total_return is summed across selected trades -- test windows are "
        "independently sampled and can overlap in calendar time, so this is not a "
        "sequential equity curve."
    )
    if n == 0:
        return {
            "note": note, "patchtst": [], "cvae": [],
            "patchtst_outcome_breakdown": {}, "cvae_outcome_breakdown": {},
        }

    true_price = true_y[:, :12].reshape(n, 3, 4)
    pt_price = pt_y[:, :12].reshape(n, 3, 4)
    cvae_price_samples = cvae_y[..., :12].reshape(cvae_y.shape[0], n, 3, 4)

    true_ohlc = reconstruct_prices(true_price, close_0)  # (N,3,4) real [open,high,low,close]

    pt_take_profit = max_close_from_components(pt_price, close_0)  # (N,)
    cvae_exit_prices = exit_price_from_components(cvae_price_samples, close_0)  # (K,N)
    cvae_take_profit = np.percentile(cvae_exit_prices, cvae_sell_quantile, axis=0)  # (N,)

    pt_outcomes = model_trade_outcomes(true_ohlc, pt_take_profit, close_0)
    cvae_outcomes = model_trade_outcomes(true_ohlc, cvae_take_profit, close_0)

    cvae_conf = cvae_confidence_scores(cvae_price_samples, close_0)
    pt_conf = patchtst_confidence_scores(pt_price, close_0)

    return {
        "note": note,
        "patchtst": sweep_thresholds(
            pt_conf, pt_outcomes["trade_return"], BACKTEST_THRESHOLDS,
            pt_outcomes["hit_take_profit"], pt_outcomes["eligible"],
        ),
        "cvae": sweep_thresholds(
            cvae_conf, cvae_outcomes["trade_return"], BACKTEST_THRESHOLDS,
            cvae_outcomes["hit_take_profit"], cvae_outcomes["eligible"],
        ),
        "patchtst_outcome_breakdown": outcome_breakdown(pt_outcomes["label"]),
        "cvae_outcome_breakdown": outcome_breakdown(cvae_outcomes["label"]),
    }


def draw_horizon_box(ax, n_shown: int) -> None:
    """Red box around the last HORIZON candles -- the region being generated/predicted."""
    x0, x1 = n_shown - HORIZON - 0.5, n_shown - 0.5
    y0, y1 = ax.get_ylim()
    ax.add_patch(
        mpatches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="red", linewidth=1.8, zorder=6)
    )


def draw_trade_lines(
    ax, n_shown: int, buy_price: float, sell_targets: list[tuple[str, float, str, str]]
) -> None:
    """Buy = dashed line at close_0 (the last known candle's close), spanning the full
    panel. Each entry in sell_targets is (label, price, color, linestyle): a take-profit
    price drawn as a horizontal line starting at the left edge of the horizon box (the
    first of the 3 target/generated candles) and extending to the right edge of the
    panel -- not a line all the way through, since the price only applies once prediction
    starts. The linestyle is caller-chosen (currently always solid, one per model -- kept
    generic here so a second line per model, e.g. a future stop-loss, is just another
    tuple in the list). If a price falls outside the panel's current y-range, the line is
    pinned just inside the top/bottom edge instead of letting the axis autoscale to it,
    with a text label giving the real price and noting it's out of range -- this keeps
    the panel's own price scale intact regardless of how extreme a prediction is."""
    y0, y1 = ax.get_ylim()
    x0 = n_shown - HORIZON - 0.5
    x1 = n_shown - 0.5
    margin = (y1 - y0) * 0.08

    ax.axhline(buy_price, color="tab:blue", linestyle="--", linewidth=1, zorder=6)
    ax.text(-0.6, buy_price, "buy", color="tab:blue", fontsize=7, va="bottom", ha="left", zorder=6)

    for label, price, color, linestyle in sell_targets:
        if price > y1:
            line_y = y1 - margin
            ax.hlines(line_y, x0, x1, color=color, linestyle=linestyle, linewidth=1.4, zorder=7)
            ax.annotate(
                f"  {label}: {price:.2f} (above range)", xy=(x0, line_y), xytext=(0, -4),
                textcoords="offset points", color=color, fontsize=7, va="top", ha="left", zorder=7,
            )
        elif price < y0:
            line_y = y0 + margin
            ax.hlines(line_y, x0, x1, color=color, linestyle=linestyle, linewidth=1.4, zorder=7)
            ax.annotate(
                f"  {label}: {price:.2f} (below range)", xy=(x0, line_y), xytext=(0, 4),
                textcoords="offset points", color=color, fontsize=7, va="bottom", ha="left", zorder=7,
            )
        else:
            ax.hlines(price, x0, x1, color=color, linestyle=linestyle, linewidth=1.4, zorder=7)
            # Label sits past the right edge, not at x0 -- anchoring it at x0 (inside the
            # horizon box) used to overlap the candles it's drawn on top of, especially
            # when the target sits close to the buy price.
            ax.text(x1 + 0.3, price, f"{label}: {price:.2f}", color=color, fontsize=7, va="bottom", ha="left", zorder=7)

    ax.set_ylim(y0, y1)


def trade_table_text(
    ohlc: np.ndarray, buy_price: float, sell_limit: float | None = None,
    true_horizon_ohlc: np.ndarray | None = None,
) -> tuple[str, tuple[float, str] | None, bool | None]:
    """ohlc: (3,4) [open,high,low,close] for this panel's 3 horizon bars (used for the
    per-candle numbers in the text table -- always shown, even for the ground truth
    panel, which passes sell_limit=None since it has no trade decision of its own).
    sell_limit: this model's own predicted take-profit price, already computed by the
    caller (max-predicted-close for PatchTST, quantile-based for CVAE -- see
    run_backtest/make_plots). true_horizon_ohlc, if given, is the REAL 3 horizon bars'
    [open,high,low,close] -- used to walk the take-profit order (see take_profit_exit)
    against real price action, but only when would_enter is True: if the model's own
    target never clears the buy price, no order is ever placed, so there's nothing to
    realize -- reporting a mechanical "would it have hit anyway" outcome underneath a
    "NO TRADE" decision would read as contradictory. Returns (annotation text,
    (realized_price, outcome)-or-None, would_enter-or-None) -- would_enter is
    sell_limit > buy_price, i.e. whether this model's own predicted target even clears
    the entry price; the long-only strategy would skip the trade entirely otherwise,
    regardless of confidence. outcome is one of "take_profit", "expired"."""
    lines = [
        f"C1  O={ohlc[0,0]:.2f}  C={ohlc[0,3]:.2f}    "
        f"C2  O={ohlc[1,0]:.2f}  C={ohlc[1,3]:.2f}    "
        f"C3  O={ohlc[2,0]:.2f}  C={ohlc[2,3]:.2f}",
    ]
    realized = None
    would_enter = None
    if sell_limit is not None:
        lines.append(f"Buy = {buy_price:.2f}   Take-profit = {sell_limit:.2f}")
        would_enter = sell_limit > buy_price
        decision = "ENTER long (target > buy)" if would_enter else "NO TRADE (target <= buy -- no expected upside)"
        lines.append(f"Trade decision: {decision}")

        if would_enter and true_horizon_ohlc is not None:
            sell_price, hit_tp = take_profit_exit(true_horizon_ohlc[None, :, :], np.array([sell_limit]))
            realized_price = float(sell_price[0])
            outcome = "take_profit" if hit_tp[0] else "expired"
            realized_return = (realized_price / buy_price - 1.0) * 100
            realized = (realized_price, outcome)
            status = {
                "take_profit": "TAKE PROFIT -> filled at target",
                "expired": "EXPIRED -> forced exit at real C3 close",
            }[outcome]
            lines.append(
                f"vs. real price action: {status}  |  realized sell = {realized_price:.2f}  |  "
                f"return = {realized_return:+.2f}%"
            )
    return "\n".join(lines), realized, would_enter


def render_panel(
    fig,
    gs_chart,
    gs_text,
    sub_df: pd.DataFrame,
    buy_price: float,
    title: str,
    sell_targets: list[tuple[str, float, str, str]],
    true_horizon_ohlc: np.ndarray | None = None,
    sell_limit: float | None = None,
) -> tuple[tuple[float, str] | None, bool | None]:
    """sell_limit is this model's own predicted take-profit price (already computed by
    the caller -- None for the ground truth panel, which has no trade decision of its
    own). Returns (realized, would_enter) from trade_table_text -- realized is
    (realized_price, outcome) when true_horizon_ohlc was given, else None; would_enter is
    likewise None for the ground truth panel -- so callers that need these (e.g. to log
    them alongside the plot) don't have to recompute them."""
    ax = fig.add_subplot(gs_chart)
    mpf.plot(sub_df, type="candle", ax=ax, style="yahoo", volume=False)
    ax.set_title(title, fontsize=10)
    ax.tick_params(axis="x", labelrotation=30, labelsize=7)
    draw_horizon_box(ax, len(sub_df))

    ohlc = sub_df.iloc[-HORIZON:][["open", "high", "low", "close"]].to_numpy()
    text, realized, would_enter = trade_table_text(ohlc, buy_price, sell_limit, true_horizon_ohlc)
    draw_trade_lines(ax, len(sub_df), buy_price, sell_targets)

    if would_enter is not None:
        label = "ENTER LONG" if would_enter else "NO TRADE"
        color = "tab:green" if would_enter else "tab:red"
        ax.text(
            0.02, 0.97, label, transform=ax.transAxes, ha="left", va="top", fontsize=10,
            fontweight="bold", color=color, zorder=9,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=color, alpha=0.9),
        )

    ax_txt = fig.add_subplot(gs_text)
    ax_txt.axis("off")
    ax_txt.text(0.0, 1.0, text, transform=ax_txt.transAxes, ha="left", va="top", fontsize=7.5, family="monospace")
    return realized, would_enter


def make_plots(
    df, test_pairs, pt_y, cvae_y, close_0, pt_take_profit, cvae_take_profit, cvae_exit_prices, cvae_label, args
):
    """Selects 8 illustrative windows from the full fixed test set -- one per (case,
    context-length) combination, choosing uniformly at random among the windows that
    match, where case is one of CASE_LABELS and context-length is "narrow" or "wide" (see
    context_bucket; "moderate" is skipped to keep a clean narrow-vs-wide contrast --
    named to avoid colliding with "long"/"short" as trading directions elsewhere in this
    doc, e.g. long-only, ENTER LONG). Cases are categorized by CVAE's own trade outcome;
    PatchTST's panel on the same figure just
    shows whatever actually happened on that same window, uncontrolled, so the two
    models' contrasting behavior on identical real price action is visible side by side.

    Reuses the already-computed, already sell_bound-shrunk predictions from the fixed
    evaluation set (pt_y, cvae_y, close_0, pt_take_profit, cvae_take_profit, cvae_exit_prices,
    cvae_label -- all computed once in main(), aligned index-for-index with test_pairs)
    rather than re-running the models on freshly sampled windows: re-sampling CVAE fresh
    could give a different set of k draws -- and therefore a different take-profit
    quantile and a different outcome -- than the one that actually earned this window its
    case label. Only *which* qualifying window is picked per (case, ctx-length) slot is
    randomized (unseeded, so a different concrete example each run), always from the same
    fixed, reproducible population used for the metrics/backtest above.

    Each figure has 3 candlestick panels: ground truth (left), PatchTST's generated
    horizon candles (top right), CVAE's generated horizon candles (bottom right). CVAE's
    take-profit target is a percentile across all k sampled draws' own exit prices (see
    run_backtest), not tied to any single draw -- so rather than always rendering an
    arbitrary draw (which could show a target line sitting outside the candles drawn
    under it), the rendered draw is whichever of the k has its own exit price closest to
    that target (see nearest_draw_index), keeping the illustration visually consistent
    with the line drawn over it. Each panel gets a red box around the generated/predicted
    3 candles, a dashed buy-price line, and a solid take-profit line per model, starting
    at the left edge of that box and running to the panel's right edge -- the ground
    truth panel shows both models' lines together for comparison. A line pinned outside
    the panel's y-range is drawn just inside the edge instead, annotated with its real
    price. The PatchTST and CVAE panels additionally report in the text table whether
    that model's take-profit order actually filled or expired against the REAL price
    action (see take_profit_exit) -- this verdict is always computed from real
    ground-truth OHLC, independent of which draw is shown or whether the drawn candles
    themselves reach the target line. Also writes samples.json alongside the PNGs -- the
    same per-sample numbers (candles, buy/take-profit prices, outcome, spread) plus its
    case and ctx bucket, structured for update_report.py to regenerate v1.md's Results
    tables without transcribing PNGs by hand."""
    out_dir = Path(args.plots_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Clear last run's PNGs/samples.json first -- update_report.py rewrites v1.md to
    # reference only the current batch, so a stale prior run's files would otherwise
    # just accumulate unreferenced.
    for stale in out_dir.glob("*_start*_ctx*.png"):
        stale.unlink()
    (out_dir / "samples.json").unlink(missing_ok=True)

    rng = np.random.default_rng()
    ohlc_cols = ["open", "high", "low", "close"]
    ctx_bars_arr = np.array([cb for _, cb in test_pairs])
    ctx_bucket_arr = np.array([context_bucket(cb) for cb in ctx_bars_arr])

    def spread(ohlc: np.ndarray) -> float:
        """max - min of the 3 bars' 6 open/close values -- how wide a candle path is."""
        return float(ohlc[:, [0, 3]].max() - ohlc[:, [0, 3]].min())

    records = []
    plotted = 0
    for case in CASE_LABELS:
        for ctx_bucket_name in ("narrow", "wide"):
            candidates = np.where((cvae_label == case) & (ctx_bucket_arr == ctx_bucket_name))[0]
            if len(candidates) == 0:
                logger.warning(
                    "no test-set windows found for case=%s ctx_bucket=%s -- skipping", case, ctx_bucket_name
                )
                continue
            idx = int(rng.choice(candidates))
            start_idx, ctx_bars = test_pairs[idx]

            pt_ohlc = reconstruct_prices(pt_y[idx, :12].reshape(3, 4), close_0[idx])  # (3,4)
            buy_price = float(close_0[idx])
            pt_tp = float(pt_take_profit[idx])
            cvae_tp = float(cvae_take_profit[idx])
            # Render whichever draw's own exit price is closest to the target actually
            # used (a percentile across all k draws) -- not always draw 0 -- so the
            # candles shown are visually consistent with the line drawn over them.
            draw_idx = nearest_draw_index(cvae_exit_prices[:, idx], cvae_tp)
            cvae_ohlc = reconstruct_prices(cvae_y[draw_idx, idx, :12].reshape(3, 4), close_0[idx])  # (3,4)

            ctx_tail = min(ctx_bars, 20)
            hz_start = start_idx + ctx_bars
            plot_rows = df.iloc[hz_start - ctx_tail : hz_start + 3]
            true_df = plot_rows.set_index("datetime")[["open", "high", "low", "close", "volume"]]

            pt_df = true_df.copy()
            pt_df.loc[pt_df.index[-3:], ohlc_cols] = pt_ohlc
            cvae_df = true_df.copy()
            cvae_df.loc[cvae_df.index[-3:], ohlc_cols] = cvae_ohlc

            true_horizon_ohlc = true_df.iloc[-HORIZON:][["open", "high", "low", "close"]].to_numpy()

            fig = plt.figure(figsize=(16, 8))
            outer = fig.add_gridspec(6, 2, width_ratios=[1, 1], hspace=0.7, wspace=0.25)

            render_panel(
                fig, outer[0:4, 0], outer[4:6, 0], true_df, buy_price, "Ground truth",
                sell_targets=[
                    ("PatchTST TP", pt_tp, "tab:orange", "-"),
                    ("CVAE TP", cvae_tp, "tab:green", "-"),
                ],
            )
            pt_realized, pt_would_enter = render_panel(
                fig, outer[0:2, 1], outer[2, 1], pt_df, buy_price, "PatchTST generated",
                sell_targets=[("PatchTST TP", pt_tp, "tab:orange", "-")],
                true_horizon_ohlc=true_horizon_ohlc, sell_limit=pt_tp,
            )
            cvae_realized, cvae_would_enter = render_panel(
                fig, outer[3:5, 1], outer[5, 1], cvae_df, buy_price, "CVAE generated (draw nearest target)",
                sell_targets=[("CVAE TP", cvae_tp, "tab:green", "-")],
                true_horizon_ohlc=true_horizon_ohlc, sell_limit=cvae_tp,
            )

            fig.suptitle(
                f"CVAE case: {CASE_TITLES[case]}  |  {ctx_bucket_name} ctx "
                f"(ctx_bars={ctx_bars}, start_idx={start_idx})"
            )
            out_path = out_dir / f"{case}_{ctx_bucket_name}_start{start_idx}_ctx{ctx_bars}.png"
            fig.savefig(out_path, bbox_inches="tight")
            plt.close(fig)
            plotted += 1

            # realized is None whenever would_enter is False -- no order was ever placed,
            # so there's nothing to report as realized (see trade_table_text).
            pt_realized_price, pt_outcome = pt_realized if pt_realized is not None else (None, None)
            cvae_realized_price, cvae_outcome = cvae_realized if cvae_realized is not None else (None, None)
            records.append({
                "case": case,
                "ctx_bucket": ctx_bucket_name,
                "file": out_path.name,
                "ctx_bars": ctx_bars,
                "start_idx": start_idx,
                "buy_price": buy_price,
                "ground_truth": {"candles": true_horizon_ohlc.tolist(), "spread": spread(true_horizon_ohlc)},
                "patchtst": {
                    "candles": pt_ohlc.tolist(),
                    "spread": spread(pt_ohlc),
                    "sell_limit": pt_tp,
                    "would_enter": pt_would_enter,
                    "outcome": pt_outcome,
                    "realized_price": pt_realized_price,
                    "realized_return_pct": (
                        (pt_realized_price / buy_price - 1.0) * 100 if pt_realized_price is not None else None
                    ),
                },
                "cvae": {
                    "candles": cvae_ohlc.tolist(),
                    "spread": spread(cvae_ohlc),
                    "sell_limit": cvae_tp,
                    "would_enter": cvae_would_enter,
                    "outcome": cvae_outcome,
                    "realized_price": cvae_realized_price,
                    "realized_return_pct": (
                        (cvae_realized_price / buy_price - 1.0) * 100 if cvae_realized_price is not None else None
                    ),
                },
            })
    logger.info("wrote %d sample plots to %s", plotted, out_dir)

    samples_path = out_dir / "samples.json"
    with open(samples_path, "w") as f:
        json.dump(records, f, indent=2)
    logger.info("wrote %d sample records to %s", len(records), samples_path)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    logger.info("device: %s", device)

    # CVAE.sample()'s reparameterize() draws from torch's global RNG (torch.randn_like) --
    # without seeding it, CVAE's backtest/metrics numbers drift between runs on the exact
    # same fixed test set, even though the window sampling below is already seeded.
    torch.manual_seed(args.seed)

    pt_ckpt = torch.load(args.patchtst_checkpoint, map_location=device, weights_only=False)
    cvae_ckpt = torch.load(args.cvae_checkpoint, map_location=device, weights_only=False)

    pt_cfg = pt_ckpt["config"]
    cvae_cfg = cvae_ckpt["config"]
    assert pt_cfg["data_path"] == cvae_cfg["data_path"], "both checkpoints must share the same data_path"

    df, bounds, stats = build_dataset(pt_cfg["data_path"])
    feat, opens, closes = extract_arrays(df)

    sell_bound = train_exit_return_bound(opens, closes, bounds["train"], percentile=args.sell_bound_percentile)
    logger.info(
        "sell-price shrink bound: p%.1f of |anchored log return| over train = %.4f (vs. model's own MAX_LOG_RETURN)",
        args.sell_bound_percentile, sell_bound,
    )

    test_sampler = WindowSampler(*bounds["test"])
    rng = np.random.default_rng(args.seed)
    test_pairs = test_sampler.draw(args.n_test_windows, rng)
    logger.info("evaluating on %d fixed test windows", len(test_pairs))

    test_ds = WindowDataset(feat, opens, closes, test_pairs)
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    patchtst = PatchTST(**pt_cfg["model"]).to(device)
    patchtst.load_state_dict(pt_ckpt["model_state"])
    patchtst.eval()

    cvae = CVAEInpainting(**cvae_cfg["model"]).to(device)
    cvae.load_state_dict(cvae_ckpt["model_state"])
    cvae.eval()

    true_y, pt_y, cvae_y, close_0, ctx_bars = collect_predictions(
        patchtst, cvae, test_loader, device, args.num_samples
    )

    n, k = len(pt_y), cvae_y.shape[0]
    pt_y[:, :12] = shrink_components(pt_y[:, :12].reshape(n, 3, 4), sell_bound).reshape(n, 12)
    cvae_y[..., :12] = shrink_components(cvae_y[..., :12].reshape(k, n, 3, 4), sell_bound).reshape(k, n, 12)

    # Computed once here (not just inside run_backtest) so make_plots can select
    # illustrative windows using the exact same take-profit targets and case labels the
    # backtest reports, instead of re-deriving them from a freshly re-sampled CVAE draw.
    pt_price = pt_y[:, :12].reshape(n, 3, 4)
    cvae_price_samples = cvae_y[..., :12].reshape(k, n, 3, 4)
    true_ohlc_full = reconstruct_prices(true_y[:, :12].reshape(n, 3, 4), close_0)
    pt_take_profit = max_close_from_components(pt_price, close_0)
    cvae_exit_prices = exit_price_from_components(cvae_price_samples, close_0)
    cvae_take_profit = np.percentile(cvae_exit_prices, args.cvae_sell_quantile, axis=0)
    cvae_outcomes = model_trade_outcomes(true_ohlc_full, cvae_take_profit, close_0)

    results = {"overall": metrics_for_slice(true_y, pt_y, cvae_y, close_0, stats)}
    results["overall"]["backtest"] = run_backtest(true_y, pt_y, cvae_y, close_0, args.cvae_sell_quantile)
    results["overall"]["backtest"]["buy_and_hold"] = buy_and_hold_benchmark(df, bounds)

    buckets = np.array([context_bucket(c) for c in ctx_bars])
    for bucket in BUCKETS:
        mask = buckets == bucket
        results[bucket] = metrics_for_slice(
            true_y[mask], pt_y[mask], cvae_y[:, mask], close_0[mask], stats
        )
        results[bucket]["backtest"] = run_backtest(
            true_y[mask], pt_y[mask], cvae_y[:, mask], close_0[mask], args.cvae_sell_quantile
        )

    Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.metrics_out, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("wrote metrics to %s", args.metrics_out)
    logger.info("overall: %s", json.dumps(results["overall"], indent=2))

    make_plots(
        df, test_pairs, pt_y, cvae_y, close_0, pt_take_profit, cvae_take_profit,
        cvae_exit_prices, cvae_outcomes["label"], args,
    )


if __name__ == "__main__":
    main()
