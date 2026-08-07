"""Walk-forward backtest of PatchTST vs. CVAE over the SPY test period.

Replays the test period once, in real chronological order, one decision at a time --
the way an actual trading account would experience it -- rather than evaluating a large
batch of independently-sampled windows (an earlier version of this script did both; the
random-sample version was dropped as uninformative once the walk-forward backtest below
existed: overlapping windows, no real equity curve, and no way to reuse for the
sample-plot illustrations without a second, disconnected population).

Usage:
    python steven/src/evaluate.py \\
        --patchtst-checkpoint steven/outputs/patchtst_checkpoint.pt \\
        --cvae-checkpoint steven/outputs/cvae_checkpoint.pt
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mplfinance as mpf

from src.data_pipeline import (
    HORIZON,
    MAX_CONTEXT,
    build_dataset,
    build_window,
    exit_price_from_components,
    extract_arrays,
    max_close_from_components,
    per_bar_close_return,
    reconstruct_prices,
    shrink_components,
    to_patchtst_input,
    train_exit_return_bound,
)
from src.models.cvae_inpainting import CVAEInpainting
from src.models.patchtst import PatchTST

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

WALK_FORWARD_CTX_BARS = MAX_CONTEXT  # 70 -- see the walk-forward module note below


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--patchtst-checkpoint", type=str, default="steven/outputs/patchtst_checkpoint.pt")
    p.add_argument("--cvae-checkpoint", type=str, default="steven/outputs/cvae_checkpoint.pt")
    p.add_argument("--num-samples", type=int, default=5, help="K sampled draws per CVAE decision.")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--metrics-out", type=str, default="steven/outputs/metrics.json")
    p.add_argument("--plots-dir", type=str, default="steven/outputs/sample_plots")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument(
        "--sell-bound-percentile", type=float, default=99.0,
        help="Percentile (over train data) of |close_0-anchored log return| used to shrink predicted "
        "price components -- a tighter, empirically-calibrated bound on top of the model's own "
        "MAX_LOG_RETURN, applied everywhere. See train_exit_return_bound.",
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
    p.add_argument(
        "--confidence-threshold", type=float, default=0.5,
        help="Confidence threshold used by the walk-forward backtest (see run_walk_forward). This "
        "changes which decisions actually get made along the way (trade vs. skip 1 bar), so it's a "
        "single fixed value rather than a sweep -- a different threshold produces a genuinely "
        "different simulated path, not just a different slice of the same one.",
    )
    p.add_argument(
        "--min-return-threshold", type=float, default=0.001,
        help="Minimum model-predicted exit return (fraction, e.g. 0.001 = 0.1%%) required to bother "
        "trading at all, on top of the plain target>buy eligibility check -- filters out trades with "
        "a technically-positive but trivially small predicted edge (see the 'skipped' case in "
        "classify_walk_forward_decision). A first-pass heuristic, not swept/tuned yet.",
    )
    return p.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def cvae_confidence_scores(cvae_price_samples: np.ndarray, close_0: np.ndarray) -> np.ndarray:
    """cvae_price_samples: (K,N,3,4). Returns (N,): fraction of the K sampled paths
    whose predicted exit price comes out above close_0. ~1 = strong sample consensus on
    an upward exit, ~0.5 = samples disagree (chop), ~0 = consensus downtrend -- one
    score captures all three of the "don't trade this" cases at once. Unlike PatchTST's
    confidence (see patchtst_walk_forward_confidence), this is inherently a per-window
    quantity -- no cross-window ranking needed -- so it works unmodified whether called on
    a batch or one window (N=1) at a time."""
    if cvae_price_samples.shape[1] == 0:
        return np.empty(0)
    exit_prices = exit_price_from_components(cvae_price_samples, close_0)  # (K, N)
    sample_return = exit_prices / close_0[None, :] - 1.0
    return (sample_return > 0).mean(axis=0)


def take_profit_exit(
    true_ohlc: np.ndarray, take_profit: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Simulates a take-profit-only limit sell order (no stop-loss -- see the module note
    above run_walk_forward for why) placed at the same time as the close_0 buy. Walks the
    3 REAL horizon bars in order; the first bar that reaches take_profit fills the order,
    in one of two ways:
    - **Touched mid-bar** (low <= take_profit <= high): fills at take_profit itself --
      the bar's range crossed the limit price but didn't open beyond it.
    - **Gapped through** (low > take_profit, i.e. the entire bar -- including its open --
      already sits above the target): a real limit sell order guarantees *at least* the
      limit price, so if the market gaps straight past it, the order fills immediately at
      the open, not at the stale take_profit level (which the market never actually
      touched, since low already clears it). This is still a take-profit win, just filled
      at a better-than-target price -- not an expiry.
    If neither ever happens across all 3 bars, the position is force-closed at the 3rd
    bar's real close instead (order expiry). true_ohlc: (N,3,4) real [open,high,low,close].
    take_profit: (N,). Returns (realized_sell_price (N,), hit_take_profit (N,) bool)."""
    n = true_ohlc.shape[0]
    open_ = true_ohlc[:, :, 0]
    low = true_ohlc[:, :, 2]
    high = true_ohlc[:, :, 1]

    sell_price = true_ohlc[:, HORIZON - 1, 3].copy()  # default: last horizon bar's real close
    hit_take_profit = np.zeros(n, dtype=bool)
    resolved = np.zeros(n, dtype=bool)

    for bar in range(HORIZON):
        touched = (~resolved) & (low[:, bar] <= take_profit) & (take_profit <= high[:, bar])
        gapped = (~resolved) & (low[:, bar] > take_profit)  # whole bar, incl. open, already above target

        sell_price[touched] = take_profit[touched]
        sell_price[gapped] = open_[:, bar][gapped]
        hit_take_profit[touched | gapped] = True
        resolved |= touched | gapped

    return sell_price, hit_take_profit


CASE_LABELS = ["win_take_profit", "win_expiry", "lose_expiry", "skipped", "no_trade"]
CASE_TITLES = {
    "win_take_profit": "Win -- take-profit hit",
    "win_expiry": "Win -- expiry (gain)",
    "lose_expiry": "Lose -- expiry (loss)",
    "skipped": "Skipped (low confidence or return too small)",
    "no_trade": "No trade (target <= buy)",
}


def classify_walk_forward_decision(
    eligible: bool, meets_return_threshold: bool, confident_enough: bool,
    hit_take_profit: bool, trade_return: float,
) -> tuple[str, bool]:
    """Sorts one walk-forward decision point into exactly one of CASE_LABELS. Returns
    (label, would_trade) -- would_trade is eligible AND meets_return_threshold AND
    confident_enough, and is exactly the condition run_walk_forward uses to decide
    whether to actually place the trade (and therefore lock out the next HORIZON bars).

    Two distinct reasons never make it to a trade: 'no_trade' (this model's own take-profit
    target never even clears the buy price -- no expected upside at all, confidence and
    predicted-return-size are irrelevant) and 'skipped' (the target does clear the buy
    price, but the predicted edge is smaller than min_return_threshold and/or the model
    isn't confident enough). An earlier version of this backtest only had 'no_trade' and
    silently folded every confidence/return-gated skip into whichever win/lose label the
    window would have gotten *had* it been traded -- which answered "what if everything
    eligible were always traded" rather than "what actually happened," and made a highly
    selective model look like it was winning/losing trades it never actually placed."""
    if not eligible:
        return "no_trade", False
    if not (meets_return_threshold and confident_enough):
        return "skipped", False
    if hit_take_profit:
        return "win_take_profit", True
    return ("win_expiry" if trade_return > 0 else "lose_expiry"), True


def outcome_breakdown(labels: np.ndarray) -> dict:
    """labels: (N,) from classify_walk_forward_decision. Returns {case: fraction of all N
    decisions in that case} -- rows sum to 1.0 (or empty dict if N=0)."""
    n = len(labels)
    if n == 0:
        return {}
    return {case: float((labels == case).mean()) for case in CASE_LABELS}


def nearest_draw_index(draw_exit_prices: np.ndarray, target: float) -> int:
    """draw_exit_prices: (K,) each of CVAE's K sampled draws' own exit_price_from_components
    value for one window. target: the take-profit price actually used for that window (a
    percentile across all K draws -- see make_cvae_predict_fn -- not tied to any single
    draw). Returns the index of whichever draw's own exit price is closest to target --
    so the one rendered/reported is visually consistent with a target that's really a
    statistic over the whole ensemble."""
    return int(np.argmin(np.abs(draw_exit_prices - target)))


def equity_stats(
    df: pd.DataFrame, entry_idx: int, exit_idx: int, entry_price: float, exit_price: float, total_return: float
) -> dict:
    """Shared entry/exit date + annualized-return bookkeeping for every benchmark/strategy
    below (buy-and-hold, the naive periodic benchmark, and each model's walk-forward run) --
    so every one of them is reported in the same shape and can be dropped into the same
    comparison table regardless of how its own total_return was actually produced."""
    entry, exit_ = df.iloc[entry_idx], df.iloc[exit_idx]
    # total_seconds(), not .days -- .days truncates to 0 for any exit less than 24h after
    # entry (routine for a single naive-periodic/walk-forward trade block), which would
    # divide by zero below. Every caller here spans real chronological bars from the same
    # df, so exit_idx > entry_idx guarantees a strictly positive elapsed time.
    elapsed_years = (exit_["datetime"] - entry["datetime"]).total_seconds() / (365.25 * 86400)
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


def buy_and_hold_benchmark(df: pd.DataFrame, entry_idx: int, exit_idx: int) -> dict:
    """Naive baseline: buy SPY at entry_idx's close, hold to exit_idx's close -- no model,
    no confidence threshold, no trade selectivity. entry_idx is the caller's choice so this
    lines up with the walk-forward's own first tradeable decision point (see main()) --
    not the test split's literal first bar."""
    entry_price = float(df.iloc[entry_idx]["close"])
    exit_price = float(df.iloc[exit_idx]["close"])
    total_return = exit_price / entry_price - 1.0
    return equity_stats(df, entry_idx, exit_idx, entry_price, exit_price, total_return)


def naive_periodic_benchmark(df: pd.DataFrame, closes: np.ndarray, t0: int, test_hi: int) -> dict:
    """No model, no signal: tiles the test range in non-overlapping 3-bar (HORIZON) blocks
    starting right after t0 (the same anchor the walk-forward strategies use) -- buy at the
    close of each block's 1st bar, sell at the close of its 3rd, then immediately start the
    next block (buy(t0+1)/sell(t0+3), buy(t0+4)/sell(t0+6), ...). Always trades, every
    block, unconditionally -- no take-profit target, no confidence gate, no expiry logic.
    Same trade cadence/shape as the walk-forward strategies (one round trip per
    HORIZON-bar block) but zero selectivity, to check whether the models' confidence-gated
    entries actually beat blind periodic exposure to the same instrument. Equity compounds
    block to block, same as run_walk_forward."""
    equity = 1.0
    trades: list[dict] = []
    k = 0
    while True:
        buy_idx, sell_idx = t0 + 3 * k + 1, t0 + 3 * k + 3
        if sell_idx >= test_hi:
            break
        buy_price, sell_price = float(closes[buy_idx]), float(closes[sell_idx])
        trade_return = sell_price / buy_price - 1.0
        equity *= 1.0 + trade_return
        trades.append({"buy_idx": buy_idx, "sell_idx": sell_idx, "buy_price": buy_price, "trade_return": trade_return})
        k += 1

    out = {"n_trades": len(trades), "total_return": equity - 1.0}
    if not trades:
        out.update(win_rate=None, take_profit_rate=None, avg_return=None)
        return out
    rets = np.array([t["trade_return"] for t in trades])
    out["win_rate"] = float((rets > 0).mean())
    out["take_profit_rate"] = None  # no take-profit order in this strategy at all
    out["avg_return"] = float(rets.mean())
    out.update(equity_stats(
        df, trades[0]["buy_idx"], trades[-1]["sell_idx"],
        trades[0]["buy_price"], float(closes[trades[-1]["sell_idx"]]), out["total_return"],
    ))
    return out


# ---------------------------------------------------------------------------
# Walk-forward backtest -- a realistic, sequential simulation
#
# Walks the test range in real chronological order, one hour at a time, always using the
# max context length (WALK_FORWARD_CTX_BARS=70): "a trader checks in every hour with the
# most recent 70 candles." No trade -> advance 1 bar and re-check next hour. Trade -> lock
# out the next HORIZON bars regardless of whether/when the take-profit fills within them
# (an intentional simplification so the next decision point is always exactly 3 bars after
# the last, whether that trade won early, won late, or lost), then resume. Equity compounds
# trade to trade starting from 1.0, so total_return is a real simulated account balance
# over time, never a sum across possibly-overlapping trades.
#
# The first decision point is the earliest start_idx with a full 70-bar context entirely
# inside the test split (no candle before the test period is ever used), so it lands
# ~WALK_FORWARD_CTX_BARS/7 trading days after the test split's first bar -- the
# buy-and-hold and naive-periodic benchmarks are deliberately anchored to that same point
# (see main()) rather than the test split's literal first bar, so every number in the
# comparison table covers the identical calendar span.
#
# PatchTST's confidence score needs special handling: it has no sample distribution, so
# its "how big a move, relative to what this model usually predicts" score has to be a
# rank against *some* reference population of past predicted-move magnitudes. Ranking
# against a big batch of independently-sampled windows (an earlier version did this) is a
# subtle lookahead bug: at the very first decision in January, you'd already be ranking
# against predictions the model makes on data from as late as May -- information a live
# trader would not have yet. patchtst_walk_forward_confidence instead ranks each decision
# against an *online, expanding* reference built purely from this same walk's own prior
# decisions, in chronological order -- nothing from the future ever leaks in.
# ---------------------------------------------------------------------------


def patchtst_walk_forward_confidence(
    pt_price: np.ndarray, close_0: np.ndarray, magnitude_reference: list[float]
) -> float:
    """Single-decision-point confidence: (1) require all 3 predicted bar-close-returns to
    agree on the up direction (else auto-dismissed as choppy/down -- confidence 0), (2)
    rank the predicted exit-return magnitude against magnitude_reference, an ONLINE,
    expanding list of every prior coherent-up decision's own predicted magnitude from this
    same walk (see the module note above) -- not a separate batch, so nothing from the
    future ever leaks into a live decision's confidence. `magnitude_reference` is mutated
    in place: this window's own value is appended AFTER computing its rank, so later
    decisions can rank against it but it never ranks against itself. The very first
    coherent-up decision of the whole walk has nothing to rank against yet, so confidence
    defaults to a neutral 0.5 rather than 0.0 (which a naive percentile rank of an empty/
    singleton reference would give, silently zeroing out early trading opportunities)."""
    coherent_up = bool((per_bar_close_return(pt_price) > 0).all(axis=1)[0])
    if not coherent_up:
        return 0.0
    predicted_exit_return = float(exit_price_from_components(pt_price, close_0)[0] / close_0[0] - 1.0)
    confidence = float(np.mean(np.array(magnitude_reference) <= predicted_exit_return)) if magnitude_reference else 0.5
    magnitude_reference.append(predicted_exit_return)
    return confidence


def make_patchtst_predict_fn(patchtst, device: torch.device, sell_bound: float):
    """Returns a predict(w) -> {take_profit, confidence, price} closure, single-window
    (batch=1) at a time. `price` is this decision's own predicted (3,4) price components,
    kept around so make_plots can render exactly what this model predicted at this
    decision without re-running inference later (see make_plots' module note). Maintains
    its own magnitude_reference list across calls (see patchtst_walk_forward_confidence) --
    one closure instance = one independent walk = one independent reference."""
    magnitude_reference: list[float] = []

    def predict(w: dict) -> dict:
        context, patch_pad = to_patchtst_input(w["masked_tensor"])
        context_t = torch.from_numpy(context)[None].to(device)
        patch_pad_t = torch.from_numpy(patch_pad)[None].to(device)
        with torch.no_grad():
            pt_price_t, _ = patchtst(context_t, patch_pad_t)
        pt_price = shrink_components(pt_price_t.cpu().numpy(), sell_bound)  # (1,3,4)
        close_0 = np.array([w["close_0"]])
        take_profit = float(max_close_from_components(pt_price, close_0)[0])
        confidence = patchtst_walk_forward_confidence(pt_price, close_0, magnitude_reference)
        return {"take_profit": take_profit, "confidence": confidence, "price": pt_price[0]}

    return predict


def make_cvae_predict_fn(
    cvae, device: torch.device, sell_bound: float, num_samples: int, cvae_sell_quantile: float
):
    """Returns a predict(w) -> {take_profit, confidence, price} closure, single-window
    (batch=1) at a time. `price` is whichever of the K sampled draws' own exit price is
    closest to the take-profit target actually used (see nearest_draw_index) -- chosen
    right here, at generation time, using the exact RNG draws this decision consumed, so
    later re-rendering it in make_plots never risks a different (freshly re-sampled) set
    of draws implying a different target/outcome than the one that actually decided this
    window's case label."""

    def predict(w: dict) -> dict:
        masked_t = torch.from_numpy(w["masked_tensor"])[None].to(device)
        with torch.no_grad():
            cvae_price_t, _ = cvae.sample(masked_t, k=num_samples)  # (K,1,3,4)
        cvae_price = shrink_components(cvae_price_t.cpu().numpy(), sell_bound)
        close_0 = np.array([w["close_0"]])
        exit_prices = exit_price_from_components(cvae_price, close_0)  # (K,1)
        take_profit = float(np.percentile(exit_prices[:, 0], cvae_sell_quantile))
        confidence = float(cvae_confidence_scores(cvae_price, close_0)[0])
        draw_idx = nearest_draw_index(exit_prices[:, 0], take_profit)
        return {"take_profit": take_profit, "confidence": confidence, "price": cvae_price[draw_idx, 0]}

    return predict


def run_walk_forward(
    predict_fn, feat: np.ndarray, opens: np.ndarray, closes: np.ndarray,
    test_lo: int, test_hi: int, confidence_threshold: float, min_return_threshold: float,
) -> dict:
    """predict_fn(w: dict) -> {take_profit, confidence, price} is model-specific -- see
    make_patchtst_predict_fn/make_cvae_predict_fn. Returns {trades, decisions,
    equity_final}: `decisions` is every decision point visited (traded or not), each
    classified into one of CASE_LABELS (see classify_walk_forward_decision) and carrying
    everything make_plots needs to render it later (price components, real OHLC, buy/sell
    prices) without re-running inference; `trades` is the subset that actually got
    traded -- disjoint models can and do follow different real-time paths here, since a
    trade vs. no-trade call changes how far this model's own clock advances next."""
    start_idx = test_lo
    equity = 1.0
    trades: list[dict] = []
    decisions: list[dict] = []

    while start_idx + WALK_FORWARD_CTX_BARS + HORIZON <= test_hi:
        w = build_window(feat, opens, closes, start_idx, WALK_FORWARD_CTX_BARS)
        close_0 = w["close_0"]
        true_price = w["y"][:12].reshape(1, 3, 4)
        true_ohlc = reconstruct_prices(true_price, np.array([close_0]))[0]

        pred = predict_fn(w)
        take_profit, confidence, price = pred["take_profit"], pred["confidence"], pred["price"]
        predicted_return = take_profit / close_0 - 1.0
        eligible = take_profit > close_0
        meets_return_threshold = predicted_return >= min_return_threshold
        confident_enough = confidence >= confidence_threshold

        sell_price_arr, hit_tp_arr = take_profit_exit(true_ohlc[None], np.array([take_profit]))
        sell_price, hit_tp = float(sell_price_arr[0]), bool(hit_tp_arr[0])
        trade_return = float(sell_price / close_0 - 1.0)
        label, would_trade = classify_walk_forward_decision(
            eligible, meets_return_threshold, confident_enough, hit_tp, trade_return
        )

        decision = {
            "start_idx": start_idx,
            "close_0": float(close_0),
            "take_profit": float(take_profit),
            "confidence": float(confidence),
            "price": price,  # (3,4) predicted components, for plotting
            "true_ohlc": true_ohlc,  # (3,4) real components, for plotting
            "label": label,
            "would_trade": would_trade,
            "sell_price": sell_price,
            "hit_take_profit": hit_tp,
            "trade_return": trade_return,
        }
        decisions.append(decision)

        if would_trade:
            equity *= 1.0 + trade_return
            trades.append(decision)
            start_idx += HORIZON
        else:
            start_idx += 1

    return {"trades": trades, "decisions": decisions, "equity_final": equity}


def walk_forward_stats(df: pd.DataFrame, result: dict) -> dict:
    """Reduces run_walk_forward's raw trade/decision log into the same reportable shape as
    buy_and_hold_benchmark/naive_periodic_benchmark, plus the case breakdown across every
    decision point checked (see outcome_breakdown) -- independent of confidence_threshold
    and min_return_threshold in the sense that it reports exactly which of the 5 cases
    each decision landed in, rather than re-deriving anything."""
    trades, decisions, equity_final = result["trades"], result["decisions"], result["equity_final"]
    labels = np.array([d["label"] for d in decisions], dtype=object)
    out = {
        "n_decisions": len(decisions),
        "n_trades": len(trades),
        "outcome_breakdown": outcome_breakdown(labels),
        "total_return": equity_final - 1.0,
    }
    if not trades:
        out.update(win_rate=None, take_profit_rate=None, avg_return=None)
        return out

    rets = np.array([t["trade_return"] for t in trades])
    hits = np.array([t["hit_take_profit"] for t in trades])
    out["win_rate"] = float((rets > 0).mean())
    out["take_profit_rate"] = float(hits.mean())
    out["avg_return"] = float(rets.mean())

    first, last = trades[0], trades[-1]
    entry_idx = first["start_idx"] + WALK_FORWARD_CTX_BARS - 1  # this trade's close_0 bar
    exit_idx = last["start_idx"] + WALK_FORWARD_CTX_BARS + HORIZON - 1  # last horizon bar
    out.update(equity_stats(df, entry_idx, exit_idx, first["close_0"], last["sell_price"], out["total_return"]))
    return out


# ---------------------------------------------------------------------------
# Sample plots
# ---------------------------------------------------------------------------


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
    would_enter: bool | None = None, realized: tuple[float, str] | None = None,
) -> str:
    """ohlc: (3,4) [open,high,low,close] for this panel's 3 horizon bars (used for the
    per-candle numbers in the text table -- always shown). sell_limit: this model's own
    predicted take-profit price -- None for the ground truth panel and for a model panel
    the walk-forward never actually evaluated at this exact window (see make_plots).
    would_enter/realized are the walk-forward's own precomputed decision (see
    run_walk_forward), passed straight through rather than recomputed here, so this text
    always matches exactly what the walk actually decided and realized. realized is
    (realized_price, outcome) where outcome is 'take_profit' or 'expired' -- only
    meaningful when would_enter is True."""
    lines = [
        f"C1  O={ohlc[0,0]:.2f}  C={ohlc[0,3]:.2f}    "
        f"C2  O={ohlc[1,0]:.2f}  C={ohlc[1,3]:.2f}    "
        f"C3  O={ohlc[2,0]:.2f}  C={ohlc[2,3]:.2f}",
    ]
    if sell_limit is not None:
        lines.append(f"Buy = {buy_price:.2f}   Take-profit = {sell_limit:.2f}")
        decision = "ENTER long" if would_enter else "NO TRADE / SKIPPED"
        lines.append(f"Trade decision: {decision}")

        if would_enter and realized is not None:
            realized_price, outcome = realized
            realized_return = (realized_price / buy_price - 1.0) * 100
            status = {
                "take_profit": "TAKE PROFIT -> filled at target",
                "expired": "EXPIRED -> forced exit at real C3 close",
            }[outcome]
            lines.append(
                f"vs. real price action: {status}  |  realized sell = {realized_price:.2f}  |  "
                f"return = {realized_return:+.2f}%"
            )
    return "\n".join(lines)


def render_panel(
    fig, gs_chart, gs_text, sub_df: pd.DataFrame, buy_price: float, title: str,
    sell_targets: list[tuple[str, float, str, str]],
    sell_limit: float | None = None, would_enter: bool | None = None,
    realized: tuple[float, str] | None = None,
) -> None:
    """sell_limit/would_enter/realized are the walk-forward's own precomputed decision for
    this model at this window (None/None/None for the ground truth panel, which has no
    trade decision of its own, and for a model panel the walk never actually evaluated
    here -- see make_plots)."""
    ax = fig.add_subplot(gs_chart)
    mpf.plot(sub_df, type="candle", ax=ax, style="yahoo", volume=False)
    ax.set_title(title, fontsize=10)
    ax.tick_params(axis="x", labelrotation=30, labelsize=7)
    draw_horizon_box(ax, len(sub_df))

    ohlc = sub_df.iloc[-HORIZON:][["open", "high", "low", "close"]].to_numpy()
    text = trade_table_text(ohlc, buy_price, sell_limit, would_enter, realized)
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


def spread(ohlc: np.ndarray) -> float:
    """max - min of the 3 bars' 6 open/close values -- how wide a candle path is."""
    return float(ohlc[:, [0, 3]].max() - ohlc[:, [0, 3]].min())


def make_plots(df: pd.DataFrame, pt_decisions: list[dict], cvae_decisions: list[dict], args) -> None:
    """Selects one illustrative example per CASE_LABELS, sourced from CVAE's own
    walk-forward decision log (not a separate random-sample population -- an earlier
    version drew these from an independently-sampled 3000-window batch, which the report
    dropped as uninformative once the walk-forward became the only backtest; reusing that
    separate batch here would also mean the illustrated cases could disagree with the
    walk-forward table's own outcome-breakdown numbers). Cases are categorized by CVAE's
    own decision; PatchTST's panel on the same figure shows whatever it separately decided
    at that same real start_idx, uncontrolled.

    PatchTST and CVAE walk independently and can visit different start_idx sequences once
    their trade/no-trade calls diverge (see run_walk_forward's module note) -- so a given
    CVAE decision isn't guaranteed to have a matching PatchTST decision at the exact same
    start_idx. This prefers CVAE candidates where PatchTST does have a decision at that
    start_idx (so the 3-panel comparison is on identical real price action); if none of a
    case's CVAE candidates overlap with a PatchTST decision, one is still picked and
    PatchTST's panel instead shows the plain real candles, titled to say so.

    Reuses each decision's own already-generated `price` (and, for CVAE, the specific draw
    nearest its own take-profit target -- see make_cvae_predict_fn) rather than re-running
    inference after the fact: CVAE's sampling is stochastic, so a fresh re-sample here
    could disagree with the draws that actually earned this window its case label.

    Each figure has 3 candlestick panels: ground truth (left), PatchTST's generated
    horizon candles (top right), CVAE's generated horizon candles (bottom right). Every
    panel gets a red box around the generated/predicted 3 candles, a dashed buy-price
    line, and a solid take-profit line per model, starting at the left edge of that box
    and running to the panel's right edge -- the ground truth panel shows both models'
    lines together for comparison. A line pinned outside the panel's y-range is drawn just
    inside the edge instead, annotated with its real price. The PatchTST and CVAE panels
    additionally report in the text table whether that model's take-profit order actually
    filled or expired against the real price action -- computed once, during the walk
    itself (see run_walk_forward), not recomputed here. Also writes samples.json alongside
    the PNGs -- the same per-sample numbers (candles, buy/take-profit prices, outcome,
    spread) plus its case, structured for update_report.py to regenerate v1.md's Results
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
    pt_by_start = {d["start_idx"]: d for d in pt_decisions}

    records = []
    plotted = 0
    for case in CASE_LABELS:
        candidates = [d for d in cvae_decisions if d["label"] == case]
        if not candidates:
            logger.warning("no CVAE walk-forward decisions found for case=%s -- skipping", case)
            continue
        overlapping = [d for d in candidates if d["start_idx"] in pt_by_start]
        if not overlapping:
            logger.warning(
                "case=%s: no CVAE candidate lines up with a PatchTST decision at the same "
                "start_idx -- PatchTST panel will show real (uncontrolled) candles only", case,
            )
        pool = overlapping if overlapping else candidates
        cvae_d = pool[int(rng.integers(len(pool)))]
        pt_d = pt_by_start.get(cvae_d["start_idx"])

        start_idx, ctx_bars = cvae_d["start_idx"], WALK_FORWARD_CTX_BARS
        buy_price = cvae_d["close_0"]
        cvae_tp = cvae_d["take_profit"]
        cvae_ohlc = reconstruct_prices(cvae_d["price"], buy_price)  # (3,4)

        ctx_tail = min(ctx_bars, 20)
        hz_start = start_idx + ctx_bars
        plot_rows = df.iloc[hz_start - ctx_tail : hz_start + 3]
        true_df = plot_rows.set_index("datetime")[["open", "high", "low", "close", "volume"]]

        cvae_df = true_df.copy()
        cvae_df.loc[cvae_df.index[-3:], ohlc_cols] = cvae_ohlc

        true_horizon_ohlc = true_df.iloc[-HORIZON:][["open", "high", "low", "close"]].to_numpy()

        sell_targets = [("CVAE TP", cvae_tp, "tab:green", "-")]
        if pt_d is not None:
            pt_tp = pt_d["take_profit"]
            pt_ohlc = reconstruct_prices(pt_d["price"], buy_price)  # (3,4)
            pt_df = true_df.copy()
            pt_df.loc[pt_df.index[-3:], ohlc_cols] = pt_ohlc
            sell_targets.insert(0, ("PatchTST TP", pt_tp, "tab:orange", "-"))
        else:
            pt_tp = pt_ohlc = None
            pt_df = true_df

        fig = plt.figure(figsize=(16, 8))
        outer = fig.add_gridspec(6, 2, width_ratios=[1, 1], hspace=0.7, wspace=0.25)

        render_panel(
            fig, outer[0:4, 0], outer[4:6, 0], true_df, buy_price, "Ground truth",
            sell_targets=sell_targets,
        )
        render_panel(
            fig, outer[0:2, 1], outer[2, 1], pt_df, buy_price,
            "PatchTST generated" if pt_d is not None else "PatchTST (not evaluated this window)",
            sell_targets=[("PatchTST TP", pt_tp, "tab:orange", "-")] if pt_d is not None else [],
            sell_limit=pt_tp, would_enter=(pt_d["would_trade"] if pt_d is not None else None),
            realized=((pt_d["sell_price"], "take_profit" if pt_d["hit_take_profit"] else "expired") if pt_d is not None else None),
        )
        render_panel(
            fig, outer[3:5, 1], outer[5, 1], cvae_df, buy_price, "CVAE generated (draw nearest target)",
            sell_targets=[("CVAE TP", cvae_tp, "tab:green", "-")],
            sell_limit=cvae_tp, would_enter=cvae_d["would_trade"],
            realized=(cvae_d["sell_price"], "take_profit" if cvae_d["hit_take_profit"] else "expired"),
        )

        fig.suptitle(f"CVAE case: {CASE_TITLES[case]}  |  ctx_bars={ctx_bars}, start_idx={start_idx}")
        out_path = out_dir / f"{case}_start{start_idx}_ctx{ctx_bars}.png"
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        plotted += 1

        def model_record(d, ohlc, tp):
            if d is None:
                return {
                    "candles": true_horizon_ohlc.tolist(), "spread": spread(true_horizon_ohlc),
                    "sell_limit": None, "would_enter": False, "outcome": None,
                    "realized_price": None, "realized_return_pct": None,
                }
            outcome = "take_profit" if d["hit_take_profit"] else "expired"
            return {
                "candles": ohlc.tolist(),
                "spread": spread(ohlc),
                "sell_limit": tp,
                "would_enter": d["would_trade"],
                "outcome": outcome if d["would_trade"] else None,
                "realized_price": d["sell_price"] if d["would_trade"] else None,
                "realized_return_pct": (
                    (d["sell_price"] / buy_price - 1.0) * 100 if d["would_trade"] else None
                ),
            }

        records.append({
            "case": case,
            "file": out_path.name,
            "ctx_bars": ctx_bars,
            "start_idx": start_idx,
            "buy_price": buy_price,
            "ground_truth": {"candles": true_horizon_ohlc.tolist(), "spread": spread(true_horizon_ohlc)},
            "patchtst": model_record(pt_d, pt_ohlc, pt_tp),
            "cvae": model_record(cvae_d, cvae_ohlc, cvae_tp),
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
    # without seeding it, CVAE's walk-forward numbers drift between runs on unchanged
    # checkpoints.
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

    patchtst = PatchTST(**pt_cfg["model"]).to(device)
    patchtst.load_state_dict(pt_ckpt["model_state"])
    patchtst.eval()

    cvae = CVAEInpainting(**cvae_cfg["model"]).to(device)
    cvae.load_state_dict(cvae_ckpt["model_state"])
    cvae.eval()

    test_lo, test_hi = bounds["test"]
    wf_entry_idx = test_lo + WALK_FORWARD_CTX_BARS - 1  # first decision point's close_0 bar

    logger.info(
        "running walk-forward backtest (ctx=%d bars, confidence>=%.2f, min_return>=%.3f%%, %d..%d)...",
        WALK_FORWARD_CTX_BARS, args.confidence_threshold, args.min_return_threshold * 100, test_lo, test_hi,
    )
    pt_predict = make_patchtst_predict_fn(patchtst, device, sell_bound)
    cvae_predict = make_cvae_predict_fn(cvae, device, sell_bound, args.num_samples, args.cvae_sell_quantile)
    pt_wf = run_walk_forward(
        pt_predict, feat, opens, closes, test_lo, test_hi, args.confidence_threshold, args.min_return_threshold
    )
    cvae_wf = run_walk_forward(
        cvae_predict, feat, opens, closes, test_lo, test_hi, args.confidence_threshold, args.min_return_threshold
    )
    logger.info(
        "walk-forward: PatchTST %d trades / %d decisions, CVAE %d trades / %d decisions",
        len(pt_wf["trades"]), len(pt_wf["decisions"]), len(cvae_wf["trades"]), len(cvae_wf["decisions"]),
    )

    results = {
        "walk_forward": {
            "ctx_bars": WALK_FORWARD_CTX_BARS,
            "confidence_threshold": args.confidence_threshold,
            "min_return_threshold": args.min_return_threshold,
            "buy_and_hold": buy_and_hold_benchmark(df, wf_entry_idx, test_hi - 1),
            "naive_periodic": naive_periodic_benchmark(df, closes, wf_entry_idx, test_hi),
            "patchtst": walk_forward_stats(df, pt_wf),
            "cvae": walk_forward_stats(df, cvae_wf),
        }
    }

    Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.metrics_out, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("wrote metrics to %s", args.metrics_out)
    logger.info("walk_forward: %s", json.dumps(results["walk_forward"], indent=2))

    make_plots(df, pt_wf["decisions"], cvae_wf["decisions"], args)


if __name__ == "__main__":
    main()
