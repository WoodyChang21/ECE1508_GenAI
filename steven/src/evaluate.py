"""Random-sample evaluation of PatchTST vs. CVAE over the SPY test period (no walk-forward,
no confidence-threshold sweep -- see the module note above main() for why, and
cvae_direction_collapse.md's "revisit the pre-walk-forward era" discussion for the full
context of switching back to this style).

Draws N independently-sampled (start_idx, ctx_bars) windows from the test period (across all
9 context lengths, see WindowSampler.draw) rather than replaying it sequentially. This is
deliberately NOT a real equity curve -- windows can overlap in calendar time, so there's no
single coherent account balance to compound, which is why no total_return/buy-and-hold
comparison is reported here (see random_sample_stats). The walk-forward backtest this
replaced (run_walk_forward/walk_forward_stats/make_plots below) stays in this file, unused by
main() by default, in case a sequential/compounding evaluation is wanted again later.

PRE-PIVOT STRATEGY, kept for history/v1.md only (steven4): bracket_exit/run_walk_forward/
make_cvae_predict_fn/make_patchtst_predict_fn/classify_walk_forward_decision/make_plots/
naive_periodic_benchmark below all assume a fixed HORIZON-bar bracket order (buy, place a
take-profit/stop-loss, walk exactly HORIZON real bars, see which fills first). steven4
replaced this *strategy* with a rolling hour-by-hour hold (see src/rolling_backtest.py and
steven/rolling_hour_backtest.md) that predicts and re-predicts one candle at a time and
holds open-endedly rather than for a fixed window -- the bracket-order *design* here is not
being carried forward. Its mechanical bugs found during the steven4 migration audit
(several literal-3/12/15 shape and cadence assumptions that didn't derive from HORIZON --
naive_periodic_benchmark's trade-block width, run_walk_forward's y-unpack, make_plots'/
make_random_sample_plots' plotted-region width) WERE fixed, so this file still loads and
runs correctly against a HORIZON=1 checkpoint and stays testable -- but the strategy design
itself (fixed-width bracket order vs. open-ended rolling hold) was deliberately not
redesigned here; that redesign lives in rolling_backtest.py instead.

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

import src.data_pipeline as dp
import src.momentum_pipeline as momentum_pipeline
from src.data_pipeline import (
    HORIZON,
    MAX_CONTEXT,
    RollingWindowSampler,
    WindowSampler,
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
N_PRICE = HORIZON * 4  # open_ret, body_ret, upper_wick, lower_wick, per horizon bar -- w["y"]'s price prefix width


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--patchtst-checkpoint", type=str, default="steven/outputs/patchtst_checkpoint.pt")
    p.add_argument("--cvae-checkpoint", type=str, default="steven/outputs/cvae_checkpoint.pt")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument(
        "--ctx-bars", type=int, default=MAX_CONTEXT,
        help="Fixed context length (bars) used both for the trend classification lookback and for "
        "every rendered panel -- MAX_CONTEXT (70) by default, same window PatchTST/CVAE were "
        "trained across.",
    )
    p.add_argument(
        "--trend-lookback", type=int, default=20,
        help="Bars of causal context (never the horizon) used to classify a window as uptrend/"
        "downtrend/choppy -- see classify_trend.",
    )
    p.add_argument(
        "--step", type=int, default=5,
        help="Stride (bars) when scanning the test split for candidate windows to classify -- doesn't "
        "need to be exhaustive, just enough variety to find a clear example of each trend label.",
    )
    p.add_argument("--charts-dir", type=str, default="steven/outputs/scenario_charts")
    p.add_argument(
        "--n-examples", type=int, default=3,
        help="Number of example windows to render per trend label (spread across that label's "
        "candidate pool, not clustered near the median) -- e.g. 3 -> 9 charts total (3 labels).",
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


def bracket_exit(
    true_ohlc: np.ndarray, take_profit: np.ndarray, stop_loss: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulates a bracket order (take-profit limit sell above entry, stop-loss below it --
    see the module note above run_walk_forward for the current stop-loss level and why it's
    shared across models) placed at the same time as the close_0 buy. Walks the 3 REAL
    horizon bars in order; the first bar that reaches either level fills the order, in one
    of two ways per level:
    - **Touched mid-bar** (low <= level <= high): fills at that level itself -- the bar's
      range crossed it but didn't open beyond it.
    - **Gapped through** (take-profit: low > take_profit; stop-loss: high < stop_loss,
      i.e. the entire bar -- including its open -- already sits beyond the level): a real
      limit/stop order guarantees *at least* that price, so if the market gaps straight
      past it, the order fills immediately at the open, not at the stale level (which the
      market never actually touched). Still counts as that level being hit, not an expiry.
    Pass stop_loss=-inf for a row to disable its stop-loss entirely (it can then never
    touch or gap, by construction).

    **Same-bar tie-break**: OHLC bars don't record intrabar order, so a bar whose range
    would trigger both levels at once is genuinely ambiguous. Resolved pessimistically --
    stop-loss wins -- by checking stop-loss first each bar and excluding any row it just
    resolved from that bar's take-profit check. Assuming the optimistic order (take-profit
    first) would quietly inflate the backtest's own numbers with an assumption we can't
    verify from OHLC data; the worst-case assumption is the defensible one for a strategy's
    self-reported results.

    If neither level is ever hit across all 3 bars, the position is force-closed at the 3rd
    bar's real close instead (order expiry). true_ohlc: (N,3,4) real [open,high,low,close].
    take_profit, stop_loss: (N,). Returns (realized_sell_price (N,), hit_take_profit (N,)
    bool, hit_stop_loss (N,) bool)."""
    n = true_ohlc.shape[0]
    open_ = true_ohlc[:, :, 0]
    low = true_ohlc[:, :, 2]
    high = true_ohlc[:, :, 1]

    sell_price = true_ohlc[:, HORIZON - 1, 3].copy()  # default: last horizon bar's real close
    hit_take_profit = np.zeros(n, dtype=bool)
    hit_stop_loss = np.zeros(n, dtype=bool)
    resolved = np.zeros(n, dtype=bool)

    for bar in range(HORIZON):
        sl_touched = (~resolved) & (low[:, bar] <= stop_loss) & (stop_loss <= high[:, bar])
        sl_gapped = (~resolved) & (high[:, bar] < stop_loss)  # whole bar, incl. open, already below stop
        sl_hit = sl_touched | sl_gapped

        tp_touched = (~resolved) & (~sl_hit) & (low[:, bar] <= take_profit) & (take_profit <= high[:, bar])
        tp_gapped = (~resolved) & (~sl_hit) & (low[:, bar] > take_profit)  # whole bar already above target

        sell_price[sl_touched] = stop_loss[sl_touched]
        sell_price[sl_gapped] = open_[:, bar][sl_gapped]
        sell_price[tp_touched] = take_profit[tp_touched]
        sell_price[tp_gapped] = open_[:, bar][tp_gapped]
        hit_stop_loss |= sl_hit
        hit_take_profit |= tp_touched | tp_gapped
        resolved |= sl_hit | tp_touched | tp_gapped

    return sell_price, hit_take_profit, hit_stop_loss


CASE_LABELS = ["win_take_profit", "win_expiry", "lose_expiry", "lose_stop_loss", "skipped", "no_trade"]
CASE_TITLES = {
    "win_take_profit": "Win -- take-profit hit",
    "win_expiry": "Win -- expiry (gain)",
    "lose_expiry": "Lose -- expiry (loss)",
    "lose_stop_loss": "Lose -- stop-loss hit",
    "skipped": "Skipped (quality gate or return too small)",
    "no_trade": "No trade (target <= buy)",
}


def classify_walk_forward_decision(
    eligible: bool, meets_return_threshold: bool, passes_quality_gate: bool,
    hit_take_profit: bool, hit_stop_loss: bool, trade_return: float,
) -> tuple[str, bool]:
    """Sorts one walk-forward decision point into exactly one of CASE_LABELS. Returns
    (label, would_trade) -- would_trade is eligible AND meets_return_threshold AND
    passes_quality_gate, and is exactly the condition run_walk_forward uses to decide
    whether to actually place the trade (and therefore lock out the next HORIZON bars).
    passes_quality_gate is model-specific (PatchTST: boolean coherence; CVAE: always True
    -- see the module note above run_walk_forward for why CVAE has no gate at all anymore),
    fed in generically here since this function doesn't care which model produced it.

    Two distinct reasons never make it to a trade: 'no_trade' (this model's own take-profit
    target never even clears the buy price -- no expected upside at all, the quality gate
    and predicted-return-size are irrelevant) and 'skipped' (the target does clear the buy
    price, but the predicted edge is smaller than min_return_threshold and/or the model's
    own quality gate didn't clear). An earlier version of this backtest only had 'no_trade'
    and silently folded every gated skip into whichever win/lose label the window would
    have gotten *had* it been traded -- which answered "what if everything eligible were
    always traded" rather than "what actually happened," and made a highly selective model
    look like it was winning/losing trades it never actually placed.

    hit_stop_loss takes priority over trade_return's sign: a stop-loss hit is always a loss
    by construction (that's what triggered it), so it gets its own case rather than folding
    into 'lose_expiry' -- see bracket_exit for the fill logic and the module note above
    run_walk_forward for the current stop-loss level."""
    if not eligible:
        return "no_trade", False
    if not (meets_return_threshold and passes_quality_gate):
        return "skipped", False
    if hit_stop_loss:
        return "lose_stop_loss", True
    if hit_take_profit:
        return "win_take_profit", True
    return ("win_expiry" if trade_return > 0 else "lose_expiry"), True


def trade_outcome_label(hit_take_profit: bool, hit_stop_loss: bool) -> str:
    """A decision's raw exit type, for the sample-plot rendering (make_plots/
    trade_table_text) -- distinct from classify_walk_forward_decision's full CASE_LABELS,
    which also folds in eligibility/gate/threshold context this doesn't need. Same
    stop-loss-takes-priority ordering as classify_walk_forward_decision, for the same
    reason (a stop-loss hit is always a loss by construction)."""
    if hit_stop_loss:
        return "stop_loss"
    if hit_take_profit:
        return "take_profit"
    return "expired"


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
    """No model, no signal: tiles the test range in non-overlapping HORIZON-bar blocks
    starting right after t0 (the same anchor the walk-forward strategies use) -- buy at the
    close of each block's 1st bar, sell at the close of its last, then immediately start
    the next block. Always trades, every block, unconditionally -- no take-profit target,
    no confidence gate, no expiry logic. Same trade cadence/shape as the walk-forward
    strategies (one round trip per HORIZON-bar block) but zero selectivity, to check
    whether the models' confidence-gated entries actually beat blind periodic exposure to
    the same instrument. Equity compounds block to block, same as run_walk_forward.

    Derived from HORIZON, not hardcoded to 3 -- a hardcoded literal here would silently
    stop matching the walk-forward strategies' own cadence the moment HORIZON changes,
    exactly the failure mode found (and fixed) during the steven4 HORIZON=1 migration
    audit. At HORIZON=1 this degenerates to buying and selling at the same bar's close
    every block (0 return per block, by construction) -- an honest reflection of what
    "same cadence as a 1-bar strategy" means, not a bug."""
    equity = 1.0
    trades: list[dict] = []
    k = 0
    while True:
        buy_idx = t0 + HORIZON * k + 1
        sell_idx = buy_idx + HORIZON - 1
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
# "Trade confidence" redesign (see backlog.md's entry of that name): there is no shared,
# cross-model "confidence" score anymore. An earlier version squeezed PatchTST's and
# CVAE's very different notions of confidence onto the same 0-1 scale and applied one
# shared threshold to both, which implied a comparability neither model actually had.
# PatchTST kept a plain boolean quality gate (do its 3 predicted bars agree on
# direction?) -- no history dependence, nothing to tune. CVAE's sample-consensus
# fraction was tried as its own independently-thresholded gate, then DROPPED entirely
# after direct A/B testing showed it wasn't just noisy but actively net-harmful: bucketing
# real trades by consensus level found the *unanimous* (5/5 samples agree) bucket was the
# worst-performing one, not the best, and winners/losers had statistically indistinguishable
# consensus distributions. Removing the gate outright (keep only eligibility + return
# threshold) outperformed every consensus-threshold value tried, including the gate's
# original default -- see backlog.md for the full numbers. CVAE's quality gate is now
# always True; only the return-size and eligibility checks apply. min_return_threshold is
# per-model (--patchtst-min-return-threshold / --cvae-min-return-threshold, see main()):
# a shared absolute threshold silently zeroed out CVAE's trades entirely after the
# posterior-collapse retrain shifted its predicted-edge scale far below PatchTST's (see
# backlog.md) -- one absolute number was never going to fit two models with different and
# potentially shifting output scales.
#
# Stop-loss (--stop-loss-pct, default 0.02): removing CVAE's quality gate above fixed its
# total return but exposed a real asymmetry underneath -- CVAE's average loss runs ~7x its
# average win (payoff ratio ~0.14; PatchTST's is healthier at ~0.64 but still loss-skewed).
# Both models place a stop-loss at a fixed percentage below entry now, SHARED across
# models rather than per-model like min_return_threshold: min_return_threshold calibrates
# to each model's own predicted-edge scale, but the stop-loss calibrates to how large a
# genuine adverse move looks like in this one instrument -- the same market for both
# models, so there's no a priori reason to expect their optimal stop level to differ.
# Chosen deliberately loose relative to typical trade sizes (median trade returns are well
# under 1%): 2% sits just outside the training data's own empirical p99 |anchored log
# return| (~1.9%, see train_exit_return_bound, the same bound already used to calibrate the
# take-profit shrink), so in practice it only fires on genuine tail moves (~1% of trades),
# not on everyday intrabar noise. An earlier 1:1-mirrored-to-target stop-loss (much
# tighter, since take-profit targets are themselves small) was tried and removed for
# backfiring -- see backlog.md's stop-loss entry for that history and for the full sweep
# (0.5%-3%) that motivated this fixed, wider level.
# ---------------------------------------------------------------------------


def make_patchtst_predict_fn(patchtst, device: torch.device, sell_bound: float):
    """Returns a predict(w) -> {take_profit, passes_quality_gate, price} closure,
    single-window (batch=1) at a time. `price` is this decision's own predicted (3,4)
    price components, kept around so make_plots can render exactly what this model
    predicted at this decision without re-running inference later (see make_plots'
    module note). `passes_quality_gate` is a plain boolean, not a score: PatchTST has no
    sample distribution and no history-dependent ranking (see the module note above) --
    it either predicts a self-consistent upward path or it doesn't."""

    def predict(w: dict) -> dict:
        context, patch_pad = to_patchtst_input(w["masked_tensor"])
        context_t = torch.from_numpy(context)[None].to(device)
        patch_pad_t = torch.from_numpy(patch_pad)[None].to(device)
        with torch.no_grad():
            pt_price_t, _ = patchtst(context_t, patch_pad_t)
        pt_price = shrink_components(pt_price_t.cpu().numpy(), sell_bound)  # (1,3,4)
        close_0 = np.array([w["close_0"]])
        take_profit = float(max_close_from_components(pt_price, close_0)[0])
        coherent_up = bool((per_bar_close_return(pt_price) > 0).all(axis=1)[0])
        return {"take_profit": take_profit, "passes_quality_gate": coherent_up, "price": pt_price[0]}

    return predict


def make_cvae_predict_fn(
    cvae, device: torch.device, sell_bound: float, num_samples: int, cvae_sell_quantile: float,
):
    """Returns a predict(w) -> {take_profit, passes_quality_gate, price} closure,
    single-window (batch=1) at a time. `price` is whichever of the K sampled draws' own
    exit price is closest to the take-profit target actually used (see
    nearest_draw_index) -- chosen right here, at generation time, using the exact RNG
    draws this decision consumed, so later re-rendering it in make_plots never risks a
    different (freshly re-sampled) set of draws implying a different target/outcome than
    the one that actually decided this window's case label. `passes_quality_gate` is
    always True: CVAE's sample-consensus fraction was tried here as its own thresholded
    gate and then removed after direct A/B testing found it net-harmful, not just
    unreliable -- see the module note above run_walk_forward and backlog.md's "trade
    confidence" entry for the numbers. Eligibility and the return-size threshold alone
    decide whether a CVAE window trades."""

    def predict(w: dict) -> dict:
        masked_t = torch.from_numpy(w["masked_tensor"])[None].to(device)
        with torch.no_grad():
            cvae_price_t, _ = cvae.sample(masked_t, k=num_samples)  # (K,1,3,4)
        cvae_price = shrink_components(cvae_price_t.cpu().numpy(), sell_bound)
        close_0 = np.array([w["close_0"]])
        exit_prices = exit_price_from_components(cvae_price, close_0)  # (K,1)
        take_profit = float(np.percentile(exit_prices[:, 0], cvae_sell_quantile))
        draw_idx = nearest_draw_index(exit_prices[:, 0], take_profit)
        return {
            "take_profit": take_profit,
            "passes_quality_gate": True,
            "price": cvae_price[draw_idx, 0],
        }

    return predict


def run_walk_forward(
    predict_fn, feat: np.ndarray, opens: np.ndarray, closes: np.ndarray,
    test_lo: int, test_hi: int, min_return_threshold: float, stop_loss_pct: float,
) -> dict:
    """predict_fn(w: dict) -> {take_profit, passes_quality_gate, price} is model-specific
    -- see make_patchtst_predict_fn/make_cvae_predict_fn. min_return_threshold is this
    model's own (see main() -- no longer a shared value across models). stop_loss_pct is
    shared across models (see the module note above -- unlike min_return_threshold, the
    stop level calibrates to real market volatility, not a model-specific edge scale);
    <= 0 disables it entirely. Returns {trades, decisions, equity_final}: `decisions` is
    every decision point visited (traded or not), each classified into one of CASE_LABELS
    (see classify_walk_forward_decision) and carrying everything make_plots needs to
    render it later (price components, real OHLC, buy/sell prices) without re-running
    inference; `trades` is the subset that actually got traded -- disjoint models can and
    do follow different real-time paths here, since a trade vs. no-trade call changes how
    far this model's own clock advances next."""
    start_idx = test_lo
    equity = 1.0
    trades: list[dict] = []
    decisions: list[dict] = []

    while start_idx + WALK_FORWARD_CTX_BARS + HORIZON <= test_hi:
        w = build_window(feat, opens, closes, start_idx, WALK_FORWARD_CTX_BARS)
        close_0 = w["close_0"]
        true_price = w["y"][:N_PRICE].reshape(1, HORIZON, 4)
        true_ohlc = reconstruct_prices(true_price, np.array([close_0]))[0]

        pred = predict_fn(w)
        take_profit, passes_quality_gate, price = pred["take_profit"], pred["passes_quality_gate"], pred["price"]
        predicted_return = take_profit / close_0 - 1.0
        eligible = take_profit > close_0
        meets_return_threshold = predicted_return >= min_return_threshold
        stop_loss = close_0 * (1.0 - stop_loss_pct) if stop_loss_pct > 0 else -np.inf

        sell_price_arr, hit_tp_arr, hit_sl_arr = bracket_exit(
            true_ohlc[None], np.array([take_profit]), np.array([stop_loss])
        )
        sell_price, hit_tp, hit_sl = float(sell_price_arr[0]), bool(hit_tp_arr[0]), bool(hit_sl_arr[0])
        trade_return = float(sell_price / close_0 - 1.0)
        label, would_trade = classify_walk_forward_decision(
            eligible, meets_return_threshold, passes_quality_gate, hit_tp, hit_sl, trade_return
        )

        decision = {
            "start_idx": start_idx,
            "close_0": float(close_0),
            "take_profit": float(take_profit),
            "passes_quality_gate": bool(passes_quality_gate),
            "price": price,  # (3,4) predicted components, for plotting
            "true_ohlc": true_ohlc,  # (3,4) real components, for plotting
            "label": label,
            "would_trade": would_trade,
            "sell_price": sell_price,
            "hit_take_profit": hit_tp,
            "hit_stop_loss": hit_sl,
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
    decision point checked (see outcome_breakdown) -- reports exactly which of the 5 cases
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
# Random-sample backtest -- the current default (see module docstring)
# ---------------------------------------------------------------------------


def run_random_sample_backtest(
    predict_fn, feat: np.ndarray, opens: np.ndarray, closes: np.ndarray,
    test_pairs: list[tuple[int, int]], min_return_threshold: float, stop_loss_pct: float,
) -> dict:
    """Same per-decision logic as run_walk_forward (classify_walk_forward_decision +
    bracket_exit -- eligibility check, return-threshold filter, take-profit/stop-loss
    bracket exit against the 3 REAL horizon bars, stop-loss winning same-bar ties), but over
    a GIVEN list of independently-sampled (start_idx, ctx_bars) pairs instead of a
    sequential walk -- no advance-rule bookkeeping, no equity compounding (see module
    docstring for why: these windows can overlap in calendar time, so there's no single
    coherent account balance). `test_pairs` is shared verbatim across both models' calls
    (see main()), so PatchTST and CVAE are always evaluated on the identical windows,
    unlike run_walk_forward where a trade/no-trade call can put each model on a different
    real-time path."""
    decisions = []
    for start_idx, ctx_bars in test_pairs:
        w = build_window(feat, opens, closes, start_idx, ctx_bars)
        close_0 = w["close_0"]
        true_price = w["y"][:N_PRICE].reshape(1, HORIZON, 4)
        true_ohlc = reconstruct_prices(true_price, np.array([close_0]))[0]

        pred = predict_fn(w)
        take_profit, passes_quality_gate, price = pred["take_profit"], pred["passes_quality_gate"], pred["price"]
        predicted_return = take_profit / close_0 - 1.0
        eligible = take_profit > close_0
        meets_return_threshold = predicted_return >= min_return_threshold
        stop_loss = close_0 * (1.0 - stop_loss_pct) if stop_loss_pct > 0 else -np.inf

        sell_price_arr, hit_tp_arr, hit_sl_arr = bracket_exit(
            true_ohlc[None], np.array([take_profit]), np.array([stop_loss])
        )
        sell_price, hit_tp, hit_sl = float(sell_price_arr[0]), bool(hit_tp_arr[0]), bool(hit_sl_arr[0])
        trade_return = float(sell_price / close_0 - 1.0)
        label, would_trade = classify_walk_forward_decision(
            eligible, meets_return_threshold, passes_quality_gate, hit_tp, hit_sl, trade_return
        )

        decisions.append({
            "start_idx": start_idx,
            "ctx_bars": ctx_bars,
            "close_0": float(close_0),
            "take_profit": float(take_profit),
            "passes_quality_gate": bool(passes_quality_gate),
            "price": price,
            "true_ohlc": true_ohlc,
            "label": label,
            "would_trade": would_trade,
            "sell_price": sell_price,
            "hit_take_profit": hit_tp,
            "hit_stop_loss": hit_sl,
            "trade_return": trade_return,
        })
    return {"decisions": decisions}


def random_sample_stats(result: dict) -> dict:
    """Reduces run_random_sample_backtest's decision log into reportable stats -- same
    outcome_breakdown/win_rate/take_profit_rate/avg_return shape as walk_forward_stats,
    deliberately WITHOUT total_return/annualized_return/buy-and-hold: these windows can
    overlap in calendar time, so summing or compounding returns across them wouldn't be a
    real, realizable account balance -- see module docstring."""
    decisions = result["decisions"]
    trades = [d for d in decisions if d["would_trade"]]
    labels = np.array([d["label"] for d in decisions], dtype=object)
    out = {
        "n_decisions": len(decisions),
        "n_trades": len(trades),
        "outcome_breakdown": outcome_breakdown(labels),
    }
    if not trades:
        out.update(win_rate=None, take_profit_rate=None, avg_return=None)
        return out

    rets = np.array([t["trade_return"] for t in trades])
    hits = np.array([t["hit_take_profit"] for t in trades])
    out["win_rate"] = float((rets > 0).mean())
    out["take_profit_rate"] = float(hits.mean())
    out["avg_return"] = float(rets.mean())
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
    starts. The linestyle is caller-chosen (take-profit solid, stop-loss dashed/dotted per
    model -- kept generic here so each model's second line is just another tuple in the
    list). If a price falls outside the panel's current y-range, the line is
    pinned just inside the top/bottom edge instead of letting the axis autoscale to it,
    with a text label giving the real price and noting it's out of range -- this keeps
    the panel's own price scale intact regardless of how extreme a prediction is. Multiple
    targets pinned to the same edge (now routine with a stop-loss line added alongside
    each take-profit line) are stacked with a small vertical offset per line instead of
    landing exactly on top of each other -- with only take-profit lines this rarely came
    up, but two lines per model doubles how often two land out-of-range together."""
    y0, y1 = ax.get_ylim()
    x0 = n_shown - HORIZON - 0.5
    x1 = n_shown - 0.5
    margin = (y1 - y0) * 0.08
    stack_step = (y1 - y0) * 0.04

    ax.axhline(buy_price, color="tab:blue", linestyle="--", linewidth=1, zorder=6)
    ax.text(-0.6, buy_price, "buy", color="tab:blue", fontsize=7, va="bottom", ha="left", zorder=6)

    n_above = n_below = 0
    for label, price, color, linestyle in sell_targets:
        if price > y1:
            line_y = y1 - margin - n_above * stack_step
            n_above += 1
            ax.hlines(line_y, x0, x1, color=color, linestyle=linestyle, linewidth=1.4, zorder=7)
            ax.annotate(
                f"  {label}: {price:.2f} (above range)", xy=(x0, line_y), xytext=(0, -4),
                textcoords="offset points", color=color, fontsize=7, va="top", ha="left", zorder=7,
            )
        elif price < y0:
            line_y = y0 + margin + n_below * stack_step
            n_below += 1
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
    (realized_price, outcome) where outcome is 'stop_loss', 'take_profit', or 'expired' --
    only meaningful when would_enter is True."""
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
                "stop_loss": "STOP LOSS -> filled at stop level",
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


def trend_z_score(feat: np.ndarray, ctx_end: int, ctx_bars: int, body_ret_col: int, lookback: int = 20) -> float:
    """Causal-only trend strength from context rows [ctx_end-lookback, ctx_end) -- never
    the horizon: a one-sample z-score of the lookback window's mean body_ret (is the
    average per-bar move significantly non-zero, relative to its own noise). body_ret_col
    matches FEATURE_COLS' fixed ordering (open_ret, body_ret, ...) in both the base and
    momentum-enriched pipelines -- same assumption as generative_metrics.regime_indicators."""
    n = min(lookback, ctx_bars)
    window = feat[ctx_end - n : ctx_end, body_ret_col]
    std = window.std()
    if std == 0:
        return 0.0
    return float(window.mean() / (std / np.sqrt(n)))


def classify_trend(z: float) -> str:
    """z > 0.5 -> uptrend, z < -0.5 -> downtrend, otherwise choppy. See trend_z_score."""
    if z > 0.5:
        return "uptrend"
    if z < -0.5:
        return "downtrend"
    return "choppy"


def render_scenario_panel(ax, sub_df: pd.DataFrame, title: str) -> None:
    """Minimal candle panel -- candlesticks + a red box around the horizon region + a
    title. No buy/sell lines, no trade-decision text, no metrics: this comparison is
    purely visual (see build_trend_comparison_chart)."""
    mpf.plot(sub_df, type="candle", ax=ax, style="yahoo", volume=False)
    ax.set_title(title, fontsize=9)
    ax.tick_params(axis="x", labelrotation=30, labelsize=6)
    draw_horizon_box(ax, len(sub_df))


def build_trend_comparison_chart(
    df: pd.DataFrame, feat: np.ndarray, opens: np.ndarray, closes: np.ndarray,
    patchtst, cvae, bounds: dict, device: torch.device, ctx_bars: int,
    trend_lookback: int, step: int, out_dir: Path, n_examples: int = 3,
) -> None:
    """n_examples windows per trend label (uptrend/downtrend/choppy), each rendered as its
    own 3-panel image: ground truth spans the left column, PatchTST's single predicted
    path top right, one CVAE sampled draw bottom right -- the classic comparison layout
    from this file's original make_plots, minus every bit of trade-decision markup (no
    buy/sell lines, no ENTER/NO TRADE label, no text table). Purely visual: no metrics, no
    trade decisions, no backtest.

    Examples within a label are spread evenly across that label's own candidate pool (by
    position in chronological scan order, not by picking the n_examples closest to the
    median trend-z) -- windows near each other in time tend to share the same label, so
    picking by z-closeness alone would mostly return near-duplicate, overlapping windows
    instead of genuinely different illustrations of the same scenario type."""
    body_ret_col = dp.FEATURE_COLS.index("body_ret")
    lo, hi = bounds["test"]
    pairs = RollingWindowSampler(lo, hi, ctx_bars, step=step).pairs()

    # Trend classification is cheap (pure feat-array slicing, no model calls) and run over
    # every scanned window; inference is only run on the windows actually picked below --
    # to_patchtst_input is single-window only (slices axis 0 as time), so batching it over
    # every candidate window would be both wrong and wasteful.
    trend_z = np.array([
        trend_z_score(feat, s + c, c, body_ret_col, lookback=trend_lookback) for s, c in pairs
    ])
    labels = np.array([classify_trend(z) for z in trend_z])

    out_dir.mkdir(parents=True, exist_ok=True)
    for label in ["uptrend", "downtrend", "choppy"]:
        idx_pool = np.where(labels == label)[0]
        if len(idx_pool) == 0:
            logger.warning("no windows classified as %s -- skipping", label)
            continue
        n = min(n_examples, len(idx_pool))
        positions = np.unique(np.linspace(0, len(idx_pool) - 1, n).astype(int))
        if len(positions) < n_examples:
            logger.warning("only %d distinct %s windows available (wanted %d)", len(positions), label, n_examples)
        picks = idx_pool[positions]

        for example_num, i in enumerate(picks, start=1):
            start_idx, this_ctx = pairs[i]

            w = build_window(feat, opens, closes, start_idx, this_ctx)
            masked_t = torch.from_numpy(w["masked_tensor"])[None].to(device)
            with torch.no_grad():
                pt_context, pt_patch_pad = to_patchtst_input(w["masked_tensor"])
                pt_price_t, _ = patchtst(
                    torch.from_numpy(pt_context)[None].to(device), torch.from_numpy(pt_patch_pad)[None].to(device)
                )
                cvae_price_t, _ = cvae.sample(masked_t, k=1)  # (1,1,3,4)
            pt_price = pt_price_t.cpu().numpy()[0]  # (3,4)
            cvae_price = cvae_price_t.cpu().numpy()[0, 0]  # (3,4)

            ctx_tail = min(this_ctx, 20)
            hz_start = start_idx + this_ctx
            plot_rows = df.iloc[hz_start - ctx_tail : hz_start + HORIZON]
            true_df = plot_rows.set_index("datetime")[["open", "high", "low", "close", "volume"]]
            close_0 = float(df.iloc[hz_start - 1]["close"])

            pt_ohlc = reconstruct_prices(pt_price, close_0)
            pt_df = true_df.copy()
            pt_df.loc[pt_df.index[-HORIZON:], ["open", "high", "low", "close"]] = pt_ohlc

            gen_ohlc = reconstruct_prices(cvae_price, close_0)
            cvae_df = true_df.copy()
            cvae_df.loc[cvae_df.index[-HORIZON:], ["open", "high", "low", "close"]] = gen_ohlc

            fig = plt.figure(figsize=(14, 7))
            outer = fig.add_gridspec(2, 2, width_ratios=[1, 1], hspace=0.4, wspace=0.25)
            render_scenario_panel(fig.add_subplot(outer[:, 0]), true_df, "Ground truth")
            render_scenario_panel(fig.add_subplot(outer[0, 1]), pt_df, "PatchTST")
            render_scenario_panel(fig.add_subplot(outer[1, 1]), cvae_df, "CVAE")
            fig.suptitle(f"{label} #{example_num} (ctx_bars={ctx_bars}, start_idx={start_idx})")

            out_path = out_dir / f"{label}_{example_num}.png"
            fig.savefig(out_path, bbox_inches="tight")
            plt.close(fig)
            logger.info("wrote %s example %d chart to %s", label, example_num, out_path)


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
        plot_rows = df.iloc[hz_start - ctx_tail : hz_start + HORIZON]
        true_df = plot_rows.set_index("datetime")[["open", "high", "low", "close", "volume"]]

        cvae_df = true_df.copy()
        cvae_df.loc[cvae_df.index[-HORIZON:], ohlc_cols] = cvae_ohlc

        true_horizon_ohlc = true_df.iloc[-HORIZON:][["open", "high", "low", "close"]].to_numpy()

        # Both models share one stop-loss level (see the module note above run_walk_forward
        # -- it's calibrated to real market volatility, not either model's own predicted
        # edge), so it's numerically identical whenever both are evaluated at the same
        # window. Drawing it twice on the ground-truth panel would put two identical lines
        # exactly on top of each other -- shown there just once, model-agnostic; each
        # model's own panel still shows its own copy since there's no collision there.
        stop_loss = buy_price * (1.0 - args.stop_loss_pct) if args.stop_loss_pct > 0 else None

        cvae_sell_targets = [("CVAE TP", cvae_tp, "tab:green", "-")]
        if stop_loss is not None:
            cvae_sell_targets.append(("CVAE SL", stop_loss, "tab:red", "--"))

        ground_truth_targets = [("CVAE TP", cvae_tp, "tab:green", "-")]
        if pt_d is not None:
            pt_tp = pt_d["take_profit"]
            pt_ohlc = reconstruct_prices(pt_d["price"], buy_price)  # (3,4)
            pt_df = true_df.copy()
            pt_df.loc[pt_df.index[-HORIZON:], ohlc_cols] = pt_ohlc
            pt_sell_targets = [("PatchTST TP", pt_tp, "tab:orange", "-")]
            if stop_loss is not None:
                pt_sell_targets.append(("PatchTST SL", stop_loss, "tab:red", ":"))
            ground_truth_targets = [("PatchTST TP", pt_tp, "tab:orange", "-")] + ground_truth_targets
        else:
            pt_tp = pt_ohlc = None
            pt_df = true_df
            pt_sell_targets = []
        if stop_loss is not None:
            ground_truth_targets.append(("Stop-loss (shared)", stop_loss, "tab:red", "--"))

        fig = plt.figure(figsize=(16, 8))
        outer = fig.add_gridspec(6, 2, width_ratios=[1, 1], hspace=0.7, wspace=0.25)

        render_panel(
            fig, outer[0:4, 0], outer[4:6, 0], true_df, buy_price, "Ground truth",
            sell_targets=ground_truth_targets,
        )
        render_panel(
            fig, outer[0:2, 1], outer[2, 1], pt_df, buy_price,
            "PatchTST generated" if pt_d is not None else "PatchTST (not evaluated this window)",
            sell_targets=pt_sell_targets,
            sell_limit=pt_tp, would_enter=(pt_d["would_trade"] if pt_d is not None else None),
            realized=(
                (pt_d["sell_price"], trade_outcome_label(pt_d["hit_take_profit"], pt_d["hit_stop_loss"]))
                if pt_d is not None else None
            ),
        )
        render_panel(
            fig, outer[3:5, 1], outer[5, 1], cvae_df, buy_price, "CVAE generated (draw nearest target)",
            sell_targets=cvae_sell_targets,
            sell_limit=cvae_tp, would_enter=cvae_d["would_trade"],
            realized=(
                cvae_d["sell_price"],
                trade_outcome_label(cvae_d["hit_take_profit"], cvae_d["hit_stop_loss"]),
            ),
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
            outcome = trade_outcome_label(d["hit_take_profit"], d["hit_stop_loss"])
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


def make_random_sample_plots(df: pd.DataFrame, pt_decisions: list[dict], cvae_decisions: list[dict], args) -> None:
    """Random-sample equivalent of make_plots (see module docstring for why this replaced
    the walk-forward version) -- selects one illustrative example per (context bucket,
    outcome) combination from CVAE's own decision log: narrow/wide context buckets
    (dp.context_bucket -- moderate is skipped, narrow/wide only, per request) x
    {win_take_profit, win_expiry, loss (lose_expiry or lose_stop_loss, whichever this
    category has an example of)}, 6 total.

    Unlike make_plots, PatchTST and CVAE are GUARANTEED to have a decision at the exact
    same (start_idx, ctx_bars) here -- both were evaluated against the identical
    `test_pairs` list (see main()), not two independently-advancing walks that can diverge
    -- so there's no "not evaluated this window" fallback to handle."""
    out_dir = Path(args.plots_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("*_start*_ctx*.png"):
        stale.unlink()
    (out_dir / "samples.json").unlink(missing_ok=True)

    rng = np.random.default_rng()
    ohlc_cols = ["open", "high", "low", "close"]
    pt_by_key = {(d["start_idx"], d["ctx_bars"]): d for d in pt_decisions}

    loss_labels = {"lose_expiry", "lose_stop_loss"}
    outcome_groups = {
        "win_take_profit": lambda label: label == "win_take_profit",
        "win_expiry": lambda label: label == "win_expiry",
        "loss": lambda label: label in loss_labels,
    }

    records = []
    plotted = 0
    for bucket in ("narrow", "wide"):
        for outcome_name, matches in outcome_groups.items():
            candidates = [
                d for d in cvae_decisions
                if dp.context_bucket(d["ctx_bars"]) == bucket and matches(d["label"])
            ]
            if not candidates:
                logger.warning("no CVAE decisions found for bucket=%s outcome=%s -- skipping", bucket, outcome_name)
                continue
            cvae_d = candidates[int(rng.integers(len(candidates)))]
            pt_d = pt_by_key[(cvae_d["start_idx"], cvae_d["ctx_bars"])]

            start_idx, ctx_bars = cvae_d["start_idx"], cvae_d["ctx_bars"]
            buy_price = cvae_d["close_0"]
            cvae_tp = cvae_d["take_profit"]
            cvae_ohlc = reconstruct_prices(cvae_d["price"], buy_price)  # (3,4)

            ctx_tail = min(ctx_bars, 20)
            hz_start = start_idx + ctx_bars
            plot_rows = df.iloc[hz_start - ctx_tail : hz_start + HORIZON]
            true_df = plot_rows.set_index("datetime")[["open", "high", "low", "close", "volume"]]

            cvae_df = true_df.copy()
            cvae_df.loc[cvae_df.index[-HORIZON:], ohlc_cols] = cvae_ohlc

            true_horizon_ohlc = true_df.iloc[-HORIZON:][["open", "high", "low", "close"]].to_numpy()

            # Shared 2% stop-loss for both models -- see module note above run_walk_forward.
            stop_loss = buy_price * (1.0 - args.stop_loss_pct) if args.stop_loss_pct > 0 else None

            cvae_sell_targets = [("CVAE TP", cvae_tp, "tab:green", "-")]
            if stop_loss is not None:
                cvae_sell_targets.append(("CVAE SL", stop_loss, "tab:red", "--"))

            pt_tp = pt_d["take_profit"]
            pt_ohlc = reconstruct_prices(pt_d["price"], buy_price)  # (3,4)
            pt_df = true_df.copy()
            pt_df.loc[pt_df.index[-HORIZON:], ohlc_cols] = pt_ohlc
            pt_sell_targets = [("PatchTST TP", pt_tp, "tab:orange", "-")]
            if stop_loss is not None:
                pt_sell_targets.append(("PatchTST SL", stop_loss, "tab:red", ":"))

            ground_truth_targets = [("PatchTST TP", pt_tp, "tab:orange", "-"), ("CVAE TP", cvae_tp, "tab:green", "-")]
            if stop_loss is not None:
                ground_truth_targets.append(("Stop-loss (shared)", stop_loss, "tab:red", "--"))

            fig = plt.figure(figsize=(16, 8))
            outer = fig.add_gridspec(6, 2, width_ratios=[1, 1], hspace=0.7, wspace=0.25)

            render_panel(
                fig, outer[0:4, 0], outer[4:6, 0], true_df, buy_price, "Ground truth",
                sell_targets=ground_truth_targets,
            )
            render_panel(
                fig, outer[0:2, 1], outer[2, 1], pt_df, buy_price, "PatchTST generated",
                sell_targets=pt_sell_targets, sell_limit=pt_tp, would_enter=pt_d["would_trade"],
                realized=(pt_d["sell_price"], trade_outcome_label(pt_d["hit_take_profit"], pt_d["hit_stop_loss"])),
            )
            render_panel(
                fig, outer[3:5, 1], outer[5, 1], cvae_df, buy_price, "CVAE generated (draw nearest target)",
                sell_targets=cvae_sell_targets, sell_limit=cvae_tp, would_enter=cvae_d["would_trade"],
                realized=(cvae_d["sell_price"], trade_outcome_label(cvae_d["hit_take_profit"], cvae_d["hit_stop_loss"])),
            )

            fig.suptitle(f"CVAE outcome: {outcome_name}  |  bucket={bucket}, ctx_bars={ctx_bars}, start_idx={start_idx}")
            out_path = out_dir / f"{outcome_name}_{bucket}_start{start_idx}_ctx{ctx_bars}.png"
            fig.savefig(out_path, bbox_inches="tight")
            plt.close(fig)
            plotted += 1

            def model_record(d, ohlc, tp):
                outcome = trade_outcome_label(d["hit_take_profit"], d["hit_stop_loss"])
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
                "outcome": outcome_name,
                "bucket": bucket,
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
    # without seeding it, CVAE's numbers drift between runs on unchanged checkpoints. Also
    # seeds the window draw below (np.random.default_rng(args.seed) is separate, but keeping
    # both tied to the same --seed value keeps a whole run reproducible from one flag).
    torch.manual_seed(args.seed)

    pt_ckpt = torch.load(args.patchtst_checkpoint, map_location=device, weights_only=False)
    cvae_ckpt = torch.load(args.cvae_checkpoint, map_location=device, weights_only=False)

    pt_cfg = pt_ckpt["config"]
    cvae_cfg = cvae_ckpt["config"]
    assert pt_cfg["data_path"] == cvae_cfg["data_path"], "both checkpoints must share the same data_path"

    # Momentum-aware build, mirroring train_cvae.py/train_patchtst.py's own opt-in toggle --
    # both checkpoints' configs must agree (asserted above on data_path; same assumption
    # extends to momentum_features since a mismatch would mean the two models were trained
    # on different channel layouts, an evaluate.py can't reconcile).
    momentum_cfg = cvae_cfg.get("momentum_features")
    if momentum_cfg and momentum_cfg.get("enabled"):
        df, bounds, stats, momentum_stats = momentum_pipeline.build_momentum_dataset(
            cvae_cfg["data_path"], momentum_cfg["vix_data_path"]
        )
        logger.info("momentum features enabled: ema_cross/trend_position/rsi/vix, N_CHANNELS=%d", dp.N_CHANNELS)
    else:
        df, bounds, stats = build_dataset(pt_cfg["data_path"])
    feat, opens, closes = extract_arrays(df)

    patchtst = PatchTST(**pt_cfg["model"], n_feature_channels=dp.N_FEATURE_CHANNELS).to(device)
    patchtst.load_state_dict(pt_ckpt["model_state"])
    patchtst.eval()

    cvae = CVAEInpainting(**cvae_cfg["model"], in_channels=dp.N_CHANNELS).to(device)
    cvae.load_state_dict(cvae_ckpt["model_state"])
    cvae.eval()

    logger.info(
        "rendering %d trend scenario charts per label (ctx_bars=%d) -- purely visual, "
        "no metrics/backtest/trade decisions",
        args.n_examples, args.ctx_bars,
    )
    build_trend_comparison_chart(
        df, feat, opens, closes, patchtst, cvae, bounds, device,
        args.ctx_bars, args.trend_lookback, args.step, Path(args.charts_dir), args.n_examples,
    )


if __name__ == "__main__":
    main()
