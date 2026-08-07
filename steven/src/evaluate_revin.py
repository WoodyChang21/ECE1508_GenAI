"""Walk-forward backtest for hf_patchtst_revin_no_volume (PatchTSTOHLCVRevIN) checkpoints.

Ports the walk-forward backtest methodology from the `steven` branch's `evaluate.py`
(chronological single-equity-path simulation, take-profit-only orders with gap-through
handling, 5-way outcome classification) -- adapted for a model that predicts raw OHLC
directly (via RevIN) instead of the anchored log-return component decomposition every
function in `steven`'s original file assumes.

The generic trading mechanics (`take_profit_exit`, `classify_walk_forward_decision`,
`outcome_breakdown`, `equity_stats`, `buy_and_hold_benchmark`, `naive_periodic_benchmark`,
`run_walk_forward`'s shape, `walk_forward_stats`) are ported essentially unchanged -- they
only ever operate on close_0/take_profit/confidence/real OHLC, never on the anchored
component representation. Only the prediction/confidence function and window builder are
new (`patchtst_revin_walk_forward_confidence`, `make_patchtst_revin_predict_fn`,
`build_raw_ohlc_window`), since those are the parts actually tied to the return-component
format this checkpoint doesn't use.

`patchtst_revin_walk_forward_confidence` originally mirrored steven's magnitude-percentile
ranking (rank predicted move size against this same walk's own prior coherent-up
decisions). `confidence_calibration()` showed that ranking was actively counterproductive
here -- win rate fell monotonically as that confidence rose -- so it's now a plain binary
coherent-up signal (1.0/0.0); see the note above that function for the full reasoning.

No CVAE support here -- no CVAE checkpoint exists for this experiment line (this notebook
family never trained one), unlike `steven`'s own evaluate.py which compares PatchTST + CVAE.

Usage:
    python steven/src/evaluate_revin.py \\
        --checkpoint steven/outputs/patchtst_revin_novolume_channel_attention_false_checkpoint.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_pipeline import HORIZON, build_dataset
from src.models.patchtst_hf import CLOSE_IDX, OHLC_COLS, build_model

OPEN_IDX = OHLC_COLS.index("open")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument(
        "--metrics-out", type=str, default=None,
        help="Defaults to steven/outputs/backtest_<checkpoint filename stem>.json",
    )
    p.add_argument(
        "--confidence-threshold", type=float, default=0.5,
        help="Confidence is now binary -- 1.0 if all HORIZON predicted bars agree on "
        "'up' (coherent), else 0.0 (see patchtst_revin_walk_forward_confidence) -- so any "
        "threshold in (0.0, 1.0] behaves identically (coherent-only trading), and a "
        "threshold of 0.0 additionally allows non-coherent windows through. Kept as a "
        "float, not a bool flag, so this still reads as 'the same knob' as every other "
        "notebook/script in this family that sweeps a confidence threshold.",
    )
    p.add_argument(
        "--min-return-threshold", type=float, default=0.001,
        help="Minimum model-predicted exit return (fraction, e.g. 0.001 = 0.1%%) required "
        "to bother trading at all, on top of the plain take_profit>close_0 eligibility "
        "check -- filters out trades with a technically-positive but trivially small "
        "predicted edge.",
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


# ---------------------------------------------------------------------------
# Walk-forward mechanics -- ported from steven branch's evaluate.py essentially
# unchanged. Generic: only ever operate on close_0/take_profit/real OHLC/confidence, no
# dependency on the anchored-return component representation.
# ---------------------------------------------------------------------------

CASE_LABELS = ["win_take_profit", "win_expiry", "lose_expiry", "skipped", "no_trade"]


def take_profit_exit(
    true_ohlc: np.ndarray, take_profit: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulates a take-profit-only limit sell order (no stop-loss -- dropped on steven
    branch, see docs/experiments.md there for rationale) placed at the same time as the
    close_0 buy. Walks the HORIZON real bars in order; the first bar that reaches
    take_profit fills the order, in one of two ways:
    - **Touched mid-bar** (low <= take_profit <= high): fills at take_profit itself.
    - **Gapped through** (low > take_profit, the whole bar including its open already
      sits above the target): fills at the open instead of the stale take_profit level,
      since a real limit sell guarantees *at least* the limit price.
    If neither ever happens across all HORIZON bars, the position is force-closed at the
    last bar's real close instead (order expiry). true_ohlc: (N,HORIZON,4) real
    [open,high,low,close]. take_profit: (N,). Returns (realized_sell_price (N,),
    hit_take_profit (N,) bool, bars_held (N,) int -- 1-indexed bar the position actually
    closed on, HORIZON on expiry -- lets the caller resume the walk as soon as a position
    is actually closed instead of always waiting the full HORIZON bars, see
    run_walk_forward)."""
    n = true_ohlc.shape[0]
    open_ = true_ohlc[:, :, OPEN_IDX]
    low = true_ohlc[:, :, OHLC_COLS.index("low")]
    high = true_ohlc[:, :, OHLC_COLS.index("high")]

    sell_price = true_ohlc[:, HORIZON - 1, CLOSE_IDX].copy()  # default: last horizon bar's real close
    hit_take_profit = np.zeros(n, dtype=bool)
    bars_held = np.full(n, HORIZON, dtype=int)  # default: full horizon (expiry)
    resolved = np.zeros(n, dtype=bool)

    for bar in range(HORIZON):
        touched = (~resolved) & (low[:, bar] <= take_profit) & (take_profit <= high[:, bar])
        gapped = (~resolved) & (low[:, bar] > take_profit)

        sell_price[touched] = take_profit[touched]
        sell_price[gapped] = open_[:, bar][gapped]
        hit_take_profit[touched | gapped] = True
        bars_held[touched | gapped] = bar + 1
        resolved |= touched | gapped

    return sell_price, hit_take_profit, bars_held


def classify_walk_forward_decision(
    eligible: bool, meets_return_threshold: bool, confident_enough: bool,
    hit_take_profit: bool, trade_return: float,
) -> tuple[str, bool]:
    """Sorts one walk-forward decision point into exactly one of CASE_LABELS. Returns
    (label, would_trade). 'no_trade' = this model's own take-profit target never even
    clears the buy price (no expected upside at all); 'skipped' = the target does clear
    the buy price, but the predicted edge is smaller than min_return_threshold and/or the
    model isn't confident enough."""
    if not eligible:
        return "no_trade", False
    if not (meets_return_threshold and confident_enough):
        return "skipped", False
    if hit_take_profit:
        return "win_take_profit", True
    return ("win_expiry" if trade_return > 0 else "lose_expiry"), True


def outcome_breakdown(labels: np.ndarray) -> dict:
    n = len(labels)
    if n == 0:
        return {}
    return {case: float((labels == case).mean()) for case in CASE_LABELS}


def equity_stats(
    df: pd.DataFrame, entry_idx: int, exit_idx: int, entry_price: float, exit_price: float, total_return: float
) -> dict:
    entry, exit_ = df.iloc[entry_idx], df.iloc[exit_idx]
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
    """Buy SPY at entry_idx's close, hold to exit_idx's close -- no model, no confidence
    threshold, no trade selectivity. entry_idx is anchored to the walk-forward's own
    first tradeable decision point (see main()), not the test split's literal first bar,
    so every number in the comparison covers the identical calendar span."""
    entry_price = float(df.iloc[entry_idx]["close"])
    exit_price = float(df.iloc[exit_idx]["close"])
    total_return = exit_price / entry_price - 1.0
    return equity_stats(df, entry_idx, exit_idx, entry_price, exit_price, total_return)


def naive_periodic_benchmark(df: pd.DataFrame, closes: np.ndarray, t0: int, test_hi: int) -> dict:
    """No model, no signal: tiles the test range in non-overlapping HORIZON-bar blocks
    starting right after t0 -- buy at each block's 1st bar close, sell at its last bar
    close, then immediately start the next block. Always trades, unconditionally -- same
    trade cadence as the walk-forward strategy but zero selectivity, to check whether the
    model's confidence-gated entries beat blind periodic exposure to the same instrument."""
    equity = 1.0
    trades: list[dict] = []
    k = 0
    while True:
        buy_idx, sell_idx = t0 + HORIZON * k + 1, t0 + HORIZON * k + HORIZON
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
    out["take_profit_rate"] = None
    out["avg_return"] = float(rets.mean())
    out.update(equity_stats(
        df, trades[0]["buy_idx"], trades[-1]["sell_idx"],
        trades[0]["buy_price"], float(closes[trades[-1]["sell_idx"]]), out["total_return"],
    ))
    return out


# ---------------------------------------------------------------------------
# RevIN-specific prediction + confidence -- NEW. Replaces steven's
# make_patchtst_predict_fn/patchtst_walk_forward_confidence, which assume the anchored
# return-component representation this model doesn't use.
#
# Confidence is coherence-only (binary), not a magnitude-percentile rank -- an earlier
# version ranked predicted move size against this same walk's own prior coherent-up
# decisions (mirroring steven's own patchtst_walk_forward_confidence), but
# confidence_calibration() showed that ranking was actively counterproductive: win rate
# fell *monotonically* as that magnitude-based confidence rose (86% in the lowest bin down
# to 65% in the highest), while direction accuracy stayed flat/noisy across bins. A bigger
# predicted move sets a more aggressive (harder-to-reach) take-profit target -- since
# take_profit is also derived from the predicted magnitude -- so "confidence" was really
# just measuring target aggressiveness, not correctness. Coherence itself is NOT the
# problem: non-coherent windows have a clearly worse win rate (25-43%) than any coherent
# bin (65-86%), so that gate is kept -- only the magnitude ranking on top of it is dropped.
# ---------------------------------------------------------------------------


def patchtst_revin_walk_forward_confidence(pred_ohlc: np.ndarray, close_0: float) -> float:
    """pred_ohlc: (HORIZON, 4) predicted real OHLC for one window. Binary: 1.0 if all
    HORIZON predicted closes land above close_0 ("coherent up"), else 0.0. See the module
    note above for why this replaced the earlier magnitude-percentile ranking."""
    pred_close = pred_ohlc[:, CLOSE_IDX]
    coherent_up = bool((pred_close > close_0).all())
    return 1.0 if coherent_up else 0.0


def make_patchtst_revin_predict_fn(model: torch.nn.Module, device: torch.device):
    """Returns a predict(context_raw, close_0) -> {take_profit, confidence, price}
    closure, single-window (batch=1) at a time."""

    def predict(context_raw: np.ndarray, close_0: float) -> dict:
        context_t = torch.from_numpy(context_raw)[None].to(device)  # (1, ctx_bars, 4)
        with torch.no_grad():
            pred_norm = model(context_t)
            pred_price = model.revin.denormalize(pred_norm)  # (1, HORIZON, 4) real OHLC
        pred_ohlc = pred_price[0].cpu().numpy()  # (HORIZON, 4)
        take_profit = float(pred_ohlc[:, CLOSE_IDX].max())  # max of predicted closes, matches max_close_from_components
        confidence = patchtst_revin_walk_forward_confidence(pred_ohlc, close_0)
        return {"take_profit": take_profit, "confidence": confidence, "price": pred_ohlc}

    return predict


def build_raw_ohlc_window(ohlc_arr: np.ndarray, start_idx: int, ctx_bars: int) -> dict:
    """Raw-OHLC equivalent of steven's build_window -- no anchored-return decomposition,
    no padding_mask handling (this checkpoint family always uses a fixed context length,
    see the fixed-context caveat throughout this notebook family)."""
    context = ohlc_arr[start_idx : start_idx + ctx_bars]
    close_0 = float(ohlc_arr[start_idx + ctx_bars - 1, CLOSE_IDX])
    true_ohlc = ohlc_arr[start_idx + ctx_bars : start_idx + ctx_bars + HORIZON]
    return {"context": context, "close_0": close_0, "true_ohlc": true_ohlc}


def run_walk_forward(
    predict_fn, ohlc_arr: np.ndarray, test_lo: int, test_hi: int, ctx_bars: int,
    confidence_threshold: float, min_return_threshold: float,
) -> dict:
    """Walks the test range in real chronological order, one hour at a time, always
    using the full ctx_bars context: "a trader checks in every hour with the most recent
    ctx_bars candles." No trade -> advance 1 bar and re-check next hour. Trade -> resume
    as soon as the position actually closes (bars_held from take_profit_exit -- 1, 2, or
    HORIZON on expiry), not always after a fixed HORIZON bars -- this is a real, live-
    knowable event (the trader sees their position close and is free to look again), not
    a lookahead leak, and avoids leaving capital idle for bars after an early take-profit
    fill. Equity compounds trade to trade starting from 1.0, so total_return is a real
    simulated account balance over time, never a sum across possibly-overlapping trades.

    Each decision also records direction_correct (did price actually end up above close_0
    by the last horizon bar, independent of the take-profit mechanics) -- used by
    confidence_calibration to check whether the confidence score tracks correctness at
    all, separately from whether a trade was actually taken."""
    start_idx = test_lo
    equity = 1.0
    trades: list[dict] = []
    decisions: list[dict] = []

    while start_idx + ctx_bars + HORIZON <= test_hi:
        w = build_raw_ohlc_window(ohlc_arr, start_idx, ctx_bars)
        close_0 = w["close_0"]
        true_ohlc = w["true_ohlc"]  # (HORIZON, 4)

        pred = predict_fn(w["context"], close_0)
        take_profit, confidence, price = pred["take_profit"], pred["confidence"], pred["price"]
        predicted_return = take_profit / close_0 - 1.0
        eligible = take_profit > close_0
        meets_return_threshold = predicted_return >= min_return_threshold
        confident_enough = confidence >= confidence_threshold

        sell_price_arr, hit_tp_arr, bars_held_arr = take_profit_exit(true_ohlc[None], np.array([take_profit]))
        sell_price, hit_tp = float(sell_price_arr[0]), bool(hit_tp_arr[0])
        bars_held = int(bars_held_arr[0])
        trade_return = float(sell_price / close_0 - 1.0)
        label, would_trade = classify_walk_forward_decision(
            eligible, meets_return_threshold, confident_enough, hit_tp, trade_return
        )
        direction_correct = bool(true_ohlc[HORIZON - 1, CLOSE_IDX] > close_0)

        decision = {
            "start_idx": start_idx,
            "close_0": float(close_0),
            "take_profit": float(take_profit),
            "confidence": float(confidence),
            "label": label,
            "would_trade": would_trade,
            "sell_price": sell_price,
            "hit_take_profit": hit_tp,
            "bars_held": bars_held,
            "trade_return": trade_return,
            "direction_correct": direction_correct,
        }
        decisions.append(decision)

        if would_trade:
            equity *= 1.0 + trade_return
            trades.append(decision)
            start_idx += bars_held
        else:
            start_idx += 1

    return {"trades": trades, "decisions": decisions, "equity_final": equity}


def walk_forward_stats(df: pd.DataFrame, result: dict, ctx_bars: int) -> dict:
    """Reduces run_walk_forward's raw trade/decision log into a reportable shape, plus
    the case breakdown across every decision point checked."""
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
    held = np.array([t["bars_held"] for t in trades])
    out["win_rate"] = float((rets > 0).mean())
    out["take_profit_rate"] = float(hits.mean())
    out["avg_return"] = float(rets.mean())
    out["avg_bars_held"] = float(held.mean())  # < HORIZON means the early-resume change (see run_walk_forward) is doing something

    first, last = trades[0], trades[-1]
    entry_idx = first["start_idx"] + ctx_bars - 1  # this trade's close_0 bar
    exit_idx = last["start_idx"] + ctx_bars - 1 + last["bars_held"]  # the bar this trade actually closed on
    out.update(equity_stats(df, entry_idx, exit_idx, first["close_0"], last["sell_price"], out["total_return"]))
    return out


def confidence_calibration(decisions: list[dict]) -> list[dict]:
    """Checks whether `confidence` (now binary -- coherent-up or not, see
    patchtst_revin_walk_forward_confidence) tracks correctness, independent of whether a
    decision was actually traded (would_trade is also gated by min_return_threshold,
    which would otherwise confound this check). Uses every decision point's
    trade_return/direction_correct -- both computed unconditionally in run_walk_forward
    regardless of would_trade, i.e. "what would have happened had this been traded" -- so
    this runs on the full decision log without re-simulating anything.

    An earlier version of this function binned a continuous magnitude-percentile
    confidence score; that ranking was dropped (see the module note above
    patchtst_revin_walk_forward_confidence) once it turned out to be inversely related to
    win rate. With confidence now binary, this just reports the two populations directly:
    coherent-up vs. not."""
    groups = [("coherent-up (confidence=1)", 1.0), ("not coherent-up (confidence=0)", 0.0)]
    rows = []
    for label, value in groups:
        in_group = [d for d in decisions if d["confidence"] == value]
        if not in_group:
            rows.append({"bin": label, "n": 0, "win_rate": None, "direction_accuracy": None})
            continue
        rets = np.array([d["trade_return"] for d in in_group])
        correct = np.array([d["direction_correct"] for d in in_group])
        rows.append({
            "bin": label,
            "n": len(in_group),
            "win_rate": float((rets > 0).mean()),
            "direction_accuracy": float(correct.mean()),
        })
    return rows


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    logger.info("device: %s", device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    if cfg.get("target") != "raw_price_revin_no_volume":
        raise ValueError(
            f"checkpoint's config['target']={cfg.get('target')!r} -- expected "
            "'raw_price_revin_no_volume' (hf_patchtst_revin_no_volume checkpoints only)."
        )

    df, bounds, stats = build_dataset(cfg["data_path"])
    ohlc_arr = df[OHLC_COLS].to_numpy(dtype=np.float32)
    closes = df["close"].to_numpy(dtype=np.float64)
    ctx_bars = cfg["context_length"]

    model = build_model(context_length=ctx_bars, **cfg["model"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    test_lo, test_hi = bounds["test"]
    wf_entry_idx = test_lo + ctx_bars - 1  # first decision point's close_0 bar

    logger.info(
        "running walk-forward backtest (ctx=%d bars, confidence>=%.2f, min_return>=%.3f%%, %d..%d)...",
        ctx_bars, args.confidence_threshold, args.min_return_threshold * 100, test_lo, test_hi,
    )
    predict_fn = make_patchtst_revin_predict_fn(model, device)
    wf = run_walk_forward(
        predict_fn, ohlc_arr, test_lo, test_hi, ctx_bars, args.confidence_threshold, args.min_return_threshold
    )
    logger.info("walk-forward: %d trades / %d decisions", len(wf["trades"]), len(wf["decisions"]))

    calibration = confidence_calibration(wf["decisions"])
    logger.info("confidence calibration (does confidence track correctness?):")
    for row in calibration:
        logger.info("  %s", row)

    results = {
        "walk_forward": {
            "checkpoint": args.checkpoint,
            "checkpoint_config": cfg,
            "ctx_bars": ctx_bars,
            "confidence_threshold": args.confidence_threshold,
            "min_return_threshold": args.min_return_threshold,
            "buy_and_hold": buy_and_hold_benchmark(df, wf_entry_idx, test_hi - 1),
            "naive_periodic": naive_periodic_benchmark(df, closes, wf_entry_idx, test_hi),
            "patchtst_revin": walk_forward_stats(df, wf, ctx_bars),
            "confidence_calibration": calibration,
        }
    }

    if args.metrics_out:
        out_path = Path(args.metrics_out)
    else:
        out_path = Path("steven/outputs") / f"backtest_{Path(args.checkpoint).stem}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("wrote metrics to %s", out_path)
    logger.info("walk_forward: %s", json.dumps(results["walk_forward"], indent=2, default=str))


if __name__ == "__main__":
    main()
