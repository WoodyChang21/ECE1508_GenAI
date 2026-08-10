"""Hysteresis-band backtest for hf_patchtst_revin_no_volume checkpoints.

Ports the entry/exit rule used by this project's Mamba return-forecasting model to our
OHLC-forecasting PatchTST checkpoints:

    - flat & forecast >= ENTER_BPS: enter long
    - long & forecast >  EXIT_BPS:  stay long
    - long & forecast <= EXIT_BPS:  exit to cash
    - negative forecasts never open a short position (long-only, same as every other
      backtest on this branch)

Mamba predicts a single scalar (next-hour simple return) directly, so "forecast" there
is that number. This model predicts the next HORIZON (3) real OHLC bars instead, so
"forecast" here is the model's **predicted bar-1 close return** (the nearest, most
directly comparable single-step quantity: `pred_close[0] / close_0 - 1`, in bps) --
bars 2/3 of the prediction are not used by this strategy at all.

Structurally this is a different walk than `evaluate_revin.py`'s take-profit strategy:
there, a position always resolves within HORIZON bars (take-profit fill or expiry) and
the walk resumes only once it does. Here, a position can stay open for however many
consecutive hours the rolling forecast keeps clearing EXIT_BPS -- there is no fixed
holding horizon and no take-profit limit-order simulation at all. The walk therefore
re-forecasts and re-evaluates the rule **every hour**, regardless of position state.

Explicitly NOT modeled in this variant (by request): transaction costs. The reference
Mamba rule this is ported from also charges 1bp on entry and 1bp on exit; this version
omits both, so it is not a fully faithful reproduction of that strategy -- see the
"transaction_costs" field in the output JSON and steven/experiments.md once this is
logged.

Reuses `evaluate_revin.py`'s model loading, per-window prediction closure, and
benchmark functions (buy-and-hold, naive periodic, equity_stats) unchanged -- only the
decision rule and walk-forward loop are new. Does not modify evaluate_revin.py.

Usage:
    python steven/src/evaluate_revin_hysteresis.py \\
        --checkpoint steven/outputs/patchtst_revin_novolume_patch14_14_channel_attention_false_checkpoint.pt
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
from src.evaluate_revin import (
    resolve_device,
    buy_and_hold_benchmark,
    naive_periodic_benchmark,
    equity_stats,
    make_patchtst_revin_predict_fn,
    build_raw_ohlc_window,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument(
        "--metrics-out", type=str, default=None,
        help="Defaults to steven/outputs/backtest_hysteresis_<checkpoint filename stem>.json",
    )
    p.add_argument(
        "--enter-bps", type=float, default=2.0,
        help="Minimum predicted bar-1 close return (bps) required to enter long from flat.",
    )
    p.add_argument(
        "--exit-bps", type=float, default=0.0,
        help="Exit an open long once the predicted bar-1 close return drops to/below this "
        "(bps). Between exit_bps and enter_bps is the 'stay long, don't re-enter' band -- "
        "the hysteresis gap that keeps a noisy forecast from flip-flopping the position.",
    )
    return p.parse_args()


def run_hysteresis_walk_forward(
    predict_fn, ohlc_arr: np.ndarray, test_lo: int, test_hi: int, ctx_bars: int,
    enter_bps: float, exit_bps: float,
) -> dict:
    """Walks the test range one hour at a time, always -- no resume-on-resolution, since
    a position's holding length isn't fixed here (unlike the take-profit strategy's
    HORIZON-bar horizon). At each hour: roll the context window forward, get the
    predicted bar-1 close return in bps, and apply the enter/stay/exit rule against the
    current position state. If a position is still open when the test range ends, it's
    force-closed at the last decision's close so total_return reflects a fully realized
    equity curve (flagged via position_open_at_end)."""
    start_idx = test_lo
    equity = 1.0
    position = "flat"
    entry_price = None
    entry_idx = None
    trades: list[dict] = []
    decisions: list[dict] = []
    last_close_0 = None
    last_decision_idx = None

    while start_idx + ctx_bars + HORIZON <= test_hi:
        w = build_raw_ohlc_window(ohlc_arr, start_idx, ctx_bars)
        close_0 = w["close_0"]
        decision_idx = start_idx + ctx_bars - 1  # this decision's close_0 bar, matches evaluate_revin.py's convention

        pred = predict_fn(w["context"], close_0)
        pred_close_bar1 = float(pred["price"][0, CLOSE_IDX])
        forecast_bps = (pred_close_bar1 / close_0 - 1.0) * 10000.0

        if position == "flat":
            if forecast_bps >= enter_bps:
                position = "long"
                entry_price = close_0
                entry_idx = decision_idx
        else:  # position == "long"
            if forecast_bps <= exit_bps:
                trade_return = close_0 / entry_price - 1.0
                equity *= 1.0 + trade_return
                trades.append({
                    "entry_idx": entry_idx, "exit_idx": decision_idx,
                    "entry_price": entry_price, "exit_price": close_0,
                    "trade_return": trade_return, "bars_held": decision_idx - entry_idx,
                })
                position = "flat"
                entry_price = None
                entry_idx = None
            # else: forecast_bps > exit_bps -- stay long, no action (covers both the
            # 0-2bps "stay" band and forecast_bps >= enter_bps while already long)

        decisions.append({
            "start_idx": start_idx, "close_0": float(close_0),
            "forecast_bps": forecast_bps, "position_after": position,
        })
        last_close_0 = close_0
        last_decision_idx = decision_idx
        start_idx += 1

    position_open_at_end = position == "long"
    if position_open_at_end:
        trade_return = last_close_0 / entry_price - 1.0
        equity *= 1.0 + trade_return
        trades.append({
            "entry_idx": entry_idx, "exit_idx": last_decision_idx,
            "entry_price": entry_price, "exit_price": last_close_0,
            "trade_return": trade_return, "bars_held": last_decision_idx - entry_idx,
            "force_closed_at_end": True,
        })

    return {
        "trades": trades, "decisions": decisions, "equity_final": equity,
        "position_open_at_end": position_open_at_end,
    }


def hysteresis_stats(df: pd.DataFrame, result: dict) -> dict:
    trades, decisions, equity_final = result["trades"], result["decisions"], result["equity_final"]
    out = {
        "n_decisions": len(decisions),
        "n_trades": len(trades),
        "total_return": equity_final - 1.0,
        "position_open_at_end": result["position_open_at_end"],
    }
    if not trades:
        out.update(win_rate=None, avg_return=None, avg_bars_held=None)
        return out

    rets = np.array([t["trade_return"] for t in trades])
    held = np.array([t["bars_held"] for t in trades])
    out["win_rate"] = float((rets > 0).mean())
    out["avg_return"] = float(rets.mean())
    out["avg_bars_held"] = float(held.mean())

    first, last = trades[0], trades[-1]
    out.update(equity_stats(
        df, first["entry_idx"], last["exit_idx"], first["entry_price"], last["exit_price"], out["total_return"],
    ))
    return out


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
    wf_entry_idx = test_lo + ctx_bars - 1

    logger.info(
        "running hysteresis-band backtest (ctx=%d bars, enter>=%.1fbps, exit<=%.1fbps, "
        "no transaction costs, %d..%d)...",
        ctx_bars, args.enter_bps, args.exit_bps, test_lo, test_hi,
    )
    predict_fn = make_patchtst_revin_predict_fn(model, device)
    wf = run_hysteresis_walk_forward(
        predict_fn, ohlc_arr, test_lo, test_hi, ctx_bars, args.enter_bps, args.exit_bps
    )
    logger.info(
        "hysteresis walk-forward: %d trades / %d decisions%s",
        len(wf["trades"]), len(wf["decisions"]),
        " (final position force-closed for equity calc)" if wf["position_open_at_end"] else "",
    )

    results = {
        "hysteresis_walk_forward": {
            "checkpoint": args.checkpoint,
            "checkpoint_config": cfg,
            "ctx_bars": ctx_bars,
            "enter_bps": args.enter_bps,
            "exit_bps": args.exit_bps,
            "transaction_costs": "none -- not modeled in this variant (requested)",
            "buy_and_hold": buy_and_hold_benchmark(df, wf_entry_idx, test_hi - 1),
            "naive_periodic": naive_periodic_benchmark(df, closes, wf_entry_idx, test_hi),
            "patchtst_revin_hysteresis": hysteresis_stats(df, wf),
        }
    }

    if args.metrics_out:
        out_path = Path(args.metrics_out)
    else:
        out_path = Path("steven/outputs") / f"backtest_hysteresis_{Path(args.checkpoint).stem}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("wrote metrics to %s", out_path)
    logger.info("hysteresis_walk_forward: %s", json.dumps(results["hysteresis_walk_forward"], indent=2, default=str))


if __name__ == "__main__":
    main()
