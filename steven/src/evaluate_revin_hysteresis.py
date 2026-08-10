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

**Threshold sweep** (`--sweep`): `enter_bps`/`exit_bps` were carried over from Mamba's
own tuned rule, not calibrated to this model's forecast distribution at all -- worth
checking. The model forward pass (one per hour, the expensive part) is run exactly
once regardless of how many threshold combinations are swept: `compute_forecast_sequence`
walks the test range and caches `(decision_idx, close_0, forecast_bps)` per hour, and
`simulate_hysteresis` cheaply replays the enter/stay/exit state machine against that
cached sequence for each `(enter_bps, exit_bps)` pair -- no repeated model calls.

Read the sweep result looking for a broad, stable region of good combinations, not just
the single best cell -- with only one continuous test window, picking the one grid cell
that happens to score highest risks curve-fitting the threshold to this specific
historical path rather than finding a genuinely better rule.

Usage:
    python steven/src/evaluate_revin_hysteresis.py \\
        --checkpoint steven/outputs/patchtst_revin_novolume_patch14_14_channel_attention_false_checkpoint.pt

    python steven/src/evaluate_revin_hysteresis.py \\
        --checkpoint steven/outputs/patchtst_revin_novolume_patch14_14_channel_attention_false_checkpoint.pt \\
        --sweep --enter-bps-grid 1,2,3,4,5 --exit-bps-grid -1,0,1
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
        help="Defaults to steven/outputs/backtest_hysteresis_<checkpoint filename stem>.json "
        "(single-run mode) or steven/outputs/backtest_hysteresis_sweep_<stem>.json (--sweep).",
    )
    p.add_argument(
        "--enter-bps", type=float, default=2.0,
        help="Single-run mode only. Minimum predicted bar-1 close return (bps) required "
        "to enter long from flat.",
    )
    p.add_argument(
        "--exit-bps", type=float, default=0.0,
        help="Single-run mode only. Exit an open long once the predicted bar-1 close "
        "return drops to/below this (bps). Between exit_bps and enter_bps is the 'stay "
        "long, don't re-enter' band -- the hysteresis gap that keeps a noisy forecast "
        "from flip-flopping the position.",
    )
    p.add_argument(
        "--sweep", action="store_true",
        help="Instead of one run at --enter-bps/--exit-bps, grid-sweep --enter-bps-grid x "
        "--exit-bps-grid (only combinations with exit_bps < enter_bps are evaluated). "
        "The model forward pass still only runs once -- see module docstring.",
    )
    p.add_argument(
        "--enter-bps-grid", type=str, default="1,2,3,4,5",
        help="Comma-separated enter_bps values to sweep (--sweep only).",
    )
    p.add_argument(
        "--exit-bps-grid", type=str, default="-1,0,1",
        help="Comma-separated exit_bps values to sweep (--sweep only).",
    )
    return p.parse_args()


def compute_forecast_sequence(
    predict_fn, ohlc_arr: np.ndarray, test_lo: int, test_hi: int, ctx_bars: int,
) -> list[dict]:
    """The one expensive pass: walks the test range one hour at a time and records each
    hour's predicted bar-1 close return in bps, with no decision logic attached at all.
    Shared by both single-run mode and --sweep -- every threshold combination replays
    against this same cached sequence instead of re-running the model."""
    forecast_seq: list[dict] = []
    start_idx = test_lo
    while start_idx + ctx_bars + HORIZON <= test_hi:
        w = build_raw_ohlc_window(ohlc_arr, start_idx, ctx_bars)
        close_0 = w["close_0"]
        decision_idx = start_idx + ctx_bars - 1  # matches evaluate_revin.py's convention

        pred = predict_fn(w["context"], close_0)
        pred_close_bar1 = float(pred["price"][0, CLOSE_IDX])
        forecast_bps = (pred_close_bar1 / close_0 - 1.0) * 10000.0

        forecast_seq.append({
            "decision_idx": decision_idx, "close_0": float(close_0), "forecast_bps": forecast_bps,
        })
        start_idx += 1

    return forecast_seq


def simulate_hysteresis(forecast_seq: list[dict], enter_bps: float, exit_bps: float) -> dict:
    """Cheap replay of the enter/stay/exit state machine against an already-computed
    forecast sequence -- no model calls, so this is fast enough to call once per grid
    cell in a sweep. Same rule and same force-close-at-end behavior as the original
    single-pass walk (see module docstring)."""
    equity = 1.0
    position = "flat"
    entry_price = None
    entry_idx = None
    trades: list[dict] = []

    for d in forecast_seq:
        close_0, forecast_bps, decision_idx = d["close_0"], d["forecast_bps"], d["decision_idx"]

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
            # exit_bps-to-enter_bps "stay" band and forecast_bps >= enter_bps while long)

    position_open_at_end = position == "long"
    if position_open_at_end and forecast_seq:
        last = forecast_seq[-1]
        trade_return = last["close_0"] / entry_price - 1.0
        equity *= 1.0 + trade_return
        trades.append({
            "entry_idx": entry_idx, "exit_idx": last["decision_idx"],
            "entry_price": entry_price, "exit_price": last["close_0"],
            "trade_return": trade_return, "bars_held": last["decision_idx"] - entry_idx,
            "force_closed_at_end": True,
        })

    return {
        "trades": trades, "n_decisions": len(forecast_seq), "equity_final": equity,
        "position_open_at_end": position_open_at_end,
    }


def run_hysteresis_walk_forward(
    predict_fn, ohlc_arr: np.ndarray, test_lo: int, test_hi: int, ctx_bars: int,
    enter_bps: float, exit_bps: float,
) -> dict:
    """Single-run convenience wrapper: compute the forecast sequence once, then replay
    the rule for this one (enter_bps, exit_bps) pair. Kept for backwards compatibility
    (single-run CLI mode) -- --sweep calls compute_forecast_sequence/simulate_hysteresis
    directly instead, to avoid recomputing the sequence per grid cell."""
    forecast_seq = compute_forecast_sequence(predict_fn, ohlc_arr, test_lo, test_hi, ctx_bars)
    result = simulate_hysteresis(forecast_seq, enter_bps, exit_bps)
    return {
        "trades": result["trades"], "decisions": forecast_seq,
        "equity_final": result["equity_final"], "position_open_at_end": result["position_open_at_end"],
    }


def hysteresis_stats(df: pd.DataFrame, result: dict, n_decisions: int | None = None) -> dict:
    trades, equity_final = result["trades"], result["equity_final"]
    out = {
        "n_decisions": n_decisions if n_decisions is not None else len(result.get("decisions", [])),
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


def run_sweep(
    df: pd.DataFrame, forecast_seq: list[dict], enter_grid: list[float], exit_grid: list[float],
) -> list[dict]:
    """Evaluates every (enter_bps, exit_bps) combination with exit_bps < enter_bps
    against the cached forecast_seq (no model calls), sorted by total_return
    descending. See module docstring for how to read this without overfitting to the
    one test window."""
    rows = []
    for enter_bps in enter_grid:
        for exit_bps in exit_grid:
            if exit_bps >= enter_bps:
                continue  # degenerate: no real "stay" band, skip
            result = simulate_hysteresis(forecast_seq, enter_bps, exit_bps)
            stats = hysteresis_stats(df, result, n_decisions=len(forecast_seq))
            rows.append({"enter_bps": enter_bps, "exit_bps": exit_bps, **stats})
    rows.sort(key=lambda r: r["total_return"], reverse=True)
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
    wf_entry_idx = test_lo + ctx_bars - 1
    predict_fn = make_patchtst_revin_predict_fn(model, device)

    if args.sweep:
        enter_grid = sorted(float(x) for x in args.enter_bps_grid.split(","))
        exit_grid = sorted(float(x) for x in args.exit_bps_grid.split(","))
        logger.info(
            "computing forecast sequence once (ctx=%d bars, %d..%d)...", ctx_bars, test_lo, test_hi,
        )
        forecast_seq = compute_forecast_sequence(predict_fn, ohlc_arr, test_lo, test_hi, ctx_bars)
        logger.info(
            "sweeping enter_bps=%s x exit_bps=%s (%d decisions cached, no further model calls)...",
            enter_grid, exit_grid, len(forecast_seq),
        )
        sweep_rows = run_sweep(df, forecast_seq, enter_grid, exit_grid)
        logger.info("sweep results (sorted by total_return, best first):")
        for row in sweep_rows:
            logger.info(
                "  enter=%.1fbps exit=%.1fbps  n_trades=%-4d win_rate=%s  total_return=%.4f  avg_bars_held=%s",
                row["enter_bps"], row["exit_bps"], row["n_trades"],
                f'{row["win_rate"]:.4f}' if row["win_rate"] is not None else "None",
                row["total_return"],
                f'{row["avg_bars_held"]:.2f}' if row.get("avg_bars_held") is not None else "None",
            )

        results = {
            "hysteresis_sweep": {
                "checkpoint": args.checkpoint,
                "checkpoint_config": cfg,
                "ctx_bars": ctx_bars,
                "enter_bps_grid": enter_grid,
                "exit_bps_grid": exit_grid,
                "transaction_costs": "none -- not modeled in this variant (requested)",
                "buy_and_hold": buy_and_hold_benchmark(df, wf_entry_idx, test_hi - 1),
                "naive_periodic": naive_periodic_benchmark(df, closes, wf_entry_idx, test_hi),
                "sweep_results": sweep_rows,
            }
        }
        if args.metrics_out:
            out_path = Path(args.metrics_out)
        else:
            out_path = Path("steven/outputs") / f"backtest_hysteresis_sweep_{Path(args.checkpoint).stem}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("wrote sweep metrics to %s", out_path)
        return

    logger.info(
        "running hysteresis-band backtest (ctx=%d bars, enter>=%.1fbps, exit<=%.1fbps, "
        "no transaction costs, %d..%d)...",
        ctx_bars, args.enter_bps, args.exit_bps, test_lo, test_hi,
    )
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
            "patchtst_revin_hysteresis": hysteresis_stats(df, wf, n_decisions=len(wf["decisions"])),
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
