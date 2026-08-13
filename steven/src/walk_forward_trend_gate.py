"""PRE-PIVOT STRATEGY (steven4): this script re-runs evaluate.py's fixed-HORIZON-bar
bracket-order walk-forward (see the PRE-PIVOT note atop evaluate.py). Its own logic is
HORIZON-agnostic (delegates entirely to evaluate.py's now-HORIZON=1-safe functions), but it
still can't run against any checkpoint saved BEFORE the steven4 migration (cvae_checkpoint.pt,
cvae_checkpoint_generative.pt, etc.): CVAEInpainting's decoder width is derived from the
current *global* HORIZON constant at construction time, not saved per-checkpoint, so
`cvae.load_state_dict(...)` against an old (HORIZON=3, decoder width 30) checkpoint's
state dict will now raise a size-mismatch error against the newly-constructed
(HORIZON=1, decoder width 10) model, the same failure mode as a wrong in_channels. This
isn't fixable without either retraining every existing checkpoint at HORIZON=1 or making
HORIZON an instance parameter instead of a module constant (out of scope for steven4) --
check out steven3 (or an earlier steven4 commit, before the migration) to re-run this
script against those older checkpoints. Kept here, not deleted, because it's the exact
script that produced the results already written up in
generative_checkpoint_uptrend_gate.md; the active strategy going forward is the rolling
hour-by-hour hold in src/rolling_backtest.py/src/rolling_trend_backtest.py (see
steven/rolling_hour_backtest.md), evaluated against a freshly-trained HORIZON=1 checkpoint.

Re-runs evaluate.py's walk-forward backtest (orphaned since "evaluate.py: replace the
backtest with a chart-only scenario comparison" -- main() there only renders scenario
charts now) against a given CVAE checkpoint, once unrestricted and once with trades gated
to only fire when the trailing context is classified "uptrend" (or whichever labels are
passed) by the same trend_z_score/classify_trend used for sample_candle_grid.py's and
evaluate.py's scenario-chart labels.

Both runs use identical seeded RNG state and reuse evaluate.py's own
run_walk_forward/make_cvae_predict_fn/walk_forward_stats unmodified -- the trend gate is
a predict_fn wrapper that forces passes_quality_gate=False on disallowed-label windows
(classify_walk_forward_decision then always resolves those to "skipped"), not a change to
the backtest loop itself.

Caveats (see generative_checkpoint_uptrend_gate.md for the full writeup):
- trend_z_score is a REALIZED momentum filter on the context preceding the decision, not
  anything the model predicts -- "uptrend-only" here means "only trade when the recent
  past was already trending up," independent of what CVAE thinks happens next.
- n_decisions differs between the two runs because run_walk_forward's advance rule steps
  1 bar on no-trade vs. HORIZON bars on a trade (see cvae_direction_collapse.md's
  still-open to-do about this) -- a less-active strategy visits more decision points over
  the same calendar span, so outcome_breakdown fractions aren't directly comparable
  between runs. total_return/equity accounting is unaffected (it compounds over trades
  taken, not decision count).
- Single seed, single checkpoint, one test period (2024-2025 SPY) that itself drifted up
  ~24% -- not enough to distinguish a real edge from a momentum filter riding the market's
  own drift (see cvae_direction_collapse.md's "r=+0.202 was training-seed noise" finding
  for why this project treats single before/after comparisons like this with suspicion
  until checked across seeds).

Usage:
    python steven/src/walk_forward_trend_gate.py \\
        --cvae-checkpoint steven/outputs/cvae_checkpoint_generative.pt \\
        --allowed-labels uptrend
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.data_pipeline as dp
import src.momentum_pipeline as mp
from src.data_pipeline import build_dataset, extract_arrays, train_exit_return_bound
from src.evaluate import (
    WALK_FORWARD_CTX_BARS,
    buy_and_hold_benchmark,
    classify_trend,
    make_cvae_predict_fn,
    naive_periodic_benchmark,
    run_walk_forward,
    trend_z_score,
    walk_forward_stats,
)
from src.models.cvae_inpainting import CVAEInpainting

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cvae-checkpoint", type=str, default="steven/outputs/cvae_checkpoint_generative.pt")
    p.add_argument("--allowed-labels", type=str, default="uptrend", help="Comma-separated subset of {uptrend,downtrend,choppy} CVAE is allowed to trade in.")
    p.add_argument("--num-samples", type=int, default=None, help="Defaults to the checkpoint's own config['inference']['num_samples'] (falls back to 5).")
    p.add_argument("--cvae-sell-quantile", type=float, default=70.0)
    p.add_argument("--cvae-min-return-threshold", type=float, default=0.0002)
    p.add_argument("--stop-loss-pct", type=float, default=0.02)
    p.add_argument("--sell-bound-percentile", type=float, default=99.0)
    p.add_argument("--trend-lookback", type=int, default=20)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--metrics-out", type=str, default="steven/outputs/walk_forward_trend_gate_metrics.json")
    return p.parse_args()


def make_trend_gated_predict_fn(base_predict_fn, feat, body_ret_col, trend_lookback, allowed_labels):
    """Still calls base_predict_fn for every window (so the model's own take-profit/price
    are computed identically to the unrestricted run) but forces passes_quality_gate=False
    whenever the window's trailing-context trend label isn't in allowed_labels."""
    def predict(w):
        pred = base_predict_fn(w)
        z = trend_z_score(feat, w["start_idx"] + w["ctx_bars"], w["ctx_bars"], body_ret_col, lookback=trend_lookback)
        label = classify_trend(z)
        if label not in allowed_labels:
            pred = copy.copy(pred)
            pred["passes_quality_gate"] = False
        return pred
    return predict


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    allowed_labels = set(args.allowed_labels.split(","))

    ckpt = torch.load(args.cvae_checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    num_samples = args.num_samples if args.num_samples is not None else cfg.get("inference", {}).get("num_samples", 5)
    logger.info("checkpoint=%s reconstruction=%s num_samples=%d allowed_labels=%s", Path(args.cvae_checkpoint).name, cfg.get("loss", {}).get("reconstruction"), num_samples, allowed_labels)

    momentum_cfg = cfg.get("momentum_features")
    if momentum_cfg and momentum_cfg.get("enabled"):
        df, bounds, stats, momentum_stats = mp.build_momentum_dataset(cfg["data_path"], momentum_cfg["vix_data_path"])
    else:
        df, bounds, stats = build_dataset(cfg["data_path"])
    feat, opens, closes = extract_arrays(df)

    sell_bound = train_exit_return_bound(opens, closes, bounds["train"], percentile=args.sell_bound_percentile)
    logger.info("sell_bound (p%.1f |anchored log return| over train) = %.4f", args.sell_bound_percentile, sell_bound)

    cvae = CVAEInpainting(**cfg["model"], in_channels=dp.N_CHANNELS).to(device)
    cvae.load_state_dict(ckpt["model_state"])
    cvae.eval()

    test_lo, test_hi = bounds["test"]
    wf_entry_idx = test_lo + WALK_FORWARD_CTX_BARS - 1
    body_ret_col = dp.FEATURE_COLS.index("body_ret")

    torch.manual_seed(args.seed)
    baseline_predict = make_cvae_predict_fn(cvae, device, sell_bound, num_samples, args.cvae_sell_quantile)
    baseline_wf = run_walk_forward(baseline_predict, feat, opens, closes, test_lo, test_hi, args.cvae_min_return_threshold, args.stop_loss_pct)
    baseline_stats = walk_forward_stats(df, baseline_wf)

    torch.manual_seed(args.seed)  # identical seeded draws as the baseline run, isolating the gate's effect
    gated_predict_base = make_cvae_predict_fn(cvae, device, sell_bound, num_samples, args.cvae_sell_quantile)
    gated_predict = make_trend_gated_predict_fn(gated_predict_base, feat, body_ret_col, args.trend_lookback, allowed_labels)
    gated_wf = run_walk_forward(gated_predict, feat, opens, closes, test_lo, test_hi, args.cvae_min_return_threshold, args.stop_loss_pct)
    gated_stats = walk_forward_stats(df, gated_wf)

    results = {
        "checkpoint": args.cvae_checkpoint,
        "allowed_labels": sorted(allowed_labels),
        "num_samples": num_samples,
        "baseline": baseline_stats,
        "trend_gated": gated_stats,
        "buy_and_hold": buy_and_hold_benchmark(df, wf_entry_idx, test_hi - 1),
        "naive_periodic": naive_periodic_benchmark(df, closes, wf_entry_idx, test_hi),
    }

    Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metrics_out).write_text(json.dumps(results, indent=2))
    logger.info("wrote %s", args.metrics_out)
    logger.info("baseline total_return=%.4f  trend_gated total_return=%.4f  buy_and_hold total_return=%.4f",
                baseline_stats["total_return"], gated_stats["total_return"], results["buy_and_hold"]["total_return"])


if __name__ == "__main__":
    main()
