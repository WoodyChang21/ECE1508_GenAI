"""CLI entry point for the steven4 rolling hour-by-hour backtest (see src/rolling_backtest.py
for the state machine and steven/rolling_hour_backtest.md for the full writeup/results).

Usage:
    python steven/src/rolling_trend_backtest.py \\
        --cvae-checkpoint steven/outputs/cvae_checkpoint_h1.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.data_pipeline as dp
import src.momentum_pipeline as mp
from src.data_pipeline import build_dataset, extract_arrays
from src.evaluate import WALK_FORWARD_CTX_BARS, buy_and_hold_benchmark
from src.models.cvae_inpainting import CVAEInpainting
from src.rolling_backtest import make_cvae_rolling_predict_fn, rolling_backtest_stats, run_rolling_backtest

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cvae-checkpoint", type=str, default="steven/outputs/cvae_checkpoint_h1.pt")
    p.add_argument("--num-samples", type=int, default=None, help="Defaults to the checkpoint's own config['inference']['num_samples'] (falls back to 5).")
    p.add_argument("--consensus-up-threshold", type=float, default=0.6, help="frac_up >= this -> 'uptrend' (entry signal).")
    p.add_argument("--consensus-down-threshold", type=float, default=0.4, help="frac_up <= this -> 'downtrend' (forced-exit signal).")
    p.add_argument("--stop-loss-pct", type=float, default=0.01, help="Hard stop-loss, fraction below entry (0.01 = 1%%).")
    p.add_argument("--ctx-bars", type=int, default=WALK_FORWARD_CTX_BARS)
    p.add_argument("--split", type=str, default="test", choices=["val", "test"])
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--metrics-out", type=str, default="steven/outputs/rolling_backtest_metrics.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    logger.info("device: %s", device)

    ckpt = torch.load(args.cvae_checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    num_samples = args.num_samples if args.num_samples is not None else cfg.get("inference", {}).get("num_samples", 5)
    logger.info(
        "checkpoint=%s reconstruction=%s num_samples=%d consensus_up=%.2f consensus_down=%.2f stop_loss_pct=%.3f",
        Path(args.cvae_checkpoint).name, cfg.get("loss", {}).get("reconstruction"),
        num_samples, args.consensus_up_threshold, args.consensus_down_threshold, args.stop_loss_pct,
    )

    momentum_cfg = cfg.get("momentum_features")
    if momentum_cfg and momentum_cfg.get("enabled"):
        df, bounds, stats, momentum_stats = mp.build_momentum_dataset(cfg["data_path"], momentum_cfg["vix_data_path"])
        logger.info("momentum features enabled, N_CHANNELS=%d", dp.N_CHANNELS)
    else:
        df, bounds, stats = build_dataset(cfg["data_path"])
    feat, opens, closes = extract_arrays(df)

    cvae = CVAEInpainting(**cfg["model"], in_channels=dp.N_CHANNELS).to(device)
    cvae.load_state_dict(ckpt["model_state"])
    cvae.eval()

    predict_fn = make_cvae_rolling_predict_fn(
        cvae, device, num_samples, args.consensus_up_threshold, args.consensus_down_threshold,
    )

    test_lo, test_hi = bounds[args.split]
    result = run_rolling_backtest(predict_fn, feat, opens, closes, test_lo, test_hi, args.ctx_bars, args.stop_loss_pct)
    stats_out = rolling_backtest_stats(df, result)

    entry_idx = test_lo + args.ctx_bars - 1
    bh = buy_and_hold_benchmark(df, entry_idx, test_hi - 1)

    results = {
        "checkpoint": args.cvae_checkpoint,
        "consensus_up_threshold": args.consensus_up_threshold,
        "consensus_down_threshold": args.consensus_down_threshold,
        "stop_loss_pct": args.stop_loss_pct,
        "num_samples": num_samples,
        "ctx_bars": args.ctx_bars,
        "split": args.split,
        "rolling_backtest": stats_out,
        "buy_and_hold": bh,
    }

    Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metrics_out).write_text(json.dumps(results, indent=2))
    logger.info("wrote %s", args.metrics_out)
    logger.info(
        "rolling backtest: n_trades=%d total_return=%.4f win_rate=%s  |  buy_and_hold total_return=%.4f",
        stats_out["n_trades"], stats_out["total_return"], stats_out.get("win_rate"), bh["total_return"],
    )


if __name__ == "__main__":
    main()
