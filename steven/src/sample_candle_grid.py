"""Ground truth + k sampled candle completions, side by side, for one example window per
trend label (uptrend/downtrend/choppy) -- a single-context view of CVAE's own sample
diversity, in the same visual language (candlesticks) as the main slide diagram, as
opposed to build_diversity_fan's k=32 semi-transparent close-price lines (which show
density across many draws, not individual candle shapes) or build_regime_grid's
volatility-bucketed rows (which compare regimes, not one context at a glance).

Reuses render_candle_panel (generative_plots.py) and reconstruct_prices
(data_pipeline.py) exactly as build_regime_grid already does -- this script is just that
row-of-panels pattern applied to one context per trend label instead of one context per
volatility bucket.

Usage:
    python steven/src/sample_candle_grid.py \\
        --cvae-checkpoint steven/outputs/cvae_checkpoint_generative.pt
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.data_pipeline as dp
import src.momentum_pipeline as mp
from src.data_pipeline import HORIZON, RollingWindowSampler, build_dataset, build_window, extract_arrays, reconstruct_prices
from src.evaluate import classify_trend, trend_z_score
from src.generative_plots import render_candle_panel
from src.models.cvae_inpainting import CVAEInpainting

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cvae-checkpoint", type=str, default="steven/outputs/cvae_checkpoint_generative.pt")
    p.add_argument("--k", type=int, default=None, help="Samples per window. Defaults to the checkpoint's own config['inference']['num_samples'] (falls back to 5 if absent).")
    p.add_argument("--ctx-bars", type=int, default=70)
    p.add_argument("--trend-lookback", type=int, default=20)
    p.add_argument("--step", type=int, default=5, help="Rolling-window stride (bars) when scanning the test split for candidate windows to classify.")
    p.add_argument("--n-examples", type=int, default=1, help="Examples per trend label, spread evenly across that label's candidate pool (ignored when --pick-mode=random).")
    p.add_argument("--labels", type=str, default="uptrend,downtrend,choppy", help="Comma-separated subset of {uptrend,downtrend,choppy} to render.")
    p.add_argument("--pick-mode", type=str, default="spread", choices=["spread", "random"], help="'spread': --n-examples windows spread evenly across the label's pool (deterministic, default). 'random': one uniformly random window per label (start_idx varies run to run unless --seed is fixed).")
    p.add_argument("--layout", type=str, default="row", choices=["row", "grid"], help="'row': 1x(k+1) panels. 'grid': 2x((k+1)/2) panels -- fits a slide page better for k=5 (2x3).")
    p.add_argument("--generic-title", action="store_true", help="Suptitle omits the trend label/start_idx/checkpoint name -- use when only showing one example and don't want to imply it's representative or reveal internal labeling.")
    p.add_argument("--out-dir", type=str, default="steven/outputs/generative_plots")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    ckpt = torch.load(args.cvae_checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    k = args.k if args.k is not None else cfg.get("inference", {}).get("num_samples", 5)
    logger.info("checkpoint=%s reconstruction=%s k=%d", Path(args.cvae_checkpoint).name, cfg.get("loss", {}).get("reconstruction"), k)

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

    lo, hi = bounds["test"]
    pairs = RollingWindowSampler(lo, hi, args.ctx_bars, step=args.step).pairs()
    body_ret_col = dp.FEATURE_COLS.index("body_ret")
    trend_z = np.array([trend_z_score(feat, s + c, c, body_ret_col, lookback=args.trend_lookback) for s, c in pairs])
    labels = np.array([classify_trend(z) for z in trend_z])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    pick_rng = np.random.default_rng(args.seed)
    requested_labels = args.labels.split(",")

    for label in requested_labels:
        idx_pool = np.where(labels == label)[0]
        if len(idx_pool) == 0:
            logger.warning("no %s windows found -- skipping", label)
            continue
        if args.pick_mode == "random":
            picks = [idx_pool[pick_rng.integers(len(idx_pool))]]
        else:
            n = min(args.n_examples, len(idx_pool))
            positions = np.unique(np.linspace(0, len(idx_pool) - 1, n).astype(int))
            picks = idx_pool[positions]

        for example_num, i in enumerate(picks, start=1):
            start_idx, this_ctx = pairs[i]
            w = build_window(feat, opens, closes, start_idx, this_ctx)
            masked_t = torch.from_numpy(w["masked_tensor"])[None].to(device)
            with torch.no_grad():
                price_samples_t, _ = cvae.sample(masked_t, k=k)  # (K,1,3,4)
            price_samples = price_samples_t.cpu().numpy()[:, 0]  # (K,3,4)

            ctx_tail = min(this_ctx, 20)
            hz_start = start_idx + this_ctx
            plot_rows = df.iloc[hz_start - ctx_tail : hz_start + HORIZON]
            true_df = plot_rows.set_index("datetime")[["open", "high", "low", "close", "volume"]]
            close_0 = float(df.iloc[hz_start - 1]["close"])

            n_panels = k + 1
            if args.layout == "grid":
                rows = 2
                cols = -(-n_panels // rows)  # ceil
                fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3.2 * rows))
                axes = axes.flatten()
                for extra_ax in axes[n_panels:]:
                    extra_ax.axis("off")
            else:
                fig, axes = plt.subplots(1, n_panels, figsize=(3 * n_panels, 3.2))

            render_candle_panel(axes[0], true_df, "Ground truth")
            for col in range(k):
                gen_ohlc = reconstruct_prices(price_samples[col], close_0)
                gen_df = true_df.copy()
                gen_df.loc[gen_df.index[-HORIZON:], ["open", "high", "low", "close"]] = gen_ohlc
                render_candle_panel(axes[col + 1], gen_df, f"sample {col + 1}")

            if args.generic_title:
                fig.suptitle("One context -- ground truth + 5 sampled completions")
            else:
                fig.suptitle(f"{label} #{example_num} -- ctx_bars={this_ctx}, start_idx={start_idx} -- {Path(args.cvae_checkpoint).name}")
            fig.tight_layout()
            suffix = "" if args.n_examples == 1 and args.pick_mode == "spread" else f"_{start_idx}"
            out_path = out_dir / f"sample_candles_{label}{suffix}.png"
            fig.savefig(out_path, bbox_inches="tight")
            plt.close(fig)
            written.append(out_path)
            logger.info("wrote %s", out_path)

    logger.info("done: %s", [str(p) for p in written])


if __name__ == "__main__":
    main()
