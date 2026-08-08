"""Offline generative-quality evaluation for a CVAE checkpoint -- diversity, calibration
(CRPS/rank-histogram/PIT-coverage), and context-sensitivity (regime bucket effect ratio),
computed over a full deterministic rolling-window pass at one fixed context length. See
cvae_direction_collapse.md's "generative pivot" discussion and generative_metrics.py's
module docstring for what these measure and why, as distinct from evaluate.py's trading-
framed PatchTST-vs-CVAE backtest comparison (no trade decisions here at all).

Usage:
    python steven/src/evaluate_generative.py \\
        --cvae-checkpoint steven/outputs/cvae_checkpoint_generative.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.data_pipeline as dp
import src.momentum_pipeline as momentum_pipeline
from src.data_pipeline import RollingWindowSampler, build_dataset, build_window, extract_arrays
from src.generative_metrics import (
    bucket_effect_ratio,
    coverage,
    crps_from_samples,
    crps_skill_score,
    diversity_stats,
    extreme_rank_fraction,
    pit_values,
    rank_histogram,
    regime_indicators,
    variance_ratio,
)
from src.generative_plots import build_diversity_fan, build_regime_grid
from src.models.cvae_inpainting import CVAEInpainting

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

COMPONENT_NAMES = ["open_ret", "body_ret", "upper_wick", "lower_wick"]
NLL_FAMILY = {"open_ret": "laplace", "body_ret": "laplace", "upper_wick": "gaussian", "lower_wick": "gaussian"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cvae-checkpoint", type=str, default="steven/outputs/cvae_checkpoint_generative.pt")
    p.add_argument("--ctx-bars", type=int, default=70)
    p.add_argument(
        "--k", type=int, default=32,
        help="Samples per window -- higher than production's default of 5, since this is an "
        "offline diagnostic with no latency constraint and CRPS/rank-histogram Monte Carlo "
        "variance benefits from more draws.",
    )
    p.add_argument("--split", type=str, default="test", choices=["val", "test"])
    p.add_argument("--step", type=int, default=1, help="Rolling-window stride, in bars.")
    p.add_argument("--regime-lookback", type=int, default=20)
    p.add_argument(
        "--climatology-size", type=int, default=200,
        help="Number of train-set body_ret draws used as the context-blind climatology baseline "
        "for the CRPS skill score.",
    )
    p.add_argument("--metrics-out", type=str, default="steven/outputs/generative_metrics.json")
    p.add_argument("--plots-dir", type=str, default="steven/outputs/generative_plots")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    ckpt = torch.load(args.cvae_checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    momentum_cfg = cfg.get("momentum_features")
    if momentum_cfg and momentum_cfg.get("enabled"):
        df, bounds, stats, momentum_stats = momentum_pipeline.build_momentum_dataset(
            cfg["data_path"], momentum_cfg["vix_data_path"]
        )
        logger.info("momentum features enabled: ema_cross/trend_position/rsi/vix, N_CHANNELS=%d", dp.N_CHANNELS)
    else:
        df, bounds, stats = build_dataset(cfg["data_path"])
        df = momentum_pipeline.add_momentum_features(df)  # reporting-only regime columns
    feat, opens, closes = extract_arrays(df)

    is_nll = cfg.get("loss", {}).get("reconstruction") == "nll"
    cvae = CVAEInpainting(**cfg["model"], in_channels=dp.N_CHANNELS).to(device)
    cvae.load_state_dict(ckpt["model_state"])
    cvae.eval()

    lo, hi = bounds[args.split]
    pairs = RollingWindowSampler(lo, hi, args.ctx_bars, step=args.step).pairs()
    logger.info(
        "evaluating on %d windows (ctx_bars=%d, split=%s, k=%d, reconstruction=%s)",
        len(pairs), args.ctx_bars, args.split, args.k, "nll" if is_nll else "mse",
    )

    windows = [build_window(feat, opens, closes, s, c) for s, c in pairs]
    masked = torch.from_numpy(np.stack([w["masked_tensor"] for w in windows])).to(device)
    true_price = np.stack([w["y"][:12].reshape(3, 4) for w in windows])  # (N,3,4)
    true_volume = np.stack([w["y"][12:15] for w in windows])  # (N,3)

    with torch.no_grad():
        price_samples_t, vol_samples_t = cvae.sample(masked, k=args.k)  # (K,N,3,4), (K,N,3)
    price_samples = price_samples_t.cpu().numpy()
    vol_samples = vol_samples_t.cpu().numpy()

    body_ret_col = dp.FEATURE_COLS.index("body_ret")
    train_lo, train_hi = bounds["train"]

    per_component = {}
    for bar in range(3):
        for ci, comp in enumerate(COMPONENT_NAMES):
            samples = price_samples[:, :, bar, ci]  # (K,N)
            y = true_price[:, bar, ci]
            div = diversity_stats(samples)
            per_component[f"bar{bar}_{comp}"] = {
                "diversity_std": div["std"],
                "diversity_pairwise": div["pairwise_mean_abs_diff"],
                "variance_ratio": variance_ratio(samples, y),
                "crps": float(crps_from_samples(samples, y).mean()),
                "extreme_rank_fraction": extreme_rank_fraction(rank_histogram(samples, y, rng=rng), k=args.k),
            }
    for bar in range(3):
        samples = vol_samples[:, :, bar]
        y = true_volume[:, bar]
        div = diversity_stats(samples)
        per_component[f"bar{bar}_volume"] = {
            "diversity_std": div["std"],
            "diversity_pairwise": div["pairwise_mean_abs_diff"],
            "variance_ratio": variance_ratio(samples, y),
            "crps": float(crps_from_samples(samples, y).mean()),
            "extreme_rank_fraction": extreme_rank_fraction(rank_histogram(samples, y, rng=rng), k=args.k),
        }

    # CRPS skill score vs. a context-blind climatology baseline, on the headline component
    # (bar 0's body_ret): does conditioning on context do anything at all?
    climatology_pool = feat[train_lo:train_hi, body_ret_col]
    climatology_draw = rng.choice(climatology_pool, size=min(args.climatology_size, len(climatology_pool)), replace=False)
    climatology_samples = np.tile(climatology_draw[:, None], (1, len(pairs)))
    model_crps_bar0_body = crps_from_samples(price_samples[:, :, 0, 1], true_price[:, 0, 1])
    climatology_crps_bar0_body = crps_from_samples(climatology_samples, true_price[:, 0, 1])
    skill_score = crps_skill_score(model_crps_bar0_body, climatology_crps_bar0_body)

    # Context-sensitivity: bucket windows by causal realized-vol tercile, compare the real
    # vs. generated bucket-to-bucket shift (bucket_effect_ratio) per component/bar.
    realized_vols = np.array([
        regime_indicators(feat, s + c, c, body_ret_col=body_ret_col, lookback=args.regime_lookback)["realized_vol"]
        for s, c in pairs
    ])
    low_thresh, high_thresh = np.percentile(realized_vols, [33.3, 66.7])
    low_mask = realized_vols <= low_thresh
    high_mask = realized_vols >= high_thresh

    context_sensitivity = {}
    for bar in range(3):
        for ci, comp in enumerate(COMPONENT_NAMES):
            y = true_price[:, bar, ci]
            gen_mean = price_samples[:, :, bar, ci].mean(axis=0)  # (N,) per-window model mean
            gen_std = price_samples[:, :, bar, ci].std(axis=0)  # (N,) per-window model spread
            real_bucket_mean = {"low": float(y[low_mask].mean()), "high": float(y[high_mask].mean())}
            gen_bucket_mean = {"low": float(gen_mean[low_mask].mean()), "high": float(gen_mean[high_mask].mean())}
            real_bucket_std = {"low": float(y[low_mask].std()), "high": float(y[high_mask].std())}
            gen_bucket_std = {"low": float(gen_std[low_mask].mean()), "high": float(gen_std[high_mask].mean())}
            context_sensitivity[f"bar{bar}_{comp}"] = {
                "effect_ratio_mean": bucket_effect_ratio(real_bucket_mean, gen_bucket_mean),
                "effect_ratio_spread": bucket_effect_ratio(real_bucket_std, gen_bucket_std),
            }

    results = {
        "checkpoint": args.cvae_checkpoint,
        "reconstruction": "nll" if is_nll else "mse",
        "n_windows": len(pairs),
        "ctx_bars": args.ctx_bars,
        "k": args.k,
        "split": args.split,
        "per_component": per_component,
        "crps_skill_score_bar0_body_ret": skill_score,
        "context_sensitivity_realized_vol_tercile": context_sensitivity,
    }

    # Calibration curves (PIT/coverage) require the model's own learned variance -- only
    # meaningful for a reconstruction="nll" checkpoint. Uses the prior's mean z (a natural
    # deterministic "point estimate" decode), not a stochastic sample() draw.
    if is_nll:
        with torch.no_grad():
            mu_p, logvar_p, ctx_repr = cvae.encode_prior(masked)
            price_mean, price_logvar, vol_mean, vol_logvar = cvae.decode(mu_p, ctx_repr)
        price_mean_np = price_mean.cpu().numpy()
        price_logvar_np = price_logvar.cpu().numpy()
        calibration = {}
        for bar in range(3):
            for ci, comp in enumerate(COMPONENT_NAMES):
                pit = pit_values(
                    true_price[:, bar, ci], price_mean_np[:, bar, ci], price_logvar_np[:, bar, ci],
                    family=NLL_FAMILY[comp],
                )
                calibration[f"bar{bar}_{comp}"] = coverage(pit)
        results["calibration_coverage"] = calibration

    metrics_out = Path(args.metrics_out)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.write_text(json.dumps(results, indent=2))
    logger.info("wrote metrics to %s", metrics_out)

    plots_dir = Path(args.plots_dir)
    build_regime_grid(df, pairs, price_samples, realized_vols, args.ctx_bars, plots_dir / "regime_grid.png", seed=args.seed)
    fan_idx = int(rng.integers(len(pairs)))
    fan_start, fan_ctx = pairs[fan_idx]
    build_diversity_fan(df, fan_start, fan_ctx, price_samples[:, fan_idx], plots_dir / "diversity_fan.png")


if __name__ == "__main__":
    main()
