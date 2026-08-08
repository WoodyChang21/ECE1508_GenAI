"""Standalone recheck of CVAE's direction-collapse symptoms against any checkpoint --
formalizes the ad hoc diagnostic used repeatedly in cvae_direction_collapse.md (across the
price_scale fix, the decoder_ctx_dim bottleneck, and the auxiliary direction loss) into a
reusable script instead of a one-off scratch script rewritten each time.

Usage:
    python steven/src/diagnose_cvae_direction.py --cvae-checkpoint steven/outputs/cvae_checkpoint.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_pipeline import (
    FEATURE_COLS,
    MAX_CONTEXT,
    build_dataset,
    build_window,
    exit_price_from_components,
    extract_arrays,
    shrink_components,
    train_exit_return_bound,
)
from src.models.cvae_inpainting import CVAEInpainting

BODY_RET_COL = FEATURE_COLS.index("body_ret")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cvae-checkpoint", type=str, default="steven/outputs/cvae_checkpoint.pt")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--num-windows", type=int, default=300)
    p.add_argument("--num-samples", type=int, default=5, help="K sampled draws per window.")
    p.add_argument("--cvae-sell-quantile", type=float, default=70.0, help="Same default as evaluate.py.")
    p.add_argument("--sell-bound-percentile", type=float, default=99.0, help="Same default as evaluate.py.")
    p.add_argument("--trend-lookback", type=int, default=10, help="Bars of preceding body_ret summed as the trend signal.")
    p.add_argument("--seed", type=int, default=123)
    return p.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def variance_ratio_and_correlation(draws: np.ndarray, trend: np.ndarray) -> dict:
    """draws: (N, K) -- K sampled values of one price component (e.g. body_ret) per
    window, N windows. trend: (N,) -- a real, context-derived signal to correlate
    against. Returns across-window std of per-window means (real, context-driven
    variation) vs. within-window std of per-window stds (the model's own sampling
    noise) -- ratio > 1 means context explains more variance than the model's own
    sampling noise does; ratio < 1 means the model looks more like sampling noise
    around a near-constant mean (collapsed). Also returns the Pearson correlation
    between per-window means and trend, and the overall mean (a large overall mean
    relative to `draws`' own spread indicates a systematic constant bias, not small
    noise around zero)."""
    per_window_mean = draws.mean(axis=1)
    per_window_std = draws.std(axis=1)
    across = float(per_window_mean.std())
    within = float(per_window_std.mean())
    return {
        "across_window_std": across,
        "within_window_std": within,
        "ratio": across / within if within > 0 else float("inf"),
        "correlation_with_trend": float(np.corrcoef(per_window_mean, trend)[0, 1]),
        "mean": float(draws.mean()),
    }


def predicted_return_stats(predicted_returns: np.ndarray) -> dict:
    """predicted_returns: (N,) -- each window's (take_profit / close_0 - 1), the exact
    quantity run_walk_forward's eligibility check (`take_profit > close_0`) depends on
    the sign of. Returns summary stats plus the fraction that would actually be
    eligible to trade (long-only)."""
    return {
        "mean": float(predicted_returns.mean()),
        "pct_eligible": float((predicted_returns > 0).mean()),
        "p5": float(np.percentile(predicted_returns, 5)),
        "p25": float(np.percentile(predicted_returns, 25)),
        "p50": float(np.percentile(predicted_returns, 50)),
        "p75": float(np.percentile(predicted_returns, 75)),
        "p95": float(np.percentile(predicted_returns, 95)),
    }


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    ckpt = torch.load(args.cvae_checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    print(f"checkpoint: {args.cvae_checkpoint}")
    print(f"model config: {cfg['model']}")
    print(f"loss config: {cfg['loss']}\n")

    df, bounds, stats = build_dataset(cfg["data_path"])
    feat, opens, closes = extract_arrays(df)

    cvae = CVAEInpainting(**cfg["model"]).to(device)
    cvae.load_state_dict(ckpt["model_state"])
    cvae.eval()

    sell_bound = train_exit_return_bound(opens, closes, bounds["train"], percentile=args.sell_bound_percentile)
    train_lo, train_hi = bounds["train"]
    test_lo, test_hi = bounds["test"]

    rng = np.random.default_rng(args.seed)
    last_valid_start = test_hi - MAX_CONTEXT - 3
    starts = rng.choice(np.arange(test_lo, last_valid_start), size=args.num_windows, replace=False)

    body_ret_draws = []  # (N, K)
    predicted_returns = []  # (N,)
    trend = []  # (N,)

    with torch.no_grad():
        for start_idx in starts:
            w = build_window(feat, opens, closes, int(start_idx), MAX_CONTEXT)
            masked_t = torch.from_numpy(w["masked_tensor"])[None].to(device)
            close_0 = np.array([w["close_0"]])

            price_t, _ = cvae.sample(masked_t, k=args.num_samples)  # (K,1,3,4)
            price = shrink_components(price_t.cpu().numpy(), sell_bound)
            body_ret_draws.append(price[:, 0, 0, 1])  # bar 0's body_ret, all K draws

            exit_prices = exit_price_from_components(price, close_0)  # (K,1)
            take_profit = np.percentile(exit_prices[:, 0], args.cvae_sell_quantile)
            predicted_returns.append(take_profit / close_0[0] - 1.0)

            ctx_end = start_idx + MAX_CONTEXT
            trend.append(float(feat[ctx_end - args.trend_lookback : ctx_end, BODY_RET_COL].sum()))

    body_ret_draws = np.array(body_ret_draws)
    predicted_returns = np.array(predicted_returns)
    trend = np.array(trend)
    real_train_std = float(feat[train_lo:train_hi, BODY_RET_COL].std())

    variance = variance_ratio_and_correlation(body_ret_draws, trend)
    returns = predicted_return_stats(predicted_returns)

    print(f"=== body_ret variance/correlation, N={args.num_windows} windows, k={args.num_samples} ===")
    print(f"  across-window std: {variance['across_window_std']:.6f}")
    print(f"  within-window std: {variance['within_window_std']:.6f}")
    print(f"  ratio (>1 = context beats sampling noise): {variance['ratio']:.3f}")
    print(f"  correlation with {args.trend_lookback}-bar trend: {variance['correlation_with_trend']:+.4f}")
    print(f"  mean: {variance['mean']:+.6f}  (real train std: {real_train_std:.6f})")

    print(f"\n=== predicted_return (take_profit vs close_0), N={args.num_windows} windows ===")
    print(f"  mean: {returns['mean']:+.4%}")
    print(f"  pct eligible (>0): {returns['pct_eligible']:.1%}")
    print(
        f"  percentiles 5/25/50/75/95: {returns['p5']:+.4%} / {returns['p25']:+.4%} / "
        f"{returns['p50']:+.4%} / {returns['p75']:+.4%} / {returns['p95']:+.4%}"
    )


if __name__ == "__main__":
    main()
