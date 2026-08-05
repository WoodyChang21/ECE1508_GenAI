"""Evaluate PatchTST and the CVAE on the same fixed test window set.

Usage:
    python steven/src/evaluate.py \\
        --patchtst-checkpoint steven/outputs/patchtst_checkpoint.pt \\
        --cvae-checkpoint steven/outputs/cvae_checkpoint.pt \\
        --n-test-windows 300 --num-plot-windows-per-bucket 2
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
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mplfinance as mpf

from src.data_pipeline import (
    WindowDataset,
    WindowSampler,
    build_dataset,
    context_bucket,
    exit_price_from_components,
    extract_arrays,
    per_bar_close_return,
    reconstruct_prices,
    reconstruct_volume,
    to_patchtst_input,
)
from src.models.cvae_inpainting import CVAEInpainting
from src.models.patchtst import PatchTST

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

BUCKETS = ["short", "medium", "long"]
BACKTEST_THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--patchtst-checkpoint", type=str, default="steven/outputs/patchtst_checkpoint.pt")
    p.add_argument("--cvae-checkpoint", type=str, default="steven/outputs/cvae_checkpoint.pt")
    p.add_argument("--n-test-windows", type=int, default=3000)
    p.add_argument("--num-samples", type=int, default=5)
    p.add_argument("--num-plot-windows-per-bucket", type=int, default=2)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--metrics-out", type=str, default="steven/outputs/metrics.json")
    p.add_argument("--plots-dir", type=str, default="steven/outputs/sample_plots")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--batch-size", type=int, default=256)
    # default 0: benchmarked in steven/configs/patchtst.yaml -- DataLoader multiprocessing
    # is slower here, not faster, since build_window() is cheap numpy per sample.
    p.add_argument("--num-workers", type=int, default=0)
    return p.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def collate(batch: list[dict]) -> dict:
    context, patch_pad = zip(*(to_patchtst_input(b["masked_tensor"].numpy()) for b in batch))
    return {
        "masked_tensor": torch.stack([b["masked_tensor"] for b in batch]),
        "context": torch.stack([torch.from_numpy(c) for c in context]),
        "patch_key_padding_mask": torch.stack([torch.from_numpy(m) for m in patch_pad]),
        "y": torch.stack([b["y"] for b in batch]),
        "close_0": torch.tensor([b["close_0"] for b in batch], dtype=torch.float64),
        "ctx_bars": torch.tensor([b["ctx_bars"] for b in batch], dtype=torch.int64),
    }


def mae_rmse(pred: np.ndarray, true: np.ndarray) -> tuple[float, float]:
    diff = pred - true
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff**2)))
    return mae, rmse


def directional_accuracy(pred_close_ret: np.ndarray, true_close_ret: np.ndarray) -> np.ndarray:
    """pred/true_close_ret: (N, 3). Returns (3,) accuracy per horizon step."""
    return (np.sign(pred_close_ret) == np.sign(true_close_ret)).mean(axis=0)


def collect_predictions(patchtst, cvae, loader, device, num_samples):
    """Runs both models over the fixed test loader, returns flat numpy arrays."""
    all_true_y, all_pt_y = [], []
    all_cvae_y = []  # (K, N, 15)
    all_close0, all_ctx = [], []

    for batch in loader:
        context = batch["context"].to(device)
        patch_pad = batch["patch_key_padding_mask"].to(device)
        masked_tensor = batch["masked_tensor"].to(device)
        y = batch["y"]

        with torch.no_grad():
            pt_price, pt_vol = patchtst(context, patch_pad)
            pt_y = torch.cat([pt_price.reshape(pt_price.shape[0], -1), pt_vol], dim=1)

            cvae_price, cvae_vol = cvae.sample(masked_tensor, k=num_samples)  # (K,B,3,4),(K,B,3)
            K, B = cvae_price.shape[0], cvae_price.shape[1]
            cvae_y = torch.cat(
                [cvae_price.reshape(K, B, -1), cvae_vol], dim=-1
            )  # (K, B, 15)

        all_true_y.append(y.numpy())
        all_pt_y.append(pt_y.cpu().numpy())
        all_cvae_y.append(cvae_y.cpu().numpy())
        all_close0.append(batch["close_0"].numpy())
        all_ctx.append(batch["ctx_bars"].numpy())

    true_y = np.concatenate(all_true_y, axis=0)
    pt_y = np.concatenate(all_pt_y, axis=0)
    cvae_y = np.concatenate(all_cvae_y, axis=1)  # (K, N, 15)
    close_0 = np.concatenate(all_close0, axis=0)
    ctx_bars = np.concatenate(all_ctx, axis=0)
    return true_y, pt_y, cvae_y, close_0, ctx_bars


def metrics_for_slice(true_y, pt_y, cvae_y, close_0, stats) -> dict:
    """true_y/pt_y: (N,15). cvae_y: (K,N,15). close_0: (N,). Returns a metrics dict."""
    n = len(true_y)
    if n == 0:
        return {"n_windows": 0}

    true_price = true_y[:, :12].reshape(n, 3, 4)
    true_vol_norm = true_y[:, 12:15]
    pt_price = pt_y[:, :12].reshape(n, 3, 4)
    pt_vol_norm = pt_y[:, 12:15]
    cvae_price_mean = cvae_y[..., :12].reshape(cvae_y.shape[0], n, 3, 4).mean(axis=0)
    cvae_vol_mean = cvae_y[..., 12:15].mean(axis=0)

    out = {"n_windows": n}

    # reparam-target MAE/RMSE (15-dim vector directly)
    out["patchtst_reparam_mae_rmse"] = mae_rmse(pt_y, true_y)
    out["cvae_reparam_mae_rmse"] = mae_rmse(cvae_y.mean(axis=0), true_y)

    # reconstructed OHLCV MAE/RMSE
    true_ohlc = reconstruct_prices(true_price, close_0)
    pt_ohlc = reconstruct_prices(pt_price, close_0)
    cvae_ohlc = reconstruct_prices(cvae_price_mean, close_0)
    out["patchtst_ohlc_mae_rmse"] = mae_rmse(pt_ohlc, true_ohlc)
    out["cvae_ohlc_mae_rmse"] = mae_rmse(cvae_ohlc, true_ohlc)

    true_vol = reconstruct_volume(true_vol_norm, stats)
    pt_vol = reconstruct_volume(pt_vol_norm, stats)
    cvae_vol = reconstruct_volume(cvae_vol_mean, stats)
    out["patchtst_volume_mae_rmse"] = mae_rmse(pt_vol, true_vol)
    out["cvae_volume_mae_rmse"] = mae_rmse(cvae_vol, true_vol)

    # directional accuracy on close_0-anchored close return, per horizon step
    true_close_ret = true_price[:, :, 0] + true_price[:, :, 1]  # open_ret + body_ret
    pt_close_ret = pt_price[:, :, 0] + pt_price[:, :, 1]
    cvae_close_ret = cvae_price_mean[:, :, 0] + cvae_price_mean[:, :, 1]
    out["patchtst_directional_accuracy"] = directional_accuracy(pt_close_ret, true_close_ret).tolist()
    out["cvae_directional_accuracy"] = directional_accuracy(cvae_close_ret, true_close_ret).tolist()

    # CVAE sample spread vs realized variance of true close price, per horizon step
    cvae_price_samples = cvae_y[..., :12].reshape(cvae_y.shape[0], n, 3, 4)
    cvae_close_price_samples = reconstruct_prices(cvae_price_samples, close_0)[..., 3]  # (K, N, 3)
    sample_var_per_window = cvae_close_price_samples.var(axis=0)  # (N, 3)
    avg_sample_var = sample_var_per_window.mean(axis=0)  # (3,)
    realized_var = true_ohlc[..., 3].var(axis=0)  # (3,) cross-window variance of true close
    out["cvae_avg_sample_variance"] = avg_sample_var.tolist()
    out["realized_close_variance"] = realized_var.tolist()
    out["cvae_spread_to_realized_ratio"] = (avg_sample_var / np.maximum(realized_var, 1e-12)).tolist()

    return out


# ---------------------------------------------------------------------------
# Long-only selective backtest
#
# Entry is always close_0 (real, known). Exit (for realized PnL) is the mean of the
# 3 horizon bars' real open+close prices -- a simple deterministic stand-in for "some
# reasonable fill within the window" rather than a specific exit-timing rule. Each
# model's own confidence score decides which windows are worth trading at all; the
# realized return used for PnL always comes from ground truth, never predictions.
# ---------------------------------------------------------------------------


def percentile_rank(values: np.ndarray) -> np.ndarray:
    """0..1 rank (0 = smallest). Used to put PatchTST's point-forecast magnitude on the
    same [0,1] threshold scale as CVAE's sample-agreement fraction."""
    n = len(values)
    if n <= 1:
        return np.zeros(n)
    order = np.argsort(values)
    ranks = np.empty(n)
    ranks[order] = np.arange(n)
    return ranks / (n - 1)


def cvae_confidence_scores(cvae_price_samples: np.ndarray, close_0: np.ndarray) -> np.ndarray:
    """cvae_price_samples: (K,N,3,4). Returns (N,): fraction of the K sampled paths
    whose predicted exit price comes out above close_0. ~1 = strong sample consensus on
    an upward exit, ~0.5 = samples disagree (chop), ~0 = consensus downtrend -- one
    score captures all three of the "don't trade this" cases at once."""
    if cvae_price_samples.shape[1] == 0:
        return np.empty(0)
    exit_prices = exit_price_from_components(cvae_price_samples, close_0)  # (K, N)
    sample_return = exit_prices / close_0[None, :] - 1.0
    return (sample_return > 0).mean(axis=0)


def patchtst_confidence_scores(pt_price: np.ndarray, close_0: np.ndarray) -> np.ndarray:
    """pt_price: (N,3,4). PatchTST has no sample distribution, so: (1) require all 3
    predicted bar-close-returns to agree on the up direction (else auto-dismissed as
    choppy/down -- confidence 0), (2) rank surviving windows by predicted exit-return
    magnitude and convert to a [0,1] percentile, matching CVAE's threshold scale."""
    n = len(pt_price)
    if n == 0:
        return np.empty(0)
    coherent_up = (per_bar_close_return(pt_price) > 0).all(axis=1)
    exit_price = exit_price_from_components(pt_price, close_0)
    predicted_exit_return = exit_price / close_0 - 1.0

    confidence = np.zeros(n)
    if coherent_up.any():
        confidence[coherent_up] = percentile_rank(predicted_exit_return[coherent_up])
    return confidence


def sweep_thresholds(
    confidence: np.ndarray, true_trade_return: np.ndarray, thresholds: list[float]
) -> list[dict]:
    n = len(confidence)
    rows = []
    for th in thresholds:
        mask = confidence >= th
        k = int(mask.sum())
        row = {"threshold": th, "n_trades": k, "selectivity": (k / n) if n else 0.0}
        if k > 0:
            rets = true_trade_return[mask]
            row["win_rate"] = float((rets > 0).mean())
            row["avg_return"] = float(rets.mean())
            row["total_return"] = float(rets.sum())
        else:
            row["win_rate"] = None
            row["avg_return"] = None
            row["total_return"] = None
        rows.append(row)
    return rows


def run_backtest(true_y: np.ndarray, pt_y: np.ndarray, cvae_y: np.ndarray, close_0: np.ndarray) -> dict:
    """true_y/pt_y: (N,15). cvae_y: (K,N,15). close_0: (N,)."""
    n = len(true_y)
    note = (
        "long-only: entry=close_0 (real); exit=mean of the 3 horizon bars' real "
        "open+close prices (ground truth). confidence (per model) decides which "
        "windows to trade. total_return is summed across selected trades -- test "
        "windows are independently sampled and can overlap in calendar time, so this "
        "is not a sequential equity curve."
    )
    if n == 0:
        return {"note": note, "patchtst": [], "cvae": []}

    true_price = true_y[:, :12].reshape(n, 3, 4)
    pt_price = pt_y[:, :12].reshape(n, 3, 4)
    cvae_price_samples = cvae_y[..., :12].reshape(cvae_y.shape[0], n, 3, 4)

    true_trade_return = exit_price_from_components(true_price, close_0) / close_0 - 1.0

    cvae_conf = cvae_confidence_scores(cvae_price_samples, close_0)
    pt_conf = patchtst_confidence_scores(pt_price, close_0)

    return {
        "note": note,
        "patchtst": sweep_thresholds(pt_conf, true_trade_return, BACKTEST_THRESHOLDS),
        "cvae": sweep_thresholds(cvae_conf, true_trade_return, BACKTEST_THRESHOLDS),
    }


def make_plots(df, feat, opens, closes, stats, patchtst, cvae, test_pairs, args, device):
    from src.data_pipeline import build_window

    out_dir = Path(args.plots_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_bucket = {b: [] for b in BUCKETS}
    for start_idx, ctx_bars in test_pairs:
        by_bucket[context_bucket(ctx_bars)].append((start_idx, ctx_bars))

    plotted = 0
    for bucket in BUCKETS:
        chosen = by_bucket[bucket][: args.num_plot_windows_per_bucket]
        for start_idx, ctx_bars in chosen:
            w = build_window(feat, opens, closes, start_idx, ctx_bars)
            masked_tensor = torch.from_numpy(w["masked_tensor"]).unsqueeze(0).to(device)
            context_np, patch_pad_np = to_patchtst_input(w["masked_tensor"])
            context_t = torch.from_numpy(context_np).unsqueeze(0).to(device)
            patch_pad_t = torch.from_numpy(patch_pad_np).unsqueeze(0).to(device)

            with torch.no_grad():
                pt_price, _ = patchtst(context_t, patch_pad_t)
                cvae_price, _ = cvae.sample(masked_tensor, k=args.num_samples)

            pt_ohlc = reconstruct_prices(pt_price[0].cpu().numpy(), w["close_0"])  # (3,4)
            cvae_ohlc = reconstruct_prices(cvae_price[:, 0].cpu().numpy(), w["close_0"])  # (K,3,4)

            ctx_tail = min(ctx_bars, 10)
            hz_start = start_idx + ctx_bars
            plot_rows = df.iloc[hz_start - ctx_tail : hz_start + 3]
            true_df = plot_rows.set_index("datetime")[["open", "high", "low", "close", "volume"]]

            horizon_idx = true_df.index[-3:]
            pt_close_line = pd.Series(np.nan, index=true_df.index)
            pt_close_line.loc[horizon_idx] = pt_ohlc[:, 3]

            addplots = [mpf.make_addplot(pt_close_line, type="line", color="blue", width=2, marker="o")]
            for k in range(cvae_ohlc.shape[0]):
                line = pd.Series(np.nan, index=true_df.index)
                line.loc[horizon_idx] = cvae_ohlc[k, :, 3]
                addplots.append(
                    mpf.make_addplot(line, type="line", color="orange", width=1, alpha=0.6)
                )

            title = f"{bucket} ctx (ctx_bars={ctx_bars}) - true vs PatchTST(blue) vs CVAE x{args.num_samples}(orange)"
            out_path = out_dir / f"{bucket}_start{start_idx}_ctx{ctx_bars}.png"
            mpf.plot(
                true_df,
                type="candle",
                addplot=addplots,
                style="yahoo",
                title=title,
                savefig=str(out_path),
                volume=False,
            )
            plotted += 1
    logger.info("wrote %d sample plots to %s", plotted, out_dir)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    logger.info("device: %s", device)

    pt_ckpt = torch.load(args.patchtst_checkpoint, map_location=device, weights_only=False)
    cvae_ckpt = torch.load(args.cvae_checkpoint, map_location=device, weights_only=False)

    pt_cfg = pt_ckpt["config"]
    cvae_cfg = cvae_ckpt["config"]
    assert pt_cfg["data_path"] == cvae_cfg["data_path"], "both checkpoints must share the same data_path"

    df, bounds, stats = build_dataset(pt_cfg["data_path"])
    feat, opens, closes = extract_arrays(df)

    test_sampler = WindowSampler(*bounds["test"])
    rng = np.random.default_rng(args.seed)
    test_pairs = test_sampler.draw(args.n_test_windows, rng)
    logger.info("evaluating on %d fixed test windows", len(test_pairs))

    test_ds = WindowDataset(feat, opens, closes, test_pairs)
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    patchtst = PatchTST(**pt_cfg["model"]).to(device)
    patchtst.load_state_dict(pt_ckpt["model_state"])
    patchtst.eval()

    cvae = CVAEInpainting(**cvae_cfg["model"]).to(device)
    cvae.load_state_dict(cvae_ckpt["model_state"])
    cvae.eval()

    true_y, pt_y, cvae_y, close_0, ctx_bars = collect_predictions(
        patchtst, cvae, test_loader, device, args.num_samples
    )

    results = {"overall": metrics_for_slice(true_y, pt_y, cvae_y, close_0, stats)}
    results["overall"]["backtest"] = run_backtest(true_y, pt_y, cvae_y, close_0)

    buckets = np.array([context_bucket(c) for c in ctx_bars])
    for bucket in BUCKETS:
        mask = buckets == bucket
        results[bucket] = metrics_for_slice(
            true_y[mask], pt_y[mask], cvae_y[:, mask], close_0[mask], stats
        )
        results[bucket]["backtest"] = run_backtest(
            true_y[mask], pt_y[mask], cvae_y[:, mask], close_0[mask]
        )

    Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.metrics_out, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("wrote metrics to %s", args.metrics_out)
    logger.info("overall: %s", json.dumps(results["overall"], indent=2))

    make_plots(df, feat, opens, closes, stats, patchtst, cvae, test_pairs, args, device)


if __name__ == "__main__":
    main()
