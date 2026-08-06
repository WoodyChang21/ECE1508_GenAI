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

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mplfinance as mpf

from src.data_pipeline import (
    HORIZON,
    WindowDataset,
    WindowSampler,
    build_dataset,
    context_bucket,
    exit_price_from_components,
    extract_arrays,
    per_bar_close_return,
    reconstruct_prices,
    reconstruct_volume,
    shrink_components,
    to_patchtst_input,
    train_exit_return_bound,
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
    p.add_argument("--num-plot-samples", type=int, default=5)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--metrics-out", type=str, default="steven/outputs/metrics.json")
    p.add_argument("--plots-dir", type=str, default="steven/outputs/sample_plots")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument(
        "--sell-bound-percentile", type=float, default=99.0,
        help="Percentile (over train data) of |close_0-anchored log return| used to shrink predicted "
        "price components -- a tighter, empirically-calibrated bound on top of the model's own "
        "MAX_LOG_RETURN, applied to metrics/backtest/plots alike. See train_exit_return_bound.",
    )
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


def limit_order_exit(true_ohlc: np.ndarray, sell_limit: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Simulates placing a limit sell order at sell_limit (the model's own predicted exit
    price) at the same time as the close_0 buy, then walking the 3 REAL horizon bars in
    order: if any bar's real [low,high] range reaches sell_limit, the order fills there (a
    "hit"); if none of the 3 bars ever reach it, the position is force-closed at the 3rd
    bar's real close instead. true_ohlc: (N,3,4) real [open,high,low,close]. sell_limit:
    (N,). Returns (realized_sell_price (N,), hit (N,) bool)."""
    low = true_ohlc[:, :, 2]
    high = true_ohlc[:, :, 1]
    reachable = (low <= sell_limit[:, None]) & (sell_limit[:, None] <= high)  # (N,3)
    hit = reachable.any(axis=1)
    forced_close = true_ohlc[:, 2, 3]  # 3rd horizon bar's real close
    sell_price = np.where(hit, sell_limit, forced_close)
    return sell_price, hit


def sweep_thresholds(
    confidence: np.ndarray, trade_return: np.ndarray, thresholds: list[float], hit: np.ndarray | None = None
) -> list[dict]:
    n = len(confidence)
    rows = []
    for th in thresholds:
        mask = confidence >= th
        k = int(mask.sum())
        row = {"threshold": th, "n_trades": k, "selectivity": (k / n) if n else 0.0}
        if k > 0:
            rets = trade_return[mask]
            row["win_rate"] = float((rets > 0).mean())
            row["avg_return"] = float(rets.mean())
            row["total_return"] = float(rets.sum())
            row["limit_hit_rate"] = float(hit[mask].mean()) if hit is not None else None
        else:
            row["win_rate"] = None
            row["avg_return"] = None
            row["total_return"] = None
            row["limit_hit_rate"] = None
        rows.append(row)
    return rows


def run_backtest(true_y: np.ndarray, pt_y: np.ndarray, cvae_y: np.ndarray, close_0: np.ndarray) -> dict:
    """true_y/pt_y: (N,15). cvae_y: (K,N,15). close_0: (N,)."""
    n = len(true_y)
    note = (
        "long-only limit-order backtest: entry=close_0 (real). Each model's own predicted "
        "exit price (mean of its predicted 3 horizon bars' open+close; for CVAE, averaged "
        "across the k sampled draws) is placed as a limit sell order at the same time as "
        "entry. If any of the 3 REAL horizon bars' [low,high] range reaches that price, "
        "the order fills there; otherwise the position is force-closed at the 3rd REAL "
        "bar's close instead. confidence (per model) decides which windows to trade. "
        "total_return is summed across selected trades -- test windows are independently "
        "sampled and can overlap in calendar time, so this is not a sequential equity curve."
    )
    if n == 0:
        return {"note": note, "patchtst": [], "cvae": []}

    true_price = true_y[:, :12].reshape(n, 3, 4)
    pt_price = pt_y[:, :12].reshape(n, 3, 4)
    cvae_price_samples = cvae_y[..., :12].reshape(cvae_y.shape[0], n, 3, 4)

    true_ohlc = reconstruct_prices(true_price, close_0)  # (N,3,4) real [open,high,low,close]

    pt_sell_limit = exit_price_from_components(pt_price, close_0)  # (N,)
    cvae_sell_limit = exit_price_from_components(cvae_price_samples, close_0).mean(axis=0)  # (N,)

    pt_sell_price, pt_hit = limit_order_exit(true_ohlc, pt_sell_limit)
    cvae_sell_price, cvae_hit = limit_order_exit(true_ohlc, cvae_sell_limit)

    pt_trade_return = pt_sell_price / close_0 - 1.0
    cvae_trade_return = cvae_sell_price / close_0 - 1.0

    cvae_conf = cvae_confidence_scores(cvae_price_samples, close_0)
    pt_conf = patchtst_confidence_scores(pt_price, close_0)

    return {
        "note": note,
        "patchtst": sweep_thresholds(pt_conf, pt_trade_return, BACKTEST_THRESHOLDS, pt_hit),
        "cvae": sweep_thresholds(cvae_conf, cvae_trade_return, BACKTEST_THRESHOLDS, cvae_hit),
    }


def draw_horizon_box(ax, n_shown: int) -> None:
    """Red box around the last HORIZON candles -- the region being generated/predicted."""
    x0, x1 = n_shown - HORIZON - 0.5, n_shown - 0.5
    y0, y1 = ax.get_ylim()
    ax.add_patch(
        mpatches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="red", linewidth=1.8, zorder=6)
    )


def draw_trade_lines(
    ax, n_shown: int, buy_price: float, sell_targets: list[tuple[str, float, str]]
) -> None:
    """Buy = dashed line at close_0 (the last known candle's close), spanning the full
    panel. Each entry in sell_targets is (label, price, color): a limit-sell price drawn
    as a horizontal line starting at the left edge of the horizon box (the first of the
    3 target/generated candles) and extending to the right edge of the panel -- not a
    line all the way through, since the price only applies once prediction starts. If a
    sell price falls outside the panel's current y-range, the line is pinned just inside
    the top/bottom edge instead of letting the axis autoscale to it, with a text label
    giving the real price and noting it's out of range -- this keeps the panel's own
    price scale intact regardless of how extreme a prediction is."""
    y0, y1 = ax.get_ylim()
    x0 = n_shown - HORIZON - 0.5
    x1 = n_shown - 0.5
    margin = (y1 - y0) * 0.08

    ax.axhline(buy_price, color="tab:blue", linestyle="--", linewidth=1, zorder=6)
    ax.text(-0.6, buy_price, "buy", color="tab:blue", fontsize=7, va="bottom", ha="left", zorder=6)

    for label, price, color in sell_targets:
        if price > y1:
            line_y = y1 - margin
            ax.hlines(line_y, x0, x1, color=color, linewidth=1.4, zorder=7)
            ax.annotate(
                f"  {label}: {price:.2f} (above range)", xy=(x0, line_y), xytext=(0, -4),
                textcoords="offset points", color=color, fontsize=7, va="top", ha="left", zorder=7,
            )
        elif price < y0:
            line_y = y0 + margin
            ax.hlines(line_y, x0, x1, color=color, linewidth=1.4, zorder=7)
            ax.annotate(
                f"  {label}: {price:.2f} (below range)", xy=(x0, line_y), xytext=(0, 4),
                textcoords="offset points", color=color, fontsize=7, va="bottom", ha="left", zorder=7,
            )
        else:
            ax.hlines(price, x0, x1, color=color, linewidth=1.4, zorder=7)
            ax.text(x0, price, f"  {label}: {price:.2f}", color=color, fontsize=7, va="bottom", ha="left", zorder=7)

    ax.set_ylim(y0, y1)


def trade_table_text(
    ohlc: np.ndarray, buy_price: float, true_horizon_ohlc: np.ndarray | None = None
) -> tuple[str, float, tuple[float, bool] | None]:
    """ohlc: (3,4) [open,high,low,close] for this panel's 3 horizon bars (used for the
    per-candle numbers and to set the limit-order target price). true_horizon_ohlc, if
    given, is the REAL 3 horizon bars' [open,high,low,close] -- used to simulate whether
    the limit order set from `ohlc` would actually have filled against real price action
    (see limit_order_exit). Returns (annotation text, sell_limit, realized-or-None)."""
    sell_limit = float(ohlc[:, [0, 3]].mean())
    lines = [
        f"C1  O={ohlc[0,0]:.2f}  C={ohlc[0,3]:.2f}    "
        f"C2  O={ohlc[1,0]:.2f}  C={ohlc[1,3]:.2f}    "
        f"C3  O={ohlc[2,0]:.2f}  C={ohlc[2,3]:.2f}",
        f"Buy = {buy_price:.2f}      Limit sell target (avg of the 6 open/close) = {sell_limit:.2f}",
    ]
    realized = None
    if true_horizon_ohlc is not None:
        low, high = true_horizon_ohlc[:, 2], true_horizon_ohlc[:, 1]
        hit = bool(((low <= sell_limit) & (sell_limit <= high)).any())
        realized_price = sell_limit if hit else float(true_horizon_ohlc[2, 3])
        realized_return = (realized_price / buy_price - 1.0) * 100
        realized = (realized_price, hit)
        status = "HIT -> filled at target" if hit else "MISSED -> forced exit at real C3 close"
        lines.append(
            f"vs. real price action: {status}  |  realized sell = {realized_price:.2f}  |  "
            f"return = {realized_return:+.2f}%"
        )
    return "\n".join(lines), sell_limit, realized


def render_panel(
    fig,
    gs_chart,
    gs_text,
    sub_df: pd.DataFrame,
    buy_price: float,
    title: str,
    sell_targets: list[tuple[str, float, str]],
    true_horizon_ohlc: np.ndarray | None = None,
) -> tuple[float, tuple[float, bool] | None]:
    """Returns (sell_limit, realized) from trade_table_text -- realized is
    (realized_price, hit) when true_horizon_ohlc was given, else None -- so callers that
    need the simulated outcome (e.g. to log it alongside the plot) don't have to
    recompute it."""
    ax = fig.add_subplot(gs_chart)
    mpf.plot(sub_df, type="candle", ax=ax, style="yahoo", volume=False)
    ax.set_title(title, fontsize=10)
    ax.tick_params(axis="x", labelrotation=30, labelsize=7)
    draw_horizon_box(ax, len(sub_df))

    ohlc = sub_df.iloc[-HORIZON:][["open", "high", "low", "close"]].to_numpy()
    text, sell_limit, realized = trade_table_text(ohlc, buy_price, true_horizon_ohlc)
    draw_trade_lines(ax, len(sub_df), buy_price, sell_targets)

    ax_txt = fig.add_subplot(gs_text)
    ax_txt.axis("off")
    ax_txt.text(0.0, 1.0, text, transform=ax_txt.transAxes, ha="left", va="top", fontsize=7.5, family="monospace")
    return sell_limit, realized


def make_plots(df, feat, opens, closes, patchtst, cvae, test_sampler, args, device, sell_bound):
    """Draws num_plot_samples fresh random (start_idx, ctx_bars) windows -- NOT the
    fixed/seeded eval test_pairs -- so each run's sample plots differ. Each figure has
    3 candlestick panels: ground truth (left), PatchTST's generated horizon candles
    (top right), CVAE's generated horizon candles from one sampled draw (bottom right).
    Each model's predicted price components are shrunk toward sell_bound (see
    train_exit_return_bound/shrink_components in main) before reconstruction, so the
    plotted candles and lines match the same recalibrated prediction used in metrics and
    the backtest. Each panel gets a red box around the generated/predicted 3 candles, a
    dashed buy-price line (close of the last known candle), and one limit-sell line per
    model (its own predicted exit price) starting at the left edge of that box and
    running to the panel's right edge -- the ground truth panel shows both models' lines
    together for comparison, while the PatchTST/CVAE panels each show only their own. A
    line pinned outside the panel's y-range is drawn just inside the edge instead,
    annotated with its real price, so one wild prediction can't blow out the axis scale.
    The PatchTST and CVAE panels additionally report in the text table whether that
    model's limit order would actually have filled against the REAL price action (see
    limit_order_exit). Also writes samples.json alongside the PNGs -- the same per-sample
    numbers (candles, buy/sell prices, hit/realized outcome, spread), structured for
    update_report.py to regenerate v1.md's Results tables without transcribing PNGs by
    hand."""
    from src.data_pipeline import CONTEXT_LENGTHS, build_window

    out_dir = Path(args.plots_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Clear last run's PNGs/samples.json first -- these are illustrative, regenerated
    # fresh every run, and update_report.py rewrites v1.md to reference only the current
    # batch, so a stale prior run's files would otherwise just accumulate unreferenced.
    for stale in out_dir.glob("sample*_start*_ctx*.png"):
        stale.unlink()
    (out_dir / "samples.json").unlink(missing_ok=True)

    # Fresh entropy each call -- intentionally independent of the fixed eval seed, since
    # these are illustrative samples, not part of the reproducible metrics.
    rng = np.random.default_rng()
    ohlc_cols = ["open", "high", "low", "close"]

    def spread(ohlc: np.ndarray) -> float:
        """max - min of the 3 bars' 6 open/close values -- how wide a candle path is."""
        return float(ohlc[:, [0, 3]].max() - ohlc[:, [0, 3]].min())

    records = []
    plotted = 0
    for i in range(args.num_plot_samples):
        ctx_bars = int(rng.choice(CONTEXT_LENGTHS))
        starts = test_sampler.valid_starts(ctx_bars)
        if len(starts) == 0:
            logger.warning("no valid starts for ctx_bars=%d, skipping sample %d", ctx_bars, i)
            continue
        start_idx = int(rng.choice(starts))

        w = build_window(feat, opens, closes, start_idx, ctx_bars)
        masked_tensor = torch.from_numpy(w["masked_tensor"]).unsqueeze(0).to(device)
        context_np, patch_pad_np = to_patchtst_input(w["masked_tensor"])
        context_t = torch.from_numpy(context_np).unsqueeze(0).to(device)
        patch_pad_t = torch.from_numpy(patch_pad_np).unsqueeze(0).to(device)

        with torch.no_grad():
            pt_price, _ = patchtst(context_t, patch_pad_t)
            cvae_price, _ = cvae.sample(masked_tensor, k=1)  # one generated draw, not the mean

        pt_components = shrink_components(pt_price[0].cpu().numpy(), sell_bound)  # (3,4)
        cvae_components = shrink_components(cvae_price[0, 0].cpu().numpy(), sell_bound)  # (3,4)
        pt_ohlc = reconstruct_prices(pt_components, w["close_0"])  # (3,4)
        cvae_ohlc = reconstruct_prices(cvae_components, w["close_0"])  # (3,4)
        buy_price = float(w["close_0"])
        pt_sell_limit = float(pt_ohlc[:, [0, 3]].mean())
        cvae_sell_limit = float(cvae_ohlc[:, [0, 3]].mean())

        ctx_tail = min(ctx_bars, 20)
        hz_start = start_idx + ctx_bars
        plot_rows = df.iloc[hz_start - ctx_tail : hz_start + 3]
        true_df = plot_rows.set_index("datetime")[["open", "high", "low", "close", "volume"]]

        pt_df = true_df.copy()
        pt_df.loc[pt_df.index[-3:], ohlc_cols] = pt_ohlc
        cvae_df = true_df.copy()
        cvae_df.loc[cvae_df.index[-3:], ohlc_cols] = cvae_ohlc

        true_horizon_ohlc = true_df.iloc[-HORIZON:][["open", "high", "low", "close"]].to_numpy()

        fig = plt.figure(figsize=(16, 8))
        outer = fig.add_gridspec(6, 2, width_ratios=[1, 1], hspace=0.7, wspace=0.25)

        render_panel(
            fig, outer[0:4, 0], outer[4:6, 0], true_df, buy_price, "Ground truth",
            sell_targets=[("PatchTST", pt_sell_limit, "tab:orange"), ("CVAE", cvae_sell_limit, "tab:green")],
        )
        _, pt_realized = render_panel(
            fig, outer[0:2, 1], outer[2, 1], pt_df, buy_price, "PatchTST generated",
            sell_targets=[("PatchTST", pt_sell_limit, "tab:orange")], true_horizon_ohlc=true_horizon_ohlc,
        )
        _, cvae_realized = render_panel(
            fig, outer[3:5, 1], outer[5, 1], cvae_df, buy_price, "CVAE generated (1 draw)",
            sell_targets=[("CVAE", cvae_sell_limit, "tab:green")], true_horizon_ohlc=true_horizon_ohlc,
        )

        fig.suptitle(f"{context_bucket(ctx_bars)} ctx (ctx_bars={ctx_bars}, start_idx={start_idx})")
        out_path = out_dir / f"sample{i}_start{start_idx}_ctx{ctx_bars}.png"
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        plotted += 1

        pt_realized_price, pt_hit = pt_realized
        cvae_realized_price, cvae_hit = cvae_realized
        records.append({
            "index": i,
            "file": out_path.name,
            "bucket": context_bucket(ctx_bars),
            "ctx_bars": ctx_bars,
            "start_idx": start_idx,
            "buy_price": buy_price,
            "ground_truth": {"candles": true_horizon_ohlc.tolist(), "spread": spread(true_horizon_ohlc)},
            "patchtst": {
                "candles": pt_ohlc.tolist(),
                "spread": spread(pt_ohlc),
                "sell_limit": pt_sell_limit,
                "hit": pt_hit,
                "realized_price": pt_realized_price,
                "realized_return_pct": (pt_realized_price / buy_price - 1.0) * 100,
            },
            "cvae": {
                "candles": cvae_ohlc.tolist(),
                "spread": spread(cvae_ohlc),
                "sell_limit": cvae_sell_limit,
                "hit": cvae_hit,
                "realized_price": cvae_realized_price,
                "realized_return_pct": (cvae_realized_price / buy_price - 1.0) * 100,
            },
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

    pt_ckpt = torch.load(args.patchtst_checkpoint, map_location=device, weights_only=False)
    cvae_ckpt = torch.load(args.cvae_checkpoint, map_location=device, weights_only=False)

    pt_cfg = pt_ckpt["config"]
    cvae_cfg = cvae_ckpt["config"]
    assert pt_cfg["data_path"] == cvae_cfg["data_path"], "both checkpoints must share the same data_path"

    df, bounds, stats = build_dataset(pt_cfg["data_path"])
    feat, opens, closes = extract_arrays(df)

    sell_bound = train_exit_return_bound(opens, closes, bounds["train"], percentile=args.sell_bound_percentile)
    logger.info(
        "sell-price shrink bound: p%.1f of |anchored log return| over train = %.4f (vs. model's own MAX_LOG_RETURN)",
        args.sell_bound_percentile, sell_bound,
    )

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

    n, k = len(pt_y), cvae_y.shape[0]
    pt_y[:, :12] = shrink_components(pt_y[:, :12].reshape(n, 3, 4), sell_bound).reshape(n, 12)
    cvae_y[..., :12] = shrink_components(cvae_y[..., :12].reshape(k, n, 3, 4), sell_bound).reshape(k, n, 12)

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

    make_plots(df, feat, opens, closes, patchtst, cvae, test_sampler, args, device, sell_bound)


if __name__ == "__main__":
    main()
