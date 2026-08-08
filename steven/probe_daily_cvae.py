"""Tests whether CVAE's candle generation carries a real signal on DAILY SPY bars, sweeping
context length (5/10/15/20 trading days) with the SAME model/loss recipe already in
configs/cvae.yaml -- a single-variable test (only the data/context changes) of the hypothesis
in daily_signal_probe.md: is CVAE's direction-collapse a modeling problem, or a
sampling-frequency problem?

Data: resamples the existing steven/data/spy_ohlcv_1h.parquet (2010-2025, already on disk) to
daily OHLCV -- no new data collection. Model: CVAE only (see the module note on PatchTST
below). Reuses data_pipeline.py's and evaluate.py's actual functions/classes UNCHANGED by
monkey-patching their module-level constants (MAX_CONTEXT, CONTEXT_LENGTHS, TOTAL_LEN,
TRAIN_END/VAL_START/VAL_END/TEST_START/TEST_END, WALK_FORWARD_CTX_BARS) instead of duplicating
or refactoring the hourly pipeline. This is safe here because every function/class touched
(WindowSampler, build_window, chronological_bounds, run_walk_forward, walk_forward_stats) is
DEFINED inside data_pipeline.py/evaluate.py itself, so their free-variable lookups resolve
against the patched module globals at call time, same mechanism as any other global lookup.

This would NOT be safe for PatchTST: src/models/patchtst.py does
`from src.data_pipeline import ... N_PATCHES, PATCH_LEN` at import time, an INDEPENDENT copy
in patchtst.py's own namespace that patching data_pipeline afterward can't reach, and its
nn.Parameter shapes (pos_embed, head) are fixed to those values at model-construction time.
Testing PatchTST at a different context length would need real code changes (parametrized
constructor args), not a monkey-patch -- out of scope here, so this script is CVAE-only.

Usage:
    python steven/probe_daily_cvae.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

import src.data_pipeline as dp
import src.evaluate as ev
from src.models.cvae_inpainting import CVAEInpainting
from src.train_cvae import collate, kl_beta_schedule, resolve_device, run_epoch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent  # steven/
OUTER = HERE.parent  # ECE1508_GenAI/ -- data_path strings below are relative to this, matching
                      # configs/cvae.yaml's own "steven/data/..." convention (scripts run with
                      # cwd at the outer repo root, e.g. colab_train.ipynb's `!python steven/...`)
HOURLY_PARQUET = HERE / "data" / "spy_ohlcv_1h.parquet"
DAILY_PARQUET = HERE / "data" / "spy_ohlcv_1d_from_hourly.parquet"
CVAE_CONFIG = HERE / "configs" / "cvae.yaml"
DAILY_CHECKPOINT = HERE / "outputs" / "cvae_checkpoint_daily.pt"

CONTEXT_SWEEP = [5, 10, 15, 20]
MAX_CONTEXT_DAILY = 20  # top of the sweep -- build_window requires ctx_bars <= MAX_CONTEXT

# One more year of test data than the hourly project (2024-2025), per this experiment's
# request -- val shrinks to 2022 alone to make room; train unchanged through 2021.
TRAIN_END_DAILY = "2021-12-31"
VAL_START_DAILY = "2022-01-01"
VAL_END_DAILY = "2022-12-31"
TEST_START_DAILY = "2023-01-01"
TEST_END_DAILY = "2025-05-30"

SEED = 42


def resample_to_daily(hourly_path: Path, out_path: Path) -> Path:
    """Plain typical OHLCV aggregation per calendar day: open=first, high=max, low=min,
    close=last, volume=sum. No intraday-derived TA enrichment (that's a separate, documented,
    not-yet-built idea in daily_signal_probe.md)."""
    df = pd.read_parquet(hourly_path)
    date = df["datetime"].dt.normalize()
    daily = (
        df.groupby(date)
        .agg(open=("open", "first"), high=("high", "max"), low=("low", "min"),
             close=("close", "last"), volume=("volume", "sum"))
        .reset_index()
        .rename(columns={"datetime": "datetime"})
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(out_path, index=False)
    logger.info("resampled %d hourly bars -> %d daily bars, wrote %s", len(df), len(daily), out_path)
    return out_path


def patch_daily_constants() -> None:
    """Monkey-patches data_pipeline's module-level constants -- see module docstring for why
    this is safe for CVAE (and would NOT be for PatchTST)."""
    dp.MAX_CONTEXT = MAX_CONTEXT_DAILY
    dp.CONTEXT_LENGTHS = CONTEXT_SWEEP
    dp.TOTAL_LEN = MAX_CONTEXT_DAILY + dp.HORIZON
    dp.TRAIN_END = TRAIN_END_DAILY
    dp.VAL_START = VAL_START_DAILY
    dp.VAL_END = VAL_END_DAILY
    dp.TEST_START = TEST_START_DAILY
    dp.TEST_END = TEST_END_DAILY


def check_max_log_return_headroom(feat: np.ndarray, bounds: dict) -> None:
    """MAX_LOG_RETURN=0.15 was calibrated against hourly bars' own observed tail -- daily
    bars can plausibly have larger single-bar extremes (e.g. single-day crashes). Warn,
    don't silently trust, if the daily training data's actual tail is getting close to it."""
    lo, hi = bounds["train"]
    max_abs = float(np.abs(feat[lo:hi, :4]).max())
    logger.info(
        "max |open_ret/body_ret/wick| in daily train data: %.4f (MAX_LOG_RETURN cap: %.4f)",
        max_abs, dp.MAX_LOG_RETURN,
    )
    if max_abs > 0.8 * dp.MAX_LOG_RETURN:
        logger.warning(
            "daily training data's tail (%.4f) is within 20%% of MAX_LOG_RETURN (%.4f) -- "
            "this cap was calibrated against hourly bars and may be clipping real daily moves",
            max_abs, dp.MAX_LOG_RETURN,
        )


def train_daily_cvae(feat: np.ndarray, opens: np.ndarray, closes: np.ndarray, bounds: dict, cfg: dict, device: torch.device) -> CVAEInpainting:
    """Mirrors train_cvae.py's main() training loop (reusing its run_epoch/kl_beta_schedule
    directly) since main() itself isn't a reusable function -- everything else here is the
    same recipe as configs/cvae.yaml, just against the daily dataset built above."""
    torch.manual_seed(cfg["seed"])

    train_lo, train_hi = bounds["train"]
    if cfg["loss"]["use_price_scale"]:
        price_scale = torch.tensor(
            feat[train_lo:train_hi, :4].std(axis=0), dtype=torch.float32, device=device
        )
        logger.info("price_scale (daily, open_ret/body_ret/upper_wick/lower_wick): %s", price_scale.tolist())
    else:
        price_scale = None

    train_sampler = dp.WindowSampler(*bounds["train"])
    val_sampler = dp.WindowSampler(*bounds["val"])

    val_rng = np.random.default_rng(cfg["seed"])
    val_pairs = val_sampler.draw(cfg["train"]["windows_per_eval_set"], val_rng)
    val_ds = dp.WindowDataset(feat, opens, closes, val_pairs)
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False, collate_fn=collate)

    model = CVAEInpainting(**cfg["model"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])

    best_val_recon = float("inf")
    best_state = None
    max_epochs = cfg["train"]["max_epochs"]

    train_pool_size = train_hi - train_lo
    exposure_ratio = cfg["train"]["train_windows_per_epoch"] * max_epochs / train_pool_size
    logger.info(
        "daily train pool: %d rows -- at %d windows/epoch x %d epochs, each row seen ~%.0fx "
        "on average (vs. ~29x for the hourly baseline's ~21,000-row pool -- see "
        "daily_signal_probe.md's caveat on this)",
        train_pool_size, cfg["train"]["train_windows_per_epoch"], max_epochs, exposure_ratio,
    )

    for epoch in range(max_epochs):
        beta = kl_beta_schedule(epoch, max_epochs, cfg["loss"]["kl_cycles"], cfg["loss"]["kl_ramp_fraction"])
        train_rng = np.random.default_rng(cfg["seed"] + epoch)
        train_pairs = train_sampler.draw(cfg["train"]["train_windows_per_epoch"], train_rng)
        train_ds = dp.WindowDataset(feat, opens, closes, train_pairs)
        train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True, collate_fn=collate)

        train_metrics = run_epoch(model, train_loader, optimizer, cfg["loss"], beta, price_scale, device, train=True)
        val_metrics = run_epoch(model, val_loader, optimizer, cfg["loss"], beta, price_scale, device, train=False)

        logger.info(
            "epoch %d/%d  beta=%.2f  train_recon=%.5f (kl=%.4f dir=%.4f)  val_recon=%.5f (kl=%.4f dir=%.4f)",
            epoch + 1, max_epochs, beta,
            train_metrics["recon_loss"], train_metrics["kl_loss"], train_metrics["direction_loss"],
            val_metrics["recon_loss"], val_metrics["kl_loss"], val_metrics["direction_loss"],
        )
        if val_metrics["recon_loss"] < best_val_recon:
            best_val_recon = val_metrics["recon_loss"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    logger.info("done training daily CVAE. best val_recon=%.5f", best_val_recon)
    return model


def evaluate_at_context_length(cvae: CVAEInpainting, df: pd.DataFrame, feat, opens, closes, bounds, sell_bound, ctx_bars: int, device: torch.device) -> dict:
    """Re-patches evaluate.py's own WALK_FORWARD_CTX_BARS (computed once from MAX_CONTEXT at
    import time, so it doesn't auto-follow patch_daily_constants) and reruns the real
    walk-forward harness -- same run_walk_forward/walk_forward_stats/buy_and_hold_benchmark/
    naive_periodic_benchmark used for the hourly project, unchanged."""
    ev.WALK_FORWARD_CTX_BARS = ctx_bars
    test_lo, test_hi = bounds["test"]
    wf_entry_idx = test_lo + ctx_bars - 1

    cvae_predict = ev.make_cvae_predict_fn(cvae, device, sell_bound, num_samples=5, cvae_sell_quantile=70.0)
    cvae_wf = ev.run_walk_forward(
        cvae_predict, feat, opens, closes, test_lo, test_hi,
        min_return_threshold=0.0002, stop_loss_pct=0.02,
    )
    return {
        "ctx_bars": ctx_bars,
        "buy_and_hold": ev.buy_and_hold_benchmark(df, wf_entry_idx, test_hi - 1),
        "naive_periodic": ev.naive_periodic_benchmark(df, closes, wf_entry_idx, test_hi),
        "cvae": ev.walk_forward_stats(df, cvae_wf),
    }


def format_report(results: list[dict]) -> str:
    """Pure string-building (no printing) so callers can either print() this directly or
    render it as real Markdown (e.g. IPython.display.Markdown in a notebook cell)."""
    lines = [
        f"Test period: {TEST_START_DAILY} to {TEST_END_DAILY} "
        f"({(pd.Timestamp(TEST_END_DAILY) - pd.Timestamp(TEST_START_DAILY)).days / 365.25:.2f} years)",
        "",
        "| ctx (days) | CVAE n_trades/n_decisions | CVAE total_return | CVAE win_rate | "
        "buy&hold total_return | naive_periodic total_return |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        c, bh, naive = r["cvae"], r["buy_and_hold"], r["naive_periodic"]
        win_rate = f"{c['win_rate']:.1%}" if c["win_rate"] is not None else "n/a"
        lines.append(
            f"| {r['ctx_bars']} | {c['n_trades']}/{c['n_decisions']} | "
            f"{c['total_return']:+.2%} | {win_rate} | "
            f"{bh['total_return']:+.2%} | {naive['total_return']:+.2%} |"
        )
    return "\n".join(lines)


def main() -> None:
    resample_to_daily(HOURLY_PARQUET, DAILY_PARQUET)
    patch_daily_constants()

    cfg = yaml.safe_load(CVAE_CONFIG.read_text())
    cfg["data_path"] = str(DAILY_PARQUET.relative_to(OUTER))

    device = resolve_device(cfg["device"])
    logger.info("device: %s", device)

    df, bounds, stats = dp.build_dataset(cfg["data_path"])
    feat, opens, closes = dp.extract_arrays(df)
    check_max_log_return_headroom(feat, bounds)

    sell_bound = dp.train_exit_return_bound(opens, closes, bounds["train"], percentile=99.0)
    logger.info("daily sell-price shrink bound (p99 of |anchored log return| over train): %.4f", sell_bound)

    cvae = train_daily_cvae(feat, opens, closes, bounds, cfg, device)
    torch.save({"model_state": cvae.state_dict(), "config": cfg}, DAILY_CHECKPOINT)

    torch.manual_seed(cfg["seed"])
    results = [
        evaluate_at_context_length(cvae, df, feat, opens, closes, bounds, sell_bound, ctx, device)
        for ctx in CONTEXT_SWEEP
    ]
    return results


if __name__ == "__main__":
    print(format_report(main()))
