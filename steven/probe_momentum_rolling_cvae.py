"""Two momentum-enriched CVAE variants (EMA9/EMA21 + RSI14, see src/momentum_pipeline.py),
differing only in candle frequency and a training-window sampling change -- see
cvae_direction_collapse.md's "momentum enrichment" discussion for the full context.

- Hourly: same 70-bar (10-trading-day) context the rest of the project already evaluates at.
- Daily: also a 10-trading-day context (10 daily bars, NOT 70 -- matched calendar lookback
  for a fair frequency comparison, not the hourly model's bar count), on SPY's full 1993-2025
  daily history pulled via yfinance (src/collect_daily_yfinance.py), not the ~2010-2025 slice
  the earlier disposable daily-bars probe got by resampling the hourly parquet --
  daily_signal_probe.md's own statistical-power argument for testing daily bars at all only
  holds up with the real ~8,100-bar history behind it, not a ~3,900-bar fraction of it.

Sampling: both use the new RollingWindowSampler (data_pipeline.py) instead of WindowSampler's
random multi-length draw -- ONE fixed context length, sliding by 1 bar, covering every valid
window exactly once per real epoch. Two motivations, neither of which is "this fixes the
direction collapse" (random vs. rolling window *selection* doesn't change the underlying
context->direction relationship being learned, just which subset of it gets gradient steps):
1. Matches training's context length to WALK_FORWARD_CTX_BARS exactly -- today's random
   sampler trains across 9 lengths (14-70 bars) but evaluation only ever tests at 70.
2. Removes an oversampling risk the daily-bars probe had and didn't fix: its
   train_windows_per_epoch=20000 was a number picked for hourly's ~21k-row pool, and
   overshot the daily probe's ~3k-row pool by ~7x (each row seen ~199x per
   daily_signal_probe.md). Rolling makes "one epoch" mean "one real pass," at whatever pool
   size actually exists -- no config number to accidentally mismatch against pool size.

Run separately per frequency, never both in the same process: momentum_pipeline's dataset
builder and this script's own context/date patches mutate data_pipeline's module-level
globals as a side effect (see momentum_pipeline.py's docstring) -- safe within one
process/frequency, not safe to mix.

Usage:
    python steven/probe_momentum_rolling_cvae.py --frequency hourly
    python steven/probe_momentum_rolling_cvae.py --frequency daily
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

import src.data_pipeline as dp
import src.evaluate as ev
import src.momentum_pipeline as mp
from src.diagnose_cvae_direction import predicted_return_stats, variance_ratio_and_correlation
from src.models.cvae_inpainting import CVAEInpainting
from src.train_cvae import collate, kl_beta_schedule, resolve_device, run_epoch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent  # steven/
OUTER = HERE.parent  # ECE1508_GenAI/
CVAE_CONFIG = HERE / "configs" / "cvae_momentum.yaml"  # same model/loss recipe reused for both

CTX_BARS_DAILY = 10  # 10 trading days -- matched calendar lookback vs. the hourly model, not
                      # bar count (see module docstring)
CTX_BARS_HOURLY = 70  # 10 trading days x 7 hourly bars/day -- already WALK_FORWARD_CTX_BARS today
DAILY_PARQUET = HERE / "data" / "spy_daily_yfinance.parquet"
VIX_PARQUET = HERE / "data" / "vix_daily_yfinance.parquet"

# Same daily split boundaries as the earlier disposable daily-bars probe (probe_daily_cvae.py)
# for continuity with everything already analyzed against them -- only the underlying data
# grows (full 1993-2025 yfinance history vs. that probe's ~2010-2025 hourly resample), not the
# split dates themselves.
TRAIN_END_DAILY = "2021-12-31"
VAL_START_DAILY = "2022-01-01"
VAL_END_DAILY = "2022-12-31"
TEST_START_DAILY = "2023-01-01"
TEST_END_DAILY = "2025-05-30"

NUM_EVAL_WINDOWS = 300
SEED = 42


MIN_RETURN_THRESHOLD_HOURLY = 0.0002  # unchanged -- tuned against CVAE's own hourly predicted-
                                       # edge scale (median ~0.04% at the time), see evaluate.py
MIN_RETURN_THRESHOLD_DAILY_FRACTION = 0.01  # of sell_bound -- see main()'s comment: same
                                             # "small fraction of a real, data-grounded move
                                             # size" principle as the daily stop-loss fix,
                                             # rather than reusing the hourly constant (which
                                             # was tuned against a self-referential, and now
                                             # known-unstable, predicted-edge scale) or
                                             # re-deriving from THIS run's own noisy prediction
                                             # distribution (see cvae_direction_collapse.md's
                                             # training-seed-variance section).


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--frequency", choices=["hourly", "daily"], required=True)
    return p.parse_args()


def configure_for_frequency(frequency: str) -> tuple[str, int]:
    """Applies this run's context-length + (for daily) date-boundary patches to
    data_pipeline's module globals, BEFORE momentum_pipeline.build_momentum_dataset is
    called (which applies its own channel-layout patches on top). Returns (data_path,
    ctx_bars)."""
    if frequency == "hourly":
        return "steven/data/spy_ohlcv_1h.parquet", CTX_BARS_HOURLY

    dp.MAX_CONTEXT = CTX_BARS_DAILY
    dp.TOTAL_LEN = CTX_BARS_DAILY + dp.HORIZON
    dp.TRAIN_END = TRAIN_END_DAILY
    dp.VAL_START = VAL_START_DAILY
    dp.VAL_END = VAL_END_DAILY
    dp.TEST_START = TEST_START_DAILY
    dp.TEST_END = TEST_END_DAILY
    return str(DAILY_PARQUET.relative_to(OUTER)), CTX_BARS_DAILY


def train_rolling_cvae(feat, opens, closes, bounds, ctx_bars: int, cfg: dict, device: torch.device) -> CVAEInpainting:
    """Same recipe/reuse pattern as probe_momentum_cvae.py's train_momentum_cvae, but with
    RollingWindowSampler (computed once, since it's deterministic -- no per-epoch redraw
    needed) in place of WindowSampler's random per-epoch draw."""
    torch.manual_seed(cfg["seed"])

    train_lo, train_hi = bounds["train"]
    if cfg["loss"]["use_price_scale"]:
        price_scale = torch.tensor(
            feat[train_lo:train_hi, :4].std(axis=0), dtype=torch.float32, device=device
        )
        logger.info("price_scale (open_ret, body_ret, upper_wick, lower_wick): %s", price_scale.tolist())
    else:
        price_scale = None

    train_pairs = dp.RollingWindowSampler(*bounds["train"], ctx_bars).pairs()
    val_pairs = dp.RollingWindowSampler(*bounds["val"], ctx_bars).pairs()
    logger.info("rolling window pool: %d train / %d val windows per epoch (ctx_bars=%d)", len(train_pairs), len(val_pairs), ctx_bars)

    train_ds = dp.WindowDataset(feat, opens, closes, train_pairs)
    val_ds = dp.WindowDataset(feat, opens, closes, val_pairs)
    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False, collate_fn=collate)

    model = CVAEInpainting(**cfg["model"], in_channels=dp.N_CHANNELS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])

    best_val_recon = float("inf")
    best_state = None
    max_epochs = cfg["train"]["max_epochs"]

    for epoch in range(max_epochs):
        beta = kl_beta_schedule(epoch, max_epochs, cfg["loss"]["kl_cycles"], cfg["loss"]["kl_ramp_fraction"])
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
    logger.info("done training. best val_recon=%.5f", best_val_recon)
    return model


def run_direction_diagnostic(
    cvae, feat, opens, closes, bounds, ctx_bars: int, sell_bound, device: torch.device, eval_seed: int = SEED,
) -> dict:
    body_ret_col = dp.FEATURE_COLS.index("body_ret")
    test_lo, test_hi = bounds["test"]
    rng = np.random.default_rng(eval_seed)
    last_valid_start = test_hi - ctx_bars - dp.HORIZON
    n = min(NUM_EVAL_WINDOWS, last_valid_start - test_lo)
    starts = rng.choice(np.arange(test_lo, last_valid_start), size=n, replace=False)

    body_ret_draws, predicted_returns, trend = [], [], []
    trend_lookback = min(10, ctx_bars)
    with torch.no_grad():
        for start_idx in starts:
            w = dp.build_window(feat, opens, closes, int(start_idx), ctx_bars)
            masked_t = torch.from_numpy(w["masked_tensor"])[None].to(device)
            close_0 = np.array([w["close_0"]])

            price_t, _ = cvae.sample(masked_t, k=5)
            price = ev.shrink_components(price_t.cpu().numpy(), sell_bound)
            body_ret_draws.append(price[:, 0, 0, 1])

            exit_prices = ev.exit_price_from_components(price, close_0)
            take_profit = np.percentile(exit_prices[:, 0], 70.0)
            predicted_returns.append(take_profit / close_0[0] - 1.0)

            ctx_end = start_idx + ctx_bars
            trend.append(float(feat[ctx_end - trend_lookback : ctx_end, body_ret_col].sum()))

    variance = variance_ratio_and_correlation(np.array(body_ret_draws), np.array(trend))
    returns = predicted_return_stats(np.array(predicted_returns))

    print(f"\n=== body_ret variance/correlation, N={n} windows, k=5 ===")
    print(f"  ratio (>1 = context beats sampling noise): {variance['ratio']:.3f}")
    print(f"  correlation with {trend_lookback}-bar trend: {variance['correlation_with_trend']:+.4f}")
    print(f"\n=== predicted_return (take_profit vs close_0), N={n} windows ===")
    print(f"  mean: {returns['mean']:+.4%}  pct eligible (>0): {returns['pct_eligible']:.1%}")
    print(
        f"  percentiles 5/25/50/75/95: {returns['p5']:+.4%} / {returns['p25']:+.4%} / "
        f"{returns['p50']:+.4%} / {returns['p75']:+.4%} / {returns['p95']:+.4%}"
    )
    return {"variance": variance, "returns": returns, "n": n}


def main(frequency: str) -> dict:
    data_path, ctx_bars = configure_for_frequency(frequency)

    cfg = yaml.safe_load(CVAE_CONFIG.read_text())
    cfg["data_path"] = data_path
    device = resolve_device(cfg["device"])
    logger.info("frequency=%s  ctx_bars=%d  device=%s  data_path=%s", frequency, ctx_bars, device, data_path)

    df, bounds, stats, momentum_stats = mp.build_momentum_dataset(cfg["data_path"], VIX_PARQUET)
    logger.info(
        "momentum stats (train-set): ema_cross mean=%.5f std=%.5f, trend_position mean=%.5f std=%.5f, "
        "rsi mean=%.2f std=%.2f, vix mean=%.2f std=%.2f",
        momentum_stats.ema_cross_mean, momentum_stats.ema_cross_std,
        momentum_stats.trend_position_mean, momentum_stats.trend_position_std,
        momentum_stats.rsi_mean, momentum_stats.rsi_std,
        momentum_stats.vix_mean, momentum_stats.vix_std,
    )
    feat, opens, closes = dp.extract_arrays(df)
    row_counts = {k: bounds[k][1] - bounds[k][0] for k in ("train", "val", "test")}
    logger.info("train/val/test row counts: %d / %d / %d", row_counts["train"], row_counts["val"], row_counts["test"])

    sell_bound = dp.train_exit_return_bound(opens, closes, bounds["train"], percentile=99.0)

    cvae = train_rolling_cvae(feat, opens, closes, bounds, ctx_bars, cfg, device)
    ckpt_path = HERE / "outputs" / f"cvae_checkpoint_momentum_rolling_{frequency}.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": cvae.state_dict(), "config": cfg}, ckpt_path)

    diagnostic = run_direction_diagnostic(cvae, feat, opens, closes, bounds, ctx_bars, sell_bound, device)

    torch.manual_seed(cfg["seed"])
    ev.WALK_FORWARD_CTX_BARS = ctx_bars
    test_lo, test_hi = bounds["test"]
    wf_entry_idx = test_lo + ctx_bars - 1
    cvae_predict = ev.make_cvae_predict_fn(cvae, device, sell_bound, num_samples=5, cvae_sell_quantile=70.0)

    # Both thresholds below are calibrated PER FREQUENCY, not shared -- see
    # MIN_RETURN_THRESHOLD_HOURLY/_DAILY_FRACTION and module note.
    # stop_loss: NOT the shared 0.02 evaluate.py otherwise defaults to -- that was calibrated
    # against HOURLY 3-bar moves ("just outside the training data's own empirical p99
    # |anchored log return|, ~1.9%", per evaluate.py's module note). Daily 3-bar (3-day) moves
    # are naturally much bigger; reusing 0.02 there tightens the stop far past what daily
    # volatility actually looks like, converting ordinary daily-scale noise into forced
    # stop-outs. Reuse `sell_bound` itself -- already the correct per-frequency p99
    # anchored-return bound (computed above from THIS run's own opens/closes), the same
    # quantity already calibrating the take-profit side. Confirmed empirically: on the daily
    # checkpoint this fixed, 0.02 produced -4.90% total_return; sell_bound (0.0515 for that
    # run) produced +1.80% on the identical trades otherwise -- see
    # cvae_direction_collapse.md.
    # min_return_threshold: the hourly constant (0.0002) was tuned against CVAE's own hourly
    # predicted-edge scale at the time -- a self-referential quantity we've since shown is
    # unstable run-to-run (training-seed variance dominates it, see
    # cvae_direction_collapse.md's robustness-check section), so re-deriving an analogous
    # "daily predicted-edge scale" from THIS run's own diagnostic would just be re-measuring
    # noise. Deriving it from sell_bound instead keeps it anchored to something stable and
    # real (measured daily market volatility) rather than the model's own noisy output.
    if frequency == "hourly":
        stop_loss_pct = 0.02
        min_return_threshold = MIN_RETURN_THRESHOLD_HOURLY
    else:
        stop_loss_pct = sell_bound
        min_return_threshold = MIN_RETURN_THRESHOLD_DAILY_FRACTION * sell_bound
    logger.info("stop_loss_pct=%.4f  min_return_threshold=%.5f", stop_loss_pct, min_return_threshold)

    cvae_wf = ev.run_walk_forward(
        cvae_predict, feat, opens, closes, test_lo, test_hi,
        min_return_threshold=min_return_threshold, stop_loss_pct=stop_loss_pct,
    )
    bh = ev.buy_and_hold_benchmark(df, wf_entry_idx, test_hi - 1)
    naive = ev.naive_periodic_benchmark(df, closes, wf_entry_idx, test_hi)
    wf_stats = ev.walk_forward_stats(df, cvae_wf)

    return {
        "frequency": frequency, "ctx_bars": ctx_bars, "row_counts": row_counts,
        "sell_bound": sell_bound, "stop_loss_pct": stop_loss_pct, "min_return_threshold": min_return_threshold,
        "momentum_stats": momentum_stats, "diagnostic": diagnostic,
        "checkpoint_path": str(ckpt_path),
        "wf_stats": wf_stats, "buy_and_hold": bh, "naive_periodic": naive,
        "test_start": TEST_START_DAILY if frequency == "daily" else None,
        "test_end": TEST_END_DAILY if frequency == "daily" else None,
    }


def format_report(results: dict) -> str:
    """Pure string-building (no printing), same pattern as probe_daily_cvae.py's
    format_report -- callers either print() this directly or render it as real Markdown
    (IPython.display.Markdown in a notebook cell)."""
    wf, bh, naive = results["wf_stats"], results["buy_and_hold"], results["naive_periodic"]
    diag = results["diagnostic"]
    win_rate_str = f"{wf['win_rate']:.1%}" if wf["win_rate"] is not None else "n/a"
    take_profit_rate_str = f"{wf['take_profit_rate']:.1%}" if wf["take_profit_rate"] is not None else "n/a"
    avg_return_str = f"{wf['avg_return']:+.4%}" if wf["avg_return"] is not None else "n/a"

    lines = [
        f"### {results['frequency']} CVAE, ctx_bars={results['ctx_bars']}"
        + (f", test {results['test_start']} to {results['test_end']}" if results["test_start"] else ""),
        "",
        f"- stop_loss_pct={results['stop_loss_pct']:.4f}, min_return_threshold={results['min_return_threshold']:.5f} "
        f"(sell_bound={results['sell_bound']:.4f})",
        f"- direction diagnostic (N={diag['n']} test windows): variance ratio={diag['variance']['ratio']:.3f}, "
        f"correlation with trend={diag['variance']['correlation_with_trend']:+.4f}",
        "",
        "| | CVAE | buy&hold | naive_periodic |",
        "|---|---|---|---|",
        f"| n_trades / n_decisions | {wf['n_trades']}/{wf['n_decisions']} | -- | {naive['n_trades']} |",
        f"| total_return | {wf['total_return']:+.2%} | {bh['total_return']:+.2%} | {naive['total_return']:+.2%} |",
        f"| win_rate | {win_rate_str} | -- | "
        + (f"{naive['win_rate']:.1%}" if naive["win_rate"] is not None else "n/a") + " |",
        f"| take_profit_rate | {take_profit_rate_str} | -- | -- |",
        f"| avg_return per trade | {avg_return_str} | -- | "
        + (f"{naive['avg_return']:+.4%}" if naive["avg_return"] is not None else "n/a") + " |",
        "",
        "**outcome breakdown (fraction of all decisions):**",
        "",
        "| " + " | ".join(wf["outcome_breakdown"].keys()) + " |",
        "|" + "---|" * len(wf["outcome_breakdown"]),
        "| " + " | ".join(f"{v:.1%}" for v in wf["outcome_breakdown"].values()) + " |",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    args = parse_args()
    results = main(args.frequency)
    print(format_report(results))
