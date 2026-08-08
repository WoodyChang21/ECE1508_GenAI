"""Sample-plot rendering for generative-quality evaluation -- see evaluate_generative.py
and cvae_direction_collapse.md's "generative pivot" discussion. Deliberately does NOT
reuse evaluate.py's render_panel/draw_trade_lines/trade_table_text (all trade-decision-
specific -- buy/sell lines, ENTER/NO TRADE labels, realized-return text); only
draw_horizon_box is graphics-generic enough to share.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd

from src.data_pipeline import HORIZON, reconstruct_prices
from src.evaluate import draw_horizon_box

logger = logging.getLogger(__name__)


def render_candle_panel(ax, sub_df: pd.DataFrame, title: str) -> None:
    """Minimal candle panel -- candlesticks + a red box around the horizon region + a
    title. No buy/sell lines, no trade-decision text: this project's generative-quality
    plots aren't about a trade decision, just "what did/would get generated here."""
    mpf.plot(sub_df, type="candle", ax=ax, style="yahoo", volume=False)
    ax.set_title(title, fontsize=9)
    ax.tick_params(axis="x", labelrotation=30, labelsize=6)
    draw_horizon_box(ax, len(sub_df))


def build_regime_grid(
    df: pd.DataFrame,
    pairs: list[tuple[int, int]],
    price_samples: np.ndarray,
    realized_vols: np.ndarray,
    ctx_bars: int,
    out_path: Path,
    k_shown: int = 5,
    seed: int = 0,
) -> None:
    """Rows = low/mid/high realized-volatility tercile buckets, columns = [ground truth,
    k_shown generated candle panels] -- directly visualizes "does this context type
    produce visibly different generated candles than that context type" (rows) and "how
    diverse are this regime's own draws" (across a row). One representative window per
    bucket (closest to that bucket's own median realized_vol, for reproducibility rather
    than a fully random pick each run).

    price_samples: (K, N, 3, 4) -- reuses evaluate_generative.py's already-sampled draws
    rather than resampling. realized_vols: (N,), aligned with `pairs`/`price_samples`'
    second axis."""
    rng = np.random.default_rng(seed)
    terciles = np.percentile(realized_vols, [33.3, 66.7])
    buckets = {
        "low vol": realized_vols <= terciles[0],
        "mid vol": (realized_vols > terciles[0]) & (realized_vols < terciles[1]),
        "high vol": realized_vols >= terciles[1],
    }

    fig, axes = plt.subplots(len(buckets), k_shown + 1, figsize=(3 * (k_shown + 1), 3.2 * len(buckets)))
    for row, (bucket_name, mask) in enumerate(buckets.items()):
        idx_pool = np.where(mask)[0]
        if len(idx_pool) == 0:
            logger.warning("no windows in bucket=%s -- skipping row", bucket_name)
            continue
        median_vol = np.median(realized_vols[idx_pool])
        i = idx_pool[np.argmin(np.abs(realized_vols[idx_pool] - median_vol))]
        start_idx, this_ctx = pairs[i]

        ctx_tail = min(this_ctx, 20)
        hz_start = start_idx + this_ctx
        plot_rows = df.iloc[hz_start - ctx_tail : hz_start + HORIZON]
        true_df = plot_rows.set_index("datetime")[["open", "high", "low", "close", "volume"]]
        close_0 = float(df.iloc[hz_start - 1]["close"])

        render_candle_panel(axes[row, 0], true_df, f"{bucket_name}\nGround truth")
        for col in range(k_shown):
            draw_idx = int(rng.integers(price_samples.shape[0]))
            gen_ohlc = reconstruct_prices(price_samples[draw_idx, i], close_0)
            gen_df = true_df.copy()
            gen_df.loc[gen_df.index[-HORIZON:], ["open", "high", "low", "close"]] = gen_ohlc
            render_candle_panel(axes[row, col + 1], gen_df, f"draw {col}")

    fig.suptitle(
        f"Regime x k-samples (ctx_bars={ctx_bars}) -- rows: does context change what's "
        "generated? columns: how diverse are one context's own draws?"
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote regime grid to %s", out_path)


def build_diversity_fan(
    df: pd.DataFrame,
    start_idx: int,
    ctx_bars: int,
    price_samples_for_window: np.ndarray,
    out_path: Path,
) -> None:
    """One context window, k sampled close-price paths overlaid as semi-transparent lines
    over the horizon region, real path in a solid contrasting color -- shows within-
    context diversity at a glance without k separate candle panels (candlesticks don't
    overlay legibly the way a simple line does).

    price_samples_for_window: (K, 3, 4) -- one window's k sampled price-component draws."""
    ctx_tail = min(ctx_bars, 20)
    hz_start = start_idx + ctx_bars
    plot_rows = df.iloc[hz_start - ctx_tail : hz_start + HORIZON]
    true_df = plot_rows.set_index("datetime")[["open", "high", "low", "close", "volume"]]
    close_0 = float(df.iloc[hz_start - 1]["close"])

    fig, ax = plt.subplots(figsize=(8, 4))
    mpf.plot(true_df, type="candle", ax=ax, style="yahoo", volume=False)
    draw_horizon_box(ax, len(true_df))

    x0 = len(true_df) - HORIZON
    xs = range(x0 - 1, x0 - 1 + HORIZON + 1)
    for draw in price_samples_for_window:
        gen_ohlc = reconstruct_prices(draw, close_0)
        closes = np.concatenate([[close_0], gen_ohlc[:, 3]])
        ax.plot(xs, closes, color="tab:orange", alpha=0.35, linewidth=1.2, zorder=8)

    real_closes = np.concatenate([[close_0], true_df["close"].to_numpy()[-HORIZON:]])
    ax.plot(xs, real_closes, color="tab:blue", linewidth=2, zorder=9, label="real")
    ax.legend(fontsize=7)
    ax.set_title(f"Diversity fan -- ctx_bars={ctx_bars}, start_idx={start_idx}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote diversity fan to %s", out_path)
