"""Is the daily momentum model's r=+0.202 correlation (EMA9/EMA21+RSI14, no VIX) a real,
reproducible signal, or noise that happens to land somewhere in a wide range depending on
what's fed in and how training goes? Motivated directly by cvae_direction_collapse.md's
"Adding VIX" section: adding one more feature swung the same measurement from +0.202 to
-0.104, which is itself evidence the number wasn't stable to begin with -- this script checks
how much of that swing is even attributable to training randomness alone, before blaming VIX
specifically.

Separates two distinct sources of variance that a single before/after comparison conflates:
1. TRAINING variance: retrains the identical daily config (include_vix=False, reproducing
   the pre-VIX +0.202 result) from N_SEEDS different seeds, each evaluated with the SAME
   fixed eval_seed -- isolates how much the correlation estimate moves just from training
   randomness (weight init, minibatch shuffling), holding the evaluation sample fixed.
2. EVALUATION-SAMPLING variance: takes ONE trained model (seed=42) and re-runs the direction
   diagnostic with N_EVAL_RESAMPLES different eval_seeds -- isolates how much the correlation
   estimate moves just from which 300 (of ~590 available, heavily overlapping) test windows
   get drawn, holding the model fixed. This is the "effective N is smaller than 300 suggests"
   concern flagged every time this correlation has been reported.

If (1) is small and results cluster near +0.202 across seeds, that's real evidence of a
reproducible signal. If (1) is large (results span, say, -0.2 to +0.3), the original +0.202
was likely just a favorable training-seed draw, and VIX's "regression" to -0.104 is
unremarkable -- within the same noise a totally different feature set couldn't be blamed for.

Usage:
    python steven/robustness_check_momentum.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

import probe_momentum_rolling_cvae as pr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

TRAIN_SEEDS = [42, 43, 44, 45, 46]
EVAL_RESEED_SEEDS = [123, 456, 789, 111, 222]  # applied only to the seed=42 model
FIXED_EVAL_SEED = 123  # used for every cross-seed training comparison, for a fair comparison


def train_and_eval(seed: int, eval_seed: int, data_path: str, ctx_bars: int, cfg: dict, device) -> dict:
    cfg = dict(cfg)
    cfg["seed"] = seed
    df, bounds, stats, momentum_stats = pr.mp.build_momentum_dataset(data_path, include_vix=False)
    feat, opens, closes = pr.dp.extract_arrays(df)
    sell_bound = pr.dp.train_exit_return_bound(opens, closes, bounds["train"], percentile=99.0)
    cvae = pr.train_rolling_cvae(feat, opens, closes, bounds, ctx_bars, cfg, device)
    result = pr.run_direction_diagnostic(cvae, feat, opens, closes, bounds, ctx_bars, sell_bound, device, eval_seed=eval_seed)
    return result


def main() -> None:
    data_path, ctx_bars = pr.configure_for_frequency("daily")
    cfg = yaml.safe_load(pr.CVAE_CONFIG.read_text())
    cfg["data_path"] = data_path
    device = pr.resolve_device(cfg["device"])
    logger.info("device=%s  ctx_bars=%d  include_vix=False (reproducing the pre-VIX config)", device, ctx_bars)

    logger.info("=== training-seed variance: %d seeds, fixed eval_seed=%d ===", len(TRAIN_SEEDS), FIXED_EVAL_SEED)
    train_variance_results = []
    for seed in TRAIN_SEEDS:
        torch.manual_seed(seed)
        result = train_and_eval(seed, FIXED_EVAL_SEED, data_path, ctx_bars, cfg, device)
        corr = result["variance"]["correlation_with_trend"]
        logger.info("seed=%d  correlation=%+.4f  pct_eligible=%.1f%%", seed, corr, 100 * result["returns"]["pct_eligible"])
        train_variance_results.append(result)

    corrs = np.array([r["variance"]["correlation_with_trend"] for r in train_variance_results])
    print(f"\n=== training-seed variance summary (N={len(TRAIN_SEEDS)} seeds) ===")
    print(f"  correlations: {[f'{c:+.4f}' for c in corrs]}")
    print(f"  mean={corrs.mean():+.4f}  std={corrs.std():.4f}  min={corrs.min():+.4f}  max={corrs.max():+.4f}")

    logger.info("=== evaluation-sampling variance: seed=42 model, %d eval reseeds ===", len(EVAL_RESEED_SEEDS))
    torch.manual_seed(42)
    df, bounds, stats, momentum_stats = pr.mp.build_momentum_dataset(data_path, include_vix=False)
    feat, opens, closes = pr.dp.extract_arrays(df)
    sell_bound = pr.dp.train_exit_return_bound(opens, closes, bounds["train"], percentile=99.0)
    cfg_seed42 = dict(cfg)
    cfg_seed42["seed"] = 42
    cvae = pr.train_rolling_cvae(feat, opens, closes, bounds, ctx_bars, cfg_seed42, device)

    eval_variance_results = []
    for eval_seed in EVAL_RESEED_SEEDS:
        result = pr.run_direction_diagnostic(cvae, feat, opens, closes, bounds, ctx_bars, sell_bound, device, eval_seed=eval_seed)
        corr = result["variance"]["correlation_with_trend"]
        logger.info("eval_seed=%d  correlation=%+.4f", eval_seed, corr)
        eval_variance_results.append(result)

    eval_corrs = np.array([r["variance"]["correlation_with_trend"] for r in eval_variance_results])
    print(f"\n=== evaluation-sampling variance summary (N={len(EVAL_RESEED_SEEDS)} reseeds, one fixed model) ===")
    print(f"  correlations: {[f'{c:+.4f}' for c in eval_corrs]}")
    print(f"  mean={eval_corrs.mean():+.4f}  std={eval_corrs.std():.4f}  min={eval_corrs.min():+.4f}  max={eval_corrs.max():+.4f}")


if __name__ == "__main__":
    main()
