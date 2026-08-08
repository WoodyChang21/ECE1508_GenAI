"""Pure numeric functions for evaluating CVAE's *generative* quality -- diversity,
calibration, and context-sensitivity -- as distinct from the trading-framed diagnostics in
cvae_direction_collapse.md (variance-ratio-vs-trend, sample-consensus-as-a-gate). A model
can have near-zero correlation with real market direction (plausible if 3-bar-ahead
direction really is close to a coin flip) while still being a good or bad *generative*
model, depending on whether its sampled diversity is calibrated to real uncertainty and
actually shifts with context -- neither property has been measured anywhere in this
project until now. See cvae_direction_collapse.md's "generative pivot" discussion.

Deliberately no I/O, no torch.nn -- importable cheaply from train_cvae.py's epoch loop
(epoch_diagnostics is the one function here that does touch a model, via plain forward
calls) without pulling in matplotlib/mplfinance the way evaluate_generative.py's plotting
does.
"""

from __future__ import annotations

import math

import numpy as np
import torch


def crps_from_samples(samples: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Continuous Ranked Probability Score, the unbiased ensemble (NRG, Gneiting & Raftery
    2007) estimator -- a proper score computable directly from K empirical samples, no
    learned/predicted variance required. samples: (K, N), y: (N,). Returns per-window CRPS
    (N,); lower is better. Jointly penalizes bias and miscalibrated spread, unlike a raw
    variance-ratio, which can look fine for a model whose spread happens to match by
    accident rather than by tracking context."""
    k = samples.shape[0]
    term1 = np.abs(samples - y[None, :]).mean(axis=0)
    diff = np.abs(samples[:, None, :] - samples[None, :, :])  # (K, K, N)
    term2 = diff.sum(axis=(0, 1)) / (2 * k * k)
    return term1 - term2


def crps_skill_score(model_crps: np.ndarray, climatology_crps: np.ndarray) -> float:
    """1 - mean(model_crps)/mean(climatology_crps). >0 means the model's context-conditioned
    samples are more informative than just knowing the context-blind (train-set marginal)
    distribution; ~0 means context isn't doing anything useful even if the model's own
    samples look internally diverse -- the sharpest single number for "is conditioning on
    context doing anything at all." climatology_crps should be computed by this same
    function against samples drawn from the train-set marginal, ignoring context entirely
    (see evaluate_generative.py)."""
    denom = float(climatology_crps.mean())
    if denom == 0:
        return float("nan")
    return 1.0 - float(model_crps.mean()) / denom


def rank_histogram(samples: np.ndarray, y: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    """Talagrand diagram data: for each window, how many of its K samples are less than
    the true value (random tiebreak on exact ties -- negligible in practice with
    continuous float samples). Ranks in {0,...,K}. Under a well-calibrated ensemble, ranks
    are uniform over K+1 bins; see extreme_rank_fraction for the scalar summary."""
    rng = rng or np.random.default_rng()
    less = (samples < y[None, :]).sum(axis=0)
    ties = (samples == y[None, :]).sum(axis=0)
    if np.any(ties):
        less = less + rng.binomial(ties, 0.5)
    return less


def extreme_rank_fraction(ranks: np.ndarray, k: int) -> float:
    """Fraction of windows where the true value fell outside the entire sample cluster
    (rank 0 or k). For a collapsed model, this is close to 1 (real value almost always
    outside the tight cluster) instead of the well-calibrated ~2/(k+1)."""
    return float(np.mean((ranks == 0) | (ranks == k)))


def diversity_stats(samples: np.ndarray) -> dict:
    """samples: (K, N) (or (K, ...), any trailing shape). Returns the within-window std
    (averaged over windows) and the mean pairwise absolute difference between two of the
    model's own draws for the same window -- same information as std, but more robust and
    intuitive for small K ("typical distance between two of the model's own generated
    candles for the same context")."""
    k = samples.shape[0]
    std = float(samples.std(axis=0).mean())
    if k > 1:
        diff = np.abs(samples[:, None, ...] - samples[None, :, ...])  # (K, K, ...)
        pairwise = float(diff.sum(axis=(0, 1)).mean() / (k * (k - 1)))
    else:
        pairwise = 0.0
    return {"std": std, "pairwise_mean_abs_diff": pairwise}


def variance_ratio(samples: np.ndarray, y: np.ndarray) -> float:
    """mean within-window sample variance / across-window (unconditional) variance of the
    true values. >1: the model's own sampling noise exceeds real cross-window variability
    (a collapse signature -- the model's samples vary more among themselves than real
    outcomes vary across genuinely different contexts). <1 alone doesn't prove good
    calibration (the model could still be under-dispersed relative to the *true
    conditional* spread and just happen to sit under the *unconditional* one) -- treat
    this as a coarse sanity check alongside CRPS/rank_histogram, not a sufficient
    calibration test on its own."""
    within = float(samples.var(axis=0).mean())
    across = float(y.var())
    return within / across if across > 0 else float("inf")


def regime_indicators(feat: np.ndarray, ctx_end: int, ctx_bars: int, body_ret_col: int = 1, lookback: int = 20) -> dict:
    """Causal-only regime indicator computed from context rows [ctx_end-n, ctx_end) --
    never the horizon. realized_vol is the std of body_ret over the lookback window (or
    the full context if ctx_bars is shorter than lookback). body_ret_col=1 matches
    FEATURE_COLS' fixed ordering (open_ret, body_ret, ...) in both the base and
    momentum-enriched pipelines -- momentum_pipeline.py inserts its new columns after
    index 4, never before, so this index is stable regardless of which pipeline built
    `feat`."""
    n = min(lookback, ctx_bars)
    window = feat[ctx_end - n : ctx_end, body_ret_col]
    return {"realized_vol": float(window.std())}


def bucket_effect_ratio(real_bucket_stats: dict, gen_bucket_stats: dict, low_key: str = "low", high_key: str = "high") -> float:
    """(gen[high]-gen[low]) / (real[high]-real[low]) -- the headline context-sensitivity
    number. ~1: the model's context-driven shift matches reality in sign and magnitude.
    ~0: context-blind generation (flat regardless of regime) -- the direct generative
    analogue of "direction collapse." Negative: shifts the wrong way. Far from 1 in
    magnitude: over/under-reacts."""
    real_delta = real_bucket_stats[high_key] - real_bucket_stats[low_key]
    gen_delta = gen_bucket_stats[high_key] - gen_bucket_stats[low_key]
    if real_delta == 0:
        return float("nan")
    return float(gen_delta / real_delta)


def _gaussian_cdf(z: np.ndarray) -> np.ndarray:
    erf = np.vectorize(math.erf)
    return 0.5 * (1.0 + erf(z / math.sqrt(2.0)))


def pit_values(true: np.ndarray, mu: np.ndarray, logvar: np.ndarray, family: str = "gaussian") -> np.ndarray:
    """Probability integral transform: the model's own predicted CDF evaluated at the true
    value, using the LEARNED variance (only meaningful for a reconstruction="nll"
    checkpoint -- see src/losses.py). Under a well-calibrated predictive distribution, PIT
    values are uniform over [0,1]: a bulge near 0/1 means overconfident (intervals too
    narrow), a bulge in the middle means underconfident (too wide). Strictly stronger than
    empirical-sample calibration (rank_histogram) alone, since it tests the model's own
    stated distribution, not just k realized draws from it."""
    std = np.exp(0.5 * logvar)
    z = (true - mu) / std
    if family == "gaussian":
        return _gaussian_cdf(z)
    if family == "laplace":
        b = std / math.sqrt(2.0)
        zz = (true - mu) / b
        return np.where(zz < 0, 0.5 * np.exp(zz), 1.0 - 0.5 * np.exp(-zz))
    raise ValueError(f"unknown family: {family!r}")


def coverage(pit: np.ndarray, levels: tuple[float, ...] = (0.5, 0.8, 0.9, 0.95)) -> dict:
    """Nominal-vs-empirical coverage from PIT values: for each level (e.g. 0.9 = a "90%
    interval"), what fraction of true outcomes fall inside pit in [(1-level)/2,
    (1+level)/2]. Should equal `level` for an honest predictive distribution."""
    out = {}
    for level in levels:
        lo, hi = (1 - level) / 2, (1 + level) / 2
        out[level] = float(np.mean((pit >= lo) & (pit <= hi)))
    return out


def epoch_diagnostics(model, val_pairs: list[tuple[int, int]], feat, opens, closes, device, k: int = 8, max_windows: int = 300) -> dict:
    """Pure forward-pass diagnostics (torch.no_grad, no gradient/optimizer impact) over a
    subset of the fixed val_pairs list -- cheap enough to run every epoch, giving visibility
    into whether diversity/calibration degrades mid-training rather than only after the
    fact (the offline diagnose_cvae_direction.py-style scripts this project has relied on
    so far only ever ran post-hoc). Batches all sampled windows into one model.sample()
    call rather than looping window-by-window, since build_window always pads to a fixed
    (TOTAL_LEN, N_CHANNELS) shape regardless of each window's own ctx_bars.

    Tracks body_ret (bar 0) only, to keep the epoch log line short -- the full
    per-component/per-bar/regime breakdown lives in evaluate_generative.py."""
    from src.data_pipeline import build_window

    pairs = val_pairs[:max_windows]
    windows = [build_window(feat, opens, closes, s, c) for s, c in pairs]
    masked = torch.from_numpy(np.stack([w["masked_tensor"] for w in windows])).to(device)
    true_body_ret = np.array([w["y"][1] for w in windows])

    was_training = model.training
    model.eval()
    with torch.no_grad():
        price, _ = model.sample(masked, k=k)  # (K, N, 3, 4)
    model.train(mode=was_training)

    samples = price[:, :, 0, 1].cpu().numpy()  # (K, N) -- bar 0's body_ret
    div = diversity_stats(samples)
    real_std = true_body_ret.std()
    return {
        "gen_diversity_ratio": float(div["std"] / real_std) if real_std > 0 else float("nan"),
        "gen_crps": float(crps_from_samples(samples, true_body_ret).mean()),
        "gen_var_ratio": variance_ratio(samples, true_body_ret),
    }
