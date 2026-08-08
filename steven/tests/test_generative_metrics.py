import math

import numpy as np

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


def test_crps_from_samples_zero_for_perfect_ensemble():
    """Every sample exactly equals the true value -- both terms of the NRG estimator
    vanish, CRPS must be exactly 0."""
    samples = np.full((5, 3), 2.0)
    y = np.full(3, 2.0)
    assert np.allclose(crps_from_samples(samples, y), 0.0)


def test_crps_from_samples_hand_computed_case():
    """samples (K=2, N=2): window 0 = [1,3] vs y=2; window 1 = [3,5] vs y=4 (same shape,
    shifted by 2). term1 = mean(|1-2|,|3-2|) = 1; term2 = sum(|1-1|,|1-3|,|3-1|,|3-3|)/(2*2*2)
    = 4/8 = 0.5 -> CRPS = 0.5 for both windows."""
    samples = np.array([[1.0, 3.0], [3.0, 5.0]])
    y = np.array([2.0, 4.0])
    result = crps_from_samples(samples, y)
    assert np.allclose(result, [0.5, 0.5])


def test_crps_skill_score_positive_when_model_beats_climatology():
    model_crps = np.array([0.1, 0.1])
    climatology_crps = np.array([1.0, 1.0])
    score = crps_skill_score(model_crps, climatology_crps)
    assert score > 0.0
    assert math.isclose(score, 0.9)


def test_crps_skill_score_zero_when_model_matches_climatology():
    crps = np.array([0.5, 0.7])
    assert math.isclose(crps_skill_score(crps, crps), 0.0)


def test_rank_histogram_ranks_within_valid_range():
    rng = np.random.default_rng(0)
    samples = rng.normal(size=(10, 50))
    y = rng.normal(size=50)
    ranks = rank_histogram(samples, y, rng=rng)
    assert ranks.min() >= 0
    assert ranks.max() <= 10


def test_extreme_rank_fraction_high_when_true_always_outside_tight_cluster():
    """Samples tightly clustered near 0; true value always far away at 100 -- every
    window's true value is above all K samples, so rank == K every time."""
    samples = np.zeros((5, 20)) + np.linspace(-0.01, 0.01, 5)[:, None]
    y = np.full(20, 100.0)
    ranks = rank_histogram(samples, y)
    assert extreme_rank_fraction(ranks, k=5) == 1.0


def test_extreme_rank_fraction_low_for_well_calibrated_ensemble():
    """True value drawn from the exact same distribution as the samples -- rank should
    land in the interior most of the time, not at the extremes."""
    rng = np.random.default_rng(1)
    samples = rng.normal(size=(20, 500))
    y = rng.normal(size=500)
    ranks = rank_histogram(samples, y, rng=rng)
    assert extreme_rank_fraction(ranks, k=20) < 0.3  # well below "always extreme" (1.0)


def test_diversity_stats_zero_for_identical_samples():
    samples = np.full((5, 3), 1.5)
    result = diversity_stats(samples)
    assert result["std"] == 0.0
    assert result["pairwise_mean_abs_diff"] == 0.0


def test_diversity_stats_hand_computed_pairwise_diff():
    """Single window, K=3 samples [0, 2, 4] -- pairs (0,2),(0,4),(2,4) have |diff| 2,4,2 ->
    mean = 8/3."""
    samples = np.array([[0.0], [2.0], [4.0]])
    result = diversity_stats(samples)
    assert math.isclose(result["pairwise_mean_abs_diff"], 8.0 / 3.0, rel_tol=1e-9)


def test_variance_ratio_above_one_when_sampling_noise_dominates():
    rng = np.random.default_rng(0)
    samples = rng.normal(scale=10.0, size=(50, 20))  # huge within-window noise
    y = rng.normal(scale=0.1, size=20)  # tiny across-window signal
    assert variance_ratio(samples, y) > 1.0


def test_variance_ratio_below_one_when_context_signal_dominates():
    rng = np.random.default_rng(0)
    samples = rng.normal(scale=0.01, size=(50, 20))  # tiny within-window noise
    y = rng.normal(scale=10.0, size=20)  # huge across-window signal
    assert variance_ratio(samples, y) < 1.0


def test_regime_indicators_realized_vol_matches_manual_std():
    feat = np.zeros((100, 2))
    feat[80:90, 1] = np.array([0.01, -0.02, 0.015, -0.01, 0.02, -0.015, 0.01, -0.02, 0.005, -0.005])
    result = regime_indicators(feat, ctx_end=90, ctx_bars=90, body_ret_col=1, lookback=10)
    assert math.isclose(result["realized_vol"], feat[80:90, 1].std(), rel_tol=1e-9)


def test_regime_indicators_clips_lookback_to_short_context():
    feat = np.zeros((20, 2))
    feat[:5, 1] = [0.01, -0.01, 0.02, -0.02, 0.005]
    result = regime_indicators(feat, ctx_end=5, ctx_bars=5, body_ret_col=1, lookback=20)
    assert math.isclose(result["realized_vol"], feat[:5, 1].std(), rel_tol=1e-9)


def test_bucket_effect_ratio_one_when_model_shift_matches_reality():
    real = {"low": 0.0, "high": 1.0}
    gen = {"low": 0.0, "high": 1.0}
    assert math.isclose(bucket_effect_ratio(real, gen), 1.0)


def test_bucket_effect_ratio_zero_when_context_blind():
    real = {"low": 0.0, "high": 1.0}
    gen = {"low": 0.5, "high": 0.5}  # same regardless of regime
    assert bucket_effect_ratio(real, gen) == 0.0


def test_bucket_effect_ratio_nan_when_real_delta_zero():
    real = {"low": 0.5, "high": 0.5}
    gen = {"low": 0.0, "high": 1.0}
    assert math.isnan(bucket_effect_ratio(real, gen))


def test_pit_values_known_gaussian_points():
    """true == mu -> PIT == 0.5; true == mu + std -> PIT == Phi(1) ~= 0.8413."""
    mu = np.array([0.0, 0.0])
    logvar = np.array([0.0, 0.0])  # std = 1
    true = np.array([0.0, 1.0])
    result = pit_values(true, mu, logvar, family="gaussian")
    assert math.isclose(result[0], 0.5, abs_tol=1e-9)
    assert math.isclose(result[1], 0.8413447, abs_tol=1e-6)


def test_pit_values_laplace_known_point():
    """true == mu -> PIT == 0.5 for Laplace too (median == mean for a symmetric dist)."""
    mu = np.array([0.0])
    logvar = np.array([0.0])
    true = np.array([0.0])
    result = pit_values(true, mu, logvar, family="laplace")
    assert math.isclose(result[0], 0.5, abs_tol=1e-9)


def test_coverage_matches_nominal_for_uniform_pit():
    """PIT values spread uniformly over [0,1] should give coverage(level) ~= level for
    every level, by construction."""
    pit = np.linspace(0.0005, 0.9995, 2000)
    result = coverage(pit, levels=(0.5, 0.8, 0.9))
    for level, frac in result.items():
        assert math.isclose(frac, level, abs_tol=0.02)


def test_coverage_low_when_pit_bulges_at_extremes():
    """Overconfident case: PIT values concentrated near 0/1 -- coverage of central
    intervals should be well below nominal."""
    rng = np.random.default_rng(0)
    pit = np.concatenate([rng.uniform(0, 0.05, 500), rng.uniform(0.95, 1.0, 500)])
    result = coverage(pit, levels=(0.9,))
    assert result[0.9] < 0.5
