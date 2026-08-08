import numpy as np

from src.diagnose_cvae_direction import predicted_return_stats, variance_ratio_and_correlation


def test_variance_ratio_above_one_when_context_dominates_sampling_noise():
    """3 windows with very different means (real, context-driven signal) but tiny
    within-window spread (little sampling noise) -- context clearly beats noise."""
    draws = np.array([
        [0.010, 0.011, 0.009],
        [0.000, 0.001, -0.001],
        [-0.010, -0.009, -0.011],
    ])
    result = variance_ratio_and_correlation(draws, trend=np.array([1.0, 0.0, -1.0]))
    assert result["ratio"] > 1.0
    assert result["correlation_with_trend"] > 0.9


def test_variance_ratio_below_one_when_collapsed_to_noise_around_a_constant():
    """Same overall mean in every window (no real per-window signal), large sampling
    noise -- collapsed behavior, ratio should be well below 1."""
    rng = np.random.default_rng(0)
    draws = -0.001 + rng.normal(scale=0.01, size=(50, 5))
    trend = rng.normal(size=50)
    result = variance_ratio_and_correlation(draws, trend)
    assert result["ratio"] < 1.0
    assert abs(result["mean"] - (-0.001)) < 0.005


def test_predicted_return_stats_eligibility_fraction():
    predicted_returns = np.array([-0.01, -0.005, 0.001, 0.02, -0.002])
    stats = predicted_return_stats(predicted_returns)
    assert stats["pct_eligible"] == 2 / 5
    assert stats["p50"] == np.percentile(predicted_returns, 50)


def test_predicted_return_stats_all_negative_means_zero_eligible():
    predicted_returns = np.array([-0.01, -0.02, -0.003, -0.15])
    stats = predicted_return_stats(predicted_returns)
    assert stats["pct_eligible"] == 0.0
    assert stats["mean"] < 0.0
