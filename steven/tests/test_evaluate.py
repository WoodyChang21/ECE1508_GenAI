import numpy as np

from src import evaluate as ev


def _components(open_ret: float, body_ret: float) -> list:
    """A single [open_ret, body_ret, upper_wick, lower_wick] row, wicks zeroed."""
    return [open_ret, body_ret, 0.0, 0.0]


def test_percentile_rank_basic():
    ranks = ev.percentile_rank(np.array([3.0, 1.0, 2.0]))
    np.testing.assert_allclose(ranks, [1.0, 0.0, 0.5])


def test_percentile_rank_edge_cases():
    assert ev.percentile_rank(np.array([5.0])).tolist() == [0.0]
    assert ev.percentile_rank(np.array([])).tolist() == []


def test_cvae_confidence_scores_consensus_and_chop():
    close_0 = np.array([100.0, 100.0])
    # window0: all 4 samples predict up -> confidence 1.0
    # window1: 2 up, 2 down -> confidence 0.5
    up = _components(0.01, 0.0)
    down = _components(-0.01, 0.0)
    samples = np.array(
        [
            [up, up],
            [up, up],
            [up, down],
            [up, down],
        ]
    )  # (K=4, N=2, 4) -- need (K,N,3,4): broadcast same components across the 3 bars
    price_samples = np.broadcast_to(samples[:, :, None, :], (4, 2, 3, 4))

    confidence = ev.cvae_confidence_scores(price_samples, close_0)
    np.testing.assert_allclose(confidence, [1.0, 0.5])


def test_cvae_confidence_scores_empty():
    assert ev.cvae_confidence_scores(np.empty((4, 0, 3, 4)), np.empty(0)).tolist() == []


def test_patchtst_confidence_scores_incoherent_forced_zero_and_ranked_by_magnitude():
    close_0 = np.array([100.0, 100.0, 100.0, 100.0])

    small_up = _components(0.005, 0.0)
    mid_up = _components(0.01, 0.0)
    big_up = _components(0.02, 0.0)
    # incoherent: bar1 up, bar2 down, bar3 up
    incoherent = [_components(0.01, 0.0), _components(0.01, -0.03), _components(0.01, 0.0)]

    pt_price = np.array(
        [
            [small_up, small_up, small_up],
            [mid_up, mid_up, mid_up],
            [big_up, big_up, big_up],
            incoherent,
        ]
    )

    confidence = ev.patchtst_confidence_scores(pt_price, close_0)

    assert confidence[3] == 0.0  # incoherent window auto-dismissed
    # coherent windows ranked strictly by magnitude
    assert confidence[0] < confidence[1] < confidence[2]
    np.testing.assert_allclose(sorted(confidence[:3]), [0.0, 0.5, 1.0])


def test_patchtst_confidence_scores_empty():
    assert ev.patchtst_confidence_scores(np.empty((0, 3, 4)), np.empty(0)).tolist() == []


def test_sweep_thresholds_basic():
    confidence = np.array([0.9, 0.8, 0.6, 0.4, 0.2])
    true_return = np.array([0.02, -0.01, 0.03, -0.02, 0.01])

    rows = ev.sweep_thresholds(confidence, true_return, [0.5, 0.9])
    by_threshold = {r["threshold"]: r for r in rows}

    row_50 = by_threshold[0.5]
    assert row_50["n_trades"] == 3
    assert row_50["selectivity"] == 3 / 5
    np.testing.assert_allclose(row_50["win_rate"], 2 / 3)
    np.testing.assert_allclose(row_50["avg_return"], np.mean([0.02, -0.01, 0.03]))
    np.testing.assert_allclose(row_50["total_return"], np.sum([0.02, -0.01, 0.03]))

    row_90 = by_threshold[0.9]
    assert row_90["n_trades"] == 1
    np.testing.assert_allclose(row_90["avg_return"], 0.02)


def test_sweep_thresholds_no_trades_returns_none_fields():
    confidence = np.array([0.1, 0.2])
    true_return = np.array([0.01, -0.01])
    rows = ev.sweep_thresholds(confidence, true_return, [0.9])
    assert rows[0]["n_trades"] == 0
    assert rows[0]["win_rate"] is None
    assert rows[0]["avg_return"] is None
    assert rows[0]["total_return"] is None


def test_run_backtest_empty_windows():
    result = ev.run_backtest(np.empty((0, 15)), np.empty((0, 15)), np.empty((5, 0, 15)), np.empty(0))
    assert result["patchtst"] == []
    assert result["cvae"] == []
