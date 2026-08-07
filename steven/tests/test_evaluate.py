import numpy as np
import pytest

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


def test_sweep_thresholds_eligible_mask_excludes_regardless_of_confidence():
    """A window with high confidence but eligible=False (its own take-profit doesn't
    clear the buy price) must never be traded, at any threshold."""
    confidence = np.array([0.9, 0.9, 0.9])
    trade_return = np.array([0.05, -0.05, 0.02])
    eligible = np.array([True, False, True])

    rows = ev.sweep_thresholds(confidence, trade_return, [0.5], eligible=eligible)

    assert rows[0]["n_trades"] == 2
    np.testing.assert_allclose(rows[0]["avg_return"], np.mean([0.05, 0.02]))


def test_run_backtest_empty_windows():
    result = ev.run_backtest(np.empty((0, 15)), np.empty((0, 15)), np.empty((5, 0, 15)), np.empty(0))
    assert result["patchtst"] == []
    assert result["cvae"] == []


def _bar(open_, high, low, close):
    return [open_, high, low, close]


def test_take_profit_exit_scenarios():
    """take_profit=105 for every row.
    row0: touched at bar0 only.
    row1: never touched -> expiry at bar2's real close.
    row2: touched only at bar2 (confirms the sequential, not "any bar", walk).
    row3: touched at bar0; bar1's range must NOT flip the already-resolved outcome.
    """
    true_ohlc = np.array([
        [_bar(101, 106, 100, 104), _bar(100, 101, 99, 100), _bar(100, 101, 99, 100)],  # row0
        [_bar(100, 102, 98, 101), _bar(101, 103, 99, 102), _bar(102, 104, 100, 103)],  # row1
        [_bar(100, 102, 99, 101), _bar(101, 103, 100, 102), _bar(102, 106, 101, 104)],  # row2
        [_bar(101, 106, 100, 104), _bar(90, 110, 85, 92), _bar(100, 101, 99, 100)],  # row3
    ])
    take_profit = np.full(4, 105.0)

    sell_price, hit_tp = ev.take_profit_exit(true_ohlc, take_profit)

    np.testing.assert_allclose(sell_price, [105.0, 103.0, 105.0, 105.0])
    np.testing.assert_array_equal(hit_tp, [True, False, True, True])


def test_sweep_thresholds_reports_take_profit_rate():
    confidence = np.array([0.9, 0.9, 0.9, 0.9])
    trade_return = np.array([0.05, -0.05, 0.05, -0.05])
    hit_tp = np.array([True, False, False, False])

    rows = ev.sweep_thresholds(confidence, trade_return, [0.5], hit_tp)

    assert rows[0]["take_profit_rate"] == pytest.approx(0.25)


def test_classify_outcomes_all_four_cases():
    # window0: eligible, hit -> win_take_profit
    # window1: eligible, missed, positive return -> win_expiry
    # window2: not eligible -> no_trade (regardless of hit/return, which shouldn't matter)
    # window3: eligible, missed, non-positive return -> lose_expiry (boundary: exactly 0)
    eligible = np.array([True, True, False, True])
    hit_tp = np.array([True, False, True, False])
    trade_return = np.array([0.05, 0.01, 0.05, 0.0])

    labels = ev.classify_outcomes(eligible, hit_tp, trade_return)

    np.testing.assert_array_equal(
        labels, ["win_take_profit", "win_expiry", "no_trade", "lose_expiry"]
    )


def test_outcome_breakdown_fractions_and_empty():
    labels = np.array(["win_take_profit", "win_expiry", "no_trade", "lose_expiry", "no_trade"])
    breakdown = ev.outcome_breakdown(labels)

    assert breakdown["win_take_profit"] == pytest.approx(0.2)
    assert breakdown["win_expiry"] == pytest.approx(0.2)
    assert breakdown["no_trade"] == pytest.approx(0.4)
    assert breakdown["lose_expiry"] == pytest.approx(0.2)
    assert sum(breakdown.values()) == pytest.approx(1.0)

    assert ev.outcome_breakdown(np.array([])) == {}


def test_model_trade_outcomes_matches_take_profit_exit_and_classify():
    true_ohlc = np.array([
        [_bar(101, 106, 100, 104), _bar(100, 101, 99, 100), _bar(100, 101, 99, 100)],  # hit
        [_bar(100, 102, 98, 101), _bar(101, 103, 99, 102), _bar(102, 104, 100, 103)],  # miss, close=103
    ])
    take_profit = np.array([105.0, 90.0])  # row1's target (90) never clears close_0 (100)
    close_0 = np.array([100.0, 100.0])

    out = ev.model_trade_outcomes(true_ohlc, take_profit, close_0)

    np.testing.assert_allclose(out["sell_price"], [105.0, 103.0])
    np.testing.assert_array_equal(out["hit_take_profit"], [True, False])
    np.testing.assert_array_equal(out["eligible"], [True, False])
    np.testing.assert_array_equal(out["label"], ["win_take_profit", "no_trade"])


def test_nearest_draw_index_picks_closest():
    draws = np.array([100.0, 102.0, 105.0, 110.0])
    assert ev.nearest_draw_index(draws, 106.0) == 2  # 105 is closer to 106 than 110 is
    assert ev.nearest_draw_index(draws, 100.5) == 0
    assert ev.nearest_draw_index(draws, 111.0) == 3


def test_nearest_draw_index_tie_breaks_to_first_occurrence():
    draws = np.array([90.0, 100.0, 110.0])
    assert ev.nearest_draw_index(draws, 100.0) == 1  # exact match, unambiguous
    # 95 and 105 are both exactly 5 away from 100 -- np.argmin returns the first occurrence
    tied_draws = np.array([95.0, 105.0])
    assert ev.nearest_draw_index(tied_draws, 100.0) == 0
