import numpy as np
import pandas as pd
import pytest

from src import evaluate as ev


def _components(open_ret: float, body_ret: float) -> list:
    """A single [open_ret, body_ret, upper_wick, lower_wick] row, wicks zeroed."""
    return [open_ret, body_ret, 0.0, 0.0]


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


def test_take_profit_exit_gap_through_fills_at_open_not_target():
    """take_profit=105 for every row. A bar whose entire range (including its open)
    already sits above the target -- a gap through the limit price -- must fill at that
    bar's OPEN (the market never actually touched 105, and a real limit order guarantees
    at least the limit price), not at the stale take_profit level, and must still count
    as a take-profit hit, not an expiry.
    row0: gaps through on bar0 -> fills at bar0's open (110), not 105.
    row1: bar0 doesn't even reach the target; bar1 gaps through -> fills at bar1's open (109).
    row2: gaps through on bar0; bar1's much larger range must NOT override the already-resolved fill.
    """
    true_ohlc = np.array([
        [_bar(110, 112, 108, 111), _bar(100, 101, 99, 100), _bar(100, 101, 99, 100)],  # row0
        [_bar(100, 102, 99, 101), _bar(109, 111, 107, 110), _bar(100, 101, 99, 100)],  # row1
        [_bar(110, 112, 108, 111), _bar(200, 210, 190, 205), _bar(100, 101, 99, 100)],  # row2
    ])
    take_profit = np.full(3, 105.0)

    sell_price, hit_tp = ev.take_profit_exit(true_ohlc, take_profit)

    np.testing.assert_allclose(sell_price, [110.0, 109.0, 110.0])
    np.testing.assert_array_equal(hit_tp, [True, True, True])


def test_classify_walk_forward_decision_all_five_cases():
    # not eligible at all -> no_trade, regardless of everything else
    assert ev.classify_walk_forward_decision(False, True, True, True, 0.05) == ("no_trade", False)
    # eligible, but predicted edge too small -> skipped
    assert ev.classify_walk_forward_decision(True, False, True, True, 0.05) == ("skipped", False)
    # eligible and big enough edge, but confidence too low -> skipped
    assert ev.classify_walk_forward_decision(True, True, False, True, 0.05) == ("skipped", False)
    # eligible, meets return threshold, confident, and hits target -> win_take_profit
    assert ev.classify_walk_forward_decision(True, True, True, True, 0.05) == ("win_take_profit", True)
    # eligible, confident, missed target, positive return at expiry -> win_expiry
    assert ev.classify_walk_forward_decision(True, True, True, False, 0.01) == ("win_expiry", True)
    # eligible, confident, missed target, non-positive return at expiry -> lose_expiry
    assert ev.classify_walk_forward_decision(True, True, True, False, -0.01) == ("lose_expiry", True)
    assert ev.classify_walk_forward_decision(True, True, True, False, 0.0) == ("lose_expiry", True)


def test_outcome_breakdown_fractions_and_empty():
    labels = np.array(
        ["win_take_profit", "win_expiry", "lose_expiry", "skipped", "no_trade", "no_trade"]
    )
    breakdown = ev.outcome_breakdown(labels)

    assert breakdown["win_take_profit"] == pytest.approx(1 / 6)
    assert breakdown["win_expiry"] == pytest.approx(1 / 6)
    assert breakdown["lose_expiry"] == pytest.approx(1 / 6)
    assert breakdown["skipped"] == pytest.approx(1 / 6)
    assert breakdown["no_trade"] == pytest.approx(2 / 6)
    assert sum(breakdown.values()) == pytest.approx(1.0)

    assert ev.outcome_breakdown(np.array([])) == {}


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


def _fake_df(n: int, start: str = "2024-01-02 09:30:00") -> pd.DataFrame:
    dt = pd.date_range(start, periods=n, freq="h")
    # close drifts by +1 per bar so entry/exit price math is easy to hand-check
    close = 100.0 + np.arange(n, dtype=float)
    return pd.DataFrame({"datetime": dt, "close": close})


def test_equity_stats_basic():
    df = _fake_df(400)
    stats = ev.equity_stats(df, entry_idx=0, exit_idx=399, entry_price=100.0, exit_price=200.0, total_return=1.0)
    assert stats["entry_price"] == 100.0
    assert stats["exit_price"] == 200.0
    assert stats["total_return"] == 1.0
    assert stats["elapsed_years"] > 0
    # doubling over some elapsed_years E means annual_return = 2**(1/E) - 1
    expected_annual = 2.0 ** (1.0 / stats["elapsed_years"]) - 1.0
    assert stats["annual_return"] == pytest.approx(expected_annual)


def test_buy_and_hold_benchmark_uses_given_entry_exit():
    df = _fake_df(200)
    bh = ev.buy_and_hold_benchmark(df, entry_idx=10, exit_idx=110)
    assert bh["entry_price"] == 110.0  # close[10] = 100+10
    assert bh["exit_price"] == 210.0  # close[110] = 100+110
    assert bh["total_return"] == pytest.approx(210.0 / 110.0 - 1.0)


def test_naive_periodic_benchmark_tiles_nonoverlapping_3bar_blocks():
    # closes: index 0..19 = [100, 101, ..., 119]. t0=0 -> first block buys idx1, sells idx3;
    # next block buys idx4, sells idx6; and so on, stopping once sell_idx would hit test_hi.
    df = _fake_df(20)
    closes = df["close"].to_numpy()
    result = ev.naive_periodic_benchmark(df, closes, t0=0, test_hi=20)

    expected_buys = [1, 4, 7, 10, 13, 16]
    expected_sells = [3, 6, 9, 12, 15, 18]
    assert result["n_trades"] == len(expected_buys)
    expected_return = 1.0
    for b, s in zip(expected_buys, expected_sells):
        expected_return *= closes[s] / closes[b]
    np.testing.assert_allclose(result["total_return"], expected_return - 1.0)
    assert result["win_rate"] == 1.0  # closes strictly increasing -> every leg wins
    assert result["take_profit_rate"] is None


def test_naive_periodic_benchmark_no_room_for_a_trade():
    df = _fake_df(3)
    closes = df["close"].to_numpy()
    result = ev.naive_periodic_benchmark(df, closes, t0=0, test_hi=3)
    assert result["n_trades"] == 0
    assert result["win_rate"] is None
    assert result["total_return"] == 0.0


def test_patchtst_walk_forward_confidence_online_reference():
    close_0 = np.array([100.0])
    coherent_up = np.array([_components(0.02, 0.0)] * 3)[None]
    incoherent = np.array([_components(0.02, 0.0), _components(0.01, -0.03), _components(0.02, 0.0)])[None]

    # Very first coherent-up call ever: nothing to rank against yet -> neutral 0.5, not 0.0
    # (0.0 would silently gate out every early trading opportunity -- see the historical bug
    # this replaced, where percentile_rank of a length<=1 array was hardcoded to 0).
    reference: list[float] = []
    assert ev.patchtst_walk_forward_confidence(coherent_up, close_0, reference) == 0.5
    # open_ret=body_ret combine via exp() (log-return reconstruction), not linearly -- see
    # reconstruct_prices -- so the stored magnitude is exp(0.02)-1, not exactly 0.02.
    assert reference == [pytest.approx(np.exp(0.02) - 1.0)]

    # Second coherent-up call with a bigger predicted move now has one prior value (0.02) to
    # beat -> ranks above it -> 1.0. The reference list grows in place (online/expanding).
    bigger_up = np.array([_components(0.05, 0.0)] * 3)[None]
    assert ev.patchtst_walk_forward_confidence(bigger_up, close_0, reference) == 1.0
    assert len(reference) == 2

    # Incoherent (bars disagree on direction) is auto-dismissed regardless of the
    # reference, and does NOT get appended to it.
    assert ev.patchtst_walk_forward_confidence(incoherent, close_0, reference) == 0.0
    assert len(reference) == 2


def _fake_predict_fn(take_profit: float, confidence: float):
    """A stand-in for make_patchtst_predict_fn/make_cvae_predict_fn's returned closure --
    same {take_profit, confidence, price} contract, constant regardless of the window."""

    def predict(w):
        return {"take_profit": take_profit, "confidence": confidence, "price": np.zeros((3, 4))}

    return predict


def test_run_walk_forward_no_trade_advances_one_bar_trade_advances_horizon():
    """A predict_fn that never clears eligibility should visit every start_idx from
    test_lo up to the last valid one (step 1 each time); one that's always eligible,
    confident, and above the min-return threshold should instead jump by HORIZON each
    time."""
    feat = np.zeros((200, 7), dtype=np.float32)
    opens = np.full(200, 100.0)
    closes = np.full(200, 100.0)
    test_lo, test_hi = 0, 200
    last_valid_start = test_hi - ev.WALK_FORWARD_CTX_BARS - ev.HORIZON

    never_eligible = _fake_predict_fn(take_profit=50.0, confidence=1.0)  # < close_0 (100)
    result = ev.run_walk_forward(
        never_eligible, feat, opens, closes, test_lo, test_hi,
        confidence_threshold=0.5, min_return_threshold=0.0,
    )
    visited = [d["start_idx"] for d in result["decisions"]]
    assert visited == list(range(test_lo, last_valid_start + 1))
    assert result["trades"] == []
    assert all(d["label"] == "no_trade" for d in result["decisions"])
    assert result["equity_final"] == 1.0

    always_trade = _fake_predict_fn(take_profit=150.0, confidence=1.0)  # > close_0 (100)
    result_2 = ev.run_walk_forward(
        always_trade, feat, opens, closes, test_lo, test_hi,
        confidence_threshold=0.5, min_return_threshold=0.0,
    )
    visited_2 = [d["start_idx"] for d in result_2["decisions"]]
    assert visited_2 == list(range(test_lo, last_valid_start + 1, ev.HORIZON))
    assert len(result_2["trades"]) == len(visited_2)


def test_run_walk_forward_min_return_threshold_gates_eligible_windows():
    """Eligible (target > buy) but the predicted edge is smaller than min_return_threshold
    -> 'skipped', never traded, and the walk advances 1 bar at a time just like a
    confidence-driven skip would."""
    feat = np.zeros((120, 7), dtype=np.float32)
    opens = np.full(120, 100.0)
    closes = np.full(120, 100.0)
    test_lo, test_hi = 0, 120

    tiny_edge = _fake_predict_fn(take_profit=100.05, confidence=1.0)  # +0.05% predicted edge
    result = ev.run_walk_forward(
        tiny_edge, feat, opens, closes, test_lo, test_hi,
        confidence_threshold=0.5, min_return_threshold=0.001,  # require >= 0.1%
    )
    assert result["trades"] == []
    assert all(d["label"] == "skipped" for d in result["decisions"])


def test_walk_forward_stats_no_trades():
    df = _fake_df(100)
    result = {"trades": [], "decisions": [{"start_idx": 0, "label": "no_trade"}], "equity_final": 1.0}
    stats = ev.walk_forward_stats(df, result)
    assert stats["n_trades"] == 0
    assert stats["win_rate"] is None
    assert stats["total_return"] == 0.0
    assert stats["outcome_breakdown"] == {
        "win_take_profit": 0.0, "win_expiry": 0.0, "lose_expiry": 0.0, "skipped": 0.0, "no_trade": 1.0,
    }
