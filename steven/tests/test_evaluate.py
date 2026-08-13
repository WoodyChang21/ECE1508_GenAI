import numpy as np
import pandas as pd
import pytest

from src import evaluate as ev


def _bar(open_, high, low, close):
    return [open_, high, low, close]


def test_bracket_exit_take_profit_only_scenarios():
    """HORIZON=1 (steven4): bracket_exit only ever checks bar 0 now -- the old multi-bar
    sequential-walk scenarios (touched on bar 2 only, bar 1 not flipping an already-
    resolved bar 0) are no longer reachable through the module-level HORIZON constant, so
    this covers what's still testable at a single bar: touched, and never-touched expiry.
    stop_loss=-inf (disabled) for every row.
    row0: touched mid-bar.
    row1: never touched -> expiry at bar 0's real close."""
    true_ohlc = np.array([
        [_bar(101, 106, 100, 104)],  # row0: touched (low=100 <= 105 <= high=106)
        [_bar(100, 102, 98, 101)],  # row1: high=102 < take_profit=105 -> never touched
    ])
    take_profit = np.full(2, 105.0)
    stop_loss = np.full(2, -np.inf)

    sell_price, hit_tp, hit_sl = ev.bracket_exit(true_ohlc, take_profit, stop_loss)

    np.testing.assert_allclose(sell_price, [105.0, 101.0])
    np.testing.assert_array_equal(hit_tp, [True, False])
    np.testing.assert_array_equal(hit_sl, [False, False])


def test_bracket_exit_gap_through_fills_at_open_not_target():
    """take_profit=105, stop_loss disabled. The bar's entire range (including its open)
    already sits above the target -- a gap through the limit price -- must fill at the
    bar's OPEN (the market never actually touched 105, and a real limit order guarantees
    at least the limit price), not at the stale take_profit level, and must still count as
    a take-profit hit, not an expiry."""
    true_ohlc = np.array([[_bar(110, 112, 108, 111)]])
    take_profit = np.array([105.0])
    stop_loss = np.array([-np.inf])

    sell_price, hit_tp, hit_sl = ev.bracket_exit(true_ohlc, take_profit, stop_loss)

    np.testing.assert_allclose(sell_price, [110.0])
    np.testing.assert_array_equal(hit_tp, [True])
    np.testing.assert_array_equal(hit_sl, [False])


def test_bracket_exit_stop_loss_touched_and_gapped():
    """take_profit=110 (never reached) for every row; stop_loss=95.
    row0: stop touched mid-bar (low=94 <= 95 <= high=99) -> fills at 95, hit_sl=True.
    row1: whole bar already below stop (high=94 < 95) -> gaps through, fills at the open (93)."""
    true_ohlc = np.array([
        [_bar(98, 99, 94, 96)],  # row0
        [_bar(93, 94, 90, 91)],  # row1
    ])
    take_profit = np.full(2, 110.0)
    stop_loss = np.full(2, 95.0)

    sell_price, hit_tp, hit_sl = ev.bracket_exit(true_ohlc, take_profit, stop_loss)

    np.testing.assert_allclose(sell_price, [95.0, 93.0])
    np.testing.assert_array_equal(hit_tp, [False, False])
    np.testing.assert_array_equal(hit_sl, [True, True])


def test_bracket_exit_same_bar_overlap_stop_loss_wins():
    """A single bar whose range spans both take_profit (105) and stop_loss (95) -- an
    ambiguous same-bar overlap OHLC can't resolve -- must be pessimistically resolved as
    the stop-loss, never the take-profit."""
    true_ohlc = np.array([
        [_bar(100, 110, 90, 100)],
    ])
    take_profit = np.array([105.0])
    stop_loss = np.array([95.0])

    sell_price, hit_tp, hit_sl = ev.bracket_exit(true_ohlc, take_profit, stop_loss)

    np.testing.assert_allclose(sell_price, [95.0])
    np.testing.assert_array_equal(hit_tp, [False])
    np.testing.assert_array_equal(hit_sl, [True])


def test_bracket_exit_stop_loss_disabled_never_triggers():
    """stop_loss=-inf must never touch or gap, however far price falls -- confirms
    disabling per-row via -inf, not a separate code path, is actually safe."""
    true_ohlc = np.array([
        [_bar(100, 101, 50, 60)],
    ])
    take_profit = np.array([200.0])  # never reached either -> expiry
    stop_loss = np.array([-np.inf])

    sell_price, hit_tp, hit_sl = ev.bracket_exit(true_ohlc, take_profit, stop_loss)

    np.testing.assert_allclose(sell_price, [60.0])  # bar 0's real close
    np.testing.assert_array_equal(hit_tp, [False])
    np.testing.assert_array_equal(hit_sl, [False])


def test_classify_walk_forward_decision_all_six_cases():
    # not eligible at all -> no_trade, regardless of everything else
    assert ev.classify_walk_forward_decision(False, True, True, True, False, 0.05) == ("no_trade", False)
    # eligible, but predicted edge too small -> skipped
    assert ev.classify_walk_forward_decision(True, False, True, True, False, 0.05) == ("skipped", False)
    # eligible and big enough edge, but confidence too low -> skipped
    assert ev.classify_walk_forward_decision(True, True, False, True, False, 0.05) == ("skipped", False)
    # stop-loss hit takes priority over everything else once a trade would happen -> lose_stop_loss
    assert ev.classify_walk_forward_decision(True, True, True, True, True, 0.05) == ("lose_stop_loss", True)
    assert ev.classify_walk_forward_decision(True, True, True, False, True, -0.02) == ("lose_stop_loss", True)
    # eligible, meets return threshold, confident, and hits target -> win_take_profit
    assert ev.classify_walk_forward_decision(True, True, True, True, False, 0.05) == ("win_take_profit", True)
    # eligible, confident, missed target, positive return at expiry -> win_expiry
    assert ev.classify_walk_forward_decision(True, True, True, False, False, 0.01) == ("win_expiry", True)
    # eligible, confident, missed target, non-positive return at expiry -> lose_expiry
    assert ev.classify_walk_forward_decision(True, True, True, False, False, -0.01) == ("lose_expiry", True)
    assert ev.classify_walk_forward_decision(True, True, True, False, False, 0.0) == ("lose_expiry", True)


def test_trade_outcome_label():
    assert ev.trade_outcome_label(hit_take_profit=True, hit_stop_loss=True) == "stop_loss"
    assert ev.trade_outcome_label(hit_take_profit=True, hit_stop_loss=False) == "take_profit"
    assert ev.trade_outcome_label(hit_take_profit=False, hit_stop_loss=False) == "expired"


def test_outcome_breakdown_fractions_and_empty():
    labels = np.array(
        ["win_take_profit", "win_expiry", "lose_expiry", "lose_stop_loss", "skipped", "no_trade"]
    )
    breakdown = ev.outcome_breakdown(labels)

    assert breakdown["win_take_profit"] == pytest.approx(1 / 6)
    assert breakdown["win_expiry"] == pytest.approx(1 / 6)
    assert breakdown["lose_expiry"] == pytest.approx(1 / 6)
    assert breakdown["lose_stop_loss"] == pytest.approx(1 / 6)
    assert breakdown["skipped"] == pytest.approx(1 / 6)
    assert breakdown["no_trade"] == pytest.approx(1 / 6)
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


def test_naive_periodic_benchmark_tiles_nonoverlapping_horizon_bar_blocks():
    """closes strictly increasing; blocks tile HORIZON-bar windows starting right after
    t0. Expectations are derived from ev.HORIZON dynamically (not a hardcoded 3), since
    naive_periodic_benchmark's whole point is matching the walk-forward strategies' own
    cadence -- a hardcoded literal here is exactly the bug found (and fixed) during the
    steven4 HORIZON=1 migration audit. At HORIZON=1 this degenerates to buying and selling
    at the same bar's close every block (0 return per block, by construction), which is
    why win_rate isn't asserted to be a fixed 1.0 here -- it's derived the same way."""
    df = _fake_df(20)
    closes = df["close"].to_numpy()
    result = ev.naive_periodic_benchmark(df, closes, t0=0, test_hi=20)

    expected_buys, k = [], 0
    while True:
        buy_idx = ev.HORIZON * k + 1
        sell_idx = buy_idx + ev.HORIZON - 1
        if sell_idx >= 20:
            break
        expected_buys.append(buy_idx)
        k += 1
    expected_sells = [b + ev.HORIZON - 1 for b in expected_buys]

    assert result["n_trades"] == len(expected_buys)
    rets = [closes[s] / closes[b] - 1.0 for b, s in zip(expected_buys, expected_sells)]
    expected_return = 1.0
    for r in rets:
        expected_return *= 1.0 + r
    np.testing.assert_allclose(result["total_return"], expected_return - 1.0)
    assert result["win_rate"] == pytest.approx(float(np.mean([r > 0 for r in rets])))
    assert result["take_profit_rate"] is None


def test_naive_periodic_benchmark_no_room_for_a_trade():
    """test_hi == t0 + 1: even the very first block's buy bar (t0+1) already sits at or
    past test_hi, regardless of HORIZON -- guarantees zero valid blocks generically."""
    df = _fake_df(1)
    closes = df["close"].to_numpy()
    result = ev.naive_periodic_benchmark(df, closes, t0=0, test_hi=1)
    assert result["n_trades"] == 0
    assert result["win_rate"] is None
    assert result["total_return"] == 0.0


def _fake_predict_fn(take_profit: float, passes_quality_gate: bool):
    """A stand-in for make_patchtst_predict_fn/make_cvae_predict_fn's returned closure --
    same {take_profit, passes_quality_gate, price} contract, constant regardless of the
    window."""

    def predict(w):
        return {"take_profit": take_profit, "passes_quality_gate": passes_quality_gate, "price": np.zeros((ev.HORIZON, 4))}

    return predict


def test_run_walk_forward_no_trade_advances_one_bar_trade_advances_horizon():
    """A predict_fn that never clears eligibility should visit every start_idx from
    test_lo up to the last valid one (step 1 each time); one that's always eligible,
    passes its quality gate, and is above the min-return threshold should instead jump by
    HORIZON each time."""
    feat = np.zeros((200, 7), dtype=np.float32)
    opens = np.full(200, 100.0)
    closes = np.full(200, 100.0)
    test_lo, test_hi = 0, 200
    last_valid_start = test_hi - ev.WALK_FORWARD_CTX_BARS - ev.HORIZON

    never_eligible = _fake_predict_fn(take_profit=50.0, passes_quality_gate=True)  # < close_0 (100)
    result = ev.run_walk_forward(
        never_eligible, feat, opens, closes, test_lo, test_hi, min_return_threshold=0.0, stop_loss_pct=0.0,
    )
    visited = [d["start_idx"] for d in result["decisions"]]
    assert visited == list(range(test_lo, last_valid_start + 1))
    assert result["trades"] == []
    assert all(d["label"] == "no_trade" for d in result["decisions"])
    assert result["equity_final"] == 1.0

    always_trade = _fake_predict_fn(take_profit=150.0, passes_quality_gate=True)  # > close_0 (100)
    result_2 = ev.run_walk_forward(
        always_trade, feat, opens, closes, test_lo, test_hi, min_return_threshold=0.0, stop_loss_pct=0.0,
    )
    visited_2 = [d["start_idx"] for d in result_2["decisions"]]
    assert visited_2 == list(range(test_lo, last_valid_start + 1, ev.HORIZON))
    assert len(result_2["trades"]) == len(visited_2)


def test_run_walk_forward_min_return_threshold_gates_eligible_windows():
    """Eligible (target > buy) but the predicted edge is smaller than min_return_threshold
    -> 'skipped', never traded, and the walk advances 1 bar at a time just like a
    quality-gate-driven skip would."""
    feat = np.zeros((120, 7), dtype=np.float32)
    opens = np.full(120, 100.0)
    closes = np.full(120, 100.0)
    test_lo, test_hi = 0, 120

    tiny_edge = _fake_predict_fn(take_profit=100.05, passes_quality_gate=True)  # +0.05% predicted edge
    result = ev.run_walk_forward(
        tiny_edge, feat, opens, closes, test_lo, test_hi,
        min_return_threshold=0.001, stop_loss_pct=0.0,  # require >= 0.1%
    )
    assert result["trades"] == []
    assert all(d["label"] == "skipped" for d in result["decisions"])


def test_run_walk_forward_quality_gate_alone_can_skip_a_large_edge():
    """Eligible AND well above the return threshold, but passes_quality_gate=False (e.g.
    PatchTST's 3 bars disagree on direction, or CVAE's sample consensus doesn't clear its
    threshold) -> still 'skipped', not traded -- the two gates are independent."""
    feat = np.zeros((120, 7), dtype=np.float32)
    opens = np.full(120, 100.0)
    closes = np.full(120, 100.0)
    test_lo, test_hi = 0, 120

    big_edge_bad_gate = _fake_predict_fn(take_profit=150.0, passes_quality_gate=False)
    result = ev.run_walk_forward(
        big_edge_bad_gate, feat, opens, closes, test_lo, test_hi,
        min_return_threshold=0.001, stop_loss_pct=0.0,
    )
    assert result["trades"] == []
    assert all(d["label"] == "skipped" for d in result["decisions"])


def test_run_walk_forward_stop_loss_pct_cuts_off_a_large_loss_early():
    """A predict_fn whose take-profit (105) is never reached, on a window whose real
    horizon bar 0 crashes hard (~-15% open, ~-17% close, entirely below any reasonable
    stop) would otherwise ride out to the 3rd real candle's close -- a big uncapped loss.
    With stop_loss_pct set, bar 0's crash should gap straight through the stop and close
    the trade there instead, at a smaller (though still real) loss."""
    n = 80
    feat = np.zeros((n, 7), dtype=np.float32)
    ctx_bars = ev.WALK_FORWARD_CTX_BARS  # 70 -- horizon bars land at feat[70:73]
    feat[ctx_bars, 0] = -0.15  # bar 0 open_ret: log(open/close_0)
    feat[ctx_bars, 1] = -0.02  # bar 0 body_ret: close even lower than open
    feat[ctx_bars, 2] = 0.001  # bar 0 upper_wick
    feat[ctx_bars, 3] = 0.001  # bar 0 lower_wick
    opens = np.full(n, 84.0)  # anchor-corrected open for bars 1/2 -- stays crashed
    closes = np.full(n, 100.0)  # close_0 = closes[ctx_bars - 1] = 100.0

    # exactly one decision point, generically for any HORIZON: the loop's own condition is
    # start_idx + WALK_FORWARD_CTX_BARS + HORIZON <= test_hi, so test_hi = ctx_bars +
    # HORIZON admits start_idx=0 but not the next decision point (start_idx=HORIZON, which
    # needs ctx_bars + 2*HORIZON <= test_hi -- false whenever HORIZON >= 1). A "+1" here
    # (this test's pre-steven4 form) only coincidentally gave one decision at HORIZON=3;
    # at HORIZON=1 it admits a second decision point too -- see the migration audit.
    test_lo, test_hi = 0, ctx_bars + ev.HORIZON

    def never_hits_target(w):
        return {"take_profit": 105.0, "passes_quality_gate": True, "price": np.zeros((ev.HORIZON, 4))}

    no_stop_result = ev.run_walk_forward(
        never_hits_target, feat, opens, closes, test_lo, test_hi,
        min_return_threshold=0.0, stop_loss_pct=0.0,
    )
    stopped_result = ev.run_walk_forward(
        never_hits_target, feat, opens, closes, test_lo, test_hi,
        min_return_threshold=0.0, stop_loss_pct=0.02,
    )
    assert len(no_stop_result["trades"]) == len(stopped_result["trades"]) == 1
    no_stop_trade, stopped_trade = no_stop_result["trades"][0], stopped_result["trades"][0]

    assert no_stop_trade["label"] == "lose_expiry"
    assert stopped_trade["label"] == "lose_stop_loss"
    assert stopped_trade["hit_stop_loss"] is True
    # Both are real losses, but the stop-loss version must be a SMALLER loss than riding
    # the crash all the way to bar 2's close uncapped.
    assert stopped_trade["trade_return"] < 0
    assert stopped_trade["trade_return"] > no_stop_trade["trade_return"]


def test_walk_forward_stats_no_trades():
    df = _fake_df(100)
    result = {"trades": [], "decisions": [{"start_idx": 0, "label": "no_trade"}], "equity_final": 1.0}
    stats = ev.walk_forward_stats(df, result)
    assert stats["n_trades"] == 0
    assert stats["win_rate"] is None
    assert stats["total_return"] == 0.0
    assert stats["outcome_breakdown"] == {
        "win_take_profit": 0.0, "win_expiry": 0.0, "lose_expiry": 0.0, "lose_stop_loss": 0.0,
        "skipped": 0.0, "no_trade": 1.0,
    }
