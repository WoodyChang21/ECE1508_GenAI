import numpy as np
import pytest

from src.rolling_backtest import classify_signal, run_rolling_backtest


def test_classify_signal_thresholds():
    assert classify_signal(0.7, up_threshold=0.6, down_threshold=0.4) == "uptrend"
    assert classify_signal(0.6, up_threshold=0.6, down_threshold=0.4) == "uptrend"  # boundary, inclusive
    assert classify_signal(0.3, up_threshold=0.6, down_threshold=0.4) == "downtrend"
    assert classify_signal(0.4, up_threshold=0.6, down_threshold=0.4) == "downtrend"  # boundary, inclusive
    assert classify_signal(0.5, up_threshold=0.6, down_threshold=0.4) == "neutral"


def _scripted_predict_fn(signals: list[str]):
    """Returns each of `signals` in order, one per call; falls back to "neutral" once
    exhausted (safe default -- no test here should call it more times than scripted)."""
    it = iter(signals)

    def predict(w: dict) -> dict:
        return {"signal": next(it, "neutral"), "frac_up": None}

    return predict


def test_no_trade_when_never_uptrend():
    feat = np.zeros((10, 7), dtype=np.float32)
    opens = np.zeros(10)
    closes = np.full(10, 100.0)
    predict_fn = _scripted_predict_fn(["neutral", "downtrend", "neutral"])

    result = run_rolling_backtest(predict_fn, feat, opens, closes, test_lo=0, test_hi=8, ctx_bars=5, stop_loss_pct=0.01)

    assert result["trades"] == []
    assert all(d["label"] == "no_trade" for d in result["decisions"])
    assert result["equity_final"] == 1.0
    assert result["forced_close_at_end"] is False


def test_enter_hold_then_exit_on_downtrend_signal_even_when_profitable():
    """Illustrates the user's rule 2: a downtrend consensus call forces an exit
    regardless of current P&L -- entry at 100, exit at 102 (a real gain), purely because
    the model's next-bar call flipped to "downtrend", not because of any loss."""
    feat = np.zeros((10, 7), dtype=np.float32)  # every horizon bar reconstructs flat (no wicks/moves)
    opens = np.zeros(10)
    closes = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 101.0, 102.0, 100.0, 100.0, 100.0])
    predict_fn = _scripted_predict_fn(["uptrend", "uptrend", "downtrend"])

    result = run_rolling_backtest(predict_fn, feat, opens, closes, test_lo=0, test_hi=8, ctx_bars=5, stop_loss_pct=0.01)

    labels = [d["label"] for d in result["decisions"]]
    assert labels == ["enter", "hold", "exit_downtrend_signal"]
    assert len(result["trades"]) == 1
    trade = result["trades"][0]
    assert trade["entry_price"] == 100.0
    assert trade["sell_price"] == 102.0
    assert trade["trade_return"] == pytest.approx(0.02)
    assert result["equity_final"] == pytest.approx(1.02)
    assert result["forced_close_at_end"] is False


def test_divergence_loss_forces_exit_even_with_an_uptrend_signal():
    """Rule 1: a real-price divergence loss forces an exit even if the freshly computed
    signal still says "uptrend" -- divergence is checked before the signal, not instead
    of it."""
    feat = np.zeros((10, 7), dtype=np.float32)
    opens = np.zeros(10)
    closes = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 95.0, 100.0, 100.0, 100.0, 100.0])
    predict_fn = _scripted_predict_fn(["uptrend", "uptrend"])

    result = run_rolling_backtest(predict_fn, feat, opens, closes, test_lo=0, test_hi=7, ctx_bars=5, stop_loss_pct=0.01)

    labels = [d["label"] for d in result["decisions"]]
    assert labels == ["enter", "exit_divergence_loss"]
    trade = result["trades"][0]
    assert trade["sell_price"] == 95.0
    assert trade["trade_return"] == pytest.approx(-0.05)


def test_hard_stop_loss_fires_before_divergence_or_signal_checks():
    """The bar realized right after entry gaps/touches through the 1% stop intrabar --
    must fill at the stop level (99.0), not ride through to the bar's own close, and must
    fire regardless of what the freshly computed signal says."""
    feat = np.zeros((10, 7), dtype=np.float32)
    feat[5, 3] = 0.02  # lower_wick on the bar predicted at step 0 -- low = 100*exp(-0.02) ~= 98.02
    opens = np.zeros(10)
    closes = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 98.0, 100.0, 100.0, 100.0, 100.0])
    predict_fn = _scripted_predict_fn(["uptrend", "uptrend"])

    result = run_rolling_backtest(predict_fn, feat, opens, closes, test_lo=0, test_hi=7, ctx_bars=5, stop_loss_pct=0.01)

    labels = [d["label"] for d in result["decisions"]]
    assert labels == ["enter", "exit_stop_loss"]
    trade = result["trades"][0]
    assert trade["sell_price"] == 99.0  # entry_price * (1 - 0.01), not the bar's own close
    assert trade["trade_return"] == pytest.approx(-0.01)


def test_forced_close_at_end_of_period_when_still_holding():
    feat = np.zeros((10, 7), dtype=np.float32)
    opens = np.zeros(10)
    closes = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 103.0, 100.0, 100.0, 100.0, 100.0])
    predict_fn = _scripted_predict_fn(["uptrend", "neutral"])

    result = run_rolling_backtest(predict_fn, feat, opens, closes, test_lo=0, test_hi=7, ctx_bars=5, stop_loss_pct=0.01)

    assert result["forced_close_at_end"] is True
    labels = [d["label"] for d in result["decisions"]]
    assert labels == ["enter", "hold"]
    assert len(result["trades"]) == 1
    trade = result["trades"][0]
    assert trade["label"] == "exit_end_of_period"
    assert trade["sell_price"] == 103.0
    assert trade["trade_return"] == pytest.approx(0.03)
