"""Rolling hour-by-hour backtest (steven4) -- replaces the fixed-HORIZON-bar bracket order
(see the PRE-PIVOT note atop evaluate.py) with an open-ended hold: predict one candle at a
time, re-predict every real hour as new data arrives, buy on an "uptrend" consensus call,
keep holding while nothing forces an exit, and sell on a real-price divergence loss, a
"downtrend" consensus call, or a hard stop-loss.

Per-step mechanics (one real hour per step, see run_rolling_backtest):
  not holding, signal == "uptrend" -> buy at close_0.
  holding, in order of priority:
    1. stop-loss: did the bar just realized (predicted last step, real now) touch/gap
       through entry_price * (1 - stop_loss_pct)? -- reuses evaluate.bracket_exit's
       touched-vs-gapped fill logic directly (HORIZON=1 makes it a single-bar check
       already; no need for a separate generalized version).
    2. divergence loss: is that just-realized bar's close below entry_price? (the
       "uptrend" call that triggered entry didn't pan out)
    3. downtrend consensus for the *next* bar (computed this step) -> forced exit
       regardless of current P&L.
    otherwise: keep holding.
Clock always advances exactly 1 real hour, trade or not.

Trade signal = consensus fraction of k sampled draws predicting a positive close return
(see classify_signal) -- NOT a deterministic point estimate. This deliberately re-tests a
gate structurally similar to one already killed for CVAE's old bracket-order strategy
(cvae_direction_collapse.md / backlog.md's "trade confidence" entry: the unanimous-
agreement bucket was the worst-performing one) -- see classify_signal's docstring and
steven/rolling_hour_backtest.md for why this might behave differently here, and why the
result shouldn't be trusted at face value just because it's a different framing.
"""

from __future__ import annotations

import numpy as np
import torch

from src.data_pipeline import HORIZON, build_window, per_bar_close_return, reconstruct_prices

RB_CASE_LABELS = [
    "enter", "hold", "no_trade", "exit_stop_loss", "exit_divergence_loss", "exit_downtrend_signal",
]


def classify_signal(frac_up: float, up_threshold: float, down_threshold: float) -> str:
    """frac_up: fraction of k sampled draws predicting a positive close return (see
    per_bar_close_return). >= up_threshold -> "uptrend" (entry signal); <= down_threshold
    -> "downtrend" (forced-exit signal); otherwise "neutral" (no effect on its own --
    doesn't trigger entry, doesn't force an exit while holding).

    This is a consensus-across-k-samples gate, structurally similar to one already A/B
    tested and killed for CVAE's old bracket-order strategy (the unanimous-agreement
    bucket was the *worst*-performing one there, not just unreliable -- see
    cvae_direction_collapse.md / backlog.md's "trade confidence" entry). Used here anyway,
    per discussion with the user: this is a genuinely different usage pattern (evaluated
    every hour to drive continuous entry/hold/exit decisions, not a one-shot filter on a
    single fixed-horizon bet), so it may behave differently -- but that's a hypothesis to
    check against the result, not something to assume going in. See
    steven/rolling_hour_backtest.md."""
    if frac_up >= up_threshold:
        return "uptrend"
    if frac_up <= down_threshold:
        return "downtrend"
    return "neutral"


def make_cvae_rolling_predict_fn(cvae, device: torch.device, num_samples: int, up_threshold: float, down_threshold: float):
    """Returns a predict(w) -> {"signal", "frac_up", "price"} closure, single-window
    (batch=1) at a time. `price` is the prior-mean (mu_p) deterministic decode -- kept only
    for optional plotting/logging, NOT used to compute the signal itself (frac_up is;
    see classify_signal)."""

    def predict(w: dict) -> dict:
        masked_t = torch.from_numpy(w["masked_tensor"])[None].to(device)
        with torch.no_grad():
            price_samples_t, _ = cvae.sample(masked_t, k=num_samples)  # (K,1,HORIZON,4)
            mu_p, logvar_p, ctx_repr = cvae.encode_prior(masked_t)
            point_price, _, _, _ = cvae.decode(mu_p, ctx_repr)
        close_ret = per_bar_close_return(price_samples_t.cpu().numpy())[:, 0, 0]  # (K,) -- bar 0
        frac_up = float((close_ret > 0).mean())
        return {
            "signal": classify_signal(frac_up, up_threshold, down_threshold),
            "frac_up": frac_up,
            "price": point_price.cpu().numpy()[0],  # (HORIZON,4)
        }

    return predict


def run_rolling_backtest(
    predict_fn, feat: np.ndarray, opens: np.ndarray, closes: np.ndarray,
    test_lo: int, test_hi: int, ctx_bars: int, stop_loss_pct: float,
) -> dict:
    """predict_fn(w) -> {"signal", "frac_up", ...} -- see make_cvae_rolling_predict_fn.
    Returns {"trades", "decisions", "equity_final", "forced_close_at_end"}: `decisions` is
    every real-hour step visited (one per HORIZON=1 window), each carrying a label from
    RB_CASE_LABELS; `trades` is the subset that actually opened+closed a position (a
    still-open position at test_hi is force-closed at the last available real close for
    accounting -- see forced_close_at_end)."""
    from src.evaluate import bracket_exit  # local import: evaluate.py imports a lot; avoid pulling that in at module load for a function only needed here

    start_idx = test_lo
    equity = 1.0
    trades: list[dict] = []
    decisions: list[dict] = []
    holding = False
    entry_price = None
    entry_bar_idx = None
    prev_true_ohlc = None
    last_bar_idx = None
    last_close_0 = None

    while start_idx + ctx_bars + HORIZON <= test_hi:
        w = build_window(feat, opens, closes, start_idx, ctx_bars)
        close_0 = w["close_0"]
        bar_idx = start_idx + ctx_bars - 1  # absolute index of the bar whose close is close_0
        true_price = w["y"][: HORIZON * 4].reshape(1, HORIZON, 4)
        true_ohlc = reconstruct_prices(true_price, np.array([close_0]))[0]  # this step's horizon bar

        pred = predict_fn(w)
        signal = pred["signal"]
        label, sold, sell_price = None, False, None

        if holding:
            stop_level = entry_price * (1.0 - stop_loss_pct) if stop_loss_pct > 0 else -np.inf
            sp_arr, _, hit_sl_arr = bracket_exit(
                prev_true_ohlc[None], np.array([np.inf]), np.array([stop_level])
            )
            if bool(hit_sl_arr[0]):
                sell_price, label, sold = float(sp_arr[0]), "exit_stop_loss", True
            elif close_0 < entry_price:  # just-realized bar's close (== this step's close_0) diverged
                sell_price, label, sold = close_0, "exit_divergence_loss", True
            elif signal == "downtrend":
                sell_price, label, sold = close_0, "exit_downtrend_signal", True
            else:
                label = "hold"

            if sold:
                trade_return = sell_price / entry_price - 1.0
                equity *= 1.0 + trade_return
                trades.append({
                    "entry_bar_idx": entry_bar_idx, "entry_price": entry_price,
                    "exit_bar_idx": bar_idx, "sell_price": sell_price,
                    "trade_return": trade_return, "label": label,
                })
                holding, entry_price, entry_bar_idx = False, None, None
        else:
            if signal == "uptrend":
                holding, entry_price, entry_bar_idx, label = True, close_0, bar_idx, "enter"
            else:
                label = "no_trade"

        decisions.append({
            "start_idx": start_idx, "bar_idx": bar_idx, "close_0": close_0,
            "signal": signal, "frac_up": pred.get("frac_up"), "label": label,
        })

        prev_true_ohlc, last_bar_idx, last_close_0 = true_ohlc, bar_idx, close_0
        start_idx += 1

    forced_close_at_end = holding
    if holding:
        sell_price = last_close_0
        trade_return = sell_price / entry_price - 1.0
        equity *= 1.0 + trade_return
        trades.append({
            "entry_bar_idx": entry_bar_idx, "entry_price": entry_price,
            "exit_bar_idx": last_bar_idx, "sell_price": sell_price,
            "trade_return": trade_return, "label": "exit_end_of_period",
        })

    return {"trades": trades, "decisions": decisions, "equity_final": equity, "forced_close_at_end": forced_close_at_end}


def rolling_backtest_stats(df, result: dict) -> dict:
    """Mirrors evaluate.walk_forward_stats' reportable shape (see its own docstring) but
    over RB_CASE_LABELS instead of the fixed-bracket CASE_LABELS -- outcome_breakdown
    fractions are over `decisions` (RB_CASE_LABELS only; "exit_end_of_period" isn't a
    per-step label, see forced_close_at_end instead)."""
    from src.evaluate import equity_stats

    trades, decisions, equity_final = result["trades"], result["decisions"], result["equity_final"]
    labels = np.array([d["label"] for d in decisions], dtype=object)
    n = len(labels)
    out = {
        "n_decisions": n,
        "n_trades": len(trades),
        "forced_close_at_end": result["forced_close_at_end"],
        "outcome_breakdown": (
            {case: float((labels == case).mean()) for case in RB_CASE_LABELS} if n else {}
        ),
        "total_return": equity_final - 1.0,
    }
    if not trades:
        out.update(win_rate=None, avg_return=None)
        return out

    rets = np.array([t["trade_return"] for t in trades])
    out["win_rate"] = float((rets > 0).mean())
    out["avg_return"] = float(rets.mean())

    first, last = trades[0], trades[-1]
    out.update(equity_stats(
        df, first["entry_bar_idx"], last["exit_bar_idx"], first["entry_price"], last["sell_price"], out["total_return"],
    ))
    return out
