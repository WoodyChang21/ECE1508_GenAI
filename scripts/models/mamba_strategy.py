"""Chronological strategy helpers shared by training and evaluation."""

from __future__ import annotations

import numpy as np


def compounded_return(returns: np.ndarray) -> float:
    return float(np.prod(1.0 + returns, dtype=np.float64) - 1.0)


def max_drawdown(returns: np.ndarray) -> float:
    equity = np.concatenate(
        [np.ones(1, dtype=np.float64), np.cumprod(1.0 + returns, dtype=np.float64)]
    )
    running_high = np.maximum.accumulate(equity)
    return float(np.min(equity / running_high - 1.0))


def timeline_from_horizon_windows(y_steps: np.ndarray) -> np.ndarray:
    """Recover one chronological return series from overlapping labels."""
    values = np.asarray(y_steps, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("horizon targets must be a non-empty 2D array")
    if values.shape[1] == 1:
        return values[:, 0].copy()
    return np.concatenate([values[:, 0], values[-1, 1:]])


def multi_period_strategy_metrics(
    y_true_steps: np.ndarray,
    y_pred_steps: np.ndarray,
    threshold_bps: float,
    transaction_cost_bps: float,
    periods_per_year: int,
    position_mode: str = "long_short",
    require_all_steps_agree: bool = False,
) -> dict[str, float | int | str | bool | None]:
    """Backtest non-overlapping fixed-horizon forecasts."""
    y_true_steps = np.asarray(y_true_steps, dtype=np.float64)
    y_pred_steps = np.asarray(y_pred_steps, dtype=np.float64)
    if (
        y_true_steps.shape != y_pred_steps.shape
        or y_true_steps.ndim != 2
        or y_true_steps.shape[0] == 0
        or y_true_steps.shape[1] < 2
    ):
        raise ValueError("step targets/predictions must be equal non-empty 2D arrays")
    if threshold_bps < 0 or transaction_cost_bps < 0:
        raise ValueError("thresholds and transaction costs must be non-negative")
    if periods_per_year < 1:
        raise ValueError("periods_per_year must be positive")
    if position_mode not in {"long_short", "long_only", "short_only"}:
        raise ValueError(
            "position_mode must be 'long_short', 'long_only', or 'short_only'"
        )

    horizon = y_true_steps.shape[1]
    actual_timeline = timeline_from_horizon_windows(y_true_steps)
    positions = np.zeros_like(actual_timeline)
    costs = np.zeros_like(actual_timeline)
    threshold = threshold_bps / 10_000.0
    one_way_cost = transaction_cost_bps / 10_000.0
    trades: list[tuple[int, int, float]] = []

    window = 0
    while window < len(y_pred_steps):
        predicted_path = y_pred_steps[window]
        predicted_return = float(np.prod(1.0 + predicted_path) - 1.0)
        direction = float(np.sign(predicted_return))
        selected = direction != 0.0 and abs(predicted_return) >= threshold
        if position_mode == "long_only":
            selected = selected and direction > 0
        elif position_mode == "short_only":
            selected = selected and direction < 0
        if require_all_steps_agree:
            selected = selected and bool((np.sign(predicted_path) == direction).all())

        if not selected:
            window += 1
            continue

        end = window + horizon
        positions[window:end] = direction
        costs[window] += one_way_cost
        costs[end - 1] += one_way_cost
        trades.append((window, end, direction))
        window = end

    gross_returns = positions * actual_timeline
    net_returns = gross_returns - costs
    active = positions != 0
    trade_returns = np.asarray(
        [compounded_return(net_returns[start:end]) for start, end, _ in trades],
        dtype=np.float64,
    )
    net_std = float(np.std(net_returns))
    sharpe = None
    if net_std > 0:
        sharpe = float(
            np.mean(net_returns) / net_std * np.sqrt(float(periods_per_year))
        )

    return {
        "position_mode": position_mode,
        "forecast_horizon": horizon,
        "signal_rule": (
            "cumulative_return_and_all_steps_agree"
            if require_all_steps_agree
            else "cumulative_return"
        ),
        "threshold_bps": float(threshold_bps),
        "trades": len(trades),
        "active_periods": int(active.sum()),
        "exposure": float(active.mean()),
        "position_turnover": float(2 * len(trades)),
        "directional_accuracy_when_active": (
            float(np.mean(np.sign(actual_timeline[active]) == positions[active]))
            if active.any()
            else None
        ),
        "net_trade_win_rate": (
            float(np.mean(trade_returns > 0)) if len(trade_returns) else None
        ),
        "mean_net_trade_return": (
            float(np.mean(trade_returns)) if len(trade_returns) else None
        ),
        "gross_compounded_return": compounded_return(gross_returns),
        "net_compounded_return": compounded_return(net_returns),
        "annualized_net_sharpe": sharpe,
        "net_max_drawdown": max_drawdown(net_returns),
    }


def persistent_long_strategy_metrics(
    y_true_steps: np.ndarray,
    y_pred_steps: np.ndarray,
    entry_threshold_bps: float,
    exit_threshold_bps: float,
    transaction_cost_bps: float,
    periods_per_year: int,
) -> dict[str, float | int | str | None]:
    """Backtest a receding-horizon long/cash policy without overlapping returns.

    At each candle, the compounded multi-step forecast determines the position for
    the next realized candle. A long position persists until the forecast falls to
    the exit threshold, so an unchanged signal does not create another round trip.
    """
    y_true_steps = np.asarray(y_true_steps, dtype=np.float64)
    y_pred_steps = np.asarray(y_pred_steps, dtype=np.float64)
    if (
        y_true_steps.shape != y_pred_steps.shape
        or y_true_steps.ndim != 2
        or y_true_steps.shape[0] == 0
        or y_true_steps.shape[1] < 2
    ):
        raise ValueError("step targets/predictions must be equal non-empty 2D arrays")
    if entry_threshold_bps < 0 or exit_threshold_bps < 0:
        raise ValueError("entry and exit thresholds must be non-negative")
    if exit_threshold_bps > entry_threshold_bps:
        raise ValueError("exit threshold must not exceed entry threshold")
    if transaction_cost_bps < 0:
        raise ValueError("transaction costs must be non-negative")
    if periods_per_year < 1:
        raise ValueError("periods_per_year must be positive")

    predicted_cumulative = np.prod(1.0 + y_pred_steps, axis=1) - 1.0
    actual_next = y_true_steps[:, 0]
    entry_threshold = entry_threshold_bps / 10_000.0
    exit_threshold = exit_threshold_bps / 10_000.0
    one_way_cost = transaction_cost_bps / 10_000.0
    positions = np.zeros(len(actual_next), dtype=np.float64)
    costs = np.zeros(len(actual_next), dtype=np.float64)
    turnover = np.zeros(len(actual_next), dtype=np.float64)
    trade_bounds: list[tuple[int, int]] = []
    position = 0.0
    trade_start: int | None = None

    for index, signal in enumerate(predicted_cumulative):
        next_position = position
        if position == 0.0 and signal >= entry_threshold:
            next_position = 1.0
            trade_start = index
        elif position == 1.0 and signal <= exit_threshold:
            next_position = 0.0

        change = abs(next_position - position)
        turnover[index] = change
        costs[index] = change * one_way_cost
        positions[index] = next_position
        if position == 1.0 and next_position == 0.0:
            assert trade_start is not None
            trade_bounds.append((trade_start, index + 1))
            trade_start = None
        position = next_position

    if position == 1.0:
        costs[-1] += one_way_cost
        turnover[-1] += 1.0
        assert trade_start is not None
        trade_bounds.append((trade_start, len(actual_next)))

    gross_returns = positions * actual_next
    net_returns = gross_returns - costs
    active = positions != 0.0
    trade_returns = np.asarray(
        [compounded_return(net_returns[start:end]) for start, end in trade_bounds],
        dtype=np.float64,
    )
    net_std = float(np.std(net_returns))
    sharpe = None
    if net_std > 0:
        sharpe = float(
            np.mean(net_returns) / net_std * np.sqrt(float(periods_per_year))
        )

    return {
        "position_mode": "long_only",
        "forecast_horizon": int(y_true_steps.shape[1]),
        "signal_rule": "persistent_cumulative_return_hysteresis",
        "entry_threshold_bps": float(entry_threshold_bps),
        "exit_threshold_bps": float(exit_threshold_bps),
        "trades": len(trade_bounds),
        "active_periods": int(active.sum()),
        "exposure": float(active.mean()),
        "position_turnover": float(turnover.sum()),
        "directional_accuracy_when_active": (
            float(np.mean(actual_next[active] > 0)) if active.any() else None
        ),
        "net_trade_win_rate": (
            float(np.mean(trade_returns > 0)) if len(trade_returns) else None
        ),
        "mean_net_trade_return": (
            float(np.mean(trade_returns)) if len(trade_returns) else None
        ),
        "gross_compounded_return": compounded_return(gross_returns),
        "net_compounded_return": compounded_return(net_returns),
        "annualized_net_sharpe": sharpe,
        "net_max_drawdown": max_drawdown(net_returns),
    }


def select_threshold(
    y_true_steps: np.ndarray,
    y_pred_steps: np.ndarray,
    thresholds_bps: list[float],
    transaction_cost_bps: float,
    periods_per_year: int,
    position_mode: str,
    require_all_steps_agree: bool,
    minimum_trades: int = 1,
    selection_metric: str = "annualized_net_sharpe",
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Select one threshold on pre-test calibration data."""
    if not thresholds_bps:
        raise ValueError("at least one strategy threshold is required")
    if minimum_trades < 1:
        raise ValueError("minimum_trades must be positive")
    allowed_metrics = {
        "annualized_net_sharpe",
        "net_compounded_return",
        "mean_net_trade_return",
    }
    if selection_metric not in allowed_metrics:
        raise ValueError(
            f"selection_metric must be one of {sorted(allowed_metrics)}"
        )
    results = [
        multi_period_strategy_metrics(
            y_true_steps,
            y_pred_steps,
            threshold,
            transaction_cost_bps,
            periods_per_year,
            position_mode=position_mode,
            require_all_steps_agree=require_all_steps_agree,
        )
        for threshold in thresholds_bps
    ]
    eligible = [
        row
        for row in results
        if row[selection_metric] is not None
        and int(row["trades"]) >= minimum_trades
    ]
    if eligible:
        best = max(eligible, key=lambda row: float(row[selection_metric]))
        recorded_selection_metric = selection_metric
    else:
        best = max(results, key=lambda row: int(row["trades"]))
        recorded_selection_metric = "fallback_most_trades"
    policy: dict[str, object] = {
        "position_mode": position_mode,
        "require_all_steps_agree": require_all_steps_agree,
        "threshold_bps": float(best["threshold_bps"]),
        "transaction_cost_bps": float(transaction_cost_bps),
        "periods_per_year": int(periods_per_year),
        "selection_metric": recorded_selection_metric,
        "minimum_calibration_trades": minimum_trades,
        "calibration_result": best,
    }
    return policy, results


def select_persistent_threshold(
    y_true_steps: np.ndarray,
    y_pred_steps: np.ndarray,
    thresholds_bps: list[float],
    exit_threshold_bps: float,
    transaction_cost_bps: float,
    periods_per_year: int,
    minimum_trades: int = 1,
    selection_metric: str = "annualized_net_sharpe",
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Select one persistent-policy entry threshold on calibration data."""
    if not thresholds_bps:
        raise ValueError("at least one strategy threshold is required")
    if minimum_trades < 1:
        raise ValueError("minimum_trades must be positive")
    allowed_metrics = {
        "annualized_net_sharpe",
        "net_compounded_return",
        "mean_net_trade_return",
    }
    if selection_metric not in allowed_metrics:
        raise ValueError(
            f"selection_metric must be one of {sorted(allowed_metrics)}"
        )
    results = [
        persistent_long_strategy_metrics(
            y_true_steps,
            y_pred_steps,
            entry_threshold_bps=threshold,
            exit_threshold_bps=exit_threshold_bps,
            transaction_cost_bps=transaction_cost_bps,
            periods_per_year=periods_per_year,
        )
        for threshold in thresholds_bps
        if threshold >= exit_threshold_bps
    ]
    if not results:
        raise ValueError("at least one entry threshold must reach the exit threshold")
    eligible = [
        row
        for row in results
        if row[selection_metric] is not None
        and int(row["trades"]) >= minimum_trades
    ]
    if eligible:
        best = max(eligible, key=lambda row: float(row[selection_metric]))
        recorded_selection_metric = selection_metric
    else:
        best = max(results, key=lambda row: int(row["trades"]))
        recorded_selection_metric = "fallback_most_trades"
    policy: dict[str, object] = {
        "strategy_type": "persistent_long_cash",
        "position_mode": "long_only",
        "entry_threshold_bps": float(best["entry_threshold_bps"]),
        "exit_threshold_bps": float(exit_threshold_bps),
        "transaction_cost_bps": float(transaction_cost_bps),
        "periods_per_year": int(periods_per_year),
        "selection_metric": recorded_selection_metric,
        "minimum_calibration_trades": minimum_trades,
        "calibration_result": best,
    }
    return policy, results
