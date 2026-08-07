"""Evaluate a saved Mamba checkpoint or prediction file.

Examples:
    python scripts/models/evaluate_mamba.py \
        --checkpoint data/checkpoints/mamba.pt

    python scripts/models/evaluate_mamba.py \
        --predictions data/predictions/mamba_preds.parquet

Checkpoint mode restores the exact architecture, scaler, lookback, and
validation-calibrated interval offsets written by ``train_mamba.py``. Prediction
mode evaluates the Parquet file that training always writes, so it also works
when a Colab run finished without ``--checkpoint``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Support both ``python -m scripts.models.evaluate_mamba`` and direct execution.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.models.mamba_data import (  # noqa: E402
    FEATURE_COLUMNS,
    TARGET_INDEX,
    ZScoreScaler,
    load_split,
    with_history,
)
from scripts.models.metrics import (  # noqa: E402
    directional_accuracy,
    interval_coverage,
    mae,
    rmse,
)
from scripts.models.train_mamba import (  # noqa: E402
    ModelOptions,
    compound_horizon,
    make_model,
    predict,
    resolve_device,
)


DEFAULT_THRESHOLDS_BPS = (0.0, 1.0, 2.0, 5.0, 10.0)
REQUIRED_CHECKPOINT_KEYS = {
    "state_dict",
    "model_options",
    "lookback",
    "feature_columns",
    "scaler_mean",
    "scaler_scale",
    "interval_offsets",
}
REQUIRED_INTERVAL_KEYS = {"lo_80", "hi_80", "lo_90", "hi_90"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint")
    source.add_argument(
        "--predictions",
        help="Existing prediction Parquet written by train_mamba.py.",
    )
    parser.add_argument("--train", default="data/splits/train.parquet")
    parser.add_argument(
        "--history",
        default="data/splits/val.parquet",
        help="Rows immediately before the test period, used only as causal context.",
    )
    parser.add_argument("--test", default="data/splits/test.parquet")
    parser.add_argument(
        "--predictions-out",
        default="data/predictions/mamba_eval.parquet",
        help="Prediction output in checkpoint mode.",
    )
    parser.add_argument("--metrics-out", default="data/predictions/mamba_eval.json")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--thresholds-bps",
        nargs="+",
        type=float,
        default=list(DEFAULT_THRESHOLDS_BPS),
        help="Trade only when |predicted return| is at least this many basis points.",
    )
    parser.add_argument(
        "--transaction-cost-bps",
        type=float,
        default=0.0,
        help="One-way cost per unit of position turnover, in basis points.",
    )
    parser.add_argument(
        "--periods-per-year",
        type=int,
        default=1764,
        help="Annualization factor (7 SPY hourly bars x 252 trading days).",
    )
    parser.add_argument(
        "--limit-test",
        type=int,
        default=None,
        help="Evaluate only the first N test rows; intended for smoke tests.",
    )
    return parser.parse_args()


def _torch_load(path: str | Path, device: torch.device) -> dict[str, Any]:
    """Load a trusted, locally produced training checkpoint."""
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # Compatibility with older supported PyTorch releases.
        checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must contain a dictionary")
    return checkpoint


def validate_checkpoint(checkpoint: dict[str, Any]) -> None:
    missing = sorted(REQUIRED_CHECKPOINT_KEYS - checkpoint.keys())
    if missing:
        raise ValueError(f"checkpoint is missing keys: {missing}")

    checkpoint_features = tuple(checkpoint["feature_columns"])
    if checkpoint_features != FEATURE_COLUMNS:
        raise ValueError(
            "checkpoint feature columns/order do not match this checkout: "
            f"expected {list(FEATURE_COLUMNS)}, got {list(checkpoint_features)}"
        )

    lookback = int(checkpoint["lookback"])
    if lookback < 1:
        raise ValueError("checkpoint lookback must be positive")
    forecast_horizon = int(checkpoint.get("forecast_horizon", 1))
    if forecast_horizon < 1:
        raise ValueError("checkpoint forecast_horizon must be positive")

    mean = np.asarray(checkpoint["scaler_mean"])
    scale = np.asarray(checkpoint["scaler_scale"])
    expected_shape = (len(FEATURE_COLUMNS),)
    if mean.shape != expected_shape or scale.shape != expected_shape:
        raise ValueError(
            f"checkpoint scaler arrays must have shape {expected_shape}"
        )
    if not np.isfinite(mean).all() or not np.isfinite(scale).all():
        raise ValueError("checkpoint scaler contains non-finite values")
    if np.any(scale <= 0):
        raise ValueError("checkpoint scaler scale values must be positive")

    offsets = checkpoint["interval_offsets"]
    if not isinstance(offsets, dict):
        raise ValueError("checkpoint interval_offsets must be a dictionary")
    missing_offsets = sorted(REQUIRED_INTERVAL_KEYS - offsets.keys())
    if missing_offsets:
        raise ValueError(
            f"checkpoint interval_offsets is missing keys: {missing_offsets}"
        )


def load_prediction_frame(path: str | Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    required = {"datetime", "y", "pred", "lo_80", "hi_80", "lo_90", "hi_90"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"prediction file is missing columns: {missing}")
    if len(frame) == 0:
        raise ValueError("prediction file is empty")
    values = frame.loc[:, ["y", "pred", "lo_80", "hi_80", "lo_90", "hi_90"]]
    if not np.isfinite(values.to_numpy(dtype=np.float64)).all():
        raise ValueError("prediction file contains non-finite evaluation values")
    return frame.reset_index(drop=True)


def horizon_arrays(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Read aligned per-step targets/predictions, with legacy one-step fallback."""
    target_steps = {
        int(column[3:])
        for column in frame.columns
        if column.startswith("y_h") and column[3:].isdigit()
    }
    prediction_steps = {
        int(column[6:])
        for column in frame.columns
        if column.startswith("pred_h") and column[6:].isdigit()
    }
    if not target_steps and not prediction_steps:
        return (
            frame["y"].to_numpy(dtype=np.float64).reshape(-1, 1),
            frame["pred"].to_numpy(dtype=np.float64).reshape(-1, 1),
        )
    if target_steps != prediction_steps:
        raise ValueError("prediction file has mismatched target/prediction horizon steps")
    expected_steps = set(range(1, max(target_steps) + 1))
    if target_steps != expected_steps:
        raise ValueError("prediction file horizon steps must be contiguous from h1")
    target_columns = [f"y_h{step}" for step in sorted(target_steps)]
    prediction_columns = [f"pred_h{step}" for step in sorted(target_steps)]
    values = frame.loc[:, target_columns + prediction_columns]
    if not np.isfinite(values.to_numpy(dtype=np.float64)).all():
        raise ValueError("prediction file contains non-finite horizon values")
    return (
        frame.loc[:, target_columns].to_numpy(dtype=np.float64),
        frame.loc[:, prediction_columns].to_numpy(dtype=np.float64),
    )


def point_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | None]:
    """Return scale, sign, bias, and correlation metrics for a point forecast."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.shape != y_pred.shape or y_true.ndim != 1 or len(y_true) == 0:
        raise ValueError("y_true and y_pred must be non-empty 1D arrays of equal shape")

    true_std = float(np.std(y_true))
    pred_std = float(np.std(y_pred))
    correlation = None
    if true_std > 0 and pred_std > 0:
        correlation = float(np.corrcoef(y_true, y_pred)[0, 1])

    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "directional_accuracy": directional_accuracy(y_true, y_pred),
        "mean_error": float(np.mean(y_pred - y_true)),
        "correlation": correlation,
        "prediction_std": pred_std,
        "actual_std": true_std,
    }


def interval_metrics(
    y_true: np.ndarray,
    lo_80: np.ndarray,
    hi_80: np.ndarray,
    lo_90: np.ndarray,
    hi_90: np.ndarray,
) -> dict[str, float]:
    return {
        "coverage_80": interval_coverage(y_true, lo_80, hi_80),
        "mean_width_80": float(np.mean(hi_80 - lo_80)),
        "coverage_90": interval_coverage(y_true, lo_90, hi_90),
        "mean_width_90": float(np.mean(hi_90 - lo_90)),
    }


def _compounded_return(returns: np.ndarray) -> float:
    return float(np.prod(1.0 + returns, dtype=np.float64) - 1.0)


def _max_drawdown(returns: np.ndarray) -> float:
    equity = np.concatenate(
        [np.ones(1, dtype=np.float64), np.cumprod(1.0 + returns, dtype=np.float64)]
    )
    running_high = np.maximum.accumulate(equity)
    return float(np.min(equity / running_high - 1.0))


def strategy_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold_bps: float,
    transaction_cost_bps: float,
    periods_per_year: int,
    position_mode: str = "long_short",
) -> dict[str, float | int | str | None]:
    """Evaluate a chronological one-period signal for the requested position mode.

    Position at row t is sign(pred_t), unless the prediction magnitude is below
    the threshold. ``long_only`` discards negative predictions, ``short_only``
    discards positive predictions, and ``long_short`` keeps both. Costs are charged
    on absolute position turnover, so a direct long-to-short reversal costs twice
    as much as entering from flat. The final position is closed after the final
    observed return.
    """
    if threshold_bps < 0 or transaction_cost_bps < 0:
        raise ValueError("thresholds and transaction costs must be non-negative")
    if periods_per_year < 1:
        raise ValueError("periods_per_year must be positive")
    if position_mode not in {"long_short", "long_only", "short_only"}:
        raise ValueError(
            "position_mode must be 'long_short', 'long_only', or 'short_only'"
        )

    threshold = threshold_bps / 10_000.0
    positions = np.where(np.abs(y_pred) >= threshold, np.sign(y_pred), 0.0)
    if position_mode == "long_only":
        positions = np.where(positions > 0, positions, 0.0)
    elif position_mode == "short_only":
        positions = np.where(positions < 0, positions, 0.0)
    active = positions != 0
    turnover = np.abs(np.diff(np.concatenate([[0.0], positions])))
    costs = turnover * (transaction_cost_bps / 10_000.0)
    # Realize the cost of closing any remaining position at the end of the test.
    if len(costs):
        costs[-1] += abs(positions[-1]) * (transaction_cost_bps / 10_000.0)

    gross_returns = positions * y_true
    net_returns = gross_returns - costs
    net_std = float(np.std(net_returns))
    sharpe = None
    if net_std > 0:
        sharpe = float(
            np.mean(net_returns) / net_std * np.sqrt(float(periods_per_year))
        )

    return {
        "position_mode": position_mode,
        "threshold_bps": float(threshold_bps),
        "active_periods": int(active.sum()),
        "exposure": float(active.mean()),
        "position_turnover": float(turnover.sum() + abs(positions[-1])),
        "directional_accuracy_when_active": (
            float(np.mean(np.sign(y_true[active]) == positions[active]))
            if active.any()
            else None
        ),
        "gross_compounded_return": _compounded_return(gross_returns),
        "net_compounded_return": _compounded_return(net_returns),
        "annualized_net_sharpe": sharpe,
        "net_max_drawdown": _max_drawdown(net_returns),
    }


def _timeline_from_horizon_windows(y_steps: np.ndarray) -> np.ndarray:
    """Recover the chronological return series from overlapping horizon labels."""
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
    """Backtest fixed-horizon forecasts with at most one open position.

    A forecast is considered whenever the strategy is flat. If selected, the
    position is held for the complete forecast horizon and the next decision is
    made only after it closes. This prevents the overlapping-window double
    counting present in many multi-horizon diagnostic backtests.
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
    if threshold_bps < 0 or transaction_cost_bps < 0:
        raise ValueError("thresholds and transaction costs must be non-negative")
    if periods_per_year < 1:
        raise ValueError("periods_per_year must be positive")
    if position_mode not in {"long_short", "long_only", "short_only"}:
        raise ValueError(
            "position_mode must be 'long_short', 'long_only', or 'short_only'"
        )

    horizon = y_true_steps.shape[1]
    actual_timeline = _timeline_from_horizon_windows(y_true_steps)
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
        [
            _compounded_return(net_returns[start:end])
            for start, end, _ in trades
        ],
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
        "gross_compounded_return": _compounded_return(gross_returns),
        "net_compounded_return": _compounded_return(net_returns),
        "annualized_net_sharpe": sharpe,
        "net_max_drawdown": _max_drawdown(net_returns),
    }


def build_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    lo_80: np.ndarray,
    hi_80: np.ndarray,
    lo_90: np.ndarray,
    hi_90: np.ndarray,
    previous_return: float,
    trainval_mean_return: float,
    thresholds_bps: list[float],
    transaction_cost_bps: float,
    periods_per_year: int,
    y_true_steps: np.ndarray | None = None,
    y_pred_steps: np.ndarray | None = None,
) -> dict[str, Any]:
    if y_true_steps is None:
        y_true_steps = np.asarray(y_true, dtype=np.float64).reshape(-1, 1)
    if y_pred_steps is None:
        y_pred_steps = np.asarray(y_pred, dtype=np.float64).reshape(-1, 1)
    y_true_steps = np.asarray(y_true_steps, dtype=np.float64)
    y_pred_steps = np.asarray(y_pred_steps, dtype=np.float64)
    if y_true_steps.shape != y_pred_steps.shape or y_true_steps.ndim != 2:
        raise ValueError("step targets/predictions must be equal 2D arrays")

    horizon = y_true_steps.shape[1]
    zero_pred = np.zeros_like(y_true)
    mean_cumulative_return = (1.0 + trainval_mean_return) ** horizon - 1.0
    mean_pred = np.full_like(y_true, mean_cumulative_return)
    lag_one_step = np.concatenate([[previous_return], y_true_steps[:-1, 0]])
    lag_one_steps = np.repeat(lag_one_step[:, None], horizon, axis=1)
    lag_one_pred = compound_horizon(lag_one_steps)

    model_metrics = point_metrics(y_true, y_pred)
    zero_metrics = point_metrics(y_true, zero_pred)
    mean_metrics = point_metrics(y_true, mean_pred)
    lag_metrics = point_metrics(y_true, lag_one_pred)
    baseline_mae = float(zero_metrics["mae"])
    baseline_rmse = float(zero_metrics["rmse"])

    report: dict[str, Any] = {
        "forecast": {
            "horizon_periods": horizon,
            "mamba": model_metrics,
            "baselines": {
                "zero_return": zero_metrics,
                "trainval_mean_return": mean_metrics,
                "lag_one_return": lag_metrics,
            },
            "skill_vs_zero_return": {
                "mae": 1.0 - float(model_metrics["mae"]) / baseline_mae,
                "rmse": 1.0 - float(model_metrics["rmse"]) / baseline_rmse,
            },
        },
        "prediction_intervals": interval_metrics(
            y_true, lo_80, hi_80, lo_90, hi_90
        ),
    }

    if horizon == 1:
        report["sign_strategy"] = {
            "note": (
                "Chronological sign strategies reported three ways: combined long/short, "
                "long-only, and short-only. Returns are compounded in test order. A "
                "one-way cost is charged on absolute position turnover; a direct "
                "long/short reversal has turnover 2. This is a diagnostic backtest, not "
                "an execution simulation."
            ),
            "transaction_cost_bps": float(transaction_cost_bps),
            "periods_per_year": periods_per_year,
            "baselines": {
                "always_long": strategy_metrics(
                    y_true,
                    np.ones_like(y_true),
                    0.0,
                    transaction_cost_bps,
                    periods_per_year,
                    position_mode="long_only",
                ),
                "lag_one_return": strategy_metrics(
                    y_true,
                    lag_one_pred,
                    0.0,
                    transaction_cost_bps,
                    periods_per_year,
                ),
            },
            "threshold_sweep": [
                strategy_metrics(
                    y_true,
                    y_pred,
                    threshold,
                    transaction_cost_bps,
                    periods_per_year,
                )
                for threshold in thresholds_bps
            ],
            "long_only_threshold_sweep": [
                strategy_metrics(
                    y_true,
                    y_pred,
                    threshold,
                    transaction_cost_bps,
                    periods_per_year,
                    position_mode="long_only",
                )
                for threshold in thresholds_bps
            ],
            "short_only_threshold_sweep": [
                strategy_metrics(
                    y_true,
                    y_pred,
                    threshold,
                    transaction_cost_bps,
                    periods_per_year,
                    position_mode="short_only",
                )
                for threshold in thresholds_bps
            ],
        }
        return report

    actual_timeline = _timeline_from_horizon_windows(y_true_steps)
    strategy: dict[str, Any] = {
        "note": (
            f"Chronological fixed-{horizon}-candle strategies. A signal is considered "
            "only while flat; an entered position is held for the full forecast horizon, "
            "so trades never overlap. Entry and exit each incur the configured one-way "
            "cost. The cumulative variants use the compounded predicted path; the "
            "agreement variants additionally require every predicted step to have the "
            "same sign. There is no take-profit because this model predicts returns, "
            "not future OHLC ranges."
        ),
        "forecast_horizon": horizon,
        "transaction_cost_bps": float(transaction_cost_bps),
        "periods_per_year": periods_per_year,
        "baselines": {
            "always_long": strategy_metrics(
                actual_timeline,
                np.ones_like(actual_timeline),
                0.0,
                transaction_cost_bps,
                periods_per_year,
                position_mode="long_only",
            ),
            "lag_one_return": multi_period_strategy_metrics(
                y_true_steps,
                lag_one_steps,
                0.0,
                transaction_cost_bps,
                periods_per_year,
            ),
        },
    }
    sweep_specs = {
        "threshold_sweep": ("long_short", False),
        "long_only_threshold_sweep": ("long_only", False),
        "short_only_threshold_sweep": ("short_only", False),
        "all_steps_agree_threshold_sweep": ("long_short", True),
        "all_steps_positive_threshold_sweep": ("long_only", True),
    }
    for name, (position_mode, require_agreement) in sweep_specs.items():
        strategy[name] = [
            multi_period_strategy_metrics(
                y_true_steps,
                y_pred_steps,
                threshold,
                transaction_cost_bps,
                periods_per_year,
                position_mode=position_mode,
                require_all_steps_agree=require_agreement,
            )
            for threshold in thresholds_bps
        ]
    report["multi_candle_strategy"] = strategy
    return report


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    if args.limit_test is not None and args.limit_test < 1:
        raise ValueError("limit-test must be positive")
    if any(threshold < 0 for threshold in args.thresholds_bps):
        raise ValueError("thresholds must be non-negative")

    _, history_values = load_split(args.history)
    previous_return = float(history_values[-1, TARGET_INDEX])

    if args.checkpoint:
        device = resolve_device(args.device)
        checkpoint = _torch_load(args.checkpoint, device)
        validate_checkpoint(checkpoint)

        options = ModelOptions(**checkpoint["model_options"])
        lookback = int(checkpoint["lookback"])
        forecast_horizon = int(checkpoint.get("forecast_horizon", 1))
        scaler = ZScoreScaler(
            mean=np.asarray(checkpoint["scaler_mean"], dtype=np.float32),
            scale=np.asarray(checkpoint["scaler_scale"], dtype=np.float32),
        )
        model = make_model(options, forecast_horizon).to(device)
        model.load_state_dict(checkpoint["state_dict"], strict=True)

        test_frame, test_values = load_split(args.test)
        if args.limit_test is not None:
            test_frame = test_frame.iloc[: args.limit_test].reset_index(drop=True)
            test_values = test_values[: args.limit_test]
        if len(history_values) < lookback:
            raise ValueError(
                f"history has {len(history_values)} rows but checkpoint needs {lookback}"
            )
        if len(test_values) < forecast_horizon:
            raise ValueError(
                f"test split needs at least {forecast_horizon} rows for this checkpoint"
            )

        history_scaled = scaler.transform(history_values)
        test_scaled = scaler.transform(test_values)
        test_set = with_history(
            history_scaled,
            test_scaled,
            lookback=lookback,
            forecast_horizon=forecast_horizon,
        )
        loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
        pred_norm, true_norm = predict(model, loader, device)
        y_pred_steps = scaler.inverse_target(pred_norm)
        y_true_steps = scaler.inverse_target(true_norm)
        if forecast_horizon == 1:
            y_pred_steps = np.asarray(y_pred_steps).reshape(-1, 1)
            y_true_steps = np.asarray(y_true_steps).reshape(-1, 1)
        y_pred = compound_horizon(y_pred_steps)
        y_true = compound_horizon(y_true_steps)

        offsets = {
            key: float(checkpoint["interval_offsets"][key])
            for key in REQUIRED_INTERVAL_KEYS
        }
        lo_80 = y_pred + offsets["lo_80"]
        hi_80 = y_pred + offsets["hi_80"]
        lo_90 = y_pred + offsets["lo_90"]
        hi_90 = y_pred + offsets["hi_90"]
        trainval_mean_return = float(scaler.mean[TARGET_INDEX])
        n_predictions = len(y_pred)
        prediction_data: dict[str, object] = {
            "ds": np.arange(n_predictions),
            "datetime": test_frame["datetime"].iloc[:n_predictions].to_numpy(),
            "horizon_end_datetime": (
                test_frame["datetime"]
                .iloc[forecast_horizon - 1 : forecast_horizon - 1 + n_predictions]
                .to_numpy()
            ),
            "y": y_true,
            "pred": y_pred,
            "lo_80": lo_80,
            "hi_80": hi_80,
            "lo_90": lo_90,
            "hi_90": hi_90,
            "position": np.sign(y_pred),
            "model": "Mamba",
        }
        for step in range(forecast_horizon):
            prediction_data[f"y_h{step + 1}"] = y_true_steps[:, step]
            prediction_data[f"pred_h{step + 1}"] = y_pred_steps[:, step]
        predictions = pd.DataFrame(prediction_data)
        source_metadata = {
            "mode": "checkpoint",
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "history": str(Path(args.history).resolve()),
            "test": str(Path(args.test).resolve()),
            "device": str(device),
            "lookback": lookback,
            "forecast_horizon": forecast_horizon,
            "feature_columns": list(FEATURE_COLUMNS),
            "data_identity_warning": (
                "This checkpoint format stores the feature schema but no data-file "
                "hash. Confirm that these test/history splits match the data version "
                "used in Colab."
            ),
        }
    else:
        predictions = load_prediction_frame(args.predictions)
        if args.limit_test is not None:
            predictions = predictions.iloc[: args.limit_test].reset_index(drop=True)
        y_true = predictions["y"].to_numpy(dtype=np.float64)
        y_pred = predictions["pred"].to_numpy(dtype=np.float64)
        y_true_steps, y_pred_steps = horizon_arrays(predictions)
        forecast_horizon = y_true_steps.shape[1]
        lo_80 = predictions["lo_80"].to_numpy(dtype=np.float64)
        hi_80 = predictions["hi_80"].to_numpy(dtype=np.float64)
        lo_90 = predictions["lo_90"].to_numpy(dtype=np.float64)
        hi_90 = predictions["hi_90"].to_numpy(dtype=np.float64)
        _, train_values = load_split(args.train)
        trainval_mean_return = float(
            np.concatenate([train_values, history_values], axis=0)[:, TARGET_INDEX].mean()
        )
        source_metadata = {
            "mode": "saved_predictions",
            "predictions": str(Path(args.predictions).resolve()),
            "train": str(Path(args.train).resolve()),
            "history": str(Path(args.history).resolve()),
            "forecast_horizon": forecast_horizon,
        }

    report = build_report(
        y_true=y_true,
        y_pred=y_pred,
        lo_80=lo_80,
        hi_80=hi_80,
        lo_90=lo_90,
        hi_90=hi_90,
        previous_return=previous_return,
        trainval_mean_return=trainval_mean_return,
        thresholds_bps=args.thresholds_bps,
        transaction_cost_bps=args.transaction_cost_bps,
        periods_per_year=args.periods_per_year,
        y_true_steps=y_true_steps,
        y_pred_steps=y_pred_steps,
    )
    source_metadata.update(
        {
            "n_test_rows": len(predictions),
            "test_start": str(pd.Timestamp(predictions["datetime"].iloc[0])),
            "test_end": str(
                pd.Timestamp(
                    predictions[
                        "horizon_end_datetime"
                        if "horizon_end_datetime" in predictions
                        else "datetime"
                    ].iloc[-1]
                )
            ),
        }
    )
    report["metadata"] = source_metadata

    metrics_path = Path(args.metrics_out)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    if args.checkpoint:
        predictions_path = Path(args.predictions_out)
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_parquet(predictions_path, index=False)
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)

    print(json.dumps(report, indent=2, allow_nan=False))
    if args.checkpoint:
        print(f"Saved {len(predictions)} predictions to {predictions_path}")
    print(f"Saved evaluation report to {metrics_path}")


if __name__ == "__main__":
    main()
