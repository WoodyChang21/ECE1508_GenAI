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
) -> dict[str, float | int | None]:
    """Evaluate a chronological long/short/flat one-period signal.

    Position at row t is sign(pred_t), unless the prediction magnitude is below
    the threshold. Costs are charged on absolute position turnover, so a direct
    long-to-short reversal costs twice as much as entering from flat. The final
    position is closed after the final observed return.
    """
    if threshold_bps < 0 or transaction_cost_bps < 0:
        raise ValueError("thresholds and transaction costs must be non-negative")
    if periods_per_year < 1:
        raise ValueError("periods_per_year must be positive")

    threshold = threshold_bps / 10_000.0
    positions = np.where(np.abs(y_pred) >= threshold, np.sign(y_pred), 0.0)
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
) -> dict[str, Any]:
    zero_pred = np.zeros_like(y_true)
    mean_pred = np.full_like(y_true, trainval_mean_return)
    lag_one_pred = np.concatenate([[previous_return], y_true[:-1]])

    model_metrics = point_metrics(y_true, y_pred)
    zero_metrics = point_metrics(y_true, zero_pred)
    mean_metrics = point_metrics(y_true, mean_pred)
    lag_metrics = point_metrics(y_true, lag_one_pred)
    baseline_mae = float(zero_metrics["mae"])
    baseline_rmse = float(zero_metrics["rmse"])

    return {
        "forecast": {
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
        "sign_strategy": {
            "note": (
                "Chronological long/short/flat strategy using sign(prediction). "
                "Returns are compounded in test order. A one-way cost is charged on "
                "absolute position turnover; a direct long/short reversal has turnover "
                "2. This is a diagnostic backtest, not an execution simulation."
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
        },
    }


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
        scaler = ZScoreScaler(
            mean=np.asarray(checkpoint["scaler_mean"], dtype=np.float32),
            scale=np.asarray(checkpoint["scaler_scale"], dtype=np.float32),
        )
        model = make_model(options).to(device)
        model.load_state_dict(checkpoint["state_dict"], strict=True)

        test_frame, test_values = load_split(args.test)
        if args.limit_test is not None:
            test_frame = test_frame.iloc[: args.limit_test].reset_index(drop=True)
            test_values = test_values[: args.limit_test]
        if len(history_values) < lookback:
            raise ValueError(
                f"history has {len(history_values)} rows but checkpoint needs {lookback}"
            )
        if len(test_values) == 0:
            raise ValueError("test split is empty")

        history_scaled = scaler.transform(history_values)
        test_scaled = scaler.transform(test_values)
        test_set = with_history(history_scaled, test_scaled, lookback=lookback)
        loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
        pred_norm, true_norm = predict(model, loader, device)
        y_pred = scaler.inverse_target(pred_norm)
        y_true = scaler.inverse_target(true_norm)

        offsets = {
            key: float(checkpoint["interval_offsets"][key])
            for key in REQUIRED_INTERVAL_KEYS
        }
        lo_80 = y_pred + offsets["lo_80"]
        hi_80 = y_pred + offsets["hi_80"]
        lo_90 = y_pred + offsets["lo_90"]
        hi_90 = y_pred + offsets["hi_90"]
        trainval_mean_return = float(scaler.mean[TARGET_INDEX])
        predictions = pd.DataFrame(
            {
                "ds": np.arange(len(test_frame)),
                "datetime": test_frame["datetime"].to_numpy(),
                "y": y_true,
                "pred": y_pred,
                "lo_80": lo_80,
                "hi_80": hi_80,
                "lo_90": lo_90,
                "hi_90": hi_90,
                "position": np.sign(y_pred),
                "model": "Mamba",
            }
        )
        source_metadata = {
            "mode": "checkpoint",
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "history": str(Path(args.history).resolve()),
            "test": str(Path(args.test).resolve()),
            "device": str(device),
            "lookback": lookback,
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
    )
    source_metadata.update(
        {
            "n_test_rows": len(predictions),
            "test_start": str(pd.Timestamp(predictions["datetime"].iloc[0])),
            "test_end": str(pd.Timestamp(predictions["datetime"].iloc[-1])),
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
