"""Train, tune, and evaluate a Mamba model on the SPY data splits.

Example:
    python scripts/models/train_mamba.py

For a quick CPU smoke run:
    python scripts/models/train_mamba.py --lookbacks 24 --epochs 1 \
        --d-model 16 --layers 1 --limit-train 256 --limit-val 64 \
        --limit-test 64 --output data/predictions/mamba_smoke.parquet
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

# Support both ``python -m scripts.models.train_mamba`` and direct execution.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.models.mamba_data import (
    FEATURE_COLUMNS,
    FEATURE_SCHEMA_VERSION,
    TARGET_INDEX,
    WindowDataset,
    ZScoreScaler,
    load_split,
    with_history,
)
from scripts.models.mamba_model import MambaForecaster
from scripts.models.mamba_strategy import select_threshold
from scripts.models.metrics import compute_all


@dataclass(frozen=True)
class ModelOptions:
    d_model: int
    state_size: int
    layers: int
    expand: int
    conv_kernel: int
    dropout: float


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def make_model(options: ModelOptions, forecast_horizon: int = 1) -> MambaForecaster:
    return MambaForecaster(
        n_features=len(FEATURE_COLUMNS),
        d_model=options.d_model,
        state_size=options.state_size,
        num_layers=options.layers,
        expand=options.expand,
        conv_kernel=options.conv_kernel,
        dropout=options.dropout,
        output_size=forecast_horizon,
    )


@torch.no_grad()
def predict(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for windows, target in loader:
        output = model(windows.to(device))
        predictions.append(output.cpu().numpy())
        targets.append(target.numpy())
    if not predictions:
        raise ValueError("prediction loader is empty")
    return np.concatenate(predictions), np.concatenate(targets)


class ForecastLoss(nn.Module):
    """Optimize the path, compounded horizon return, and horizon direction."""

    def __init__(
        self,
        target_mean: float,
        target_scale: float,
        forecast_horizon: int,
        cumulative_weight: float,
        direction_weight: float,
    ) -> None:
        super().__init__()
        self.target_mean = target_mean
        self.target_scale = target_scale
        self.cumulative_scale = target_scale * np.sqrt(forecast_horizon)
        self.cumulative_weight = cumulative_weight
        self.direction_weight = direction_weight
        self.huber = nn.HuberLoss(delta=1.0)
        self.direction = nn.BCEWithLogitsLoss()

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        step_loss = self.huber(output, target)
        if output.ndim == 1:
            return step_loss

        predicted_returns = output * self.target_scale + self.target_mean
        actual_returns = target * self.target_scale + self.target_mean
        predicted_cumulative = torch.prod(1.0 + predicted_returns, dim=1) - 1.0
        actual_cumulative = torch.prod(1.0 + actual_returns, dim=1) - 1.0
        cumulative_loss = self.huber(
            predicted_cumulative / self.cumulative_scale,
            actual_cumulative / self.cumulative_scale,
        )
        direction_loss = self.direction(
            predicted_cumulative / self.cumulative_scale,
            (actual_cumulative > 0).to(output.dtype),
        )
        return (
            step_loss
            + self.cumulative_weight * cumulative_loss
            + self.direction_weight * direction_loss
        )


@torch.no_grad()
def cumulative_mae(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    scaler: ZScoreScaler,
) -> float:
    predictions, targets = predict(model, loader, device)
    predicted = compound_horizon(scaler.inverse_target(predictions))
    actual = compound_horizon(scaler.inverse_target(targets))
    return float(np.mean(np.abs(predicted - actual)))


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    scaler: ZScoreScaler,
    forecast_horizon: int,
    cumulative_loss_weight: float,
    direction_loss_weight: float,
    val_loader: DataLoader | None = None,
    patience: int = 5,
) -> tuple[nn.Module, int, float | None]:
    """Train with optional early stopping and restore the best validation state."""
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    loss_fn = ForecastLoss(
        target_mean=float(scaler.mean[TARGET_INDEX]),
        target_scale=float(scaler.scale[TARGET_INDEX]),
        forecast_horizon=forecast_horizon,
        cumulative_weight=cumulative_loss_weight,
        direction_weight=direction_loss_weight,
    )
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = epochs
    best_score = float("inf")
    stale_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_rows = 0
        for windows, target in train_loader:
            windows = windows.to(device)
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(windows)
            loss = loss_fn(output, target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * len(target)
            total_rows += len(target)

        message = f"epoch {epoch:03d} train_loss={total_loss / total_rows:.6f}"
        if val_loader is not None:
            score = cumulative_mae(model, val_loader, device, scaler)
            message += f" cumulative_val_mae={score:.6f}"
            if score < best_score - 1e-7:
                best_score = score
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1
        print(message, flush=True)

        if val_loader is not None and stale_epochs >= patience:
            print(f"early stopping after epoch {epoch}", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_epoch, None if val_loader is None else best_score


def limit_rows(
    frame: pd.DataFrame, values: np.ndarray, limit: int | None, keep_tail: bool
) -> tuple[pd.DataFrame, np.ndarray]:
    if limit is None or limit >= len(frame):
        return frame.reset_index(drop=True), values
    if limit < 1:
        raise ValueError("row limits must be positive")
    selection = slice(-limit, None) if keep_tail else slice(0, limit)
    return frame.iloc[selection].reset_index(drop=True), values[selection]


def loaders_for_tuning(
    train_values: np.ndarray,
    val_values: np.ndarray,
    scaler: ZScoreScaler,
    lookback: int,
    batch_size: int,
    forecast_horizon: int = 1,
) -> tuple[DataLoader, DataLoader]:
    train_scaled = scaler.transform(train_values)
    val_scaled = scaler.transform(val_values)
    train_set = WindowDataset(
        train_scaled, lookback=lookback, forecast_horizon=forecast_horizon
    )
    val_set = with_history(
        train_scaled,
        val_scaled,
        lookback=lookback,
        forecast_horizon=forecast_horizon,
    )
    if len(train_set) == 0:
        raise ValueError(f"lookback {lookback} leaves no training windows")
    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True),
        DataLoader(val_set, batch_size=batch_size, shuffle=False),
    )


def residual_intervals(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, float]:
    residuals = y_true - y_pred
    return {
        "lo_80": float(np.quantile(residuals, 0.10)),
        "hi_80": float(np.quantile(residuals, 0.90)),
        "lo_90": float(np.quantile(residuals, 0.05)),
        "hi_90": float(np.quantile(residuals, 0.95)),
    }


def compound_horizon(returns: np.ndarray) -> np.ndarray:
    """Compound per-step simple returns into one return per forecast window."""
    values = np.asarray(returns)
    if values.ndim == 1:
        return values
    if values.ndim != 2:
        raise ValueError("returns must be a 1D or 2D array")
    return np.prod(1.0 + values, axis=1, dtype=np.float64) - 1.0


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_calibration_tail(
    frame: pd.DataFrame,
    values: np.ndarray,
    fraction: float,
    minimum_rows: int,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    calibration_rows = max(minimum_rows, int(round(len(values) * fraction)))
    if calibration_rows >= len(values):
        raise ValueError("calibration split leaves no validation-selection rows")
    boundary = len(values) - calibration_rows
    return (
        frame.iloc[:boundary].reset_index(drop=True),
        values[:boundary],
        frame.iloc[boundary:].reset_index(drop=True),
        values[boundary:],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="data/splits/train.parquet")
    parser.add_argument("--val", default="data/splits/val.parquet")
    parser.add_argument("--test", default="data/splits/test.parquet")
    parser.add_argument("--output", default="data/predictions/mamba_preds.parquet")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--lookbacks", nargs="+", type=int, default=[24, 60, 120])
    parser.add_argument(
        "--forecast-horizon",
        type=int,
        default=3,
        help="Number of future hourly returns predicted together (default: 3).",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--state-size", type=int, default=16)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--expand", type=int, default=2)
    parser.add_argument("--conv-kernel", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--cumulative-loss-weight", type=float, default=1.0)
    parser.add_argument("--direction-loss-weight", type=float, default=0.05)
    parser.add_argument(
        "--calibration-fraction",
        type=float,
        default=0.30,
        help="Tail fraction of validation reserved from all model fitting.",
    )
    parser.add_argument(
        "--strategy-thresholds-bps",
        nargs="+",
        type=float,
        default=[2.0, 4.0, 6.0, 8.0, 10.0],
    )
    parser.add_argument(
        "--strategy-position-mode",
        choices=("long_short", "long_only", "short_only"),
        default="long_only",
    )
    parser.add_argument(
        "--strategy-require-all-steps-agree",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--transaction-cost-bps", type=float, default=1.0)
    parser.add_argument("--periods-per-year", type=int, default=1764)
    parser.add_argument("--strategy-min-calibration-trades", type=int, default=30)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--limit-test", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch size must be positive")
    if any(lookback < 1 for lookback in args.lookbacks):
        raise ValueError("lookbacks must be positive")
    if args.forecast_horizon < 1:
        raise ValueError("forecast-horizon must be positive")
    if not 0.0 < args.calibration_fraction < 1.0:
        raise ValueError("calibration-fraction must be between zero and one")
    if args.cumulative_loss_weight < 0 or args.direction_loss_weight < 0:
        raise ValueError("loss weights must be non-negative")
    if any(value < 0 for value in args.strategy_thresholds_bps):
        raise ValueError("strategy thresholds must be non-negative")
    if args.transaction_cost_bps < 0 or args.periods_per_year < 1:
        raise ValueError("strategy cost must be non-negative and periods positive")
    if args.strategy_min_calibration_trades < 1:
        raise ValueError("strategy-min-calibration-trades must be positive")

    seed_everything(args.seed)
    device = resolve_device(args.device)
    options = ModelOptions(
        d_model=args.d_model,
        state_size=args.state_size,
        layers=args.layers,
        expand=args.expand,
        conv_kernel=args.conv_kernel,
        dropout=args.dropout,
    )

    train_frame, train_values = load_split(args.train)
    val_frame, val_values = load_split(args.val)
    test_frame, test_values = load_split(args.test)
    train_frame, train_values = limit_rows(
        train_frame, train_values, args.limit_train, keep_tail=True
    )
    val_frame, val_values = limit_rows(
        val_frame, val_values, args.limit_val, keep_tail=False
    )
    test_frame, test_values = limit_rows(
        test_frame, test_values, args.limit_test, keep_tail=False
    )
    (
        selection_frame,
        selection_values,
        calibration_frame,
        calibration_values,
    ) = split_calibration_tail(
        val_frame,
        val_values,
        args.calibration_fraction,
        minimum_rows=max(args.forecast_horizon + 1, 32),
    )
    print(
        f"device={device} rows: train={len(train_values)}, "
        f"selection={len(selection_values)}, calibration={len(calibration_values)}, "
        f"test={len(test_values)}"
    )

    tuning_scaler = ZScoreScaler.fit(train_values)
    tuning_results: list[dict[str, float | int]] = []
    best: dict[str, object] | None = None

    for lookback in args.lookbacks:
        print(f"\nTuning lookback={lookback}", flush=True)
        seed_everything(args.seed)
        train_loader, val_loader = loaders_for_tuning(
            train_values,
            selection_values,
            tuning_scaler,
            lookback,
            args.batch_size,
            args.forecast_horizon,
        )
        model, best_epoch, score = train_model(
            make_model(options, args.forecast_horizon),
            train_loader,
            device,
            args.epochs,
            args.learning_rate,
            args.weight_decay,
            tuning_scaler,
            args.forecast_horizon,
            args.cumulative_loss_weight,
            args.direction_loss_weight,
            val_loader=val_loader,
            patience=args.patience,
        )
        assert score is not None
        cumulative_val_mae = score
        result = {
            "lookback": lookback,
            "best_epoch": best_epoch,
            "cumulative_horizon_val_mae": cumulative_val_mae,
        }
        tuning_results.append(result)
        if best is None or cumulative_val_mae < best["selection_score"]:
            best = {
                "lookback": lookback,
                "epoch": best_epoch,
                "selection_score": cumulative_val_mae,
            }

    assert best is not None
    best_lookback = int(best["lookback"])
    final_epochs = int(best["epoch"])
    print(
        f"\nSelected lookback={best_lookback}, epochs={final_epochs}, "
        f"cumulative_horizon_val_mae={best['selection_score']:.6f}"
    )

    fit_values = np.concatenate([train_values, selection_values], axis=0)
    final_scaler = ZScoreScaler.fit(fit_values)
    fit_scaled = final_scaler.transform(fit_values)
    final_train_set = WindowDataset(
        fit_scaled,
        lookback=best_lookback,
        forecast_horizon=args.forecast_horizon,
    )
    final_loader = DataLoader(
        final_train_set, batch_size=args.batch_size, shuffle=True
    )
    seed_everything(args.seed)
    final_model, _, _ = train_model(
        make_model(options, args.forecast_horizon),
        final_loader,
        device,
        final_epochs,
        args.learning_rate,
        args.weight_decay,
        final_scaler,
        args.forecast_horizon,
        args.cumulative_loss_weight,
        args.direction_loss_weight,
    )

    calibration_scaled = final_scaler.transform(calibration_values)
    calibration_set = with_history(
        fit_scaled,
        calibration_scaled,
        lookback=best_lookback,
        forecast_horizon=args.forecast_horizon,
    )
    calibration_loader = DataLoader(
        calibration_set, batch_size=args.batch_size, shuffle=False
    )
    calibration_pred_norm, calibration_true_norm = predict(
        final_model, calibration_loader, device
    )
    calibration_pred_steps = final_scaler.inverse_target(calibration_pred_norm)
    calibration_true_steps = final_scaler.inverse_target(calibration_true_norm)
    calibration_pred = compound_horizon(calibration_pred_steps)
    calibration_true = compound_horizon(calibration_true_steps)
    offsets = residual_intervals(calibration_true, calibration_pred)
    strategy_policy: dict[str, object] | None = None
    strategy_calibration_results: list[dict[str, object]] = []
    if args.forecast_horizon > 1:
        strategy_policy, strategy_calibration_results = select_threshold(
            calibration_true_steps,
            calibration_pred_steps,
            thresholds_bps=args.strategy_thresholds_bps,
            transaction_cost_bps=args.transaction_cost_bps,
            periods_per_year=args.periods_per_year,
            position_mode=args.strategy_position_mode,
            require_all_steps_agree=args.strategy_require_all_steps_agree,
            minimum_trades=args.strategy_min_calibration_trades,
        )

    full_history_values = np.concatenate([fit_values, calibration_values], axis=0)
    full_history_scaled = final_scaler.transform(full_history_values)
    test_scaled = final_scaler.transform(test_values)
    test_set = with_history(
        full_history_scaled,
        test_scaled,
        lookback=best_lookback,
        forecast_horizon=args.forecast_horizon,
    )
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    pred_norm, true_norm = predict(final_model, test_loader, device)
    test_pred_steps = final_scaler.inverse_target(pred_norm)
    test_true_steps = final_scaler.inverse_target(true_norm)
    test_pred = compound_horizon(test_pred_steps)
    test_true = compound_horizon(test_true_steps)

    lo_80 = test_pred + offsets["lo_80"]
    hi_80 = test_pred + offsets["hi_80"]
    lo_90 = test_pred + offsets["lo_90"]
    hi_90 = test_pred + offsets["hi_90"]
    metrics = compute_all(test_true, test_pred, lo_80, hi_80, lo_90, hi_90)

    n_predictions = len(test_pred)
    output_data: dict[str, object] = {
        "ds": np.arange(n_predictions),
        "datetime": test_frame["datetime"].iloc[:n_predictions].to_numpy(),
        "horizon_end_datetime": (
            test_frame["datetime"]
            .iloc[
                args.forecast_horizon
                - 1 : args.forecast_horizon
                - 1
                + n_predictions
            ]
            .to_numpy()
        ),
        "y": test_true,
        "pred": test_pred,
        "lo_80": lo_80,
        "hi_80": hi_80,
        "lo_90": lo_90,
        "hi_90": hi_90,
        "model": "Mamba",
    }
    if args.forecast_horizon == 1:
        test_true_steps = np.asarray(test_true_steps).reshape(-1, 1)
        test_pred_steps = np.asarray(test_pred_steps).reshape(-1, 1)
    for step in range(args.forecast_horizon):
        output_data[f"y_h{step + 1}"] = test_true_steps[:, step]
        output_data[f"pred_h{step + 1}"] = test_pred_steps[:, step]
    output = pd.DataFrame(output_data)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(output_path, index=False)

    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": final_model.cpu().state_dict(),
                "model_options": asdict(options),
                "lookback": best_lookback,
                "forecast_horizon": args.forecast_horizon,
                "feature_columns": FEATURE_COLUMNS,
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "scaler_mean": final_scaler.mean,
                "scaler_scale": final_scaler.scale,
                "interval_offsets": offsets,
                "strategy_policy": strategy_policy,
                "training": {
                    "seed": args.seed,
                    "epochs": final_epochs,
                    "cumulative_loss_weight": args.cumulative_loss_weight,
                    "direction_loss_weight": args.direction_loss_weight,
                    "learning_rate": args.learning_rate,
                    "weight_decay": args.weight_decay,
                    "batch_size": args.batch_size,
                    "calibration_fraction": args.calibration_fraction,
                    "row_limits": {
                        "train": args.limit_train,
                        "val": args.limit_val,
                        "test": args.limit_test,
                    },
                    "selection_start": str(selection_frame["datetime"].iloc[0]),
                    "selection_end": str(selection_frame["datetime"].iloc[-1]),
                    "calibration_start": str(calibration_frame["datetime"].iloc[0]),
                    "calibration_end": str(calibration_frame["datetime"].iloc[-1]),
                    "tuning_results": tuning_results,
                    "strategy_calibration_results": strategy_calibration_results,
                },
                "data_sha256": {
                    "train": file_sha256(args.train),
                    "val": file_sha256(args.val),
                    "test": file_sha256(args.test),
                },
            },
            checkpoint_path,
        )

    print("\nTuning results:")
    print(json.dumps(tuning_results, indent=2))
    print("Test metrics:")
    print(json.dumps(metrics, indent=2))
    if strategy_policy is not None:
        print("Locked strategy selected on calibration data:")
        print(json.dumps(strategy_policy, indent=2))
    print(f"Saved {len(output)} predictions to {output_path}")


if __name__ == "__main__":
    main()
