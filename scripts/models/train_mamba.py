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
    TARGET_INDEX,
    WindowDataset,
    ZScoreScaler,
    load_split,
    with_history,
)
from scripts.models.mamba_model import MambaForecaster
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


@torch.no_grad()
def normalized_mae(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> float:
    predictions, targets = predict(model, loader, device)
    return float(np.mean(np.abs(predictions - targets)))


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    val_loader: DataLoader | None = None,
    patience: int = 5,
) -> tuple[nn.Module, int, float | None]:
    """Train with optional early stopping and restore the best validation state."""
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    loss_fn = nn.HuberLoss(delta=1.0)
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
            score = normalized_mae(model, val_loader, device)
            message += f" val_mae={score:.6f}"
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
    print(
        f"device={device} rows: train={len(train_values)}, "
        f"val={len(val_values)}, test={len(test_values)}"
    )

    tuning_scaler = ZScoreScaler.fit(train_values)
    tuning_results: list[dict[str, float | int]] = []
    best: dict[str, object] | None = None

    for lookback in args.lookbacks:
        print(f"\nTuning lookback={lookback}", flush=True)
        seed_everything(args.seed)
        train_loader, val_loader = loaders_for_tuning(
            train_values,
            val_values,
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
            val_loader=val_loader,
            patience=args.patience,
        )
        assert score is not None
        pred_norm, true_norm = predict(model, val_loader, device)
        val_pred_steps = tuning_scaler.inverse_target(pred_norm)
        val_true_steps = tuning_scaler.inverse_target(true_norm)
        val_pred = compound_horizon(val_pred_steps)
        val_true = compound_horizon(val_true_steps)
        cumulative_val_mae = float(np.mean(np.abs(val_pred - val_true)))
        result = {
            "lookback": lookback,
            "best_epoch": best_epoch,
            "normalized_per_step_val_mae": score,
            "cumulative_horizon_val_mae": cumulative_val_mae,
        }
        tuning_results.append(result)
        if best is None or cumulative_val_mae < best["selection_score"]:
            best = {
                "lookback": lookback,
                "epoch": best_epoch,
                "selection_score": cumulative_val_mae,
                "normalized_per_step_score": score,
                "val_pred": val_pred,
                "val_true": val_true,
            }

    assert best is not None
    best_lookback = int(best["lookback"])
    final_epochs = int(best["epoch"])
    print(
        f"\nSelected lookback={best_lookback}, epochs={final_epochs}, "
        f"cumulative_horizon_val_mae={best['selection_score']:.6f}, "
        f"normalized_per_step_val_mae={best['normalized_per_step_score']:.6f}"
    )

    trainval_values = np.concatenate([train_values, val_values], axis=0)
    final_scaler = ZScoreScaler.fit(trainval_values)
    trainval_scaled = final_scaler.transform(trainval_values)
    final_train_set = WindowDataset(
        trainval_scaled,
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
    )

    test_scaled = final_scaler.transform(test_values)
    test_set = with_history(
        trainval_scaled,
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

    offsets = residual_intervals(best["val_true"], best["val_pred"])
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
                "scaler_mean": final_scaler.mean,
                "scaler_scale": final_scaler.scale,
                "interval_offsets": offsets,
            },
            checkpoint_path,
        )

    print("\nTuning results:")
    print(json.dumps(tuning_results, indent=2))
    print("Test metrics:")
    print(json.dumps(metrics, indent=2))
    print(f"Saved {len(output)} predictions to {output_path}")


if __name__ == "__main__":
    main()
