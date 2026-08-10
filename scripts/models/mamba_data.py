"""Leakage-safe scaling and windowing for the Mamba forecaster."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


FEATURE_SCHEMA_VERSION = 2

# Names presented to the model. Price-level features are deliberately expressed
# as relative distances so a model trained when SPY traded near 250 does not see
# every 550-dollar test observation as an outlier.
FEATURE_COLUMNS = (
    "return_1h",  # target is deliberately first
    "open_to_previous_close",
    "high_to_close",
    "low_to_close",
    "close_to_open",
    "return_4h",
    "return_24h",
    "is_first_bar",
    "bar_time_sin",
    "bar_time_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "vol_24h",
    "vol_60h",
    "log_volume_ratio",
    "rsi_centered",
    "macd_relative",
    "macd_signal_relative",
    "macd_diff_relative",
    "bb_upper_distance",
    "bb_lower_distance",
    "bb_width",
    "vix_change_1h",
)
TARGET_INDEX = 0

SOURCE_COLUMNS = (
    "datetime",
    "return_1h",
    "open",
    "high",
    "low",
    "close",
    "return_4h",
    "return_24h",
    "is_first_bar",
    "vol_24h",
    "vol_60h",
    "volume_ratio",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_diff",
    "bb_upper",
    "bb_lower",
    "bb_width",
    "vix_change_1h",
)


@dataclass(frozen=True)
class ZScoreScaler:
    """Per-feature z-score scaler fitted only on the supplied training rows."""

    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "ZScoreScaler":
        values = np.asarray(values, dtype=np.float32)
        if values.ndim != 2 or len(values) == 0:
            raise ValueError("values must be a non-empty 2D array")
        mean = values.mean(axis=0, dtype=np.float64).astype(np.float32)
        scale = values.std(axis=0, dtype=np.float64).astype(np.float32)
        scale[scale < 1e-8] = 1.0
        return cls(mean=mean, scale=scale)

    def transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        return ((values - self.mean) / self.scale).astype(np.float32)

    def inverse_target(
        self, values: np.ndarray, target_index: int = TARGET_INDEX
    ) -> np.ndarray:
        values = np.asarray(values)
        return values * self.scale[target_index] + self.mean[target_index]


class WindowDataset(Dataset):
    """Causal windows whose labels begin at ``target_start``.

    ``target_start`` lets validation/test arrays include earlier context while
    ensuring that only rows from the requested split become labels.
    """

    def __init__(
        self,
        values: np.ndarray,
        lookback: int,
        target_start: int | None = None,
        target_index: int = TARGET_INDEX,
        forecast_horizon: int = 1,
    ) -> None:
        values = np.asarray(values, dtype=np.float32)
        if values.ndim != 2:
            raise ValueError("values must have shape (time, features)")
        if lookback < 1:
            raise ValueError("lookback must be positive")
        if forecast_horizon < 1:
            raise ValueError("forecast_horizon must be positive")
        if not 0 <= target_index < values.shape[1]:
            raise ValueError("target_index is out of bounds")

        self.values = torch.from_numpy(values)
        self.lookback = lookback
        self.target_index = target_index
        self.forecast_horizon = forecast_horizon
        requested_start = lookback if target_start is None else target_start
        self.first_target = max(lookback, requested_start)

    def __len__(self) -> int:
        return max(0, len(self.values) - self.first_target - self.forecast_horizon + 1)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        target_pos = self.first_target + index
        window = self.values[target_pos - self.lookback : target_pos]
        target = self.values[
            target_pos : target_pos + self.forecast_horizon, self.target_index
        ]
        if self.forecast_horizon == 1:
            target = target.squeeze(0)
        return window, target


def model_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Build stationary, calendar-aware model inputs from a raw split."""
    missing = [column for column in SOURCE_COLUMNS if column not in frame]
    if missing:
        raise ValueError(f"input frame is missing columns: {missing}")

    close = frame["close"].astype(np.float64)
    previous_close = close / (1.0 + frame["return_1h"].astype(np.float64))
    timestamps = pd.to_datetime(frame["datetime"])
    bar_number = (
        timestamps.dt.hour * 60 + timestamps.dt.minute - (9 * 60 + 30)
    ) / 60.0
    bar_phase = 2.0 * np.pi * bar_number / 7.0
    weekday_phase = 2.0 * np.pi * timestamps.dt.dayofweek / 5.0

    features = pd.DataFrame(
        {
            "return_1h": frame["return_1h"],
            "open_to_previous_close": frame["open"] / previous_close - 1.0,
            "high_to_close": frame["high"] / close - 1.0,
            "low_to_close": frame["low"] / close - 1.0,
            "close_to_open": close / frame["open"] - 1.0,
            "return_4h": frame["return_4h"],
            "return_24h": frame["return_24h"],
            "is_first_bar": frame["is_first_bar"].astype(np.float64),
            "bar_time_sin": np.sin(bar_phase),
            "bar_time_cos": np.cos(bar_phase),
            "day_of_week_sin": np.sin(weekday_phase),
            "day_of_week_cos": np.cos(weekday_phase),
            "vol_24h": frame["vol_24h"],
            "vol_60h": frame["vol_60h"],
            "log_volume_ratio": np.log(frame["volume_ratio"].clip(lower=1e-8)),
            "rsi_centered": (frame["rsi_14"] - 50.0) / 50.0,
            "macd_relative": frame["macd"] / close,
            "macd_signal_relative": frame["macd_signal"] / close,
            "macd_diff_relative": frame["macd_diff"] / close,
            "bb_upper_distance": frame["bb_upper"] / close - 1.0,
            "bb_lower_distance": close / frame["bb_lower"] - 1.0,
            "bb_width": frame["bb_width"],
            "vix_change_1h": frame["vix_change_1h"],
        },
        index=frame.index,
    )
    return features.loc[:, FEATURE_COLUMNS]


def load_split(path: str | Path) -> tuple[pd.DataFrame, np.ndarray]:
    """Load one split and return its raw frame plus derived model features."""
    frame = pd.read_parquet(path)
    try:
        feature_frame = model_feature_frame(frame)
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from exc
    values = feature_frame.astype(np.float32).to_numpy()
    if not np.isfinite(values).all():
        raise ValueError(f"{path} contains non-finite model inputs")
    return frame, values


def with_history(
    history: np.ndarray,
    current: np.ndarray,
    lookback: int,
    forecast_horizon: int = 1,
) -> WindowDataset:
    """Create a dataset where history is context and current rows are labels."""
    if len(history) < lookback:
        raise ValueError("history is shorter than the requested lookback")
    context = history[-lookback:]
    combined = np.concatenate([context, current], axis=0)
    return WindowDataset(
        combined,
        lookback=lookback,
        target_start=len(context),
        forecast_horizon=forecast_horizon,
    )
