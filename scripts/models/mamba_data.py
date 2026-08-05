"""Leakage-safe scaling and windowing for the Mamba forecaster."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


FEATURE_COLUMNS = (
    "return_1h",  # target is deliberately first
    "open",
    "high",
    "low",
    "close",
    "volume",
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
    "vix_log",
    "vix_change_1h",
)
TARGET_INDEX = 0


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
    ) -> None:
        values = np.asarray(values, dtype=np.float32)
        if values.ndim != 2:
            raise ValueError("values must have shape (time, features)")
        if lookback < 1:
            raise ValueError("lookback must be positive")
        if not 0 <= target_index < values.shape[1]:
            raise ValueError("target_index is out of bounds")

        self.values = torch.from_numpy(values)
        self.lookback = lookback
        self.target_index = target_index
        requested_start = lookback if target_start is None else target_start
        self.first_target = max(lookback, requested_start)

    def __len__(self) -> int:
        return max(0, len(self.values) - self.first_target)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        target_pos = self.first_target + index
        window = self.values[target_pos - self.lookback : target_pos]
        target = self.values[target_pos, self.target_index]
        return window, target


def load_split(path: str | Path) -> tuple[pd.DataFrame, np.ndarray]:
    """Load and validate one model split, returning frame and float features."""
    frame = pd.read_parquet(path)
    missing = [column for column in FEATURE_COLUMNS if column not in frame]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    values = frame.loc[:, FEATURE_COLUMNS].astype(np.float32).to_numpy()
    if not np.isfinite(values).all():
        raise ValueError(f"{path} contains non-finite model inputs")
    return frame, values


def with_history(
    history: np.ndarray, current: np.ndarray, lookback: int
) -> WindowDataset:
    """Create a dataset where history is context and current rows are labels."""
    if len(history) < lookback:
        raise ValueError("history is shorter than the requested lookback")
    context = history[-lookback:]
    combined = np.concatenate([context, current], axis=0)
    return WindowDataset(combined, lookback=lookback, target_start=len(context))
