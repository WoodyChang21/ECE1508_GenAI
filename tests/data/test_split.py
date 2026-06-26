import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.data.split_utils import split_dataframe, TRAIN_END, VAL_END


def _make_features_df() -> pd.DataFrame:
    dates = pd.date_range("2010-01-04 09:30:00", periods=500, freq="1h")
    np.random.seed(0)
    return pd.DataFrame({
        "datetime": dates,
        "close": 470 + np.cumsum(np.random.randn(500) * 0.5),
        "return_1h": np.random.randn(500) * 0.001,
    })


def test_splits_cover_all_rows():
    df = _make_features_df()
    train, val, test = split_dataframe(df)
    assert len(train) + len(val) + len(test) == len(df)


def test_splits_do_not_overlap():
    df = _make_features_df()
    train, val, test = split_dataframe(df)
    if len(train) > 0 and len(val) > 0:
        assert train["datetime"].max() < val["datetime"].min()
    if len(val) > 0 and len(test) > 0:
        assert val["datetime"].max() < test["datetime"].min()


def test_train_ends_at_or_before_train_end():
    df = _make_features_df()
    train, _, _ = split_dataframe(df)
    if len(train) > 0:
        assert str(train["datetime"].max().date()) <= TRAIN_END


def test_val_within_val_period():
    df = _make_features_df()
    _, val, _ = split_dataframe(df)
    if len(val) > 0:
        assert str(val["datetime"].min().date()) > TRAIN_END
        assert str(val["datetime"].max().date()) <= VAL_END


def test_empty_splits_when_data_ends_before_split_period():
    df = _make_features_df()
    early_df = df[df["datetime"] < "2022-01-01"].copy()
    train, val, test = split_dataframe(early_df)
    assert len(val) == 0
    assert len(test) == 0
