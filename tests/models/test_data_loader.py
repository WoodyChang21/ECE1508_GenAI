import pandas as pd
import numpy as np
import pytest

from scripts.models.data_loader import load_nf_dataframe, build_full_df, HIST_EXOG_COLS, FUTR_EXOG_COLS


@pytest.fixture
def sample_parquet(tmp_path):
    df = pd.DataFrame({
        'datetime': pd.date_range('2023-01-03 09:30', periods=10, freq='h'),
        'open': np.ones(10) * 400.0,
        'high': np.ones(10) * 401.0,
        'low': np.ones(10) * 399.0,
        'close': np.ones(10) * 400.0,
        'volume': np.ones(10, dtype=int) * 1_000_000,
        'return_1h': np.linspace(0.001, 0.010, 10),
        'return_4h': np.linspace(0.002, 0.020, 10),
        'return_24h': np.linspace(0.003, 0.030, 10),
        'is_first_bar': [True] + [False] * 9,
        'vol_24h': np.ones(10) * 0.002,
        'vol_60h': np.ones(10) * 0.002,
        'volume_ratio': np.ones(10) * 1.0,
        'rsi_14': np.ones(10) * 50.0,
        'macd': np.zeros(10),
        'macd_signal': np.zeros(10),
        'macd_diff': np.zeros(10),
        'bb_upper': np.ones(10) * 405.0,
        'bb_lower': np.ones(10) * 395.0,
        'bb_width': np.ones(10) * 2.5,
        'vix_log': np.ones(10) * 3.0,
        'vix_change_1h': np.zeros(10),
    })
    path = tmp_path / 'split.parquet'
    df.to_parquet(path)
    return str(path)


def test_required_columns_present(sample_parquet):
    df = load_nf_dataframe(sample_parquet)
    assert {'unique_id', 'ds', 'y'}.issubset(df.columns)


def test_unique_id_is_spy(sample_parquet):
    df = load_nf_dataframe(sample_parquet)
    assert (df['unique_id'] == 'SPY').all()


def test_ds_is_sequential_integers_from_zero(sample_parquet):
    df = load_nf_dataframe(sample_parquet)
    assert list(df['ds']) == list(range(10))


def test_y_equals_return_1h(sample_parquet):
    df = load_nf_dataframe(sample_parquet)
    raw = pd.read_parquet(sample_parquet)
    np.testing.assert_array_almost_equal(df['y'].values, raw['return_1h'].values)


def test_extra_cols_included(sample_parquet):
    df = load_nf_dataframe(sample_parquet, extra_cols=['is_first_bar', 'rsi_14'])
    assert 'is_first_bar' in df.columns
    assert 'rsi_14' in df.columns


def test_no_extra_cols_by_default(sample_parquet):
    df = load_nf_dataframe(sample_parquet)
    assert list(df.columns) == ['unique_id', 'ds', 'y']


def test_build_full_df_reassigns_contiguous_ds(sample_parquet):
    a = load_nf_dataframe(sample_parquet)
    b = load_nf_dataframe(sample_parquet)
    full = build_full_df([a, b])
    assert list(full['ds']) == list(range(20))
    assert len(full) == 20
