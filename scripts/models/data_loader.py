import pandas as pd

HIST_EXOG_COLS = [
    'open', 'high', 'low', 'close', 'volume',
    'return_4h', 'return_24h', 'is_first_bar',
    'vol_24h', 'vol_60h', 'volume_ratio',
    'rsi_14', 'macd', 'macd_signal', 'macd_diff',
    'bb_upper', 'bb_lower', 'bb_width',
    'vix_log', 'vix_change_1h',
]

FUTR_EXOG_COLS = ['is_first_bar']


def load_nf_dataframe(split_path: str, extra_cols: list[str] | None = None) -> pd.DataFrame:
    """Load a split parquet and return a NeuralForecast-compatible DataFrame.

    Columns returned: unique_id, ds (sequential int starting at 0), y (= return_1h),
    plus any columns listed in extra_cols.

    Integer ds avoids market-hours timestamp gap issues with PatchTST positional encoding.
    """
    df = pd.read_parquet(split_path)
    nf = df.copy()
    nf['unique_id'] = 'SPY'
    nf['ds'] = range(len(nf))
    nf = nf.rename(columns={'return_1h': 'y'})
    cols = ['unique_id', 'ds', 'y'] + (extra_cols or [])
    return nf[cols].reset_index(drop=True)


def build_full_df(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate NeuralForecast DataFrames and reassign a single contiguous ds sequence."""
    combined = pd.concat(dfs, ignore_index=True)
    combined['ds'] = range(len(combined))
    return combined
