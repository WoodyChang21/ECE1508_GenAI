import pandas as pd

TRAIN_END = "2022-12-31"
VAL_END = "2023-12-31"


def split_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dt = pd.to_datetime(df["datetime"])
    train = df[dt <= TRAIN_END].reset_index(drop=True)
    val = df[(dt > TRAIN_END) & (dt <= VAL_END)].reset_index(drop=True)
    test = df[dt > VAL_END].reset_index(drop=True)
    return train, val, test
