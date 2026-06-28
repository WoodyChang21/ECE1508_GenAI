"""Split features.parquet into train/val/test Parquet files.

Split boundaries:
  Train: 2010-01-01 -> 2022-12-31
  Val:   2023-01-01 -> 2023-12-31
  Test:  2024-01-01 -> 2025-06-01

Usage:
    python scripts/data/03_split.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.data.split_utils import split_dataframe

PROCESSED_DIR = Path("data/processed")
SPLITS_DIR = Path("data/splits")
SPLITS_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    df = pd.read_parquet(PROCESSED_DIR / "features.parquet")

    train, val, test = split_dataframe(df)

    train.to_parquet(SPLITS_DIR / "train.parquet", index=False)
    val.to_parquet(SPLITS_DIR / "val.parquet", index=False)
    test.to_parquet(SPLITS_DIR / "test.parquet", index=False)

    for name, split in [("Train", train), ("Val", val), ("Test", test)]:
        if len(split) > 0:
            print(
                f"{name:5s}: {len(split):6d} rows  "
                f"{split['datetime'].min().date()} -> {split['datetime'].max().date()}"
            )
        else:
            print(f"{name:5s}:      0 rows  (no data in this date range)")


if __name__ == "__main__":
    main()
