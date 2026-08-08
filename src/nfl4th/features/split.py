from __future__ import annotations

import pandas as pd


def time_based_split(
    df: pd.DataFrame,
    train_seasons: range,
    val_seasons: range,
    test_seasons: range,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df[df["season"].isin(train_seasons)].reset_index(drop=True)
    val = df[df["season"].isin(val_seasons)].reset_index(drop=True)
    test = df[df["season"].isin(test_seasons)].reset_index(drop=True)
    return train, val, test
