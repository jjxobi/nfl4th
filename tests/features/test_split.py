import pandas as pd

from nfl4th.features.split import time_based_split


def test_splits_by_season_with_no_overlap():
    df = pd.DataFrame({"season": [2018, 2019, 2020, 2021, 2022, 2023], "value": range(6)})

    train, val, test = time_based_split(
        df, train_seasons=range(2018, 2021), val_seasons=range(2021, 2022), test_seasons=range(2022, 2024)
    )

    assert sorted(train["season"].unique()) == [2018, 2019, 2020]
    assert sorted(val["season"].unique()) == [2021]
    assert sorted(test["season"].unique()) == [2022, 2023]
    assert len(train) + len(val) + len(test) == len(df)
