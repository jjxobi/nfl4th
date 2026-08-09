import numpy as np
import pandas as pd

from nfl4th.models.conversion import (
    load_conversion_model,
    predict_conversion,
    save_conversion_model,
    train_conversion_model,
)


def _synthetic_conversion_data(n_per_class: int = 30) -> pd.DataFrame:
    rows = []
    for i in range(n_per_class):
        # Short yardage: converts. Long yardage: does not.
        rows.append(
            {
                "ydstogo": 1, "yardline_100": 45, "score_differential": 0,
                "game_seconds_remaining": 1800, "qtr": 2,
                "posteam_timeouts_remaining": 3, "defteam_timeouts_remaining": 3,
                "is_home": 1, "converted": 1,
            }
        )
        rows.append(
            {
                "ydstogo": 15, "yardline_100": 60, "score_differential": 0,
                "game_seconds_remaining": 1800, "qtr": 2,
                "posteam_timeouts_remaining": 3, "defteam_timeouts_remaining": 3,
                "is_home": 1, "converted": 0,
            }
        )
    return pd.DataFrame(rows)


def test_predict_conversion_returns_valid_probabilities():
    train_df = _synthetic_conversion_data()
    model = train_conversion_model(train_df, seed=42)

    probs = predict_conversion(model, train_df)

    assert len(probs) == len(train_df)
    assert (probs >= 0).all() and (probs <= 1).all()


def test_training_is_deterministic_given_seed():
    train_df = _synthetic_conversion_data()

    model_a = train_conversion_model(train_df, seed=42)
    model_b = train_conversion_model(train_df, seed=42)

    probs_a = predict_conversion(model_a, train_df)
    probs_b = predict_conversion(model_b, train_df)

    assert np.allclose(probs_a.to_numpy(), probs_b.to_numpy())


def test_learns_the_obvious_pattern():
    train_df = _synthetic_conversion_data()
    model = train_conversion_model(train_df, seed=42)

    short_yardage_row = train_df[train_df["ydstogo"] == 1].iloc[[0]]
    long_yardage_row = train_df[train_df["ydstogo"] == 15].iloc[[0]]

    short_prob = predict_conversion(model, short_yardage_row).iloc[0]
    long_prob = predict_conversion(model, long_yardage_row).iloc[0]

    assert short_prob > long_prob


def test_save_and_load_round_trips_predictions(tmp_path):
    train_df = _synthetic_conversion_data()
    model = train_conversion_model(train_df, seed=42)
    original_probs = predict_conversion(model, train_df)

    save_conversion_model(model, tmp_path / "conversion")
    loaded_model = load_conversion_model(tmp_path / "conversion")
    loaded_probs = predict_conversion(loaded_model, train_df)

    assert np.allclose(original_probs.to_numpy(), loaded_probs.to_numpy())
