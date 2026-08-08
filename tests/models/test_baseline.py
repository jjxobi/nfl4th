import numpy as np
import pandas as pd

from nfl4th.features.build_features import DECISION_CLASSES
from nfl4th.models.baseline import (
    load_baseline,
    predict_baseline,
    save_baseline,
    train_baseline,
    train_baseline_no_coach,
)


def _synthetic_training_data(n_per_class: int = 30) -> pd.DataFrame:
    rows = []
    coaches = ["Coach A", "Coach B"]
    for i in range(n_per_class):
        coach = coaches[i % 2]
        rows.append(
            {
                "coach": coach, "decision": "go_for_it",
                "season": 2020, "week": 1,
                "ydstogo": 1, "yardline_100": 45, "score_differential": -2,
                "game_seconds_remaining": 1800, "qtr": 2,
                "posteam_timeouts_remaining": 3, "defteam_timeouts_remaining": 3,
                "is_home": 1, "career_decisions": i,
            }
        )
        rows.append(
            {
                "coach": coach, "decision": "punt",
                "season": 2020, "week": 1,
                "ydstogo": 9, "yardline_100": 65, "score_differential": 0,
                "game_seconds_remaining": 1800, "qtr": 2,
                "posteam_timeouts_remaining": 3, "defteam_timeouts_remaining": 3,
                "is_home": 0, "career_decisions": i,
            }
        )
        rows.append(
            {
                "coach": coach, "decision": "field_goal",
                "season": 2020, "week": 1,
                "ydstogo": 3, "yardline_100": 15, "score_differential": 1,
                "game_seconds_remaining": 1800, "qtr": 2,
                "posteam_timeouts_remaining": 3, "defteam_timeouts_remaining": 3,
                "is_home": 1, "career_decisions": i,
            }
        )
    return pd.DataFrame(rows)


def test_predict_baseline_returns_valid_probabilities():
    train_df = _synthetic_training_data()
    model = train_baseline(train_df, seed=42)

    probs = predict_baseline(model, train_df)

    assert list(probs.columns) == DECISION_CLASSES
    assert len(probs) == len(train_df)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_training_is_deterministic_given_seed():
    train_df = _synthetic_training_data()

    model_a = train_baseline(train_df, seed=42)
    model_b = train_baseline(train_df, seed=42)

    probs_a = predict_baseline(model_a, train_df)
    probs_b = predict_baseline(model_b, train_df)

    assert np.allclose(probs_a.to_numpy(), probs_b.to_numpy())


def test_learns_the_obvious_pattern():
    train_df = _synthetic_training_data()
    model = train_baseline(train_df, seed=42)

    go_for_it_row = train_df[train_df["decision"] == "go_for_it"].iloc[[0]]
    probs = predict_baseline(model, go_for_it_row)

    assert probs.iloc[0]["go_for_it"] == probs.iloc[0].max()


def test_predict_on_a_single_row_matches_full_batch_prediction():
    # A single-row (or otherwise category-subset) predict call must encode
    # the coach column the same way training did, not re-derive categories
    # from whatever rows happen to be present in this call.
    train_df = _synthetic_training_data()
    model = train_baseline(train_df, seed=42)

    single_row = train_df.iloc[[0]]
    batch_probs = predict_baseline(model, train_df)
    single_probs = predict_baseline(model, single_row)

    assert np.allclose(single_probs.to_numpy()[0], batch_probs.to_numpy()[0])


def test_predict_handles_a_coach_never_seen_in_training():
    train_df = _synthetic_training_data()
    model = train_baseline(train_df, seed=42)

    unseen_row = train_df.iloc[[0]].copy()
    unseen_row["coach"] = "Brand New Coach"

    probs = predict_baseline(model, unseen_row)

    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_train_baseline_no_coach_has_no_coach_categories():
    train_df = _synthetic_training_data()
    model = train_baseline_no_coach(train_df, seed=42)

    assert model.coach_categories_ is None


def test_train_baseline_no_coach_predictions_do_not_vary_by_coach_identity():
    train_df = _synthetic_training_data()
    model = train_baseline_no_coach(train_df, seed=42)

    row = train_df[train_df["decision"] == "go_for_it"].iloc[[0]].copy()
    row_other_coach = row.copy()
    row_other_coach["coach"] = "Some Other Coach"

    probs_same_coach = predict_baseline(model, row)
    probs_other_coach = predict_baseline(model, row_other_coach)

    assert np.allclose(probs_same_coach.to_numpy(), probs_other_coach.to_numpy())


def test_save_and_load_baseline_round_trips_predictions(tmp_path):
    train_df = _synthetic_training_data()
    model = train_baseline(train_df, seed=42)
    original_probs = predict_baseline(model, train_df)

    save_baseline(model, tmp_path / "baseline")
    loaded_model = load_baseline(tmp_path / "baseline")
    loaded_probs = predict_baseline(loaded_model, train_df)

    assert np.allclose(original_probs.to_numpy(), loaded_probs.to_numpy())


def test_save_and_load_baseline_no_coach_round_trips_predictions(tmp_path):
    train_df = _synthetic_training_data()
    model = train_baseline_no_coach(train_df, seed=42)
    original_probs = predict_baseline(model, train_df)

    save_baseline(model, tmp_path / "baseline_no_coach")
    loaded_model = load_baseline(tmp_path / "baseline_no_coach")
    loaded_probs = predict_baseline(loaded_model, train_df)

    assert loaded_model.coach_categories_ is None
    assert np.allclose(original_probs.to_numpy(), loaded_probs.to_numpy())
