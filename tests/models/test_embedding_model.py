import numpy as np
import pandas as pd
import torch

from nfl4th.features.build_features import DECISION_CLASSES, SITUATIONAL_FEATURES
from nfl4th.models.embedding_model import (
    CoachVocab,
    TendencyEmbeddingModel,
    load_embedding_model,
    predict_with_coldstart,
    save_embedding_model,
    train_embedding_model,
)


def _synthetic_training_data(n_per_coach: int = 25) -> pd.DataFrame:
    rows = []
    for i in range(n_per_coach):
        rows.append(
            {
                "coach": "Aggressive Al", "decision": "go_for_it",
                "season": 2020, "week": 1,
                "ydstogo": 2, "yardline_100": 50, "score_differential": 0,
                "game_seconds_remaining": 1800, "qtr": 2,
                "posteam_timeouts_remaining": 3, "defteam_timeouts_remaining": 3,
                "is_home": 1, "career_decisions": i,
            }
        )
        rows.append(
            {
                "coach": "Cautious Carl", "decision": "punt",
                "season": 2020, "week": 1,
                "ydstogo": 2, "yardline_100": 50, "score_differential": 0,
                "game_seconds_remaining": 1800, "qtr": 2,
                "posteam_timeouts_remaining": 3, "defteam_timeouts_remaining": 3,
                "is_home": 1, "career_decisions": i,
            }
        )
    return pd.DataFrame(rows)


def test_vocab_encodes_known_coaches_and_reserves_zero_for_unknown():
    vocab = CoachVocab(["Aggressive Al", "Cautious Carl"])

    assert vocab.encode("Aggressive Al") != 0
    assert vocab.encode("Cautious Carl") != 0
    assert vocab.encode("Someone New") == 0
    assert len(vocab) == 3


def test_forward_pass_shape():
    model = TendencyEmbeddingModel(n_coaches=3, n_situational=len(SITUATIONAL_FEATURES))
    coach_idx = torch.tensor([1, 2])
    situational = torch.zeros((2, len(SITUATIONAL_FEATURES)))

    logits = model(coach_idx, situational)

    assert logits.shape == (2, len(DECISION_CLASSES))


def test_training_reduces_loss():
    train_df = _synthetic_training_data()

    early_model, early_vocab = train_embedding_model(train_df, epochs=1, seed=42)
    late_model, late_vocab = train_embedding_model(train_df, epochs=100, seed=42)

    early_probs = predict_with_coldstart(early_model, early_vocab, train_df)
    late_probs = predict_with_coldstart(late_model, late_vocab, train_df)

    label_index = {label: i for i, label in enumerate(DECISION_CLASSES)}
    true_class = train_df["decision"].map(label_index).to_numpy()

    early_loss = -np.log(early_probs.to_numpy()[np.arange(len(true_class)), true_class]).mean()
    late_loss = -np.log(late_probs.to_numpy()[np.arange(len(true_class)), true_class]).mean()

    assert late_loss < early_loss


def test_coldstart_prediction_uses_mean_embedding():
    train_df = _synthetic_training_data()
    model, vocab = train_embedding_model(train_df, epochs=20, seed=42)

    unseen_row = train_df.iloc[[0]].copy()
    unseen_row["coach"] = "Brand New Coach"

    result = predict_with_coldstart(model, vocab, unseen_row)

    mean_embedding = model.mean_coach_embedding().detach()
    situational = torch.tensor(unseen_row[SITUATIONAL_FEATURES].to_numpy(dtype="float32"))
    with torch.no_grad():
        combined = torch.cat([mean_embedding.unsqueeze(0), model.normalize(situational)], dim=1)
        expected_probs = torch.softmax(model.classifier(combined), dim=1).numpy()

    assert np.allclose(result.to_numpy(), expected_probs, atol=1e-6)


def test_train_embedding_model_normalizes_features():
    # Real situational features span wildly different raw scales (season is
    # ~2010-2024, game_seconds_remaining is 0-3600). Training must fit
    # per-feature mean/std from train_df, not leave the identity transform
    # in place, or gradient descent stalls on the widest-scale features.
    train_df = _synthetic_training_data()
    train_df["game_seconds_remaining"] = [1800 + i * 10 for i in range(len(train_df))]

    model, _ = train_embedding_model(train_df, epochs=1, seed=42)

    game_seconds_index = SITUATIONAL_FEATURES.index("game_seconds_remaining")
    expected_mean = train_df["game_seconds_remaining"].astype("float32").mean()
    expected_std = train_df["game_seconds_remaining"].astype("float32").std()

    assert model.feature_mean[game_seconds_index].item() == np.float32(expected_mean)
    assert model.feature_std[game_seconds_index].item() == np.float32(expected_std)


def test_train_embedding_model_guards_against_zero_variance_features():
    # Several features in this fixture are constant across every row
    # (season, week, ydstogo, ...). Their std is 0, which would divide by
    # zero during normalization if not guarded.
    train_df = _synthetic_training_data()

    model, vocab = train_embedding_model(train_df, epochs=1, seed=42)
    result = predict_with_coldstart(model, vocab, train_df)

    assert not result.isna().any().any()


def test_save_and_load_embedding_model_round_trips_predictions(tmp_path):
    train_df = _synthetic_training_data()
    model, vocab = train_embedding_model(train_df, epochs=5, seed=42)
    original_probs = predict_with_coldstart(model, vocab, train_df)

    save_embedding_model(model, vocab, tmp_path / "embedding")
    loaded_model, loaded_vocab = load_embedding_model(tmp_path / "embedding")
    loaded_probs = predict_with_coldstart(loaded_model, loaded_vocab, train_df)

    assert np.allclose(original_probs.to_numpy(), loaded_probs.to_numpy())


def test_save_and_load_embedding_model_preserves_coldstart_behavior(tmp_path):
    train_df = _synthetic_training_data()
    model, vocab = train_embedding_model(train_df, epochs=5, seed=42)

    unseen_row = train_df.iloc[[0]].copy()
    unseen_row["coach"] = "Brand New Coach"
    original_probs = predict_with_coldstart(model, vocab, unseen_row)

    save_embedding_model(model, vocab, tmp_path / "embedding")
    loaded_model, loaded_vocab = load_embedding_model(tmp_path / "embedding")
    loaded_probs = predict_with_coldstart(loaded_model, loaded_vocab, unseen_row)

    assert loaded_vocab.encode("Brand New Coach") == 0
    assert np.allclose(original_probs.to_numpy(), loaded_probs.to_numpy())
