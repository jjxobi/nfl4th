from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import xgboost as xgb

from nfl4th.features.build_features import DECISION_CLASSES, SITUATIONAL_FEATURES


def _prepare_matrix(df: pd.DataFrame, coach_categories: pd.CategoricalDtype | None) -> pd.DataFrame:
    features = df[SITUATIONAL_FEATURES].copy()
    if coach_categories is not None:
        # A coach outside coach_categories becomes NaN here, which XGBoost treats
        # as a missing value and routes through its learned default direction.
        # That is the baseline model's cold start behavior for an unseen coach.
        features["coach"] = df["coach"].astype(coach_categories)
    return features


def _fit(train_df: pd.DataFrame, seed: int, include_coach: bool) -> xgb.XGBClassifier:
    label_index = {label: i for i, label in enumerate(DECISION_CLASSES)}
    y = train_df["decision"].map(label_index)
    coach_categories = (
        pd.CategoricalDtype(categories=sorted(train_df["coach"].unique())) if include_coach else None
    )
    X = _prepare_matrix(train_df, coach_categories)

    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=len(DECISION_CLASSES),
        tree_method="hist",
        enable_categorical=True,
        random_state=seed,
        n_estimators=200,
        max_depth=4,
    )
    model.fit(X, y)
    model.coach_categories_ = coach_categories
    return model


def train_baseline(train_df: pd.DataFrame, seed: int = 42) -> xgb.XGBClassifier:
    return _fit(train_df, seed, include_coach=True)


def train_baseline_no_coach(train_df: pd.DataFrame, seed: int = 42) -> xgb.XGBClassifier:
    return _fit(train_df, seed, include_coach=False)


def predict_baseline(model: xgb.XGBClassifier, df: pd.DataFrame) -> pd.DataFrame:
    X = _prepare_matrix(df, model.coach_categories_)
    probs = model.predict_proba(X)
    return pd.DataFrame(probs, columns=DECISION_CLASSES, index=df.index)


def save_baseline(model: xgb.XGBClassifier, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    model.save_model(path / "model.json")
    # coach_categories_ is a plain Python attribute, not part of the
    # booster, so it needs its own sidecar file to survive a reload.
    categories = list(model.coach_categories_.categories) if model.coach_categories_ is not None else None
    (path / "coach_categories.json").write_text(json.dumps(categories))


def load_baseline(path: Path) -> xgb.XGBClassifier:
    model = xgb.XGBClassifier()
    model.load_model(path / "model.json")
    categories = json.loads((path / "coach_categories.json").read_text())
    model.coach_categories_ = pd.CategoricalDtype(categories=categories) if categories is not None else None
    return model
