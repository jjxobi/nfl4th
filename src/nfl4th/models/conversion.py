from __future__ import annotations

from pathlib import Path

import pandas as pd
import xgboost as xgb

CONVERSION_FEATURES = [
    "ydstogo",
    "yardline_100",
    "score_differential",
    "game_seconds_remaining",
    "qtr",
    "posteam_timeouts_remaining",
    "defteam_timeouts_remaining",
    "is_home",
]


def train_conversion_model(train_df: pd.DataFrame, seed: int = 42) -> xgb.XGBClassifier:
    X = train_df[CONVERSION_FEATURES]
    y = train_df["converted"]

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        tree_method="hist",
        random_state=seed,
        n_estimators=200,
        max_depth=4,
    )
    model.fit(X, y)
    return model


def predict_conversion(model: xgb.XGBClassifier, df: pd.DataFrame) -> pd.Series:
    X = df[CONVERSION_FEATURES]
    probs = model.predict_proba(X)[:, 1]
    return pd.Series(probs, index=df.index, name="conversion_probability")


def save_conversion_model(model: xgb.XGBClassifier, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    model.save_model(path / "model.json")


def load_conversion_model(path: Path) -> xgb.XGBClassifier:
    model = xgb.XGBClassifier()
    model.load_model(path / "model.json")
    return model
