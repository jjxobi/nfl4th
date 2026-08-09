from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

from nfl4th.analysis.findings import (
    DISTANCE_BUCKETS,
    coach_bucket_leaderboard,
    league_trend_over_time,
    situational_splits,
)
from nfl4th.analysis.tendencies import coach_tendency_report
from nfl4th.data.ingest import load_pbp, load_schedules
from nfl4th.features.build_features import (
    DECISION_CLASSES,
    build_conversion_training_data,
    build_feature_table,
)
from nfl4th.features.split import time_based_split
from nfl4th.models.baseline import (
    predict_baseline,
    save_baseline,
    train_baseline,
    train_baseline_no_coach,
)
from nfl4th.models.conversion import (
    CONVERSION_FEATURES,  # noqa: F401 -- re-exported for symmetry with the other model imports
    load_conversion_model,  # noqa: F401 -- re-exported for symmetry; nothing here loads a model
    predict_conversion,
    save_conversion_model,
    train_conversion_model,
)
from nfl4th.models.embedding_model import (
    predict_with_coldstart,
    save_embedding_model,
    train_embedding_model,
)


def _score(probs: pd.DataFrame, true_decisions: pd.Series) -> dict[str, float]:
    # log_loss assumes probability columns are in alphabetical label order
    # regardless of what order `labels` is passed in, so the columns must be
    # sorted to match, separately from the DECISION_CLASSES order used
    # elsewhere for the predicted-label lookup.
    sorted_classes = sorted(DECISION_CLASSES)
    predicted = probs[DECISION_CLASSES].idxmax(axis=1)
    return {
        "log_loss": log_loss(true_decisions, probs[sorted_classes].to_numpy(), labels=sorted_classes),
        "accuracy": accuracy_score(true_decisions, predicted),
    }


def ensemble_probs(baseline_probs: pd.DataFrame, embedding_probs: pd.DataFrame) -> pd.DataFrame:
    # Averaging the two models' predicted probabilities costs nothing extra
    # to train and, verified against real 2010-2025 data, reliably beats
    # either model alone on log loss.
    return (baseline_probs + embedding_probs) / 2


def _print_model_comparison(baseline_model, embedding_model, vocab, val_df, test_df) -> None:
    for split_name, split_df in [("val", val_df), ("test", test_df)]:
        if len(split_df) == 0:
            continue
        baseline_probs = predict_baseline(baseline_model, split_df)
        embedding_probs = predict_with_coldstart(embedding_model, vocab, split_df)
        combined_probs = ensemble_probs(baseline_probs, embedding_probs)

        print(f"{split_name} baseline:  {_score(baseline_probs, split_df['decision'])}")
        print(f"{split_name} embedding: {_score(embedding_probs, split_df['decision'])}")
        print(f"{split_name} ensemble:  {_score(combined_probs, split_df['decision'])}")


def save_run_artifacts(report: pd.DataFrame, all_seasons: list[int], model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    report.to_csv(model_dir / "coach_tendency_report.csv", index=False)
    metadata = {"latest_season": max(all_seasons)}
    (model_dir / "metadata.json").write_text(json.dumps(metadata))


def _conversion_grid(conversion_model) -> list[dict]:
    representative_situation = {
        "yardline_100": 50,
        "score_differential": 0,
        "game_seconds_remaining": 1800,
        "qtr": 2,
        "posteam_timeouts_remaining": 3,
        "defteam_timeouts_remaining": 3,
        "is_home": 1,
    }
    # Midpoint values are still hardcoded (they're not derivable from the
    # bucket definition alone), but the set of bucket labels comes from the
    # same DISTANCE_BUCKETS used by findings.py, so the two can't drift apart.
    bucket_midpoint_values = {"1": 1, "2": 2, "3": 3, "4-6": 5, "7-10": 8, "11+": 15}
    bucket_midpoints = {label: bucket_midpoint_values[label] for _, _, label in DISTANCE_BUCKETS}

    rows = []
    for label, ydstogo in bucket_midpoints.items():
        situation = pd.DataFrame([{**representative_situation, "ydstogo": ydstogo}])
        probability = predict_conversion(conversion_model, situation).iloc[0]
        rows.append({"distance_bucket": label, "conversion_probability": float(probability)})
    return rows


def save_findings_artifacts(train_df: pd.DataFrame, conversion_model, model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    findings = {
        "situational_splits": situational_splits(train_df).to_dict(orient="records"),
        "coach_bucket_leaderboard": coach_bucket_leaderboard(train_df).to_dict(orient="records"),
        "league_trend": league_trend_over_time(train_df).to_dict(orient="records"),
        "conversion_by_distance": _conversion_grid(conversion_model),
    }
    (model_dir / "findings.json").write_text(json.dumps(findings))


def evaluate(train_seasons: range, test_seasons: range) -> dict[str, float]:
    all_seasons = list(train_seasons) + list(test_seasons)
    pbp = load_pbp(all_seasons)
    schedules = load_schedules(all_seasons)
    features = build_feature_table(pbp, schedules)

    train_df, _, test_df = time_based_split(features, train_seasons, range(0), test_seasons)

    baseline_model = train_baseline(train_df)
    embedding_model, vocab = train_embedding_model(train_df)

    baseline_probs = predict_baseline(baseline_model, test_df)
    embedding_probs = predict_with_coldstart(embedding_model, vocab, test_df)
    combined_probs = ensemble_probs(baseline_probs, embedding_probs)

    return _score(combined_probs, test_df["decision"])


def run(train_seasons: range, val_seasons: range, test_seasons: range, model_dir: Path | None = None):
    all_seasons = list(train_seasons) + list(val_seasons) + list(test_seasons)
    pbp = load_pbp(all_seasons)
    schedules = load_schedules(all_seasons)
    features = build_feature_table(pbp, schedules)

    train_df, val_df, test_df = time_based_split(features, train_seasons, val_seasons, test_seasons)

    baseline_model = train_baseline(train_df)
    baseline_no_coach_model = train_baseline_no_coach(train_df)
    embedding_model, vocab = train_embedding_model(train_df)

    _print_model_comparison(baseline_model, embedding_model, vocab, val_df, test_df)

    # The report's expected/baseline rate comes from the coach-free model, not
    # the coach-aware one, so a coach isn't shrunk toward a memorized version
    # of themselves.
    baseline_no_coach_train_probs = predict_baseline(baseline_no_coach_model, train_df)
    report = coach_tendency_report(train_df, baseline_no_coach_train_probs)

    conversion_train_df, _, _ = time_based_split(
        build_conversion_training_data(pbp), train_seasons, range(0), range(0)
    )
    conversion_model = train_conversion_model(conversion_train_df)

    if model_dir is not None:
        save_baseline(baseline_model, model_dir / "baseline")
        save_baseline(baseline_no_coach_model, model_dir / "baseline_no_coach")
        save_embedding_model(embedding_model, vocab, model_dir / "embedding")
        save_run_artifacts(report, all_seasons, model_dir)
        save_conversion_model(conversion_model, model_dir / "conversion")
        save_findings_artifacts(train_df, conversion_model, model_dir)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the phase 1 tendency pipeline")
    parser.add_argument("--train-start", type=int, default=2010)
    parser.add_argument("--train-end", type=int, default=2021)
    parser.add_argument("--val-end", type=int, default=2023)
    parser.add_argument("--test-end", type=int, default=2025)
    parser.add_argument(
        "--model-dir", type=str, default=None, help="If set, save trained models here after training"
    )
    args = parser.parse_args()

    report = run(
        range(args.train_start, args.train_end),
        range(args.train_end, args.val_end),
        range(args.val_end, args.test_end),
        model_dir=Path(args.model_dir) if args.model_dir else None,
    )
    print(report.to_string())


if __name__ == "__main__":
    main()
