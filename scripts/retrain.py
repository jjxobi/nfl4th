import json
import sys
from datetime import date
from itertools import pairwise
from pathlib import Path

from nfl4th.data.ingest import load_pbp
from nfl4th.data.season import current_target_season
from nfl4th.features.build_features import filter_fourth_down_decisions
from nfl4th.pipeline import evaluate, run

# Fixed on purpose: this is a known-good, already-played-out season (see the
# real accuracy numbers for it in README.md's Results section), so the sanity
# check compares against the same reference point every week instead of a
# moving target that could itself be affected by a bad retrain.
# Revisit this once 2025 stops being a reasonably recent season to validate
# against and roll it forward.
SANITY_CHECK_TRAIN_SEASONS = range(2010, 2025)
SANITY_CHECK_HOLDOUT_SEASON = range(2025, 2026)

# The real ensemble model scores 88.9% accuracy on a 2010-2024 train / 2025
# test split (see README.md). 75% leaves generous room for normal
# season-to-season variation while still catching a genuinely broken retrain:
# the real bug fixed during phase 1 (unnormalized features) collapsed
# accuracy to 46%, and always guessing the most common decision only gets
# 55-65%.
SANITY_CHECK_ACCURACY_FLOOR = 0.75

# A real, fully-played season has thousands of fourth-down decisions (see
# real per-season counts in data/raw/pbp); a single week alone has well over
# a hundred. This floor is set far below either, so it only trips when the
# latest season's data came back empty or badly truncated, not on normal
# early-season weeks.
MIN_LATEST_SEASON_DECISIONS = 50

TRAIN_START_SEASON = 2010
MODEL_DIR = Path("api/models")


def main() -> None:
    metrics = evaluate(SANITY_CHECK_TRAIN_SEASONS, SANITY_CHECK_HOLDOUT_SEASON)
    print(f"Sanity check: accuracy={metrics['accuracy']:.3f}, log_loss={metrics['log_loss']:.3f}")

    if metrics["accuracy"] < SANITY_CHECK_ACCURACY_FLOOR:
        print(
            f"Sanity check failed: accuracy {metrics['accuracy']:.3f} is below the "
            f"floor of {SANITY_CHECK_ACCURACY_FLOOR}. Leaving api/models untouched."
        )
        sys.exit(1)

    latest_season = current_target_season(date.today())  # noqa: DTZ011 (date, not datetime)
    print(f"Sanity check passed. Retraining on seasons {TRAIN_START_SEASON} through {latest_season}.")

    report = run(
        train_seasons=range(TRAIN_START_SEASON, latest_season + 1),
        val_seasons=range(latest_season + 1, latest_season + 1),
        test_seasons=range(latest_season + 1, latest_season + 1),
        model_dir=MODEL_DIR,
    )

    if report.empty:
        print("Retrain produced an empty tendency report. Not trusting these artifacts.")
        sys.exit(1)

    # The tendency report is aggregated per coach across every training
    # season combined, so a report that looks healthy overall can still be
    # hiding a current season that came back empty or truncated (a transient
    # fetch failure, or nflverse not having published it yet). Re-check the
    # latest season's own data directly, the same way run() built it, instead
    # of trusting the aggregate.
    latest_season_pbp = load_pbp([latest_season])
    latest_season_decisions = len(filter_fourth_down_decisions(latest_season_pbp))
    if latest_season_decisions < MIN_LATEST_SEASON_DECISIONS:
        print(
            f"Season {latest_season} has only {latest_season_decisions} fourth-down decisions "
            f"in the data actually used for training. Its play-by-play may not be published "
            f"yet, or the fetch was silently truncated. Not trusting these artifacts."
        )
        sys.exit(1)

    # Neither check above touches the conversion model or the findings
    # aggregations at all, so a corrupted conversion model or a broken
    # findings computation could still slip through. Check two cheap, strong
    # invariants on the freshly written findings.json instead of re-deriving
    # a full accuracy metric for it.
    findings = json.loads((MODEL_DIR / "findings.json").read_text())

    conversion_rates = [row["conversion_probability"] for row in findings["conversion_by_distance"]]
    if any(later > earlier for earlier, later in pairwise(conversion_rates)):
        print(
            f"Sanity check failed: conversion_by_distance is not monotonically "
            f"non-increasing by distance ({conversion_rates}). Not trusting these artifacts."
        )
        sys.exit(1)

    league_trend = findings["league_trend"]
    if not league_trend or league_trend[-1]["season"] != latest_season:
        print(
            f"Sanity check failed: league_trend is empty or its last season doesn't "
            f"match this run's latest_season ({latest_season}). Not trusting these artifacts."
        )
        sys.exit(1)

    print(f"Wrote refreshed models to {MODEL_DIR}")


if __name__ == "__main__":
    main()
