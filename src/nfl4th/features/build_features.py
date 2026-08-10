from __future__ import annotations

import pandas as pd

DECISION_MAP = {
    "run": "go_for_it",
    "pass": "go_for_it",
    "punt": "punt",
    "field_goal": "field_goal",
}
DECISION_CLASSES = ["punt", "field_goal", "go_for_it"]


def filter_fourth_down_decisions(pbp: pd.DataFrame) -> pd.DataFrame:
    fourth_down = pbp[pbp["down"] == 4]
    real_plays = fourth_down[fourth_down["penalty"] != 1]
    decisions = real_plays[real_plays["play_type"].isin(DECISION_MAP)].copy()
    decisions["decision"] = decisions["play_type"].map(DECISION_MAP)
    return decisions


SITUATIONAL_FEATURES = [
    "season",
    "week",
    "ydstogo",
    "yardline_100",
    "score_differential",
    "game_seconds_remaining",
    "qtr",
    "posteam_timeouts_remaining",
    "defteam_timeouts_remaining",
    "is_home",
    "career_decisions",
]


def attach_coach(decisions: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    # Real play-by-play data already carries its own home_coach/away_coach
    # columns, which would otherwise collide with the schedule's copies and
    # get silently suffixed by merge instead of raising.
    decisions = decisions.drop(columns=["home_coach", "away_coach"], errors="ignore")
    coach_lookup = schedules[["game_id", "home_coach", "away_coach"]]
    merged = decisions.merge(coach_lookup, on="game_id", how="left")
    merged["is_home"] = (merged["posteam"] == merged["home_team"]).astype(int)
    merged["coach"] = merged["home_coach"].where(merged["is_home"] == 1, merged["away_coach"])
    return merged.drop(columns=["home_coach", "away_coach"])


def add_career_decision_count(df: pd.DataFrame) -> pd.DataFrame:
    ordered = df.sort_values(["season", "week", "game_id", "play_id"]).copy()
    ordered["career_decisions"] = ordered.groupby("coach").cumcount()
    return ordered


def build_feature_table(pbp: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    decisions = filter_fourth_down_decisions(pbp)
    with_coach = attach_coach(decisions, schedules)
    with_experience = add_career_decision_count(with_coach)
    output_columns = ["game_id", "coach", "decision"] + SITUATIONAL_FEATURES
    return with_experience[output_columns].reset_index(drop=True)


def build_conversion_training_data(pbp: pd.DataFrame) -> pd.DataFrame:
    decisions = filter_fourth_down_decisions(pbp)
    go_for_it = decisions[decisions["decision"] == "go_for_it"].copy()
    go_for_it["converted"] = go_for_it["fourth_down_converted"].astype(int)
    go_for_it["is_home"] = (go_for_it["posteam"] == go_for_it["home_team"]).astype(int)
    return go_for_it


def build_coach_conversion_data(pbp: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    # Real observed conversion outcomes per coach, for a descriptive
    # leaderboard stat (who actually converts most often). Deliberately
    # separate from the conversion model in models/conversion.py, which is
    # situation-only by design; this is historical record-keeping, not a
    # coach feature used to predict anything.
    decisions = filter_fourth_down_decisions(pbp)
    with_coach = attach_coach(decisions, schedules)
    go_for_it = with_coach[with_coach["decision"] == "go_for_it"].copy()
    go_for_it["converted"] = go_for_it["fourth_down_converted"].astype(int)
    return go_for_it[["coach", "season", "converted"]].reset_index(drop=True)


def build_outcome_context_data(pbp: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    # wpa (win probability added by the play itself) and fixed_drive_result
    # (how the drive containing the play actually ended) are both real,
    # already-computed nflverse fields, not derived here. fourth_down_converted
    # is kept alongside them because "did the drive score" is the wrong
    # success measure for a team that's already leading: keeping the ball
    # (converting) is the actual goal there, not necessarily scoring again.
    # Scoped to go-for-it attempts only, since that's the decision whose
    # real-world payoff is in question.
    decisions = filter_fourth_down_decisions(pbp)
    with_coach = attach_coach(decisions, schedules)
    go_for_it = with_coach[with_coach["decision"] == "go_for_it"].copy()
    return go_for_it[
        ["season", "score_differential", "wpa", "fixed_drive_result", "fourth_down_converted"]
    ].reset_index(drop=True)
