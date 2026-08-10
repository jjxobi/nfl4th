import pandas as pd

from nfl4th.analysis.findings import (
    coach_bucket_leaderboard,
    league_trend_over_time,
    outcome_impact_by_context,
    situational_splits,
)


def _synthetic_decisions() -> pd.DataFrame:
    rows = []
    # Bucket "1": mostly go_for_it. Bucket "11+": mostly punt.
    for i in range(20):
        rows.append(
            {
                "coach": "Coach A" if i % 2 == 0 else "Coach B",
                "decision": "go_for_it" if i < 16 else "punt",
                "season": 2020,
                "ydstogo": 1,
            }
        )
    for i in range(20):
        rows.append(
            {
                "coach": "Coach A" if i % 2 == 0 else "Coach B",
                "decision": "punt" if i < 16 else "go_for_it",
                "season": 2021,
                "ydstogo": 15,
            }
        )
    return pd.DataFrame(rows)


def test_situational_splits_buckets_by_distance():
    decisions = _synthetic_decisions()

    result = situational_splits(decisions)

    bucket_1 = result[result["distance_bucket"] == "1"].iloc[0]
    bucket_11plus = result[result["distance_bucket"] == "11+"].iloc[0]
    assert bucket_1["n_decisions"] == 20
    assert bucket_1["go_for_it_rate"] == 0.8
    assert bucket_11plus["n_decisions"] == 20
    assert bucket_11plus["go_for_it_rate"] == 0.2


def test_situational_splits_omits_empty_buckets():
    decisions = _synthetic_decisions()

    result = situational_splits(decisions)

    assert "4-6" not in result["distance_bucket"].tolist()


def test_coach_bucket_leaderboard_respects_minimum_attempts():
    decisions = _synthetic_decisions()

    result = coach_bucket_leaderboard(decisions, min_attempts=15)

    assert len(result) == 0


def test_coach_bucket_leaderboard_includes_coaches_with_enough_attempts():
    decisions = _synthetic_decisions()

    result = coach_bucket_leaderboard(decisions, min_attempts=5)

    coach_a_bucket_1 = result[(result["coach"] == "Coach A") & (result["distance_bucket"] == "1")].iloc[0]
    assert coach_a_bucket_1["n_decisions"] == 10
    assert coach_a_bucket_1["go_for_it_rate"] == 0.8


def test_league_trend_over_time_has_one_row_per_season():
    decisions = _synthetic_decisions()

    result = league_trend_over_time(decisions)

    assert sorted(result["season"].tolist()) == [2020, 2021]
    row_2020 = result[result["season"] == 2020].iloc[0]
    assert row_2020["n_decisions"] == 20
    assert row_2020["go_for_it_rate"] == 0.8


def _synthetic_outcomes() -> pd.DataFrame:
    return pd.DataFrame(
        [
            # Trailing: one scored, one didn't convert at all.
            {"score_differential": -7, "wpa": 0.08, "fixed_drive_result": "Touchdown", "fourth_down_converted": 1},
            {"score_differential": -3, "wpa": -0.05, "fixed_drive_result": "Turnover on downs", "fourth_down_converted": 0},
            # Tied: scored via field goal.
            {"score_differential": 0, "wpa": 0.02, "fixed_drive_result": "Field goal", "fourth_down_converted": 1},
            # Leading: both attempts converted (kept the ball), but only one
            # led to another score. The other just ran clock before punting,
            # which is still a real win for a team protecting a lead.
            {"score_differential": 4, "wpa": 0.03, "fixed_drive_result": "Touchdown", "fourth_down_converted": 1},
            {"score_differential": 10, "wpa": -0.01, "fixed_drive_result": "Punt", "fourth_down_converted": 1},
        ]
    )


def test_outcome_impact_by_context_splits_trailing_tied_leading():
    result = outcome_impact_by_context(_synthetic_outcomes())

    assert sorted(result["context"].tolist()) == ["leading", "tied", "trailing"]

    trailing = result[result["context"] == "trailing"].iloc[0]
    assert trailing["n_decisions"] == 2
    assert trailing["avg_win_prob_added"] == (0.08 + -0.05) / 2
    assert trailing["converted_rate"] == 0.5
    assert trailing["drive_scored_rate"] == 0.5

    tied = result[result["context"] == "tied"].iloc[0]
    assert tied["n_decisions"] == 1
    assert tied["converted_rate"] == 1.0
    assert tied["drive_scored_rate"] == 1.0

    leading = result[result["context"] == "leading"].iloc[0]
    assert leading["n_decisions"] == 2
    # Both kept the ball, but the drive_scored metric alone would miss that
    # the punt-ending attempt still did its job for a leading team.
    assert leading["converted_rate"] == 1.0
    assert leading["drive_scored_rate"] == 0.5


def test_outcome_impact_by_context_omits_empty_contexts():
    outcomes = pd.DataFrame(
        [{"score_differential": -7, "wpa": 0.08, "fixed_drive_result": "Touchdown", "fourth_down_converted": 1}]
    )

    result = outcome_impact_by_context(outcomes)

    assert result["context"].tolist() == ["trailing"]
