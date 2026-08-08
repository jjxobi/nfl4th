import pandas as pd

from nfl4th.features.build_features import (
    DECISION_CLASSES,
    SITUATIONAL_FEATURES,
    add_career_decision_count,
    attach_coach,
    build_feature_table,
    filter_fourth_down_decisions,
)


def _row(**overrides):
    base = {
        "down": 4,
        "play_type": "run",
        "penalty": 0,
        "game_id": "2023_01_ARI_WAS",
        "season": 2023,
        "week": 1,
        "play_id": 1,
        "posteam": "ARI",
        "defteam": "WAS",
        "home_team": "WAS",
        "away_team": "ARI",
    }
    base.update(overrides)
    return base


def test_keeps_real_decisions_and_maps_labels():
    pbp = pd.DataFrame(
        [
            _row(play_type="run", play_id=1),
            _row(play_type="pass", play_id=2),
            _row(play_type="punt", play_id=3),
            _row(play_type="field_goal", play_id=4),
        ]
    )

    result = filter_fourth_down_decisions(pbp)

    assert list(result["decision"]) == ["go_for_it", "go_for_it", "punt", "field_goal"]
    assert set(result["decision"].unique()).issubset(set(DECISION_CLASSES))


def test_drops_non_decision_and_penalty_negated_plays():
    pbp = pd.DataFrame(
        [
            _row(down=3, play_type="pass", play_id=1),
            _row(play_type="no_play", penalty=1, play_id=2),
            _row(play_type="qb_kneel", play_id=3),
            _row(play_type=None, play_id=4),
            _row(play_type="punt", penalty=1, play_id=5),
        ]
    )

    result = filter_fourth_down_decisions(pbp)

    assert len(result) == 0


def _situational_row(**overrides):
    base = _row(
        ydstogo=2,
        yardline_100=40,
        score_differential=-3,
        game_seconds_remaining=1800,
        qtr=2,
        posteam_timeouts_remaining=3,
        defteam_timeouts_remaining=3,
    )
    base.update(overrides)
    return base


def test_attach_coach_picks_home_or_away_by_posteam():
    decisions = pd.DataFrame(
        [
            _row(posteam="ARI", home_team="WAS", away_team="ARI", play_id=1),
            _row(posteam="WAS", home_team="WAS", away_team="ARI", play_id=2),
        ]
    )
    decisions["decision"] = "go_for_it"
    schedules = pd.DataFrame(
        [{"game_id": "2023_01_ARI_WAS", "home_coach": "Ron Rivera", "away_coach": "Jonathan Gannon"}]
    )

    result = attach_coach(decisions, schedules)

    assert list(result["coach"]) == ["Jonathan Gannon", "Ron Rivera"]
    assert list(result["is_home"]) == [0, 1]


def test_attach_coach_ignores_pbp_own_coach_columns():
    # Real nflverse play-by-play data already carries home_coach/away_coach
    # columns of its own. Schedules must win without pandas silently
    # suffixing both to home_coach_x/home_coach_y.
    decisions = pd.DataFrame(
        [
            _row(
                posteam="ARI",
                home_team="WAS",
                away_team="ARI",
                play_id=1,
                home_coach="Wrong Coach",
                away_coach="Also Wrong",
            ),
        ]
    )
    decisions["decision"] = "go_for_it"
    schedules = pd.DataFrame(
        [{"game_id": "2023_01_ARI_WAS", "home_coach": "Ron Rivera", "away_coach": "Jonathan Gannon"}]
    )

    result = attach_coach(decisions, schedules)

    assert list(result["coach"]) == ["Jonathan Gannon"]


def test_add_career_decision_count_is_per_coach_and_chronological():
    df = pd.DataFrame(
        [
            {"coach": "A", "season": 2020, "week": 3, "game_id": "g3", "play_id": 1},
            {"coach": "B", "season": 2020, "week": 1, "game_id": "g1", "play_id": 2},
            {"coach": "A", "season": 2020, "week": 1, "game_id": "g1", "play_id": 1},
            {"coach": "B", "season": 2020, "week": 3, "game_id": "g3", "play_id": 2},
            {"coach": "A", "season": 2020, "week": 2, "game_id": "g2", "play_id": 1},
        ]
    )

    result = add_career_decision_count(df).sort_values(["coach", "week"])

    a_counts = result[result["coach"] == "A"]["career_decisions"].tolist()
    b_counts = result[result["coach"] == "B"]["career_decisions"].tolist()
    assert a_counts == [0, 1, 2]
    assert b_counts == [0, 1]


def test_build_feature_table_end_to_end():
    pbp = pd.DataFrame(
        [
            _situational_row(posteam="ARI", home_team="WAS", away_team="ARI", play_type="run", play_id=1),
            _situational_row(posteam="WAS", home_team="WAS", away_team="ARI", play_type="punt", play_id=2, week=2, game_id="g2"),
        ]
    )
    schedules = pd.DataFrame(
        [
            {"game_id": "2023_01_ARI_WAS", "home_coach": "Ron Rivera", "away_coach": "Jonathan Gannon"},
            {"game_id": "g2", "home_coach": "Ron Rivera", "away_coach": "Jonathan Gannon"},
        ]
    )

    result = build_feature_table(pbp, schedules)

    expected_columns = ["game_id", "coach", "decision"] + SITUATIONAL_FEATURES
    assert list(result.columns) == expected_columns
    assert len(result) == 2
