import json
import subprocess
import sys

import pandas as pd
import pytest

from nfl4th import pipeline


def test_ensemble_probs_averages_two_prediction_frames():
    baseline_probs = pd.DataFrame({"punt": [0.6], "field_goal": [0.3], "go_for_it": [0.1]})
    embedding_probs = pd.DataFrame({"punt": [0.4], "field_goal": [0.1], "go_for_it": [0.5]})

    result = pipeline.ensemble_probs(baseline_probs, embedding_probs)

    assert result["punt"].iloc[0] == pytest.approx(0.5)
    assert result["field_goal"].iloc[0] == pytest.approx(0.2)
    assert result["go_for_it"].iloc[0] == pytest.approx(0.3)


def test_importing_pipeline_in_a_fresh_process_does_not_crash():
    # Regression test for the pandas/torch DLL load order bug on Windows.
    # Must run in a subprocess: once torch has been imported anywhere in this
    # test process (e.g. by an earlier test file), the bug cannot reproduce.
    result = subprocess.run(
        [sys.executable, "-c", "import nfl4th.pipeline"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def _synthetic_pbp(seasons: list[int]) -> pd.DataFrame:
    # posteam always matches home_team here, so every decision in this fixture
    # belongs to the home coach ("Coach A" per _synthetic_schedules below).
    rows = []
    play_id = 1
    for season in seasons:
        for week in [1, 2]:
            for play_type in ["run", "punt", "field_goal"]:
                rows.append(
                    {
                        "down": 4, "penalty": 0, "game_id": f"{season}_{week}_g",
                        "season": season, "week": week, "play_id": play_id,
                        "play_type": play_type, "posteam": "HOME", "defteam": "AWAY",
                        "home_team": "HOME", "away_team": "AWAY",
                        "ydstogo": 2, "yardline_100": 40, "score_differential": 0,
                        "game_seconds_remaining": 1800, "qtr": 2,
                        "posteam_timeouts_remaining": 3, "defteam_timeouts_remaining": 3,
                    }
                )
                play_id += 1
    return pd.DataFrame(rows)


def _synthetic_schedules(seasons: list[int], cold_start_season: int | None = None) -> pd.DataFrame:
    # cold_start_season gets a home coach who never appears in any other
    # season, so a test-season lookup for that coach genuinely misses the
    # training vocabulary instead of just replaying an in-vocabulary coach.
    rows = []
    for season in seasons:
        home_coach = "Rookie Ray" if season == cold_start_season else "Coach A"
        for week in [1, 2]:
            rows.append(
                {"game_id": f"{season}_{week}_g", "home_coach": home_coach, "away_coach": "Coach B"}
            )
    return pd.DataFrame(rows)


def test_run_completes_and_returns_a_report(monkeypatch):
    monkeypatch.setattr(pipeline, "load_pbp", lambda seasons: _synthetic_pbp(seasons))
    monkeypatch.setattr(
        pipeline, "load_schedules", lambda seasons: _synthetic_schedules(seasons, cold_start_season=2022)
    )

    report = pipeline.run(
        train_seasons=range(2020, 2021), val_seasons=range(2021, 2022), test_seasons=range(2022, 2023)
    )

    assert "coach" in report.columns
    assert set(report["coach"]) == {"Coach A"}
    # Report is built from train_df only (season 2020), 2 weeks x 3 decisions each.
    assert report.iloc[0]["n_decisions"] == 6


def test_run_saves_models_when_model_dir_is_given(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline, "load_pbp", lambda seasons: _synthetic_pbp(seasons))
    monkeypatch.setattr(
        pipeline, "load_schedules", lambda seasons: _synthetic_schedules(seasons, cold_start_season=2022)
    )

    pipeline.run(
        train_seasons=range(2020, 2021),
        val_seasons=range(2021, 2022),
        test_seasons=range(2022, 2023),
        model_dir=tmp_path,
    )

    assert (tmp_path / "baseline" / "model.json").exists()
    assert (tmp_path / "baseline_no_coach" / "model.json").exists()
    assert (tmp_path / "embedding" / "model.pt").exists()
    assert (tmp_path / "embedding" / "vocab.json").exists()


def test_run_saves_report_and_metadata_when_model_dir_is_given(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline, "load_pbp", lambda seasons: _synthetic_pbp(seasons))
    monkeypatch.setattr(
        pipeline, "load_schedules", lambda seasons: _synthetic_schedules(seasons, cold_start_season=2022)
    )

    pipeline.run(
        train_seasons=range(2020, 2021),
        val_seasons=range(2021, 2022),
        test_seasons=range(2022, 2023),
        model_dir=tmp_path,
    )

    report_path = tmp_path / "coach_tendency_report.csv"
    metadata_path = tmp_path / "metadata.json"
    assert report_path.exists()
    assert metadata_path.exists()

    saved_report = pd.read_csv(report_path)
    assert "coach" in saved_report.columns
    assert set(saved_report["coach"]) == {"Coach A"}

    metadata = json.loads(metadata_path.read_text())
    assert metadata["latest_season"] == 2022


def test_evaluate_returns_accuracy_and_log_loss(monkeypatch):
    monkeypatch.setattr(pipeline, "load_pbp", lambda seasons: _synthetic_pbp(seasons))
    monkeypatch.setattr(
        pipeline, "load_schedules", lambda seasons: _synthetic_schedules(seasons)
    )

    metrics = pipeline.evaluate(train_seasons=range(2020, 2021), test_seasons=range(2021, 2022))

    assert set(metrics.keys()) == {"accuracy", "log_loss"}
    assert 0.0 <= metrics["accuracy"] <= 1.0
