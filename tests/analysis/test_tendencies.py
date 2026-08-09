import pandas as pd

from nfl4th.analysis.tendencies import coach_tendency_report, shrinkage_weight


def test_shrinkage_weight_boundaries():
    assert shrinkage_weight(0, k=10) == 0.0
    assert shrinkage_weight(10, k=10) == 0.5
    assert shrinkage_weight(10_000, k=10) > 0.99


def test_report_blends_toward_baseline_for_small_sample():
    decisions = pd.DataFrame({"coach": ["Rookie"], "decision": ["go_for_it"], "season": [2024]})
    baseline_probs = pd.DataFrame(
        {"punt": [0.7], "field_goal": [0.2], "go_for_it": [0.1]}, index=decisions.index
    )

    report = coach_tendency_report(decisions, baseline_probs, k=10.0)

    row = report.iloc[0]
    assert row["go_for_it_observed"] == 1.0
    assert row["go_for_it_expected"] == 0.1
    assert row["go_for_it_shrunk"] < 0.5


def test_report_converges_to_observed_for_large_sample():
    n = 200
    decisions = pd.DataFrame(
        {"coach": ["Veteran"] * n, "decision": ["go_for_it"] * n, "season": [2024] * n}
    )
    baseline_probs = pd.DataFrame(
        {"punt": [0.7] * n, "field_goal": [0.2] * n, "go_for_it": [0.1] * n}, index=decisions.index
    )

    report = coach_tendency_report(decisions, baseline_probs, k=10.0)

    row = report.iloc[0]
    assert row["go_for_it_shrunk"] > 0.95


def test_report_records_the_coachs_most_recent_season():
    decisions = pd.DataFrame(
        {
            "coach": ["Multi Year", "Multi Year", "Multi Year"],
            "decision": ["go_for_it", "punt", "field_goal"],
            "season": [2020, 2022, 2021],
        }
    )
    baseline_probs = pd.DataFrame(
        {"punt": [0.7, 0.7, 0.7], "field_goal": [0.2, 0.2, 0.2], "go_for_it": [0.1, 0.1, 0.1]},
        index=decisions.index,
    )

    report = coach_tendency_report(decisions, baseline_probs, k=10.0)

    assert report.iloc[0]["last_season"] == 2022
