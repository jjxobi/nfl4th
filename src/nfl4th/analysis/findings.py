from __future__ import annotations

import pandas as pd

# (low, high, label). high=None means "low or more".
DISTANCE_BUCKETS: list[tuple[int, int | None, str]] = [
    (1, 1, "1"),
    (2, 2, "2"),
    (3, 3, "3"),
    (4, 6, "4-6"),
    (7, 10, "7-10"),
    (11, None, "11+"),
]


def _bucket_label(ydstogo: int) -> str:
    for low, high, label in DISTANCE_BUCKETS:
        if high is None:
            if ydstogo >= low:
                return label
        elif low <= ydstogo <= high:
            return label
    raise ValueError(f"ydstogo {ydstogo} did not match any distance bucket")


def situational_splits(decisions: pd.DataFrame) -> pd.DataFrame:
    working = decisions.copy()
    working["distance_bucket"] = working["ydstogo"].apply(_bucket_label)

    rows = []
    for _, _, label in DISTANCE_BUCKETS:
        bucket_df = working[working["distance_bucket"] == label]
        if len(bucket_df) == 0:
            continue
        rows.append(
            {
                "distance_bucket": label,
                "n_decisions": len(bucket_df),
                "go_for_it_rate": (bucket_df["decision"] == "go_for_it").mean(),
            }
        )
    return pd.DataFrame(rows)


def coach_bucket_leaderboard(decisions: pd.DataFrame, min_attempts: int = 10) -> pd.DataFrame:
    working = decisions.copy()
    working["distance_bucket"] = working["ydstogo"].apply(_bucket_label)

    rows = []
    for (coach, bucket), group in working.groupby(["coach", "distance_bucket"]):
        if len(group) < min_attempts:
            continue
        rows.append(
            {
                "coach": coach,
                "distance_bucket": bucket,
                "n_decisions": len(group),
                "go_for_it_rate": (group["decision"] == "go_for_it").mean(),
            }
        )
    df = pd.DataFrame(rows, columns=["coach", "distance_bucket", "n_decisions", "go_for_it_rate"])
    if len(df) == 0:
        return df
    return df.sort_values(["distance_bucket", "go_for_it_rate"], ascending=[True, False]).reset_index(drop=True)


def league_trend_over_time(decisions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for season, group in decisions.groupby("season"):
        rows.append(
            {
                "season": int(season),
                "n_decisions": len(group),
                "go_for_it_rate": (group["decision"] == "go_for_it").mean(),
            }
        )
    return pd.DataFrame(rows).sort_values("season").reset_index(drop=True)
