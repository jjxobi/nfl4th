from __future__ import annotations

import pandas as pd

from nfl4th.features.build_features import DECISION_CLASSES


def shrinkage_weight(n: int, k: float = 10.0) -> float:
    return n / (n + k)


def coach_tendency_report(
    decisions: pd.DataFrame,
    baseline_probs: pd.DataFrame,
    k: float = 10.0,
) -> pd.DataFrame:
    rows = []
    for coach, group in decisions.groupby("coach"):
        n = len(group)
        weight = shrinkage_weight(n, k)
        baseline_for_coach = baseline_probs.loc[group.index]

        row = {
            "coach": coach,
            "n_decisions": n,
            "shrinkage_weight": weight,
            "last_season": int(group["season"].max()),
        }
        for decision_class in DECISION_CLASSES:
            observed_rate = (group["decision"] == decision_class).mean()
            expected_rate = baseline_for_coach[decision_class].mean()
            row[f"{decision_class}_observed"] = observed_rate
            row[f"{decision_class}_expected"] = expected_rate
            row[f"{decision_class}_shrunk"] = weight * observed_rate + (1 - weight) * expected_rate
        rows.append(row)

    return pd.DataFrame(rows).sort_values("coach").reset_index(drop=True)


def coach_conversion_rates(go_for_it_by_coach: pd.DataFrame) -> pd.DataFrame:
    # Raw observed rate, no shrinkage: unlike decision tendency (where
    # shrinking toward a baseline makes sense because coaches differ in how
    # aggressive they are), conversion success is treated site-wide as a
    # property of the situation, not the coach, so there's no baseline to
    # shrink toward here.
    rows = []
    for coach, group in go_for_it_by_coach.groupby("coach"):
        rows.append(
            {
                "coach": coach,
                "n_go_for_it_attempts": len(group),
                "n_conversions": int(group["converted"].sum()),
                "conversion_rate": float(group["converted"].mean()),
            }
        )
    return pd.DataFrame(rows, columns=["coach", "n_go_for_it_attempts", "n_conversions", "conversion_rate"])
