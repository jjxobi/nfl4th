from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, HTTPException

from api.loader import LoadedModels
from api.schemas import CoachProfile, DecisionRates


def _coach_profile_from_row(coach: str, row: pd.Series) -> CoachProfile:
    return CoachProfile(
        coach=coach,
        n_decisions=int(row["n_decisions"]),
        shrinkage_weight=float(row["shrinkage_weight"]),
        last_season=int(row["last_season"]),
        n_go_for_it_attempts=int(row["n_go_for_it_attempts"]),
        n_conversions=int(row["n_conversions"]),
        conversion_rate=float(row["conversion_rate"]),
        punt=DecisionRates(
            observed=float(row["punt_observed"]),
            expected=float(row["punt_expected"]),
            shrunk=float(row["punt_shrunk"]),
        ),
        field_goal=DecisionRates(
            observed=float(row["field_goal_observed"]),
            expected=float(row["field_goal_expected"]),
            shrunk=float(row["field_goal_shrunk"]),
        ),
        go_for_it=DecisionRates(
            observed=float(row["go_for_it_observed"]),
            expected=float(row["go_for_it_expected"]),
            shrunk=float(row["go_for_it_shrunk"]),
        ),
    )


def register_coach_routes(router: APIRouter, loaded: LoadedModels) -> None:
    @router.get("/coaches", response_model=list[CoachProfile])
    def list_coaches() -> list[CoachProfile]:
        return [_coach_profile_from_row(coach, row) for coach, row in loaded.report.iterrows()]

    @router.get("/coaches/{name}", response_model=CoachProfile)
    def get_coach(name: str) -> CoachProfile:
        if name not in loaded.report.index:
            raise HTTPException(status_code=404, detail=f"Unknown coach: {name}")
        return _coach_profile_from_row(name, loaded.report.loc[name])
