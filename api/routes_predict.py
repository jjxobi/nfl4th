from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, HTTPException

from api.loader import LoadedModels
from api.schemas import DecisionProbabilities, PredictResponse, SituationRequest
from nfl4th.models.baseline import predict_baseline
from nfl4th.models.conversion import predict_conversion
from nfl4th.models.embedding_model import predict_with_coldstart
from nfl4th.pipeline import ensemble_probs


def _build_situation_row(request: SituationRequest, latest_season: int, career_decisions: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "coach": request.coach,
                "season": latest_season,
                "week": request.week,
                "ydstogo": request.ydstogo,
                "yardline_100": request.yardline_100,
                "score_differential": request.score_differential,
                "game_seconds_remaining": request.game_seconds_remaining,
                "qtr": request.qtr,
                "posteam_timeouts_remaining": request.posteam_timeouts_remaining,
                "defteam_timeouts_remaining": request.defteam_timeouts_remaining,
                "is_home": int(request.is_home),
                "career_decisions": career_decisions,
            }
        ]
    )


def _probs_to_schema(probs: pd.DataFrame) -> DecisionProbabilities:
    row = probs.iloc[0]
    return DecisionProbabilities(
        punt=float(row["punt"]), field_goal=float(row["field_goal"]), go_for_it=float(row["go_for_it"])
    )


def register_predict_routes(router: APIRouter, loaded: LoadedModels) -> None:
    @router.post("/predict", response_model=PredictResponse)
    def predict(request: SituationRequest) -> PredictResponse:
        if request.coach not in loaded.report.index:
            raise HTTPException(status_code=404, detail=f"Unknown coach: {request.coach}")

        coach_row = loaded.report.loc[request.coach]
        situation = _build_situation_row(request, loaded.latest_season, int(coach_row["n_decisions"]))

        baseline_probs = predict_baseline(loaded.baseline_model, situation)
        embedding_probs = predict_with_coldstart(loaded.embedding_model, loaded.vocab, situation)
        combined_probs = ensemble_probs(baseline_probs, embedding_probs)

        league_probs = predict_baseline(loaded.baseline_no_coach_model, situation)
        conversion_probability = predict_conversion(loaded.conversion_model, situation).iloc[0]

        coach_career_average = DecisionProbabilities(
            punt=float(coach_row["punt_shrunk"]),
            field_goal=float(coach_row["field_goal_shrunk"]),
            go_for_it=float(coach_row["go_for_it_shrunk"]),
        )

        return PredictResponse(
            predicted=_probs_to_schema(combined_probs),
            coach_career_average=coach_career_average,
            league_baseline=_probs_to_schema(league_probs),
            conversion_probability=float(conversion_probability),
        )
