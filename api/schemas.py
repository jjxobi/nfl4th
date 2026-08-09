from __future__ import annotations

from pydantic import BaseModel, Field


class DecisionRates(BaseModel):
    observed: float
    expected: float
    shrunk: float


class CoachProfile(BaseModel):
    coach: str
    n_decisions: int
    shrinkage_weight: float
    last_season: int
    punt: DecisionRates
    field_goal: DecisionRates
    go_for_it: DecisionRates


class DecisionProbabilities(BaseModel):
    punt: float
    field_goal: float
    go_for_it: float


class SituationRequest(BaseModel):
    coach: str
    ydstogo: float = Field(ge=1, le=99)
    yardline_100: float = Field(ge=1, le=99)
    score_differential: float = Field(ge=-60, le=60)
    game_seconds_remaining: float = Field(ge=0, le=3600)
    qtr: int = Field(ge=1, le=5)
    posteam_timeouts_remaining: int = Field(ge=0, le=3)
    defteam_timeouts_remaining: int = Field(ge=0, le=3)
    is_home: bool
    week: int = Field(default=9, description="Defaults to a mid-season week if not specified")


class PredictResponse(BaseModel):
    predicted: DecisionProbabilities = Field(
        description="Model's predicted probabilities for this exact situation, for this coach"
    )
    coach_career_average: DecisionProbabilities = Field(
        description="This coach's shrunk career-wide average rates, not conditioned on the situation"
    )
    league_baseline: DecisionProbabilities = Field(
        description="Predicted probabilities for this exact situation, ignoring which coach is calling it"
    )
