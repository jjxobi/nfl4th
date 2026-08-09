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
    conversion_probability: float = Field(
        description="If a go-for-it attempt is made in this exact situation, the model's estimate of the "
        "chance it succeeds. Not conditioned on the predicted decision actually being go-for-it."
    )


class SituationalSplit(BaseModel):
    distance_bucket: str
    n_decisions: int
    go_for_it_rate: float


class CoachBucketEntry(BaseModel):
    coach: str
    distance_bucket: str
    n_decisions: int
    go_for_it_rate: float


class LeagueTrendPoint(BaseModel):
    season: int
    n_decisions: int
    go_for_it_rate: float


class ConversionByDistance(BaseModel):
    distance_bucket: str
    conversion_probability: float


class FindingsResponse(BaseModel):
    situational_splits: list[SituationalSplit]
    coach_bucket_leaderboard: list[CoachBucketEntry]
    league_trend: list[LeagueTrendPoint]
    conversion_by_distance: list[ConversionByDistance]
