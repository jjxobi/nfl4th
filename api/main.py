from __future__ import annotations

import os

from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.loader import LoadedModels
from api.routes_coaches import register_coach_routes
from api.routes_predict import register_predict_routes

app = FastAPI(title="NFL 4th Down Coach Tendency API")

allowed_origins = [
    o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "http://localhost:4321").split(",") if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

loaded = LoadedModels()

router = APIRouter()


@router.get("/")
def health() -> dict:
    return {"name": "nfl4th API", "latest_season": loaded.latest_season, "n_coaches": len(loaded.report)}


register_coach_routes(router, loaded)
register_predict_routes(router, loaded)
app.include_router(router)
