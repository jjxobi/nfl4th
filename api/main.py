from __future__ import annotations

from fastapi import APIRouter, FastAPI

from api.loader import LoadedModels
from api.routes_coaches import register_coach_routes
from api.routes_predict import register_predict_routes

app = FastAPI(title="NFL 4th Down Coach Tendency API")
loaded = LoadedModels()

router = APIRouter()
register_coach_routes(router, loaded)
register_predict_routes(router, loaded)
app.include_router(router)
