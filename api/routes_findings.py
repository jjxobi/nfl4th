from __future__ import annotations

from fastapi import APIRouter

from api.loader import LoadedModels
from api.schemas import FindingsResponse


def register_findings_routes(router: APIRouter, loaded: LoadedModels) -> None:
    @router.get("/findings", response_model=FindingsResponse)
    def findings() -> FindingsResponse:
        return FindingsResponse(**loaded.findings)
