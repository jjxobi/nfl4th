# Phase 2 Backend API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a FastAPI backend that loads the already-trained, persisted `nfl4th` models and a precomputed coach tendency report, and serves three read-only endpoints (`GET /coaches`, `GET /coaches/{name}`, `POST /predict`) with no live nflverse calls and no retraining at request time.

**Architecture:** A new top-level `api/` directory (not part of the installable `nfl4th` package, imported the same way the existing `scripts/` directory is run from the repo root) that imports `nfl4th` as a library. `pipeline.py` gets extended to also persist the coach tendency report and a small metadata file alongside the models it already saves. Real, small (~3MB) trained artifacts are committed at `api/models/` so the API never needs network access or retraining to start.

**Tech Stack:** FastAPI, uvicorn, httpx (for `TestClient`), all on the existing Python 3.11 / `nfl4th` stack.

## Global Constraints

- Python 3.11 exactly (`>=3.11,<3.12`), same as the rest of the project.
- Every git commit message is plain, simple, human-sounding language. No em dashes. No "Co-Authored-By" trailer, no mention of Claude or AI anywhere in any commit message. Do not push to any remote.
- Code comments: none by default. Only add a comment where the reasoning is genuinely non-obvious.
- `api/` is a plain directory, not a second installable package. Tests import it as `from api.X import Y` and this only resolves when pytest runs from the repo root, which is why Task 2 adds `pythonpath = ["."]` to `[tool.pytest.ini_options]`. `uvicorn api.main:app` run from the repo root resolves the same way (uvicorn's CLI adds the current working directory to `sys.path`). Every command in this plan assumes the repo root as the working directory, consistent with every other command in this project's existing README.
- **Real committed model artifacts must exist at `api/models/` before Task 2 begins.** This is generated directly by the controller (running `scripts/run_pipeline.py` against real cached nflverse data), not delegated to an implementer subagent — it is a data-generation operation, not a coding task, and doing it via a subagent would mean redownloading ~300-500MB of data in a fresh environment for no benefit. See the note between Task 1 and Task 2 below.
- FastAPI's `TestClient` (via Starlette) emits a `StarletteDeprecationWarning` about `httpx` being deprecated in favor of a package called `httpx2`. This is a known, accepted, third-party warning from the framework itself, not from this project's code — treat it the same way the existing `pandas`/`numpy` `DeprecationWarning`s already present in this test suite are treated: present in the test output, not a defect to chase down.
- Real nflverse play-by-play data already has its own `home_coach`/`away_coach` columns (see `attach_coach` in `src/nfl4th/features/build_features.py`) and features span wildly different raw scales (see the embedding model's normalization) — these are established, verified facts from earlier work, not new research needed here.

---

## Task 1: Persist the coach tendency report and run metadata

**Files:**
- Modify: `src/nfl4th/pipeline.py`
- Modify: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: nothing new; uses `coach_tendency_report` (already imported), `pandas`.
- Produces: `save_run_artifacts(report: pd.DataFrame, all_seasons: list[int], model_dir: Path) -> None`, called from `run()` alongside the existing model-saving calls when `model_dir` is given. Writes `model_dir / "coach_tendency_report.csv"` and `model_dir / "metadata.json"` (containing `{"latest_season": <max season used>}`). Used by `api/loader.py` in Task 2.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_pipeline.py` (needs `import json` added to the existing imports at the top of the file):

```python
def test_run_saves_report_and_metadata_when_model_dir_is_given(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline, "load_pbp", lambda seasons: _synthetic_pbp(seasons))
    monkeypatch.setattr(
        pipeline, "load_schedules", lambda seasons: _synthetic_schedules(seasons, cold_start_season=2022)
    )

    pipeline.run(
        train_seasons=range(2020, 2021),
        val_seasons=range(2021, 2022),
        test_seasons=range(2022, 2023),
        model_dir=tmp_path,
    )

    report_path = tmp_path / "coach_tendency_report.csv"
    metadata_path = tmp_path / "metadata.json"
    assert report_path.exists()
    assert metadata_path.exists()

    saved_report = pd.read_csv(report_path)
    assert "coach" in saved_report.columns
    assert set(saved_report["coach"]) == {"Coach A"}

    metadata = json.loads(metadata_path.read_text())
    assert metadata["latest_season"] == 2022
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_pipeline.py::test_run_saves_report_and_metadata_when_model_dir_is_given -v`
Expected: FAIL, `coach_tendency_report.csv` does not exist (the file is never written yet).

- [ ] **Step 3: Add `save_run_artifacts` and wire it into `run()`**

Add `import json` to the top of `src/nfl4th/pipeline.py` alongside the existing `from pathlib import Path` import.

Add this function after `_print_model_comparison` and before `run`:

```python
def save_run_artifacts(report: pd.DataFrame, all_seasons: list[int], model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    report.to_csv(model_dir / "coach_tendency_report.csv", index=False)
    metadata = {"latest_season": max(all_seasons)}
    (model_dir / "metadata.json").write_text(json.dumps(metadata))
```

Replace the body of `run` with this (the only change is moving the `if model_dir is not None` block to after the report is computed, and adding the `save_run_artifacts` call inside it):

```python
def run(train_seasons: range, val_seasons: range, test_seasons: range, model_dir: Path | None = None):
    all_seasons = list(train_seasons) + list(val_seasons) + list(test_seasons)
    pbp = load_pbp(all_seasons)
    schedules = load_schedules(all_seasons)
    features = build_feature_table(pbp, schedules)

    train_df, val_df, test_df = time_based_split(features, train_seasons, val_seasons, test_seasons)

    baseline_model = train_baseline(train_df)
    baseline_no_coach_model = train_baseline_no_coach(train_df)
    embedding_model, vocab = train_embedding_model(train_df)

    _print_model_comparison(baseline_model, embedding_model, vocab, val_df, test_df)

    # The report's expected/baseline rate comes from the coach-free model, not
    # the coach-aware one, so a coach isn't shrunk toward a memorized version
    # of themselves.
    baseline_no_coach_train_probs = predict_baseline(baseline_no_coach_model, train_df)
    report = coach_tendency_report(train_df, baseline_no_coach_train_probs)

    if model_dir is not None:
        save_baseline(baseline_model, model_dir / "baseline")
        save_baseline(baseline_no_coach_model, model_dir / "baseline_no_coach")
        save_embedding_model(embedding_model, vocab, model_dir / "embedding")
        save_run_artifacts(report, all_seasons, model_dir)

    return report
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/test_pipeline.py -v`
Expected: all tests in this file pass, including the new one and the pre-existing `test_run_saves_models_when_model_dir_is_given`.

- [ ] **Step 5: Run the full suite and lint, then commit**

Run: `.venv/Scripts/python -m pytest -v` and `.venv/Scripts/python -m ruff check .`
Expected: all tests pass (this project's full suite, not just `test_pipeline.py`), lint clean.

```bash
git add src/nfl4th/pipeline.py tests/test_pipeline.py
git commit -m "Save the coach tendency report and run metadata alongside models"
```

---

## Controller step (not a task): generate and commit real `api/models/` artifacts

Before Task 2 starts, run this from the repo root, using the real cached nflverse data already present in `data/raw/` (2010-2025 already cached from earlier work; no network access needed if that cache is still present, otherwise this downloads it):

```bash
.venv/Scripts/python scripts/run_pipeline.py --train-start 2010 --train-end 2026 --val-end 2026 --test-end 2026 --model-dir api/models
```

This trains on all available seasons (2010-2025) with no held-out val/test split — that evaluation already happened once and is documented in `README.md`; these are the actual artifacts the deployed API will serve, so they should be trained on everything available. Verify `api/models/baseline/model.json`, `api/models/baseline_no_coach/model.json`, `api/models/embedding/model.pt`, `api/models/embedding/vocab.json`, `api/models/coach_tendency_report.csv`, and `api/models/metadata.json` all exist afterward, then commit them:

```bash
git add api/models
git commit -m "Add the real trained models the backend api serves"
```

---

## Task 2: FastAPI scaffolding and the startup model/report loader

**Files:**
- Modify: `pyproject.toml`
- Create: `api/__init__.py`
- Create: `api/loader.py`
- Create: `api/main.py`
- Create: `tests/api/__init__.py`
- Create: `tests/api/test_loader.py`
- Create: `tests/api/test_main.py`

**Interfaces:**
- Consumes: `load_baseline` (`nfl4th.models.baseline`), `load_embedding_model` (`nfl4th.models.embedding_model`), and the real committed artifacts at `api/models/` from the controller step above.
- Produces: `LoadedModels` class with attributes `baseline_model`, `baseline_no_coach_model`, `embedding_model`, `vocab`, `report` (a `pandas.DataFrame` indexed by `coach`), `latest_season` (`int`). `app` (the FastAPI instance) in `api/main.py`. Used by `api/routes_coaches.py` and `api/routes_predict.py` in Tasks 3 and 4.

- [ ] **Step 1: Add the `api` dependency extra and pytest path config**

In `pyproject.toml`, add a new optional-dependencies group after the existing `dev` group:

```toml
[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "ruff>=0.4",
]
api = [
    "fastapi>=0.110",
    "uvicorn>=0.30",
    "httpx>=0.27",
]
```

Add `pythonpath = ["."]` to the existing `[tool.pytest.ini_options]` table so `from api.X import Y` resolves in tests:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
```

Run: `.venv/Scripts/python -m pip install -e ".[dev,api]"`
Expected: installs cleanly (fastapi/uvicorn/httpx were already verified compatible with this project's Python 3.11 + pinned pandas during planning).

- [ ] **Step 2: Write the failing test for the loader**

Create `tests/api/__init__.py` (empty), then `tests/api/test_loader.py`:

```python
from api.loader import LoadedModels


def test_loaded_models_loads_real_committed_artifacts():
    loaded = LoadedModels()

    assert loaded.baseline_model is not None
    assert loaded.baseline_no_coach_model is not None
    assert loaded.embedding_model is not None
    assert len(loaded.vocab) > 1
    assert loaded.report.index.name == "coach"
    assert len(loaded.report) > 0
    assert isinstance(loaded.latest_season, int)
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/api/test_loader.py -v`
Expected: FAIL with `ModuleNotFoundError` for `api.loader`.

- [ ] **Step 4: Write `api/loader.py`**

Create `api/__init__.py` (empty), then `api/loader.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from nfl4th.models.baseline import load_baseline
from nfl4th.models.embedding_model import load_embedding_model

DEFAULT_MODEL_DIR = Path(__file__).parent / "models"


class LoadedModels:
    def __init__(self, model_dir: Path = DEFAULT_MODEL_DIR):
        self.baseline_model = load_baseline(model_dir / "baseline")
        self.baseline_no_coach_model = load_baseline(model_dir / "baseline_no_coach")
        self.embedding_model, self.vocab = load_embedding_model(model_dir / "embedding")
        self.report = pd.read_csv(model_dir / "coach_tendency_report.csv").set_index("coach")
        metadata = json.loads((model_dir / "metadata.json").read_text())
        self.latest_season = metadata["latest_season"]
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/api/test_loader.py -v`
Expected: 1 passed. This proves the real committed artifacts load correctly, including through the DLL-load-order-sensitive torch import — if this hangs or crashes with a Windows DLL error, check that `import nfl4th` (which imports torch first, see `src/nfl4th/__init__.py`) happens before any bare `import pandas`/`import torch` elsewhere in the import chain. `api/loader.py` importing from `nfl4th.models.*` already triggers `nfl4th/__init__.py`'s torch-first import, so this should not be an issue in practice, but if it is, that is the mechanism to check.

- [ ] **Step 6: Write the failing test for the app boot**

Create `tests/api/test_main.py`:

```python
from fastapi.testclient import TestClient

from api.main import app


def test_app_starts_and_loads_real_models():
    client = TestClient(app)
    response = client.get("/openapi.json")
    assert response.status_code == 200
```

- [ ] **Step 7: Run the test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/api/test_main.py -v`
Expected: FAIL with `ModuleNotFoundError` for `api.main`.

- [ ] **Step 8: Write `api/main.py`**

```python
from __future__ import annotations

from fastapi import FastAPI

from api.loader import LoadedModels

app = FastAPI(title="NFL 4th Down Coach Tendency API")
loaded = LoadedModels()
```

- [ ] **Step 9: Run the test to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/api/test_main.py -v`
Expected: 1 passed.

- [ ] **Step 10: Run the full suite and lint, then commit**

Run: `.venv/Scripts/python -m pytest -v` and `.venv/Scripts/python -m ruff check .`
Expected: all tests pass, lint clean.

```bash
git add pyproject.toml api/__init__.py api/loader.py api/main.py tests/api/
git commit -m "Add the fastapi app skeleton and the model and report loader"
```

---

## Task 3: Coach schemas and the coach endpoints

**Files:**
- Create: `api/schemas.py`
- Create: `api/routes_coaches.py`
- Modify: `api/main.py`
- Create: `tests/api/test_routes_coaches.py`

**Interfaces:**
- Consumes: `LoadedModels` (Task 2).
- Produces: `DecisionRates`, `CoachProfile` (Pydantic models in `api/schemas.py`); `register_coach_routes(router: APIRouter, loaded: LoadedModels) -> None` (`api/routes_coaches.py`), registers `GET /coaches` and `GET /coaches/{name}`. `DecisionProbabilities` (also in `api/schemas.py`) is produced here for Task 4 to consume.

- [ ] **Step 1: Write the failing tests**

Create `tests/api/test_routes_coaches.py`:

```python
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_list_coaches_returns_known_coaches_with_valid_rates():
    response = client.get("/coaches")

    assert response.status_code == 200
    coaches = response.json()
    assert len(coaches) > 0
    first = coaches[0]
    assert "coach" in first
    assert first["n_decisions"] > 0
    assert 0.0 <= first["shrinkage_weight"] <= 1.0


def test_get_coach_returns_full_profile_for_a_known_coach():
    coach_name = client.get("/coaches").json()[0]["coach"]

    response = client.get(f"/coaches/{coach_name}")

    assert response.status_code == 200
    body = response.json()
    assert body["coach"] == coach_name
    assert "observed" in body["punt"]
    assert "expected" in body["punt"]
    assert "shrunk" in body["punt"]


def test_get_coach_returns_404_for_unknown_coach():
    response = client.get("/coaches/Definitely Not A Real Coach")

    assert response.status_code == 404
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/api/test_routes_coaches.py -v`
Expected: FAIL, `/coaches` returns 404 (no such route registered yet).

- [ ] **Step 3: Write `api/schemas.py`**

```python
from __future__ import annotations

from pydantic import BaseModel


class DecisionRates(BaseModel):
    observed: float
    expected: float
    shrunk: float


class CoachProfile(BaseModel):
    coach: str
    n_decisions: int
    shrinkage_weight: float
    punt: DecisionRates
    field_goal: DecisionRates
    go_for_it: DecisionRates


class DecisionProbabilities(BaseModel):
    punt: float
    field_goal: float
    go_for_it: float
```

- [ ] **Step 4: Write `api/routes_coaches.py`**

```python
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
```

- [ ] **Step 5: Wire the coach routes into `api/main.py`**

Replace the full contents of `api/main.py` with:

```python
from __future__ import annotations

from fastapi import APIRouter, FastAPI

from api.loader import LoadedModels
from api.routes_coaches import register_coach_routes

app = FastAPI(title="NFL 4th Down Coach Tendency API")
loaded = LoadedModels()

router = APIRouter()
register_coach_routes(router, loaded)
app.include_router(router)
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/api/ -v`
Expected: all tests in `tests/api/` pass, including the Task 2 tests still passing.

- [ ] **Step 7: Run the full suite and lint, then commit**

Run: `.venv/Scripts/python -m pytest -v` and `.venv/Scripts/python -m ruff check .`
Expected: all tests pass, lint clean.

```bash
git add api/schemas.py api/routes_coaches.py api/main.py tests/api/test_routes_coaches.py
git commit -m "Add the coach list and coach profile endpoints"
```

---

## Task 4: The predict endpoint

**Files:**
- Modify: `api/schemas.py`
- Create: `api/routes_predict.py`
- Modify: `api/main.py`
- Create: `tests/api/test_routes_predict.py`

**Interfaces:**
- Consumes: `DecisionProbabilities` (Task 3, `api/schemas.py`), `LoadedModels` (Task 2), `predict_baseline` (`nfl4th.models.baseline`), `predict_with_coldstart` (`nfl4th.models.embedding_model`).
- Produces: `SituationRequest`, `PredictResponse` (Pydantic models, `api/schemas.py`); `register_predict_routes(router: APIRouter, loaded: LoadedModels) -> None` (`api/routes_predict.py`), registers `POST /predict`. Nothing later in this plan depends on this task.

- [ ] **Step 1: Write the failing tests**

Create `tests/api/test_routes_predict.py`:

```python
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)

_SITUATION = {
    "ydstogo": 2,
    "yardline_100": 40,
    "score_differential": 0,
    "game_seconds_remaining": 1800,
    "qtr": 2,
    "posteam_timeouts_remaining": 3,
    "defteam_timeouts_remaining": 3,
    "is_home": True,
}


def test_predict_returns_probabilities_for_a_known_coach():
    coach_name = client.get("/coaches").json()[0]["coach"]

    response = client.post("/predict", json={"coach": coach_name, **_SITUATION})

    assert response.status_code == 200
    body = response.json()
    for key in ("predicted", "coach_tendency", "league_baseline"):
        probs = body[key]
        total = probs["punt"] + probs["field_goal"] + probs["go_for_it"]
        assert abs(total - 1.0) < 0.01


def test_predict_returns_404_for_unknown_coach():
    response = client.post("/predict", json={"coach": "Definitely Not A Real Coach", **_SITUATION})

    assert response.status_code == 404
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/api/test_routes_predict.py -v`
Expected: FAIL, `/predict` returns 404 (no such route registered yet).

- [ ] **Step 3: Add the predict schemas to `api/schemas.py`**

Append to `api/schemas.py`:

```python
class SituationRequest(BaseModel):
    coach: str
    ydstogo: float
    yardline_100: float
    score_differential: float
    game_seconds_remaining: float
    qtr: int
    posteam_timeouts_remaining: int
    defteam_timeouts_remaining: int
    is_home: bool
    week: int = 9


class PredictResponse(BaseModel):
    predicted: DecisionProbabilities
    coach_tendency: DecisionProbabilities
    league_baseline: DecisionProbabilities
```

- [ ] **Step 4: Write `api/routes_predict.py`**

```python
from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, HTTPException

from api.loader import LoadedModels
from api.schemas import DecisionProbabilities, PredictResponse, SituationRequest
from nfl4th.models.baseline import predict_baseline
from nfl4th.models.embedding_model import predict_with_coldstart


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
        # Averaging the two models' probabilities beats either alone, see the
        # phase 1 results documented in README.md.
        ensemble_probs = (baseline_probs + embedding_probs) / 2

        league_probs = predict_baseline(loaded.baseline_no_coach_model, situation)

        coach_tendency = DecisionProbabilities(
            punt=float(coach_row["punt_shrunk"]),
            field_goal=float(coach_row["field_goal_shrunk"]),
            go_for_it=float(coach_row["go_for_it_shrunk"]),
        )

        return PredictResponse(
            predicted=_probs_to_schema(ensemble_probs),
            coach_tendency=coach_tendency,
            league_baseline=_probs_to_schema(league_probs),
        )
```

- [ ] **Step 5: Wire the predict route into `api/main.py`**

Replace the full contents of `api/main.py` with:

```python
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
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/api/ -v`
Expected: all tests in `tests/api/` pass.

- [ ] **Step 7: Run the full suite and lint, then commit**

Run: `.venv/Scripts/python -m pytest -v` and `.venv/Scripts/python -m ruff check .`
Expected: all tests pass, lint clean.

```bash
git add api/schemas.py api/routes_predict.py api/main.py tests/api/test_routes_predict.py
git commit -m "Add the situation predict endpoint"
```

---

## Task 5: CI wiring and final verification

**Files:**
- Modify: `.github/workflows/ci.yml`

**Interfaces:**
- None (CI configuration only).

- [ ] **Step 1: Update the CI workflow to install the `api` extra and cache reliably**

Replace the full contents of `.github/workflows/ci.yml` with:

```yaml
name: CI

on:
  push:
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip
          cache-dependency-path: pyproject.toml
      - name: Install CPU-only torch
        run: pip install torch --index-url https://download.pytorch.org/whl/cpu
      - name: Install dependencies
        run: pip install -e ".[dev,api]"
      - name: Lint
        run: ruff check .
      - name: Test
        run: pytest
```

(The `cache-dependency-path` addition closes a previously-noted gap: without it, `actions/setup-python`'s pip cache step can fail to find a dependency file to key on, since this repo has no `requirements.txt`, only `pyproject.toml`.)

- [ ] **Step 2: Run the full suite and lint locally one more time**

Run: `.venv/Scripts/python -m pytest -v` and `.venv/Scripts/python -m ruff check .`
Expected: all tests pass (including every `tests/api/` test against the real committed `api/models/` artifacts), lint clean.

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "Install the api extra in CI so the backend tests run there too"
```
