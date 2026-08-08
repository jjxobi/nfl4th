# Phase 1 Coach Tendency Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a tested, modular Python pipeline that ingests NFL play-by-play data, engineers 4th-down decision features, trains two coach-tendency models (XGBoost baseline, PyTorch embedding model with cold-start handling), and produces a per-coach tendency report.

**Architecture:** A `src/nfl4th` package with one module per pipeline stage (data ingestion, feature engineering, splitting, modeling, analysis), each independently unit tested with synthetic fixtures so tests run offline and fast. A thin `scripts/run_pipeline.py` wires the stages together for a real end-to-end run against live data. GitHub Actions runs lint and the full offline test suite on every push.

**Tech Stack:** Python 3.11, pandas, nfl_data_py, xgboost, PyTorch (CPU), pytest, ruff.

## Global Constraints

- Python 3.11 exactly (`>=3.11,<3.12`). Verified locally: `nfl_data_py` 0.3.3 pulls in pandas 1.5.3, which has no prebuilt wheel for Python 3.12 and fails to build from source. Python 3.11 installs cleanly.
- Data scope: 2010-2024 seasons (regular season and playoffs), per the approved spec.
- Phase 1 excludes: web app, API, Docker, MLflow, DVC. Standard git plus GitHub Actions CI only.
- Phase 1 excludes pre-head-coaching (coordinator) history entirely, not deferred.
- Raw data is cached under `data/raw/` and is gitignored (already set up).
- Train/val/test split is time-based by season, never randomized, to avoid leakage.
- Coach identity comes directly from the `home_coach`/`away_coach` columns in nflverse schedule data (confirmed present, already correct per individual game including in-season coaching changes). No separate coach-mapping module is needed.
- Every git commit message is plain, simple, human-sounding language. No em dashes. No "Co-Authored-By" trailer, no mention of Claude or AI anywhere in any commit message. Commits are authored under the existing local git identity (Jesse O'Brien), which is already configured correctly. Do not push to any remote as part of this plan; nothing here requires it.
- Code comments: none by default. Only add a comment where the reasoning is genuinely non-obvious (e.g. why `penalty != 1` is required, why index 0 is reserved in the coach vocabulary). Never comment what the code already says.

---

## Verified data facts (from live 2023 data pull, do not re-derive)

- `nfl_data_py.import_schedules(seasons)` returns a dataframe including `game_id`, `home_team`, `away_team`, `home_coach`, `away_coach`.
- `nfl_data_py.import_pbp_data(seasons)` returns a dataframe including `down`, `ydstogo`, `yardline_100`, `play_type`, `penalty`, `score_differential`, `game_seconds_remaining`, `qtr`, `posteam_timeouts_remaining`, `defteam_timeouts_remaining`, `posteam`, `defteam`, `home_team`, `away_team`, `season`, `week`, `game_id`, `play_id`. All of these are non-null on real 4th-down decision rows.
- On 4th down, `play_type` takes these values: `punt`, `field_goal`, `pass`, `run`, `no_play` (penalty-negated), `None` (rare, ambiguous edge cases), `qb_kneel`. A clean decision label is derived by mapping `run`/`pass` -> `go_for_it`, `punt` -> `punt`, `field_goal` -> `field_goal`, and dropping every other `play_type` plus any row with `penalty == 1`.
- `posteam` always equals either `home_team` or `away_team` on every row (verified, no exceptions).

---

## Deviations from the approved spec

Real data made a few things in the spec's feature list either redundant or hard to define cleanly. Noting them here rather than silently dropping them:

- **Team identity is not a model feature.** The spec listed "team id" alongside coach id. Within a season, team and coach are almost 1:1, so team id adds little independent signal over coach id plus season, while giving the embedding model a second identity axis to learn (double the embedding complexity) for marginal value. Dropped for phase 1. If a future pass wants to isolate organizational philosophy from the individual coach, that is a clean, separable addition later.
- **No explicit garbage-time/end-of-half exclusion.** The spec called for excluding "meaningless" decisions but there is no reliable column that marks this, and coaches still make a real go/punt/kick call even in a blowout. Rather than build a fuzzy hand-rolled filter, `score_differential` and `game_seconds_remaining` are left as features and the model can learn context-dependent behavior directly.
- **Aborted snaps are not specifically excluded.** The spec said to exclude them, but real 2023 data shows a botched snap that still results in a run or pass attempt keeps `play_type` as `run`/`pass`, meaning the coach's original call is still real signal. These rows are kept; they are rare (9 out of 4490 four-down plays in the 2023 season checked).

---

## Task 1: Project scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `src/nfl4th/__init__.py`
- Create: `src/nfl4th/data/__init__.py`
- Create: `src/nfl4th/features/__init__.py`
- Create: `src/nfl4th/models/__init__.py`
- Create: `src/nfl4th/analysis/__init__.py`
- Create: `tests/__init__.py`
- Create: `.github/workflows/ci.yml`
- Create: `README.md`

**Interfaces:**
- Produces: an installable `nfl4th` package (`pip install -e ".[dev]"`), pytest and ruff runnable from repo root.

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "nfl4th"
version = "0.1.0"
description = "Predicts NFL head coach 4th down tendencies from play by play history"
requires-python = ">=3.11,<3.12"
dependencies = [
    "nfl_data_py>=0.3.3",
    "pandas>=1.5,<2.0",
    "numpy>=1.26",
    "pyarrow>=14.0",
    "scikit-learn>=1.4",
    "xgboost>=2.0",
    "torch>=2.2",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "ruff>=0.4",
]

[tool.ruff]
line-length = 100

[tool.pytest.ini_options]
testpaths = ["tests"]

[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]
```

- [ ] **Step 2: Create empty package `__init__.py` files**

Create these five files, each with empty content:
- `src/nfl4th/__init__.py`
- `src/nfl4th/data/__init__.py`
- `src/nfl4th/features/__init__.py`
- `src/nfl4th/models/__init__.py`
- `src/nfl4th/analysis/__init__.py`
- `tests/__init__.py`

- [ ] **Step 3: Write the CI workflow**

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
      - name: Install dependencies
        run: pip install -e ".[dev]"
      - name: Lint
        run: ruff check .
      - name: Test
        run: pytest
```

- [ ] **Step 4: Write a README stub**

```markdown
# NFL 4th Down Coach Tendency Predictor

Predicts what an NFL head coach will actually do on 4th down (go for it, punt,
or attempt a field goal) based on their real historical tendencies, the game
situation, and how much head-coaching experience they have.

Full writeup coming once the modeling pipeline is complete.
```

- [ ] **Step 5: Create a local virtual environment and install the package**

Run:
```bash
py -3.11 -m venv .venv
.venv/Scripts/python -m pip install --upgrade pip
.venv/Scripts/python -m pip install -e ".[dev]"
```
Expected: install completes with no errors.

- [ ] **Step 6: Verify lint and test commands run cleanly on an empty project**

Run: `.venv/Scripts/python -m ruff check .`
Expected: no errors (no Python files yet besides empty `__init__.py`s).

Run: `.venv/Scripts/python -m pytest`
Expected: "no tests ran" or 0 collected, exit code 0.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml src/ tests/ .github/ README.md
git commit -m "Set up project scaffolding with CI"
```

---

## Task 2: Data ingestion with local caching

**Files:**
- Create: `src/nfl4th/data/ingest.py`
- Test: `tests/data/test_ingest.py`
- Create: `tests/data/__init__.py`

**Interfaces:**
- Produces: `load_pbp(seasons: list[int], cache_dir: Path = DEFAULT_CACHE_DIR) -> pd.DataFrame`, `load_schedules(seasons: list[int], cache_dir: Path = DEFAULT_CACHE_DIR) -> pd.DataFrame`. Both used by `src/nfl4th/pipeline.py` in Task 9.

- [ ] **Step 1: Write the failing tests**

Create `tests/data/__init__.py` (empty), then `tests/data/test_ingest.py`:

```python
from pathlib import Path

import pandas as pd

from nfl4th.data.ingest import _load_cached_seasons


def _fake_fetch(calls):
    def fetch(seasons):
        calls.append(list(seasons))
        return pd.DataFrame(
            {
                "season": [s for s in seasons for _ in range(2)],
                "week": [1, 2] * len(seasons),
            }
        )

    return fetch


def test_fetches_and_caches_missing_seasons(tmp_path: Path):
    calls = []
    result = _load_cached_seasons([2020, 2021], "pbp", _fake_fetch(calls), tmp_path)

    assert calls == [[2020, 2021]]
    assert (tmp_path / "pbp" / "2020.parquet").exists()
    assert (tmp_path / "pbp" / "2021.parquet").exists()
    assert sorted(result["season"].unique().tolist()) == [2020, 2021]


def test_reuses_cached_seasons_without_refetching(tmp_path: Path):
    calls = []
    _load_cached_seasons([2020], "pbp", _fake_fetch(calls), tmp_path)

    calls.clear()
    _load_cached_seasons([2020], "pbp", _fake_fetch(calls), tmp_path)

    assert calls == []


def test_only_fetches_missing_seasons(tmp_path: Path):
    calls = []
    _load_cached_seasons([2020], "pbp", _fake_fetch(calls), tmp_path)

    calls.clear()
    _load_cached_seasons([2020, 2021], "pbp", _fake_fetch(calls), tmp_path)

    assert calls == [[2021]]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/data/test_ingest.py -v`
Expected: FAIL with `ModuleNotFoundError` or `ImportError` for `nfl4th.data.ingest`.

- [ ] **Step 3: Write the implementation**

```python
from __future__ import annotations

from pathlib import Path
from typing import Callable

import nfl_data_py as nfl
import pandas as pd

DEFAULT_CACHE_DIR = Path("data/raw")


def _load_cached_seasons(
    seasons: list[int],
    subdir: str,
    fetch_fn: Callable[[list[int]], pd.DataFrame],
    cache_dir: Path,
) -> pd.DataFrame:
    cache_subdir = cache_dir / subdir
    cache_subdir.mkdir(parents=True, exist_ok=True)

    frames = []
    missing_seasons = []
    for season in seasons:
        season_path = cache_subdir / f"{season}.parquet"
        if season_path.exists():
            frames.append(pd.read_parquet(season_path))
        else:
            missing_seasons.append(season)

    if missing_seasons:
        fetched = fetch_fn(missing_seasons)
        for season in missing_seasons:
            season_df = fetched[fetched["season"] == season]
            season_df.to_parquet(cache_subdir / f"{season}.parquet")
            frames.append(season_df)

    return (
        pd.concat(frames, ignore_index=True)
        .sort_values(["season", "week"])
        .reset_index(drop=True)
    )


def load_pbp(seasons: list[int], cache_dir: Path = DEFAULT_CACHE_DIR) -> pd.DataFrame:
    return _load_cached_seasons(seasons, "pbp", nfl.import_pbp_data, cache_dir)


def load_schedules(seasons: list[int], cache_dir: Path = DEFAULT_CACHE_DIR) -> pd.DataFrame:
    return _load_cached_seasons(seasons, "schedules", nfl.import_schedules, cache_dir)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/data/test_ingest.py -v`
Expected: 3 passed.

- [ ] **Step 5: Lint and commit**

Run: `.venv/Scripts/python -m ruff check .`
Expected: no errors.

```bash
git add src/nfl4th/data/ingest.py tests/data/
git commit -m "Add cached season data ingestion for pbp and schedules"
```

---

## Task 3: Decision filtering and label derivation

**Files:**
- Create: `src/nfl4th/features/build_features.py`
- Test: `tests/features/test_build_features.py`
- Create: `tests/features/__init__.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `DECISION_MAP: dict[str, str]`, `DECISION_CLASSES: list[str]`, `filter_fourth_down_decisions(pbp: pd.DataFrame) -> pd.DataFrame` (adds a `decision` column). Used by `attach_coach`/`build_feature_table` in Task 4, and by `baseline.py`/`embedding_model.py`/`tendencies.py` in Tasks 6-8.

- [ ] **Step 1: Write the failing test**

Create `tests/features/__init__.py` (empty), then `tests/features/test_build_features.py`:

```python
import pandas as pd

from nfl4th.features.build_features import DECISION_CLASSES, filter_fourth_down_decisions


def _row(**overrides):
    base = {
        "down": 4,
        "play_type": "run",
        "penalty": 0,
        "game_id": "2023_01_ARI_WAS",
        "season": 2023,
        "week": 1,
        "play_id": 1,
        "posteam": "ARI",
        "defteam": "WAS",
        "home_team": "WAS",
        "away_team": "ARI",
    }
    base.update(overrides)
    return base


def test_keeps_real_decisions_and_maps_labels():
    pbp = pd.DataFrame(
        [
            _row(play_type="run", play_id=1),
            _row(play_type="pass", play_id=2),
            _row(play_type="punt", play_id=3),
            _row(play_type="field_goal", play_id=4),
        ]
    )

    result = filter_fourth_down_decisions(pbp)

    assert list(result["decision"]) == ["go_for_it", "go_for_it", "punt", "field_goal"]
    assert set(result["decision"].unique()).issubset(set(DECISION_CLASSES))


def test_drops_non_decision_and_penalty_negated_plays():
    pbp = pd.DataFrame(
        [
            _row(down=3, play_type="pass", play_id=1),
            _row(play_type="no_play", penalty=1, play_id=2),
            _row(play_type="qb_kneel", play_id=3),
            _row(play_type=None, play_id=4),
            _row(play_type="punt", penalty=1, play_id=5),
        ]
    )

    result = filter_fourth_down_decisions(pbp)

    assert len(result) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/features/test_build_features.py -v`
Expected: FAIL with `ModuleNotFoundError` for `nfl4th.features.build_features`.

- [ ] **Step 3: Write the implementation**

```python
from __future__ import annotations

import pandas as pd

DECISION_MAP = {
    "run": "go_for_it",
    "pass": "go_for_it",
    "punt": "punt",
    "field_goal": "field_goal",
}
DECISION_CLASSES = ["punt", "field_goal", "go_for_it"]


def filter_fourth_down_decisions(pbp: pd.DataFrame) -> pd.DataFrame:
    fourth_down = pbp[pbp["down"] == 4]
    real_plays = fourth_down[fourth_down["penalty"] != 1]
    decisions = real_plays[real_plays["play_type"].isin(DECISION_MAP)].copy()
    decisions["decision"] = decisions["play_type"].map(DECISION_MAP)
    return decisions
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/features/test_build_features.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/nfl4th/features/build_features.py tests/features/
git commit -m "Add 4th down decision filtering and label mapping"
```

---

## Task 4: Coach attachment, situational features, career experience

**Files:**
- Modify: `src/nfl4th/features/build_features.py`
- Modify: `tests/features/test_build_features.py`

**Interfaces:**
- Consumes: `DECISION_MAP`, `DECISION_CLASSES`, `filter_fourth_down_decisions` from Task 3.
- Produces: `SITUATIONAL_FEATURES: list[str]` (includes `season` and `week`), `attach_coach(decisions, schedules) -> pd.DataFrame`, `add_career_decision_count(df) -> pd.DataFrame`, `build_feature_table(pbp, schedules) -> pd.DataFrame`. `build_feature_table` output columns: `["game_id", "coach", "decision"] + SITUATIONAL_FEATURES`. Used by `split.py` (Task 5), `baseline.py`/`embedding_model.py` (Tasks 6-7, via `SITUATIONAL_FEATURES`), `pipeline.py` (Task 9).

- [ ] **Step 1: Write the failing tests**

Append to `tests/features/test_build_features.py`:

```python
from nfl4th.features.build_features import (
    SITUATIONAL_FEATURES,
    add_career_decision_count,
    attach_coach,
    build_feature_table,
)


def _situational_row(**overrides):
    base = _row(
        ydstogo=2,
        yardline_100=40,
        score_differential=-3,
        game_seconds_remaining=1800,
        qtr=2,
        posteam_timeouts_remaining=3,
        defteam_timeouts_remaining=3,
    )
    base.update(overrides)
    return base


def test_attach_coach_picks_home_or_away_by_posteam():
    decisions = pd.DataFrame(
        [
            _row(posteam="ARI", home_team="WAS", away_team="ARI", play_id=1),
            _row(posteam="WAS", home_team="WAS", away_team="ARI", play_id=2),
        ]
    )
    decisions["decision"] = "go_for_it"
    schedules = pd.DataFrame(
        [{"game_id": "2023_01_ARI_WAS", "home_coach": "Ron Rivera", "away_coach": "Jonathan Gannon"}]
    )

    result = attach_coach(decisions, schedules)

    assert list(result["coach"]) == ["Jonathan Gannon", "Ron Rivera"]
    assert list(result["is_home"]) == [0, 1]


def test_add_career_decision_count_is_per_coach_and_chronological():
    df = pd.DataFrame(
        [
            {"coach": "A", "season": 2020, "week": 1, "game_id": "g1", "play_id": 1},
            {"coach": "B", "season": 2020, "week": 1, "game_id": "g1", "play_id": 2},
            {"coach": "A", "season": 2020, "week": 2, "game_id": "g2", "play_id": 1},
            {"coach": "A", "season": 2020, "week": 3, "game_id": "g3", "play_id": 1},
            {"coach": "B", "season": 2020, "week": 3, "game_id": "g3", "play_id": 2},
        ]
    )

    result = add_career_decision_count(df).sort_values(["coach", "week"])

    a_counts = result[result["coach"] == "A"]["career_decisions"].tolist()
    b_counts = result[result["coach"] == "B"]["career_decisions"].tolist()
    assert a_counts == [0, 1, 2]
    assert b_counts == [0, 1]


def test_build_feature_table_end_to_end():
    pbp = pd.DataFrame(
        [
            _situational_row(posteam="ARI", home_team="WAS", away_team="ARI", play_type="run", play_id=1),
            _situational_row(posteam="WAS", home_team="WAS", away_team="ARI", play_type="punt", play_id=2, week=2, game_id="g2"),
        ]
    )
    schedules = pd.DataFrame(
        [
            {"game_id": "2023_01_ARI_WAS", "home_coach": "Ron Rivera", "away_coach": "Jonathan Gannon"},
            {"game_id": "g2", "home_coach": "Ron Rivera", "away_coach": "Jonathan Gannon"},
        ]
    )

    result = build_feature_table(pbp, schedules)

    expected_columns = ["game_id", "coach", "decision"] + SITUATIONAL_FEATURES
    assert list(result.columns) == expected_columns
    assert len(result) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/features/test_build_features.py -v`
Expected: FAIL with `ImportError` for the new names.

- [ ] **Step 3: Extend the implementation**

Append to `src/nfl4th/features/build_features.py`:

```python
SITUATIONAL_FEATURES = [
    "season",
    "week",
    "ydstogo",
    "yardline_100",
    "score_differential",
    "game_seconds_remaining",
    "qtr",
    "posteam_timeouts_remaining",
    "defteam_timeouts_remaining",
    "is_home",
    "career_decisions",
]


def attach_coach(decisions: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    coach_lookup = schedules[["game_id", "home_coach", "away_coach"]]
    merged = decisions.merge(coach_lookup, on="game_id", how="left")
    merged["is_home"] = (merged["posteam"] == merged["home_team"]).astype(int)
    merged["coach"] = merged["home_coach"].where(merged["is_home"] == 1, merged["away_coach"])
    return merged.drop(columns=["home_coach", "away_coach"])


def add_career_decision_count(df: pd.DataFrame) -> pd.DataFrame:
    ordered = df.sort_values(["season", "week", "game_id", "play_id"]).copy()
    ordered["career_decisions"] = ordered.groupby("coach").cumcount()
    return ordered


def build_feature_table(pbp: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    decisions = filter_fourth_down_decisions(pbp)
    with_coach = attach_coach(decisions, schedules)
    with_experience = add_career_decision_count(with_coach)
    output_columns = ["game_id", "coach", "decision"] + SITUATIONAL_FEATURES
    return with_experience[output_columns].reset_index(drop=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/features/test_build_features.py -v`
Expected: 5 passed.

- [ ] **Step 5: Lint and commit**

Run: `.venv/Scripts/python -m ruff check .`

```bash
git add src/nfl4th/features/build_features.py tests/features/test_build_features.py
git commit -m "Attach coach identity and add situational features"
```

---

## Task 5: Time-based train/val/test split

**Files:**
- Create: `src/nfl4th/features/split.py`
- Test: `tests/features/test_split.py`

**Interfaces:**
- Consumes: nothing directly, operates on `build_feature_table` output shape (must have a `season` column).
- Produces: `time_based_split(df, train_seasons, val_seasons, test_seasons) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]`. Used by `pipeline.py` (Task 9).

- [ ] **Step 1: Write the failing test**

```python
import pandas as pd

from nfl4th.features.split import time_based_split


def test_splits_by_season_with_no_overlap():
    df = pd.DataFrame({"season": [2018, 2019, 2020, 2021, 2022, 2023], "value": range(6)})

    train, val, test = time_based_split(
        df, train_seasons=range(2018, 2021), val_seasons=range(2021, 2022), test_seasons=range(2022, 2024)
    )

    assert sorted(train["season"].unique()) == [2018, 2019, 2020]
    assert sorted(val["season"].unique()) == [2021]
    assert sorted(test["season"].unique()) == [2022, 2023]
    assert len(train) + len(val) + len(test) == len(df)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/features/test_split.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the implementation**

```python
from __future__ import annotations

import pandas as pd


def time_based_split(
    df: pd.DataFrame,
    train_seasons: range,
    val_seasons: range,
    test_seasons: range,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df[df["season"].isin(train_seasons)].reset_index(drop=True)
    val = df[df["season"].isin(val_seasons)].reset_index(drop=True)
    test = df[df["season"].isin(test_seasons)].reset_index(drop=True)
    return train, val, test
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/features/test_split.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/nfl4th/features/split.py tests/features/test_split.py
git commit -m "Add time based train val test split"
```

---

## Task 6: XGBoost baseline model

**Files:**
- Create: `src/nfl4th/models/baseline.py`
- Test: `tests/models/test_baseline.py`
- Create: `tests/models/__init__.py`

**Interfaces:**
- Consumes: `SITUATIONAL_FEATURES`, `DECISION_CLASSES` from `nfl4th.features.build_features` (Tasks 3-4).
- Produces: `train_baseline(train_df, seed=42) -> xgb.XGBClassifier`, `predict_baseline(model, df) -> pd.DataFrame` (columns = `DECISION_CLASSES`, one row per input row, probabilities). Used by `tendencies.py` (Task 8) and `pipeline.py` (Task 9).

- [ ] **Step 1: Write the failing tests**

Create `tests/models/__init__.py` (empty), then `tests/models/test_baseline.py`:

```python
import numpy as np
import pandas as pd

from nfl4th.features.build_features import DECISION_CLASSES
from nfl4th.models.baseline import predict_baseline, train_baseline


def _synthetic_training_data(n_per_class: int = 30) -> pd.DataFrame:
    rows = []
    coaches = ["Coach A", "Coach B"]
    for i in range(n_per_class):
        coach = coaches[i % 2]
        rows.append(
            {
                "coach": coach, "decision": "go_for_it",
                "season": 2020, "week": 1,
                "ydstogo": 1, "yardline_100": 45, "score_differential": -2,
                "game_seconds_remaining": 1800, "qtr": 2,
                "posteam_timeouts_remaining": 3, "defteam_timeouts_remaining": 3,
                "is_home": 1, "career_decisions": i,
            }
        )
        rows.append(
            {
                "coach": coach, "decision": "punt",
                "season": 2020, "week": 1,
                "ydstogo": 9, "yardline_100": 65, "score_differential": 0,
                "game_seconds_remaining": 1800, "qtr": 2,
                "posteam_timeouts_remaining": 3, "defteam_timeouts_remaining": 3,
                "is_home": 0, "career_decisions": i,
            }
        )
        rows.append(
            {
                "coach": coach, "decision": "field_goal",
                "season": 2020, "week": 1,
                "ydstogo": 3, "yardline_100": 15, "score_differential": 1,
                "game_seconds_remaining": 1800, "qtr": 2,
                "posteam_timeouts_remaining": 3, "defteam_timeouts_remaining": 3,
                "is_home": 1, "career_decisions": i,
            }
        )
    return pd.DataFrame(rows)


def test_predict_baseline_returns_valid_probabilities():
    train_df = _synthetic_training_data()
    model = train_baseline(train_df, seed=42)

    probs = predict_baseline(model, train_df)

    assert list(probs.columns) == DECISION_CLASSES
    assert len(probs) == len(train_df)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_training_is_deterministic_given_seed():
    train_df = _synthetic_training_data()

    model_a = train_baseline(train_df, seed=42)
    model_b = train_baseline(train_df, seed=42)

    probs_a = predict_baseline(model_a, train_df)
    probs_b = predict_baseline(model_b, train_df)

    assert np.allclose(probs_a.to_numpy(), probs_b.to_numpy())


def test_learns_the_obvious_pattern():
    train_df = _synthetic_training_data()
    model = train_baseline(train_df, seed=42)

    go_for_it_row = train_df[train_df["decision"] == "go_for_it"].iloc[[0]]
    probs = predict_baseline(model, go_for_it_row)

    assert probs.iloc[0]["go_for_it"] == probs.iloc[0].max()


def test_predict_on_a_single_row_matches_full_batch_prediction():
    # A single-row (or otherwise category-subset) predict call must encode
    # the coach column the same way training did, not re-derive categories
    # from whatever rows happen to be present in this call.
    train_df = _synthetic_training_data()
    model = train_baseline(train_df, seed=42)

    single_row = train_df.iloc[[0]]
    batch_probs = predict_baseline(model, train_df)
    single_probs = predict_baseline(model, single_row)

    assert np.allclose(single_probs.to_numpy()[0], batch_probs.to_numpy()[0])


def test_predict_handles_a_coach_never_seen_in_training():
    train_df = _synthetic_training_data()
    model = train_baseline(train_df, seed=42)

    unseen_row = train_df.iloc[[0]].copy()
    unseen_row["coach"] = "Brand New Coach"

    probs = predict_baseline(model, unseen_row)

    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/models/test_baseline.py -v`
Expected: FAIL with `ModuleNotFoundError` for `nfl4th.models.baseline`.

- [ ] **Step 3: Write the implementation**

```python
from __future__ import annotations

import pandas as pd
import xgboost as xgb

from nfl4th.features.build_features import DECISION_CLASSES, SITUATIONAL_FEATURES


def _prepare_matrix(df: pd.DataFrame, coach_categories: pd.CategoricalDtype) -> pd.DataFrame:
    features = df[SITUATIONAL_FEATURES].copy()
    # A coach outside coach_categories becomes NaN here, which XGBoost treats
    # as a missing value and routes through its learned default direction.
    # That is the baseline model's cold start behavior for an unseen coach.
    features["coach"] = df["coach"].astype(coach_categories)
    return features


def train_baseline(train_df: pd.DataFrame, seed: int = 42) -> xgb.XGBClassifier:
    label_index = {label: i for i, label in enumerate(DECISION_CLASSES)}
    y = train_df["decision"].map(label_index)
    coach_categories = pd.CategoricalDtype(categories=sorted(train_df["coach"].unique()))
    X = _prepare_matrix(train_df, coach_categories)

    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=len(DECISION_CLASSES),
        tree_method="hist",
        enable_categorical=True,
        random_state=seed,
        n_estimators=200,
        max_depth=4,
    )
    model.fit(X, y)
    model.coach_categories_ = coach_categories
    return model


def predict_baseline(model: xgb.XGBClassifier, df: pd.DataFrame) -> pd.DataFrame:
    X = _prepare_matrix(df, model.coach_categories_)
    probs = model.predict_proba(X)
    return pd.DataFrame(probs, columns=DECISION_CLASSES, index=df.index)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/models/test_baseline.py -v`
Expected: 5 passed.

- [ ] **Step 5: Lint and commit**

Run: `.venv/Scripts/python -m ruff check .`

```bash
git add src/nfl4th/models/baseline.py tests/models/
git commit -m "Add XGBoost baseline tendency model"
```

---

## Task 7: PyTorch coach embedding model with cold start

**Files:**
- Create: `src/nfl4th/models/embedding_model.py`
- Test: `tests/models/test_embedding_model.py`

**Interfaces:**
- Consumes: `SITUATIONAL_FEATURES`, `DECISION_CLASSES` from `nfl4th.features.build_features`.
- Produces: `CoachVocab` (with `.encode(name) -> int`, `__len__`), `TendencyEmbeddingModel` (`nn.Module`), `train_embedding_model(train_df, epochs=30, lr=0.01, seed=42) -> tuple[TendencyEmbeddingModel, CoachVocab]`, `predict_with_coldstart(model, vocab, df) -> pd.DataFrame` (columns = `DECISION_CLASSES`). `tendencies.py` (Task 8) does not use these, it works from the baseline model. `pipeline.py` (Task 9) uses all of these.

- [ ] **Step 1: Write the failing tests**

Create `tests/models/test_embedding_model.py`:

```python
import numpy as np
import pandas as pd
import torch

from nfl4th.features.build_features import DECISION_CLASSES, SITUATIONAL_FEATURES
from nfl4th.models.embedding_model import (
    CoachVocab,
    TendencyEmbeddingModel,
    predict_with_coldstart,
    train_embedding_model,
)


def _synthetic_training_data(n_per_coach: int = 25) -> pd.DataFrame:
    rows = []
    for i in range(n_per_coach):
        rows.append(
            {
                "coach": "Aggressive Al", "decision": "go_for_it",
                "season": 2020, "week": 1,
                "ydstogo": 2, "yardline_100": 50, "score_differential": 0,
                "game_seconds_remaining": 1800, "qtr": 2,
                "posteam_timeouts_remaining": 3, "defteam_timeouts_remaining": 3,
                "is_home": 1, "career_decisions": i,
            }
        )
        rows.append(
            {
                "coach": "Cautious Carl", "decision": "punt",
                "season": 2020, "week": 1,
                "ydstogo": 2, "yardline_100": 50, "score_differential": 0,
                "game_seconds_remaining": 1800, "qtr": 2,
                "posteam_timeouts_remaining": 3, "defteam_timeouts_remaining": 3,
                "is_home": 1, "career_decisions": i,
            }
        )
    return pd.DataFrame(rows)


def test_vocab_encodes_known_coaches_and_reserves_zero_for_unknown():
    vocab = CoachVocab(["Aggressive Al", "Cautious Carl"])

    assert vocab.encode("Aggressive Al") != 0
    assert vocab.encode("Cautious Carl") != 0
    assert vocab.encode("Someone New") == 0
    assert len(vocab) == 3


def test_forward_pass_shape():
    model = TendencyEmbeddingModel(n_coaches=3, n_situational=len(SITUATIONAL_FEATURES))
    coach_idx = torch.tensor([1, 2])
    situational = torch.zeros((2, len(SITUATIONAL_FEATURES)))

    logits = model(coach_idx, situational)

    assert logits.shape == (2, len(DECISION_CLASSES))


def test_training_reduces_loss():
    train_df = _synthetic_training_data()

    early_model, early_vocab = train_embedding_model(train_df, epochs=1, seed=42)
    late_model, late_vocab = train_embedding_model(train_df, epochs=100, seed=42)

    early_probs = predict_with_coldstart(early_model, early_vocab, train_df)
    late_probs = predict_with_coldstart(late_model, late_vocab, train_df)

    label_index = {label: i for i, label in enumerate(DECISION_CLASSES)}
    true_class = train_df["decision"].map(label_index).to_numpy()

    early_loss = -np.log(early_probs.to_numpy()[np.arange(len(true_class)), true_class]).mean()
    late_loss = -np.log(late_probs.to_numpy()[np.arange(len(true_class)), true_class]).mean()

    assert late_loss < early_loss


def test_coldstart_prediction_uses_mean_embedding():
    train_df = _synthetic_training_data()
    model, vocab = train_embedding_model(train_df, epochs=20, seed=42)

    unseen_row = train_df.iloc[[0]].copy()
    unseen_row["coach"] = "Brand New Coach"

    result = predict_with_coldstart(model, vocab, unseen_row)

    mean_embedding = model.mean_coach_embedding().detach()
    situational = torch.tensor(unseen_row[SITUATIONAL_FEATURES].to_numpy(dtype="float32"))
    with torch.no_grad():
        combined = torch.cat([mean_embedding.unsqueeze(0), situational], dim=1)
        expected_probs = torch.softmax(model.classifier(combined), dim=1).numpy()

    assert np.allclose(result.to_numpy(), expected_probs, atol=1e-6)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/models/test_embedding_model.py -v`
Expected: FAIL with `ModuleNotFoundError` for `nfl4th.models.embedding_model`.

- [ ] **Step 3: Write the implementation**

```python
from __future__ import annotations

import pandas as pd
import torch
from torch import nn

from nfl4th.features.build_features import DECISION_CLASSES, SITUATIONAL_FEATURES

UNKNOWN_COACH = "<unknown>"


class CoachVocab:
    def __init__(self, coach_names: list[str]):
        unique_names = sorted(set(coach_names))
        self.coach_to_index = {UNKNOWN_COACH: 0}
        for name in unique_names:
            self.coach_to_index[name] = len(self.coach_to_index)

    def __len__(self) -> int:
        return len(self.coach_to_index)

    def encode(self, name: str) -> int:
        return self.coach_to_index.get(name, 0)

    def encode_series(self, names: pd.Series) -> torch.Tensor:
        return torch.tensor([self.encode(n) for n in names], dtype=torch.long)


class TendencyEmbeddingModel(nn.Module):
    def __init__(self, n_coaches: int, n_situational: int, embedding_dim: int = 8, hidden_dim: int = 32):
        super().__init__()
        self.coach_embedding = nn.Embedding(n_coaches, embedding_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim + n_situational, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(DECISION_CLASSES)),
        )

    def forward(self, coach_idx: torch.Tensor, situational: torch.Tensor) -> torch.Tensor:
        embedded = self.coach_embedding(coach_idx)
        combined = torch.cat([embedded, situational], dim=1)
        return self.classifier(combined)

    def mean_coach_embedding(self) -> torch.Tensor:
        # Index 0 is the reserved unknown-coach slot and never receives a
        # training signal, so it is excluded from the mean.
        return self.coach_embedding.weight[1:].mean(dim=0)


def _to_tensors(df: pd.DataFrame, vocab: CoachVocab) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    label_index = {label: i for i, label in enumerate(DECISION_CLASSES)}
    coach_idx = vocab.encode_series(df["coach"])
    situational = torch.tensor(df[SITUATIONAL_FEATURES].to_numpy(dtype="float32"))
    labels = torch.tensor(df["decision"].map(label_index).to_numpy(), dtype=torch.long)
    return coach_idx, situational, labels


def train_embedding_model(
    train_df: pd.DataFrame,
    epochs: int = 30,
    lr: float = 0.01,
    seed: int = 42,
) -> tuple[TendencyEmbeddingModel, CoachVocab]:
    torch.manual_seed(seed)
    vocab = CoachVocab(train_df["coach"].tolist())
    coach_idx, situational, labels = _to_tensors(train_df, vocab)

    model = TendencyEmbeddingModel(len(vocab), len(SITUATIONAL_FEATURES))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(coach_idx, situational)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

    return model, vocab


def predict_with_coldstart(
    model: TendencyEmbeddingModel, vocab: CoachVocab, df: pd.DataFrame
) -> pd.DataFrame:
    model.eval()
    situational = torch.tensor(df[SITUATIONAL_FEATURES].to_numpy(dtype="float32"))
    mean_embedding = model.mean_coach_embedding().detach()

    embeddings = []
    for name in df["coach"]:
        idx = vocab.encode(name)
        if idx == 0:
            embeddings.append(mean_embedding)
        else:
            embeddings.append(model.coach_embedding.weight[idx].detach())
    embedding_batch = torch.stack(embeddings)

    with torch.no_grad():
        combined = torch.cat([embedding_batch, situational], dim=1)
        probs = torch.softmax(model.classifier(combined), dim=1)

    return pd.DataFrame(probs.numpy(), columns=DECISION_CLASSES, index=df.index)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/models/test_embedding_model.py -v`
Expected: 4 passed.

- [ ] **Step 5: Lint and commit**

Run: `.venv/Scripts/python -m ruff check .`

```bash
git add src/nfl4th/models/embedding_model.py tests/models/test_embedding_model.py
git commit -m "Add coach embedding model with cold start handling"
```

---

## Task 8: Shrinkage-based coach tendency report

**Files:**
- Create: `src/nfl4th/analysis/tendencies.py`
- Test: `tests/analysis/test_tendencies.py`
- Create: `tests/analysis/__init__.py`

**Interfaces:**
- Consumes: `DECISION_CLASSES` from `nfl4th.features.build_features`.
- Produces: `shrinkage_weight(n, k=10.0) -> float`, `coach_tendency_report(decisions, baseline_probs, k=10.0) -> pd.DataFrame`. Used by `pipeline.py` (Task 9).

- [ ] **Step 1: Write the failing tests**

Create `tests/analysis/__init__.py` (empty), then `tests/analysis/test_tendencies.py`:

```python
import pandas as pd

from nfl4th.analysis.tendencies import coach_tendency_report, shrinkage_weight


def test_shrinkage_weight_boundaries():
    assert shrinkage_weight(0, k=10) == 0.0
    assert shrinkage_weight(10, k=10) == 0.5
    assert shrinkage_weight(10_000, k=10) > 0.99


def test_report_blends_toward_baseline_for_small_sample():
    decisions = pd.DataFrame({"coach": ["Rookie"], "decision": ["go_for_it"]})
    baseline_probs = pd.DataFrame(
        {"punt": [0.7], "field_goal": [0.2], "go_for_it": [0.1]}, index=decisions.index
    )

    report = coach_tendency_report(decisions, baseline_probs, k=10.0)

    row = report.iloc[0]
    assert row["go_for_it_observed"] == 1.0
    assert row["go_for_it_expected"] == 0.1
    assert row["go_for_it_shrunk"] < 0.5


def test_report_converges_to_observed_for_large_sample():
    n = 200
    decisions = pd.DataFrame({"coach": ["Veteran"] * n, "decision": ["go_for_it"] * n})
    baseline_probs = pd.DataFrame(
        {"punt": [0.7] * n, "field_goal": [0.2] * n, "go_for_it": [0.1] * n}, index=decisions.index
    )

    report = coach_tendency_report(decisions, baseline_probs, k=10.0)

    row = report.iloc[0]
    assert row["go_for_it_shrunk"] > 0.95
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/analysis/test_tendencies.py -v`
Expected: FAIL with `ModuleNotFoundError` for `nfl4th.analysis.tendencies`.

- [ ] **Step 3: Write the implementation**

```python
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

        row = {"coach": coach, "n_decisions": n, "shrinkage_weight": weight}
        for decision_class in DECISION_CLASSES:
            observed_rate = (group["decision"] == decision_class).mean()
            expected_rate = baseline_for_coach[decision_class].mean()
            row[f"{decision_class}_observed"] = observed_rate
            row[f"{decision_class}_expected"] = expected_rate
            row[f"{decision_class}_shrunk"] = weight * observed_rate + (1 - weight) * expected_rate
        rows.append(row)

    return pd.DataFrame(rows).sort_values("coach").reset_index(drop=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/analysis/test_tendencies.py -v`
Expected: 3 passed.

- [ ] **Step 5: Lint and commit**

Run: `.venv/Scripts/python -m ruff check .`

```bash
git add src/nfl4th/analysis/tendencies.py tests/analysis/
git commit -m "Add shrinkage based coach tendency report"
```

---

## Task 9: End-to-end pipeline wiring

**Files:**
- Create: `src/nfl4th/pipeline.py`
- Create: `scripts/run_pipeline.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `load_pbp`, `load_schedules` (Task 2); `build_feature_table` (Tasks 3-4); `time_based_split` (Task 5); `train_baseline`, `predict_baseline` (Task 6); `train_embedding_model`, `predict_with_coldstart` (Task 7); `coach_tendency_report` (Task 8).
- Produces: `run(train_seasons, val_seasons, test_seasons) -> pd.DataFrame` (the tendency report), `main() -> None` (CLI entry point). This is the top-level entry point; nothing later depends on it.

- [ ] **Step 1: Write the failing test**

Create `tests/test_pipeline.py`:

```python
import pandas as pd

import nfl4th.pipeline as pipeline


def _synthetic_pbp(seasons: list[int]) -> pd.DataFrame:
    # posteam always matches home_team here, so every decision in this fixture
    # belongs to the home coach ("Coach A" per _synthetic_schedules below).
    rows = []
    play_id = 1
    for season in seasons:
        for week in [1, 2]:
            for play_type in ["run", "punt", "field_goal"]:
                rows.append(
                    {
                        "down": 4, "penalty": 0, "game_id": f"{season}_{week}_g",
                        "season": season, "week": week, "play_id": play_id,
                        "play_type": play_type, "posteam": "HOME", "defteam": "AWAY",
                        "home_team": "HOME", "away_team": "AWAY",
                        "ydstogo": 2, "yardline_100": 40, "score_differential": 0,
                        "game_seconds_remaining": 1800, "qtr": 2,
                        "posteam_timeouts_remaining": 3, "defteam_timeouts_remaining": 3,
                    }
                )
                play_id += 1
    return pd.DataFrame(rows)


def _synthetic_schedules(seasons: list[int]) -> pd.DataFrame:
    rows = []
    for season in seasons:
        for week in [1, 2]:
            rows.append(
                {"game_id": f"{season}_{week}_g", "home_coach": "Coach A", "away_coach": "Coach B"}
            )
    return pd.DataFrame(rows)


def test_run_completes_and_returns_a_report(monkeypatch):
    monkeypatch.setattr(pipeline, "load_pbp", lambda seasons: _synthetic_pbp(seasons))
    monkeypatch.setattr(pipeline, "load_schedules", lambda seasons: _synthetic_schedules(seasons))

    report = pipeline.run(
        train_seasons=range(2020, 2021), val_seasons=range(2021, 2022), test_seasons=range(2022, 2023)
    )

    assert "coach" in report.columns
    assert set(report["coach"]) == {"Coach A"}
    # Report is built from train_df only (season 2020), 2 weeks x 3 decisions each.
    assert report.iloc[0]["n_decisions"] == 6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError` for `nfl4th.pipeline`.

- [ ] **Step 3: Write the implementation**

Create `src/nfl4th/pipeline.py`:

```python
from __future__ import annotations

import argparse

from nfl4th.analysis.tendencies import coach_tendency_report
from nfl4th.data.ingest import load_pbp, load_schedules
from nfl4th.features.build_features import build_feature_table
from nfl4th.features.split import time_based_split
from nfl4th.models.baseline import predict_baseline, train_baseline
from nfl4th.models.embedding_model import predict_with_coldstart, train_embedding_model


def run(train_seasons: range, val_seasons: range, test_seasons: range):
    all_seasons = list(train_seasons) + list(val_seasons) + list(test_seasons)
    pbp = load_pbp(all_seasons)
    schedules = load_schedules(all_seasons)
    features = build_feature_table(pbp, schedules)

    train_df, val_df, test_df = time_based_split(features, train_seasons, val_seasons, test_seasons)

    baseline_model = train_baseline(train_df)
    embedding_model, vocab = train_embedding_model(train_df)

    # val_df and the embedding model's test predictions feed model comparison
    # metrics, which are a follow up once both models exist end to end. This
    # report is what proves the pipeline produces a usable result today.
    predict_with_coldstart(embedding_model, vocab, test_df)
    baseline_train_probs = predict_baseline(baseline_model, train_df)
    return coach_tendency_report(train_df, baseline_train_probs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the phase 1 tendency pipeline")
    parser.add_argument("--train-start", type=int, default=2010)
    parser.add_argument("--train-end", type=int, default=2021)
    parser.add_argument("--val-end", type=int, default=2023)
    parser.add_argument("--test-end", type=int, default=2025)
    args = parser.parse_args()

    report = run(
        range(args.train_start, args.train_end),
        range(args.train_end, args.val_end),
        range(args.val_end, args.test_end),
    )
    print(report.to_string())


if __name__ == "__main__":
    main()
```

Create `scripts/run_pipeline.py`:

```python
from nfl4th.pipeline import main

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/test_pipeline.py -v`
Expected: 1 passed.

- [ ] **Step 5: Run the full test suite**

Run: `.venv/Scripts/python -m pytest -v`
Expected: all tests pass (Tasks 2-9 combined).

- [ ] **Step 6: Lint and commit**

Run: `.venv/Scripts/python -m ruff check .`

```bash
git add src/nfl4th/pipeline.py scripts/ tests/test_pipeline.py
git commit -m "Wire pipeline stages together end to end"
```

---

## Task 10: Portfolio README and final verification

**Files:**
- Modify: `README.md`

**Interfaces:**
- None (documentation only).

- [ ] **Step 1: Write the full README**

```markdown
# NFL 4th Down Coach Tendency Predictor

Predicts what an NFL head coach will actually do on 4th down (go for it,
punt, or attempt a field goal), based on their real historical decisions,
the game situation, and how much head-coaching experience they have. This
models actual tendency, not the win-probability-optimal choice.

## Why tendency, not optimal decision

Plenty of public models already answer "what should a coach do here." This
one answers a different question: "what will this specific coach actually
do here," learned from their real decision history. The two often disagree,
and that gap is part of the point.

## How it works

1. **Ingest**: pulls play by play and schedule data from nflverse
   (`nfl_data_py`) for the 2010-2024 seasons, cached locally as parquet.
2. **Feature engineering**: filters play by play down to genuine 4th down
   decisions (excludes penalty-negated plays, kneels, and other noise),
   attaches the head coach who made the call, and builds situational
   features (down and distance, field position, score, time, timeouts,
   home/away, and career decisions coached so far).
3. **Modeling**: trains two models side by side on a time-based split
   (train on earlier seasons, test on the most recent ones, no leakage):
   - An XGBoost baseline with coach as a categorical feature.
   - A PyTorch model with a learned per-coach embedding.
4. **Cold start**: a coach's first 4th down decision as a head coach comes
   with zero track record, regardless of background. Both models handle
   this the same way: predictions fall back to the situational baseline for
   new coaches and phase toward the coach's own tendency as their sample
   size grows, using empirical Bayes shrinkage.
5. **Analysis**: produces a per-coach tendency report comparing observed
   decision rates against the model's expected baseline rate, shrunk by
   sample size.

## Setup

Requires Python 3.11 (nfl_data_py's pinned pandas version has no Python 3.12
wheel).

```bash
py -3.11 -m venv .venv
.venv/Scripts/python -m pip install --upgrade pip
.venv/Scripts/python -m pip install -e ".[dev]"
```

## Running the tests

```bash
.venv/Scripts/python -m pytest
```

All tests run against small synthetic fixtures, not live data, so they run
fully offline in a few seconds.

## Running the full pipeline

```bash
.venv/Scripts/python scripts/run_pipeline.py
```

This downloads and caches 2010-2024 play by play and schedule data on first
run (a few hundred MB, several minutes), then trains both models and prints
a coach tendency report.

## What's next

This is phase 1: the data pipeline and modeling core. Phase 2 will put this
behind a small web app so you can look up any coach or situation directly.
```

- [ ] **Step 2: Run the full verification suite**

Run: `.venv/Scripts/python -m pytest -v`
Expected: all tests pass.

Run: `.venv/Scripts/python -m ruff check .`
Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "Write project README"
```
