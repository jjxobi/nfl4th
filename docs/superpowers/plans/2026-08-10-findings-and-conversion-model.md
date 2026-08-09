# Findings Page and Conversion Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `/findings` page (superlative leaderboards, situational splits, league-wide trend over time) and a new situation-only conversion-probability model, surfaced on both the predictor and the findings page, and folded into the existing weekly automated retrain.

**Architecture:** New backend aggregation module (`src/nfl4th/analysis/findings.py`) and a new model module (`src/nfl4th/models/conversion.py`), both wired into `pipeline.py`'s `run()` alongside the existing models, producing two new committed artifacts (`api/models/conversion/`, `api/models/findings.json`). The API gains a new field on `POST /predict` and a new `GET /findings` endpoint, both serving precomputed data with zero live computation, matching every other endpoint in this project. The frontend gains a new page and one new result card on the predictor.

**Tech Stack:** Python (pandas, XGBoost, FastAPI, Pydantic), Astro, TypeScript.

## Global Constraints

- No em dashes anywhere in any UI copy, code, or commit message.
- No mention of AI/Claude/Anthropic anywhere in committed content.
- No "Co-Authored-By" trailers in any commit.
- Commit messages must be plain, human-sounding.
- No code comments unless they explain a genuinely non-obvious WHY, never a WHAT.
- New pages must use `Layout.astro`'s existing design tokens (`--color-bg`, `--color-surface`, `--color-surface-raised`, `--color-line`, `--color-text`, `--color-text-dim`, `--color-accent`, `--color-accent-2`, `--font-display`, `--font-mono`, `--font-body`, `--space-2` through `--space-6`) rather than introducing new colors, fonts, or spacing values.
- No new frontend charting library. The trend-over-time visual is a hand-built SVG line; the split/conversion visuals reuse the predictor's existing bar-style "meter" visual language.
- The conversion model is situation-only: no coach feature, no ensemble, no embedding counterpart.
- Every new piece of served data (findings, conversion probability) is precomputed at pipeline run time and read from committed artifacts at request time. Nothing in `api/` retrains or recomputes anything live.

---

## Task 1: Findings aggregation module

**Files:**
- Create: `src/nfl4th/analysis/findings.py`
- Test: `tests/analysis/test_findings.py`

**Interfaces:**
- Consumes: a `decisions`-shaped DataFrame — the same shape `coach_tendency_report()` already consumes (`src/nfl4th/analysis/tendencies.py`), i.e. one row per real 4th-down decision with at least `coach`, `decision`, `season`, `ydstogo` columns. This is `train_df` in `pipeline.py`'s existing `run()`.
- Produces: `DISTANCE_BUCKETS: list[tuple[int, int | None, str]]`, `situational_splits(decisions: pd.DataFrame) -> pd.DataFrame` (columns: `distance_bucket`, `n_decisions`, `go_for_it_rate`), `coach_bucket_leaderboard(decisions: pd.DataFrame, min_attempts: int = 10) -> pd.DataFrame` (columns: `coach`, `distance_bucket`, `n_decisions`, `go_for_it_rate`), `league_trend_over_time(decisions: pd.DataFrame) -> pd.DataFrame` (columns: `season`, `n_decisions`, `go_for_it_rate`). Task 3 (pipeline wiring) calls all three of these by name with these exact signatures.

- [ ] **Step 1: Write the failing tests**

Create `tests/analysis/test_findings.py`:

```python
import pandas as pd

from nfl4th.analysis.findings import (
    coach_bucket_leaderboard,
    league_trend_over_time,
    situational_splits,
)


def _synthetic_decisions() -> pd.DataFrame:
    rows = []
    # Bucket "1": mostly go_for_it. Bucket "11+": mostly punt.
    for i in range(20):
        rows.append(
            {
                "coach": "Coach A" if i % 2 == 0 else "Coach B",
                "decision": "go_for_it" if i < 16 else "punt",
                "season": 2020,
                "ydstogo": 1,
            }
        )
    for i in range(20):
        rows.append(
            {
                "coach": "Coach A" if i % 2 == 0 else "Coach B",
                "decision": "punt" if i < 16 else "go_for_it",
                "season": 2021,
                "ydstogo": 15,
            }
        )
    return pd.DataFrame(rows)


def test_situational_splits_buckets_by_distance():
    decisions = _synthetic_decisions()

    result = situational_splits(decisions)

    bucket_1 = result[result["distance_bucket"] == "1"].iloc[0]
    bucket_11plus = result[result["distance_bucket"] == "11+"].iloc[0]
    assert bucket_1["n_decisions"] == 20
    assert bucket_1["go_for_it_rate"] == 0.8
    assert bucket_11plus["n_decisions"] == 20
    assert bucket_11plus["go_for_it_rate"] == 0.2


def test_situational_splits_omits_empty_buckets():
    decisions = _synthetic_decisions()

    result = situational_splits(decisions)

    assert "4-6" not in result["distance_bucket"].tolist()


def test_coach_bucket_leaderboard_respects_minimum_attempts():
    decisions = _synthetic_decisions()

    result = coach_bucket_leaderboard(decisions, min_attempts=15)

    assert len(result) == 0


def test_coach_bucket_leaderboard_includes_coaches_with_enough_attempts():
    decisions = _synthetic_decisions()

    result = coach_bucket_leaderboard(decisions, min_attempts=5)

    coach_a_bucket_1 = result[(result["coach"] == "Coach A") & (result["distance_bucket"] == "1")].iloc[0]
    assert coach_a_bucket_1["n_decisions"] == 10
    assert coach_a_bucket_1["go_for_it_rate"] == 0.8


def test_league_trend_over_time_has_one_row_per_season():
    decisions = _synthetic_decisions()

    result = league_trend_over_time(decisions)

    assert sorted(result["season"].tolist()) == [2020, 2021]
    row_2020 = result[result["season"] == 2020].iloc[0]
    assert row_2020["n_decisions"] == 20
    assert row_2020["go_for_it_rate"] == 0.8
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/analysis/test_findings.py -v`
Expected: FAIL (or collection error) — `nfl4th.analysis.findings` does not exist yet.

- [ ] **Step 3: Implement the findings module**

Create `src/nfl4th/analysis/findings.py`:

```python
from __future__ import annotations

import pandas as pd

# (low, high, label). high=None means "low or more".
DISTANCE_BUCKETS: list[tuple[int, int | None, str]] = [
    (1, 1, "1"),
    (2, 2, "2"),
    (3, 3, "3"),
    (4, 6, "4-6"),
    (7, 10, "7-10"),
    (11, None, "11+"),
]


def _bucket_label(ydstogo: int) -> str:
    for low, high, label in DISTANCE_BUCKETS:
        if high is None:
            if ydstogo >= low:
                return label
        elif low <= ydstogo <= high:
            return label
    raise ValueError(f"ydstogo {ydstogo} did not match any distance bucket")


def situational_splits(decisions: pd.DataFrame) -> pd.DataFrame:
    working = decisions.copy()
    working["distance_bucket"] = working["ydstogo"].apply(_bucket_label)

    rows = []
    for _, _, label in DISTANCE_BUCKETS:
        bucket_df = working[working["distance_bucket"] == label]
        if len(bucket_df) == 0:
            continue
        rows.append(
            {
                "distance_bucket": label,
                "n_decisions": len(bucket_df),
                "go_for_it_rate": (bucket_df["decision"] == "go_for_it").mean(),
            }
        )
    return pd.DataFrame(rows)


def coach_bucket_leaderboard(decisions: pd.DataFrame, min_attempts: int = 10) -> pd.DataFrame:
    working = decisions.copy()
    working["distance_bucket"] = working["ydstogo"].apply(_bucket_label)

    rows = []
    for (coach, bucket), group in working.groupby(["coach", "distance_bucket"]):
        if len(group) < min_attempts:
            continue
        rows.append(
            {
                "coach": coach,
                "distance_bucket": bucket,
                "n_decisions": len(group),
                "go_for_it_rate": (group["decision"] == "go_for_it").mean(),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["distance_bucket", "go_for_it_rate"], ascending=[True, False])
        .reset_index(drop=True)
    )


def league_trend_over_time(decisions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for season, group in decisions.groupby("season"):
        rows.append(
            {
                "season": int(season),
                "n_decisions": len(group),
                "go_for_it_rate": (group["decision"] == "go_for_it").mean(),
            }
        )
    return pd.DataFrame(rows).sort_values("season").reset_index(drop=True)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/analysis/test_findings.py -v`
Expected: PASS, all 5 tests.

- [ ] **Step 5: Run the full test suite and lint**

Run: `pytest -q` and `ruff check .`
Expected: all tests pass (no regressions, this file is purely additive), ruff clean.

- [ ] **Step 6: Commit**

```bash
git add src/nfl4th/analysis/findings.py tests/analysis/test_findings.py
git commit -m "Add situational splits, coach leaderboards, and league trend aggregation"
```

---

## Task 2: Conversion probability model

**Files:**
- Modify: `src/nfl4th/data/ingest.py`
- Modify: `src/nfl4th/features/build_features.py`
- Create: `src/nfl4th/models/conversion.py`
- Test: `tests/features/test_build_features.py` (add one test)
- Test: `tests/models/test_conversion.py`

**Interfaces:**
- Consumes: `filter_fourth_down_decisions` (existing, `src/nfl4th/features/build_features.py`).
- Produces: `build_conversion_training_data(pbp: pd.DataFrame) -> pd.DataFrame` in `build_features.py` (one row per real go-for-it attempt, situational features plus a `converted` int column and a `season` column for time-based splitting). `CONVERSION_FEATURES: list[str]`, `train_conversion_model(train_df, seed=42) -> xgb.XGBClassifier`, `predict_conversion(model, df) -> pd.Series`, `save_conversion_model(model, path)`, `load_conversion_model(path) -> xgb.XGBClassifier` in `src/nfl4th/models/conversion.py`. Task 3 calls all of these by name with these exact signatures.

- [ ] **Step 1: Add the conversion outcome column to the pbp column allowlist**

In `src/nfl4th/data/ingest.py`, `load_pbp` currently restricts real play-by-play data to 16 columns via `PBP_COLUMNS` (added in a previous plan, to keep memory usage down). The conversion model needs one more: whether a 4th down attempt actually succeeded. Change:

```python
PBP_COLUMNS = [
    "game_id",
    "season",
    "week",
    "play_id",
    "down",
    "penalty",
    "play_type",
    "posteam",
    "home_team",
    "ydstogo",
    "yardline_100",
    "score_differential",
    "game_seconds_remaining",
    "qtr",
    "posteam_timeouts_remaining",
    "defteam_timeouts_remaining",
]
```

to:

```python
PBP_COLUMNS = [
    "game_id",
    "season",
    "week",
    "play_id",
    "down",
    "penalty",
    "play_type",
    "posteam",
    "home_team",
    "ydstogo",
    "yardline_100",
    "score_differential",
    "game_seconds_remaining",
    "qtr",
    "posteam_timeouts_remaining",
    "defteam_timeouts_remaining",
    "fourth_down_converted",
]
```

(`fourth_down_failed` is not needed: it is the exact complement of `fourth_down_converted` on every real 4th down attempt, confirmed against real cached 2025 data, so keeping just one avoids carrying redundant data.)

- [ ] **Step 2: Write the failing test for conversion training data**

Add to `tests/features/test_build_features.py` (read the existing file first to match its exact synthetic-fixture style before adding this — it already has fixtures for `filter_fourth_down_decisions` and similar functions; follow that same pattern rather than inventing a new one):

```python
def test_build_conversion_training_data_keeps_only_go_for_it_attempts():
    pbp = pd.DataFrame(
        {
            "down": [4, 4, 4],
            "penalty": [0, 0, 0],
            "play_type": ["run", "punt", "pass"],
            "season": [2024, 2024, 2024],
            "ydstogo": [2, 9, 1],
            "yardline_100": [40, 60, 30],
            "score_differential": [0, 0, 0],
            "game_seconds_remaining": [1800, 1800, 1800],
            "qtr": [2, 2, 2],
            "posteam_timeouts_remaining": [3, 3, 3],
            "defteam_timeouts_remaining": [3, 3, 3],
            "fourth_down_converted": [1, 0, 0],
        }
    )

    result = build_conversion_training_data(pbp)

    assert len(result) == 2
    assert set(result["converted"]) == {1, 0}
    assert list(result[result["ydstogo"] == 2]["converted"]) == [1]
    assert list(result[result["ydstogo"] == 1]["converted"]) == [0]
```

(Import `build_conversion_training_data` alongside whatever this test file already imports from `nfl4th.features.build_features`.)

- [ ] **Step 3: Run the test to verify it fails**

Run: `pytest tests/features/test_build_features.py::test_build_conversion_training_data_keeps_only_go_for_it_attempts -v`
Expected: FAIL with `ImportError` or `AttributeError` — the function does not exist yet.

- [ ] **Step 4: Implement `build_conversion_training_data`**

In `src/nfl4th/features/build_features.py`, add this function after `filter_fourth_down_decisions` (it reuses that function directly, so it must come after):

```python
def build_conversion_training_data(pbp: pd.DataFrame) -> pd.DataFrame:
    decisions = filter_fourth_down_decisions(pbp)
    go_for_it = decisions[decisions["decision"] == "go_for_it"].copy()
    go_for_it["converted"] = go_for_it["fourth_down_converted"].astype(int)
    return go_for_it
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `pytest tests/features/test_build_features.py -v`
Expected: PASS, including the new test and every pre-existing test in the file.

- [ ] **Step 6: Commit**

```bash
git add src/nfl4th/data/ingest.py src/nfl4th/features/build_features.py tests/features/test_build_features.py
git commit -m "Add fourth down conversion outcome to the loaded data and build training rows for it"
```

- [ ] **Step 7: Write the failing tests for the conversion model**

Create `tests/models/test_conversion.py`:

```python
import numpy as np
import pandas as pd

from nfl4th.models.conversion import (
    load_conversion_model,
    predict_conversion,
    save_conversion_model,
    train_conversion_model,
)


def _synthetic_conversion_data(n_per_class: int = 30) -> pd.DataFrame:
    rows = []
    for i in range(n_per_class):
        # Short yardage: converts. Long yardage: does not.
        rows.append(
            {
                "ydstogo": 1, "yardline_100": 45, "score_differential": 0,
                "game_seconds_remaining": 1800, "qtr": 2,
                "posteam_timeouts_remaining": 3, "defteam_timeouts_remaining": 3,
                "is_home": 1, "converted": 1,
            }
        )
        rows.append(
            {
                "ydstogo": 15, "yardline_100": 60, "score_differential": 0,
                "game_seconds_remaining": 1800, "qtr": 2,
                "posteam_timeouts_remaining": 3, "defteam_timeouts_remaining": 3,
                "is_home": 1, "converted": 0,
            }
        )
    return pd.DataFrame(rows)


def test_predict_conversion_returns_valid_probabilities():
    train_df = _synthetic_conversion_data()
    model = train_conversion_model(train_df, seed=42)

    probs = predict_conversion(model, train_df)

    assert len(probs) == len(train_df)
    assert (probs >= 0).all() and (probs <= 1).all()


def test_training_is_deterministic_given_seed():
    train_df = _synthetic_conversion_data()

    model_a = train_conversion_model(train_df, seed=42)
    model_b = train_conversion_model(train_df, seed=42)

    probs_a = predict_conversion(model_a, train_df)
    probs_b = predict_conversion(model_b, train_df)

    assert np.allclose(probs_a.to_numpy(), probs_b.to_numpy())


def test_learns_the_obvious_pattern():
    train_df = _synthetic_conversion_data()
    model = train_conversion_model(train_df, seed=42)

    short_yardage_row = train_df[train_df["ydstogo"] == 1].iloc[[0]]
    long_yardage_row = train_df[train_df["ydstogo"] == 15].iloc[[0]]

    short_prob = predict_conversion(model, short_yardage_row).iloc[0]
    long_prob = predict_conversion(model, long_yardage_row).iloc[0]

    assert short_prob > long_prob


def test_save_and_load_round_trips_predictions(tmp_path):
    train_df = _synthetic_conversion_data()
    model = train_conversion_model(train_df, seed=42)
    original_probs = predict_conversion(model, train_df)

    save_conversion_model(model, tmp_path / "conversion")
    loaded_model = load_conversion_model(tmp_path / "conversion")
    loaded_probs = predict_conversion(loaded_model, train_df)

    assert np.allclose(original_probs.to_numpy(), loaded_probs.to_numpy())
```

- [ ] **Step 8: Run the tests to verify they fail**

Run: `pytest tests/models/test_conversion.py -v`
Expected: FAIL (or collection error) — `nfl4th.models.conversion` does not exist yet.

- [ ] **Step 9: Implement the conversion model module**

Create `src/nfl4th/models/conversion.py`:

```python
from __future__ import annotations

from pathlib import Path

import pandas as pd
import xgboost as xgb

CONVERSION_FEATURES = [
    "ydstogo",
    "yardline_100",
    "score_differential",
    "game_seconds_remaining",
    "qtr",
    "posteam_timeouts_remaining",
    "defteam_timeouts_remaining",
    "is_home",
]


def train_conversion_model(train_df: pd.DataFrame, seed: int = 42) -> xgb.XGBClassifier:
    X = train_df[CONVERSION_FEATURES]
    y = train_df["converted"]

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        tree_method="hist",
        random_state=seed,
        n_estimators=200,
        max_depth=4,
    )
    model.fit(X, y)
    return model


def predict_conversion(model: xgb.XGBClassifier, df: pd.DataFrame) -> pd.Series:
    X = df[CONVERSION_FEATURES]
    probs = model.predict_proba(X)[:, 1]
    return pd.Series(probs, index=df.index, name="conversion_probability")


def save_conversion_model(model: xgb.XGBClassifier, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    model.save_model(path / "model.json")


def load_conversion_model(path: Path) -> xgb.XGBClassifier:
    model = xgb.XGBClassifier()
    model.load_model(path / "model.json")
    return model
```

- [ ] **Step 10: Run the tests to verify they pass**

Run: `pytest tests/models/test_conversion.py -v`
Expected: PASS, all 4 tests.

- [ ] **Step 11: Run the full test suite and lint**

Run: `pytest -q` and `ruff check .`
Expected: all pass, no regressions (both changes so far are additive; the `PBP_COLUMNS` addition only adds a column, it does not remove or rename anything existing).

- [ ] **Step 12: Commit**

```bash
git add src/nfl4th/models/conversion.py tests/models/test_conversion.py
git commit -m "Add the conversion probability model"
```

---

## Task 3: Wire findings and the conversion model into the pipeline

**Files:**
- Modify: `src/nfl4th/pipeline.py`
- Modify: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `situational_splits`, `coach_bucket_leaderboard`, `league_trend_over_time` (Task 1); `build_conversion_training_data`, `CONVERSION_FEATURES`, `train_conversion_model`, `predict_conversion`, `save_conversion_model`, `load_conversion_model` (Task 2); the existing `time_based_split` (`src/nfl4th/features/split.py`, unchanged, already used elsewhere in this file).
- Produces: `run()` now also writes `api/models/conversion/` (via `save_conversion_model`) and `api/models/findings.json` when given `model_dir`. Task 4 (API wiring) and Task 5 (retrain script) both depend on these two artifacts existing with the exact shape defined here.

- [ ] **Step 1: Read the current `run()` function**

Before editing, read `src/nfl4th/pipeline.py`'s current `run()` function in full. It currently does, in order: loads `pbp` and `schedules`, builds `features` via `build_feature_table`, splits into `train_df`/`val_df`/`test_df` via `time_based_split`, trains `baseline_model`/`baseline_no_coach_model`/`embedding_model`, prints a model comparison, builds `report` via `coach_tendency_report`, and (if `model_dir` is given) saves the baseline models, the embedding model, and calls `save_run_artifacts(report, all_seasons, model_dir)`. This task adds to that flow without changing anything already there.

- [ ] **Step 2: Write the failing test for the new artifacts**

Add to `tests/test_pipeline.py` (read the file first to match its existing synthetic-fixture style, in particular `_synthetic_pbp` and `_synthetic_schedules` and the existing `test_run_saves_models_when_model_dir_is_given` test, which this new test parallels; the existing `_synthetic_pbp` fixture will need a `fourth_down_converted` column added to it, since `load_pbp` no longer strips it out and the conversion training path now reads it — add `"fourth_down_converted": 1` as a constant value to every row `_synthetic_pbp` builds, since the exact value does not matter for this test, only that the column exists):

```python
def test_run_saves_findings_and_conversion_model_when_model_dir_is_given(monkeypatch, tmp_path):
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

    assert (tmp_path / "conversion" / "model.json").exists()

    findings_path = tmp_path / "findings.json"
    assert findings_path.exists()
    findings = json.loads(findings_path.read_text())
    assert "situational_splits" in findings
    assert "coach_bucket_leaderboard" in findings
    assert "league_trend" in findings
    assert "conversion_by_distance" in findings
```

(This file already imports `json` at the top per the existing `test_run_saves_report_and_metadata_when_model_dir_is_given` test — confirm that import is present, it should already be there.)

- [ ] **Step 3: Run the test to verify it fails**

Run: `pytest tests/test_pipeline.py::test_run_saves_findings_and_conversion_model_when_model_dir_is_given -v`
Expected: FAIL — `conversion/model.json` and `findings.json` are not written yet.

- [ ] **Step 4: Update the synthetic pbp fixture**

In `tests/test_pipeline.py`'s `_synthetic_pbp` function, add `"fourth_down_converted": 1` to the row dict it builds (a constant value is fine; this test suite does not exercise conversion-model accuracy, only that the pipeline wires the pieces together and produces the right files).

- [ ] **Step 5: Add the imports and wire the new steps into `run()`**

In `src/nfl4th/pipeline.py`, add to the existing import block:

```python
from nfl4th.analysis.findings import (
    coach_bucket_leaderboard,
    league_trend_over_time,
    situational_splits,
)
from nfl4th.features.build_features import build_conversion_training_data
from nfl4th.models.conversion import (
    CONVERSION_FEATURES,
    load_conversion_model,
    predict_conversion,
    save_conversion_model,
    train_conversion_model,
)
```

(`load_conversion_model` is imported here for symmetry with the rest of the file's model imports, even though `run()` itself only saves; nothing in this task calls it. `build_feature_table` and the other existing imports stay exactly as they are.)

Add a new function, near `save_run_artifacts`:

```python
def _conversion_grid(conversion_model) -> list[dict]:
    representative_situation = {
        "yardline_100": 50,
        "score_differential": 0,
        "game_seconds_remaining": 1800,
        "qtr": 2,
        "posteam_timeouts_remaining": 3,
        "defteam_timeouts_remaining": 3,
        "is_home": 1,
    }
    bucket_midpoints = {"1": 1, "2": 2, "3": 3, "4-6": 5, "7-10": 8, "11+": 15}

    rows = []
    for label, ydstogo in bucket_midpoints.items():
        situation = pd.DataFrame([{**representative_situation, "ydstogo": ydstogo}])
        probability = predict_conversion(conversion_model, situation).iloc[0]
        rows.append({"distance_bucket": label, "conversion_probability": float(probability)})
    return rows


def save_findings_artifacts(train_df: pd.DataFrame, conversion_model, model_dir: Path) -> None:
    findings = {
        "situational_splits": situational_splits(train_df).to_dict(orient="records"),
        "coach_bucket_leaderboard": coach_bucket_leaderboard(train_df).to_dict(orient="records"),
        "league_trend": league_trend_over_time(train_df).to_dict(orient="records"),
        "conversion_by_distance": _conversion_grid(conversion_model),
    }
    (model_dir / "findings.json").write_text(json.dumps(findings))
```

(`_conversion_grid`'s bucket midpoints are deliberately fixed representative values, not derived from real data per bucket — the design spec calls this out explicitly as a documented simplification for a "league average conversion by distance" chart, not a bug. `11+`'s midpoint of 15 is an arbitrary but reasonable representative value for that open-ended bucket.)

In `run()`, after the existing `report = coach_tendency_report(train_df, baseline_no_coach_train_probs)` line and before the `if model_dir is not None:` block, add:

```python
    conversion_train_df, _, _ = time_based_split(
        build_conversion_training_data(pbp), train_seasons, range(0), range(0)
    )
    conversion_model = train_conversion_model(conversion_train_df)
```

Inside the existing `if model_dir is not None:` block, after the existing `save_run_artifacts(report, all_seasons, model_dir)` line, add:

```python
        save_conversion_model(conversion_model, model_dir / "conversion")
        save_findings_artifacts(train_df, conversion_model, model_dir)
```

- [ ] **Step 6: Run the test to verify it passes**

Run: `pytest tests/test_pipeline.py -v`
Expected: PASS, including the new test and every pre-existing test in the file.

- [ ] **Step 7: Run the full test suite and lint**

Run: `pytest -q` and `ruff check .`
Expected: all pass.

- [ ] **Step 8: Regenerate the real committed model artifacts**

Run the same command used to generate every previous version of the real committed artifacts:

```bash
.venv/Scripts/python scripts/run_pipeline.py --train-start 2010 --train-end 2026 --val-end 2026 --test-end 2026 --model-dir api/models
```

This is a real training run against real cached data and will take real time and memory (a `PBP_COLUMNS`-filtered load, from prior work in this project, keeps this well within normal limits on this machine as long as nothing else is under heavy memory pressure at the same time; if it fails with a memory error, retry once, and if it fails again, stop and report rather than guessing at a workaround). After it completes, check that `api/models/conversion/model.json` and `api/models/findings.json` both exist and that `findings.json` contains plausible numbers (spot check: `situational_splits`' `go_for_it_rate` should generally increase as distance bucket gets shorter; `conversion_by_distance`'s `conversion_probability` should generally decrease as distance increases).

- [ ] **Step 9: Run the full test suite again against the regenerated artifacts**

Run: `pytest -q`
Expected: all pass (this project's API tests hit the real committed artifacts directly, not mocks, so this is a real end-to-end check, even though Task 4 hasn't wired the API to read the new fields yet).

- [ ] **Step 10: Commit**

```bash
git add src/nfl4th/pipeline.py tests/test_pipeline.py api/models
git commit -m "Compute findings and train the conversion model as part of every pipeline run"
```

---

## Task 4: Serve findings and conversion probability from the API

**Files:**
- Modify: `api/loader.py`
- Modify: `api/schemas.py`
- Modify: `api/routes_predict.py`
- Create: `api/routes_findings.py`
- Modify: `api/main.py`
- Test: `tests/api/test_routes_predict.py` (add assertions)
- Test: `tests/api/test_routes_findings.py`

**Interfaces:**
- Consumes: `api/models/conversion/` and `api/models/findings.json`, both real and committed by Task 3.
- Produces: `POST /predict`'s JSON response gains a `conversion_probability` field. A new `GET /findings` endpoint. Task 6 (frontend API client) consumes both by these exact names.

- [ ] **Step 1: Load the conversion model and findings data at startup**

In `api/loader.py`, add to the imports:

```python
from nfl4th.models.conversion import load_conversion_model
```

In `LoadedModels.__init__`, after the existing `self.latest_season = metadata["latest_season"]` line, add:

```python
        self.conversion_model = load_conversion_model(model_dir / "conversion")
        self.findings = json.loads((model_dir / "findings.json").read_text())
```

- [ ] **Step 2: Add `conversion_probability` to the predict response schema**

In `api/schemas.py`, change `PredictResponse` from:

```python
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
```

to:

```python
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
```

Also add two new schemas, used by the findings endpoint, anywhere in this file:

```python
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
```

- [ ] **Step 3: Wire `conversion_probability` into `/predict`**

In `api/routes_predict.py`, add to the imports:

```python
from nfl4th.models.conversion import predict_conversion
```

Inside the `predict` function, after the existing `league_probs = predict_baseline(loaded.baseline_no_coach_model, situation)` line, add:

```python
        conversion_probability = predict_conversion(loaded.conversion_model, situation).iloc[0]
```

Change the final `return PredictResponse(...)` call from:

```python
        return PredictResponse(
            predicted=_probs_to_schema(combined_probs),
            coach_career_average=coach_career_average,
            league_baseline=_probs_to_schema(league_probs),
        )
```

to:

```python
        return PredictResponse(
            predicted=_probs_to_schema(combined_probs),
            coach_career_average=coach_career_average,
            league_baseline=_probs_to_schema(league_probs),
            conversion_probability=float(conversion_probability),
        )
```

- [ ] **Step 4: Add the findings route**

Create `api/routes_findings.py`:

```python
from __future__ import annotations

from fastapi import APIRouter

from api.loader import LoadedModels
from api.schemas import FindingsResponse


def register_findings_routes(router: APIRouter, loaded: LoadedModels) -> None:
    @router.get("/findings", response_model=FindingsResponse)
    def findings() -> FindingsResponse:
        return FindingsResponse(**loaded.findings)
```

In `api/main.py`, add to the imports:

```python
from api.routes_findings import register_findings_routes
```

After the existing `register_predict_routes(router, loaded)` line, add:

```python
register_findings_routes(router, loaded)
```

- [ ] **Step 5: Add tests**

Add to `tests/api/test_routes_predict.py` (read the file first to match its exact existing style before adding): in whichever existing test calls `POST /predict` for a known coach and situation and checks the response body, add an assertion that `"conversion_probability" in body` and `0.0 <= body["conversion_probability"] <= 1.0`.

Create `tests/api/test_routes_findings.py`:

```python
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_findings_returns_all_four_sections():
    response = client.get("/findings")

    assert response.status_code == 200
    body = response.json()
    assert "situational_splits" in body
    assert "coach_bucket_leaderboard" in body
    assert "league_trend" in body
    assert "conversion_by_distance" in body
    assert len(body["situational_splits"]) > 0
    assert len(body["league_trend"]) > 0
    assert len(body["conversion_by_distance"]) > 0
```

- [ ] **Step 6: Run the full test suite and lint**

Run: `pytest -q` and `ruff check .`
Expected: all pass, including the new and modified tests, against the real regenerated `api/models/` artifacts from Task 3.

- [ ] **Step 7: Commit**

```bash
git add api/loader.py api/schemas.py api/routes_predict.py api/routes_findings.py api/main.py tests/api/test_routes_predict.py tests/api/test_routes_findings.py
git commit -m "Serve conversion probability and findings from the API"
```

---

## Task 5: Add the conversion model and findings to the automated retrain

**Files:**
- Modify: `scripts/retrain.py`

**Interfaces:**
- Consumes: `pipeline.run()`'s updated behavior from Task 3 (already trains and saves the conversion model and findings whenever `model_dir` is given — no new function call is needed here, `run()` already does it).
- Produces: nothing further depends on this task.

- [ ] **Step 1: Confirm no code change is actually needed for the training step itself**

Read `scripts/retrain.py`'s current `main()` function. It already calls `run(train_seasons=..., val_seasons=..., test_seasons=..., model_dir=MODEL_DIR)` for the real production retrain, unchanged by this plan. Since Task 3 made `run()` itself always train and save the conversion model and findings whenever `model_dir` is given, `scripts/retrain.py`'s actual training and saving step needs no modification at all. This step exists to make that explicit, not to change anything.

- [ ] **Step 2: Add a sanity-check note to the file for future readers**

Add one line to the comment block near the top of `scripts/retrain.py`, immediately after the existing comment that explains `SANITY_CHECK_HOLDOUT_SEASON`'s purpose (read the current file first to place this correctly relative to what is already there):

```python
# The conversion model and findings.json are trained/computed as part of the
# same run() call above; they do not have their own separate sanity check,
# since a broken run() call already fails this script before anything is
# committed (see the accuracy floor check and the latest-season decision
# count check below).
```

- [ ] **Step 3: Verify manually**

Run: `.venv/Scripts/python -m pytest -q` and `.venv/Scripts/python -m ruff check .`
Expected: both clean, no changes to test behavior since this task is documentation-only.

This task deliberately does not re-run `scripts/retrain.py` itself end to end (that requires the full real 16-season dataset and has hit real local memory pressure in past work on this project); Task 3's Step 8 already proved `run()` produces the conversion model and findings correctly against real data, and this task does not change that code path.

- [ ] **Step 4: Commit**

```bash
git add scripts/retrain.py
git commit -m "Note that the conversion model and findings ride along with the existing retrain"
```

---

## Task 6: Frontend API client for conversion probability and findings

**Files:**
- Modify: `frontend/src/lib/api.ts`

**Interfaces:**
- Consumes: `POST /predict`'s new `conversion_probability` field and the new `GET /findings` endpoint (Task 4).
- Produces: `PredictResponse.conversion_probability: number`. New types `SituationalSplit`, `CoachBucketEntry`, `LeagueTrendPoint`, `ConversionByDistance`, `FindingsData`, and a new `getFindings(): Promise<FindingsData>` function. Task 7 and Task 8 both consume these by these exact names.

- [ ] **Step 1: Add `conversion_probability` to `PredictResponse`**

In `frontend/src/lib/api.ts`, change:

```typescript
export interface PredictResponse {
  predicted: DecisionProbabilities;
  coach_career_average: DecisionProbabilities;
  league_baseline: DecisionProbabilities;
}
```

to:

```typescript
export interface PredictResponse {
  predicted: DecisionProbabilities;
  coach_career_average: DecisionProbabilities;
  league_baseline: DecisionProbabilities;
  conversion_probability: number;
}
```

- [ ] **Step 2: Add findings types and client function**

Add these interfaces and function anywhere after the existing `predict` function's definition:

```typescript
export interface SituationalSplit {
  distance_bucket: string;
  n_decisions: number;
  go_for_it_rate: number;
}

export interface CoachBucketEntry {
  coach: string;
  distance_bucket: string;
  n_decisions: number;
  go_for_it_rate: number;
}

export interface LeagueTrendPoint {
  season: number;
  n_decisions: number;
  go_for_it_rate: number;
}

export interface ConversionByDistance {
  distance_bucket: string;
  conversion_probability: number;
}

export interface FindingsData {
  situational_splits: SituationalSplit[];
  coach_bucket_leaderboard: CoachBucketEntry[];
  league_trend: LeagueTrendPoint[];
  conversion_by_distance: ConversionByDistance[];
}

export async function getFindings(): Promise<FindingsData> {
  const response = await fetch(`${API_URL}/findings`);
  if (!response.ok) throw new Error(`Failed to load findings: ${response.status}`);
  return response.json();
}
```

- [ ] **Step 3: Verify the build**

Run: `cd frontend && npm run build`
Expected: succeeds with no TypeScript errors (this step only adds types and one function; nothing yet calls `getFindings()` or reads `conversion_probability`, so there is nothing to manually verify in a browser yet).

- [ ] **Step 4: Commit**

```bash
git add frontend/src/lib/api.ts
git commit -m "Add conversion probability and findings to the frontend API client"
```

---

## Task 7: Show conversion probability on the predictor

**Files:**
- Modify: `frontend/src/pages/predict.astro`

**Interfaces:**
- Consumes: `PredictResponse.conversion_probability` (Task 6).
- Produces: nothing further depends on this task.

- [ ] **Step 1: Add a conversion probability card to the result rendering**

In `frontend/src/pages/predict.astro`'s `<script>` block, add a new function right after the existing `renderProbabilities` function:

```typescript
  function renderConversionCard(probability: number): string {
    const pct = probability * 100;
    return `
      <article class="result-card conversion-card">
        <div class="result-card-head">
          <h3>If they go for it</h3>
          <span class="result-tag">Conversion odds</span>
        </div>
        <p class="note">Chance a go-for-it attempt succeeds here, regardless of which decision is predicted above.</p>
        <p class="conversion-value">${pct.toFixed(1)}%</p>
      </article>
    `;
  }
```

Change the `result.innerHTML` assignment inside the form submit handler from:

```typescript
      result.innerHTML = `
        ${renderProbabilities("Model prediction", "This situation", "For this exact situation.", response.predicted, true)}
        ${renderProbabilities("This coach's career average", "Career-wide", "Across every 4th down they've faced, not specific to this situation.", response.coach_career_average, false)}
        ${renderProbabilities("League baseline", "This situation", "A coach-agnostic model, for this exact situation.", response.league_baseline, false)}
      `;
```

to:

```typescript
      result.innerHTML = `
        ${renderProbabilities("Model prediction", "This situation", "For this exact situation.", response.predicted, true)}
        ${renderProbabilities("This coach's career average", "Career-wide", "Across every 4th down they've faced, not specific to this situation.", response.coach_career_average, false)}
        ${renderProbabilities("League baseline", "This situation", "A coach-agnostic model, for this exact situation.", response.league_baseline, false)}
        ${renderConversionCard(response.conversion_probability)}
      `;
```

- [ ] **Step 2: Style the new card**

Add to the `<style>` block, near the existing `:global(.result-card.featured)` rule:

```css
  :global(.conversion-card) {
    grid-column: 1 / -1;
  }

  :global(.conversion-value) {
    font-family: var(--font-display);
    font-size: 2rem;
    font-weight: 600;
    color: var(--color-accent-2);
    margin: 0;
  }
```

- [ ] **Step 3: Verify manually**

With the backend running against the real regenerated artifacts (`.venv/Scripts/python scripts/run_api.py`) and the frontend dev server running (`cd frontend && npm run dev`), load `/predict`, submit the form for any coach and situation, and confirm a fourth result card appears showing a percentage between 0 and 100, and that a short-yardage situation (e.g. 1 yard to go) shows a visibly higher conversion percentage than a long-yardage one (e.g. 15 yards to go) for the same coach.

Run `cd frontend && npm run build` and confirm it completes with no errors.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/pages/predict.astro
git commit -m "Show conversion odds alongside the decision prediction"
```

---

## Task 8: Findings page

**Files:**
- Create: `frontend/src/pages/findings.astro`
- Modify: `frontend/src/layouts/Layout.astro`

**Interfaces:**
- Consumes: `getCoaches()` (existing, for the superlative leaderboards), `getFindings()` and `FindingsData` (Task 6).
- Produces: nothing further depends on this task.

- [ ] **Step 1: Add the nav link**

In `frontend/src/layouts/Layout.astro`, change the `<nav>` block from:

```astro
        <nav aria-label="Primary">
          <a href="/" aria-current={isActive("/") ? "page" : undefined}>Home</a>
          <a href="/predict" aria-current={isActive("/predict") ? "page" : undefined}>Predict</a>
          <a href="/coaches" aria-current={isActive("/coaches") ? "page" : undefined}>Coaches</a>
          <a href="/about" aria-current={isActive("/about") ? "page" : undefined}>About</a>
        </nav>
```

to:

```astro
        <nav aria-label="Primary">
          <a href="/" aria-current={isActive("/") ? "page" : undefined}>Home</a>
          <a href="/predict" aria-current={isActive("/predict") ? "page" : undefined}>Predict</a>
          <a href="/coaches" aria-current={isActive("/coaches") ? "page" : undefined}>Coaches</a>
          <a href="/findings" aria-current={isActive("/findings") ? "page" : undefined}>Findings</a>
          <a href="/about" aria-current={isActive("/about") ? "page" : undefined}>About</a>
        </nav>
```

Also add `/findings` as a fourth link card on the home page. In `frontend/src/pages/index.astro`, inside the existing `.link-cards` section, add a new card between the "Browse coaches" and "About this project" cards:

```astro
    <a class="link-card" href="/findings">
      <h2>See the findings</h2>
      <p>Leaderboards, situational splits, and how the league's aggressiveness has changed since 2010.</p>
      <span class="link-cta">Explore the findings &rarr;</span>
    </a>
```

- [ ] **Step 2: Create the findings page**

Create `frontend/src/pages/findings.astro`:

```astro
---
import Layout from "../layouts/Layout.astro";
---

<Layout title="Findings | NFL 4th Down Predictor" description="Superlative coach leaderboards, situational go-for-it splits, conversion odds by distance, and how league-wide aggressiveness has changed since 2010.">
  <section class="intro">
    <p class="eyebrow">The findings</p>
    <h1>What the data actually shows</h1>
    <p class="lede">
      Real patterns pulled from every 4th down decision since 2010, not just a lookup tool.
    </p>
  </section>

  <section class="findings-section">
    <p class="eyebrow">Superlatives</p>
    <div id="superlatives" class="superlative-grid">
      <p class="loading-note">Loading&hellip;</p>
    </div>
  </section>

  <section class="findings-section">
    <p class="eyebrow">Situational splits</p>
    <p class="section-note">How often the league goes for it, by distance to go.</p>
    <div id="splits-chart" class="bar-chart"></div>
  </section>

  <section class="findings-section">
    <p class="eyebrow">Conversion odds</p>
    <p class="section-note">If a team goes for it, how often it actually works, by distance to go.</p>
    <div id="conversion-chart" class="bar-chart"></div>
  </section>

  <section class="findings-section">
    <p class="eyebrow">League trend</p>
    <p class="section-note">League-wide go-for-it rate by season.</p>
    <div id="trend-chart" class="trend-chart"></div>
  </section>

  <section class="findings-section">
    <p class="eyebrow">By coach, by situation</p>
    <p class="section-note">The most aggressive coaches within each distance bucket, minimum sample size applied.</p>
    <div id="coach-leaderboard" class="coach-leaderboard">
      <p class="loading-note">Loading&hellip;</p>
    </div>
  </section>
</Layout>

<script>
  import { getCoaches, getFindings, type CoachProfile, type FindingsData } from "../lib/api";

  function renderSuperlatives(coaches: CoachProfile[]): string {
    const mostAggressive = [...coaches].sort((a, b) => b.go_for_it.shrunk - a.go_for_it.shrunk)[0];
    const mostConservative = [...coaches].sort((a, b) => a.go_for_it.shrunk - b.go_for_it.shrunk)[0];
    const biggestGap = [...coaches].sort(
      (a, b) =>
        Math.abs(b.go_for_it.observed - b.go_for_it.expected) - Math.abs(a.go_for_it.observed - a.go_for_it.expected)
    )[0];

    const card = (label: string, coach: CoachProfile, value: string) => `
      <div class="superlative-card">
        <span class="superlative-label">${label}</span>
        <span class="superlative-coach">${coach.coach}</span>
        <span class="superlative-value">${value}</span>
      </div>
    `;

    return (
      card("Most aggressive", mostAggressive, `${(mostAggressive.go_for_it.shrunk * 100).toFixed(1)}% go-for-it rate`) +
      card("Most conservative", mostConservative, `${(mostConservative.go_for_it.shrunk * 100).toFixed(1)}% go-for-it rate`) +
      card(
        "Biggest outlier",
        biggestGap,
        `${((biggestGap.go_for_it.observed - biggestGap.go_for_it.expected) * 100).toFixed(1)} points from expected`
      )
    );
  }

  function renderBarChart(rows: { distance_bucket: string; value: number }[]): string {
    return rows
      .map(
        (row) => `
        <div class="bar-row">
          <span class="bar-label">${row.distance_bucket}</span>
          <div class="bar-track">
            <span class="bar-fill" style="width: ${row.value * 100}%"></span>
          </div>
          <span class="bar-value">${(row.value * 100).toFixed(1)}%</span>
        </div>
      `
      )
      .join("");
  }

  function renderTrendChart(points: { season: number; go_for_it_rate: number }[]): string {
    if (points.length === 0) return "";
    const width = 600;
    const height = 200;
    const padding = 24;
    const minSeason = points[0].season;
    const maxSeason = points[points.length - 1].season;
    const seasonSpan = Math.max(maxSeason - minSeason, 1);

    const coords = points.map((point) => {
      const x = padding + ((point.season - minSeason) / seasonSpan) * (width - padding * 2);
      const y = height - padding - point.go_for_it_rate * (height - padding * 2);
      return `${x},${y}`;
    });

    return `
      <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="League go for it rate by season, from ${minSeason} to ${maxSeason}">
        <polyline points="${coords.join(" ")}" fill="none" stroke="var(--color-accent)" stroke-width="2" />
      </svg>
      <div class="trend-labels">
        <span>${minSeason}</span>
        <span>${maxSeason}</span>
      </div>
    `;
  }

  function renderCoachLeaderboard(entries: FindingsData["coach_bucket_leaderboard"]): string {
    const buckets = [...new Set(entries.map((entry) => entry.distance_bucket))];
    return buckets
      .map((bucket) => {
        const top = entries
          .filter((entry) => entry.distance_bucket === bucket)
          .slice(0, 3);
        if (top.length === 0) return "";
        const rows = top
          .map(
            (entry) => `
            <li>${entry.coach} &mdash; ${(entry.go_for_it_rate * 100).toFixed(1)}% (${entry.n_decisions} attempts)</li>
          `
          )
          .join("");
        return `
          <div class="leaderboard-bucket">
            <h3>${bucket} yards to go</h3>
            <ul>${rows}</ul>
          </div>
        `;
      })
      .join("");
  }

  Promise.all([getCoaches(), getFindings()])
    .then(([coaches, findings]) => {
      document.getElementById("superlatives")!.innerHTML = renderSuperlatives(coaches);
      document.getElementById("splits-chart")!.innerHTML = renderBarChart(
        findings.situational_splits.map((split) => ({ distance_bucket: split.distance_bucket, value: split.go_for_it_rate }))
      );
      document.getElementById("conversion-chart")!.innerHTML = renderBarChart(
        findings.conversion_by_distance.map((entry) => ({
          distance_bucket: entry.distance_bucket,
          value: entry.conversion_probability,
        }))
      );
      document.getElementById("trend-chart")!.innerHTML = renderTrendChart(findings.league_trend);
      document.getElementById("coach-leaderboard")!.innerHTML = renderCoachLeaderboard(findings.coach_bucket_leaderboard);
    })
    .catch((error) => {
      const message = `<p class="error">Could not load findings: ${error instanceof Error ? error.message : "unknown error"}</p>`;
      document.getElementById("superlatives")!.innerHTML = message;
    });
</script>

<style>
  .intro {
    max-width: 52rem;
    margin-bottom: var(--space-5);
  }

  .eyebrow {
    font-family: var(--font-mono);
    font-size: 0.75rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--color-accent);
    margin: 0 0 var(--space-2);
  }

  .lede {
    font-size: 1.05rem;
    max-width: 46rem;
  }

  .findings-section {
    background: var(--color-surface);
    border: 1px solid var(--color-line);
    border-radius: 6px;
    padding: var(--space-4);
    margin-bottom: var(--space-5);
  }

  .section-note {
    font-size: 0.9rem;
    margin-bottom: var(--space-3);
  }

  .loading-note {
    font-family: var(--font-mono);
    font-size: 0.85rem;
    color: var(--color-text-dim);
  }

  .superlative-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(14rem, 1fr));
    gap: var(--space-3);
  }

  .superlative-card {
    display: flex;
    flex-direction: column;
    gap: 0.3rem;
    background: var(--color-surface-raised);
    border: 1px solid var(--color-line);
    border-radius: 4px;
    padding: var(--space-3);
  }

  .superlative-label {
    font-family: var(--font-mono);
    font-size: 0.7rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--color-text-dim);
  }

  .superlative-coach {
    font-family: var(--font-display);
    font-size: 1.1rem;
    font-weight: 600;
  }

  .superlative-value {
    font-size: 0.85rem;
    color: var(--color-text-dim);
  }

  .bar-chart {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
  }

  .bar-row {
    display: grid;
    grid-template-columns: 4rem 1fr 4rem;
    align-items: center;
    gap: var(--space-2);
  }

  .bar-label {
    font-family: var(--font-mono);
    font-size: 0.8rem;
    color: var(--color-text-dim);
  }

  .bar-track {
    height: 0.9rem;
    background: var(--color-bg);
    border: 1px solid var(--color-line);
    border-radius: 999px;
    overflow: hidden;
  }

  .bar-fill {
    display: block;
    height: 100%;
    background: var(--color-accent);
  }

  .bar-value {
    font-family: var(--font-mono);
    font-size: 0.8rem;
    text-align: right;
  }

  .trend-chart svg {
    width: 100%;
    height: auto;
  }

  .trend-labels {
    display: flex;
    justify-content: space-between;
    font-family: var(--font-mono);
    font-size: 0.75rem;
    color: var(--color-text-dim);
    margin-top: var(--space-2);
  }

  .coach-leaderboard {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(14rem, 1fr));
    gap: var(--space-3);
  }

  .leaderboard-bucket h3 {
    font-size: 0.95rem;
    margin-bottom: var(--space-2);
  }

  .leaderboard-bucket ul {
    list-style: none;
    padding: 0;
    margin: 0;
    font-size: 0.85rem;
    color: var(--color-text-dim);
    display: flex;
    flex-direction: column;
    gap: 0.35rem;
  }

  .error {
    color: var(--color-accent);
    font-family: var(--font-mono);
  }
</style>
```

- [ ] **Step 3: Verify manually**

With the backend running against the real regenerated artifacts and the frontend dev server running, load `/findings` and confirm: three superlative cards render with real coach names, the situational splits bar chart shows six bars with distance labels and percentages, the conversion odds chart shows six bars trending downward as distance increases, the trend chart renders an SVG line, and the coach leaderboard shows real coach names grouped by distance bucket. Confirm the new "Findings" nav link and home page card both work and highlight correctly.

Run `cd frontend && npm run build` and confirm it completes with no errors and reports 5 pages.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/pages/findings.astro frontend/src/pages/index.astro frontend/src/layouts/Layout.astro
git commit -m "Add the findings page"
```
