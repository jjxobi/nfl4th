# Automated Weekly Retraining Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A scheduled GitHub Actions workflow that retrains the coach tendency models weekly against the latest available season data, sanity-checks the result, and commits refreshed `api/models/` artifacts to `master` so Render and Netlify's auto-deploy keeps the live site current with no manual step.

**Architecture:** A new pure library function `evaluate()` in `src/nfl4th/pipeline.py` trains on one season range and scores ensemble accuracy on another, reusing the same training/prediction functions `run()` already uses. A new date-only helper `current_target_season()` in `src/nfl4th/data/season.py` computes which NFL season's data should now be included as training data, without hardcoding a season number. A new orchestration script `scripts/retrain.py` uses both: it runs a sanity check against a fixed historical holdout, and only if that passes, retrains on all data through the current season and overwrites `api/models/`. A new `.github/workflows/retrain.yml` runs that script weekly and commits the result if the working tree changed.

**Tech Stack:** Python 3.11, pytest, GitHub Actions (`actions/checkout@v4`, `actions/setup-python@v5`), the existing `nfl4th` package and `pipeline.py` module.

## Global Constraints

- Python `>=3.11,<3.12` (`pyproject.toml`'s `requires-python`), matching every other Python entry point in this repo.
- No mention of AI/Claude/Anthropic anywhere in committed content (code, comments, README, commit messages) — this applies to every commit this plan's implementer makes, AND to the commit message the deployed workflow itself generates at runtime every week.
- No em dashes in any committed text, including the workflow's runtime-generated commit messages.
- No "Co-Authored-By" trailers in any commit, including the workflow's runtime-generated commits.
- Commit messages (both the implementer's and the workflow's runtime ones) must be plain and human-sounding, consistent with the rest of this repo's history (e.g. "Add CORS and a health endpoint to the backend api", "Swap the body font away from Inter").
- No code comments unless they explain a genuinely non-obvious WHY, never a WHAT.
- The retraining workflow authors its commit under a repo-scoped `GITHUB_TOKEN`, not the user's personal account. This is the one documented, pre-approved exception to this project's "no push without explicit confirmation" rule — a scheduled, spec'd, already-approved CI job, not an ad hoc push. Nothing in this plan pushes to any remote as part of implementation; the workflow only pushes once it is later running for real on GitHub, after the user connects a remote (not part of this plan).
- Design source of truth: `docs/superpowers/specs/2026-08-09-phase2-web-app-design.md`, section "Automated freshness" (lines 90-99). Do not deviate from its behavior (weekly Tuesday schedule, sanity check before any commit, fail loudly with nothing committed on a failed sanity check) without flagging the deviation.

---

## Task 1: Evaluation function and season-detection helper

**Files:**
- Modify: `src/nfl4th/pipeline.py` (add a new `evaluate()` function)
- Create: `src/nfl4th/data/season.py`
- Modify: `tests/test_pipeline.py` (add a test for `evaluate()`, reusing existing fixtures)
- Create: `tests/data/test_season.py`

**Interfaces:**
- Consumes: `src/nfl4th/pipeline.py`'s existing imports (`load_pbp`, `load_schedules` from `nfl4th.data.ingest`; `build_feature_table` from `nfl4th.features.build_features`; `time_based_split` from `nfl4th.features.split`; `train_baseline`, `predict_baseline` from `nfl4th.models.baseline`; `train_embedding_model`, `predict_with_coldstart` from `nfl4th.models.embedding_model`) and the existing private `_score()` and public `ensemble_probs()` functions already defined in that file. All of these already exist — do not redefine or re-import anything not already present at the top of `pipeline.py`.
- Produces: `evaluate(train_seasons: range, test_seasons: range) -> dict[str, float]` in `nfl4th.pipeline`, returning `{"accuracy": float, "log_loss": float}` for the ensemble model. `current_target_season(today: date) -> int` in `nfl4th.data.season`. Task 2 imports both by name from these exact locations.

- [ ] **Step 1: Write the failing test for `current_target_season`**

Create `tests/data/test_season.py`:

```python
from datetime import date

from nfl4th.data.season import current_target_season


def test_early_in_a_new_season_returns_that_seasons_year():
    assert current_target_season(date(2026, 9, 15)) == 2026


def test_september_first_is_already_the_new_season():
    assert current_target_season(date(2026, 9, 1)) == 2026


def test_end_of_calendar_year_is_still_that_seasons_year():
    assert current_target_season(date(2026, 12, 31)) == 2026


def test_playoffs_in_january_are_still_the_prior_septembers_season():
    assert current_target_season(date(2027, 1, 15)) == 2026


def test_day_before_kickoff_is_still_the_prior_season():
    assert current_target_season(date(2027, 8, 31)) == 2026


def test_offseason_in_june_is_still_the_prior_season():
    assert current_target_season(date(2027, 6, 1)) == 2026
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/data/test_season.py -v`
Expected: FAIL (or collection error) — `nfl4th.data.season` does not exist yet.

- [ ] **Step 3: Implement `current_target_season`**

Create `src/nfl4th/data/season.py`:

```python
from __future__ import annotations

from datetime import date


def current_target_season(today: date) -> int:
    # NFL season S runs from kickoff in September of year S through the
    # Super Bowl in February of year S+1, so any date before September
    # belongs to the season that started the previous calendar year.
    return today.year if today.month >= 9 else today.year - 1
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/data/test_season.py -v`
Expected: PASS, all 6 tests.

- [ ] **Step 5: Commit**

```bash
git add src/nfl4th/data/season.py tests/data/test_season.py
git commit -m "Add a helper to compute the current NFL season from a date"
```

- [ ] **Step 6: Write the failing test for `evaluate()`**

Add to `tests/test_pipeline.py` (below the existing `test_ensemble_probs_averages_two_prediction_frames` test, above `test_importing_pipeline_in_a_fresh_process_does_not_crash` — the file already imports `pandas as pd`, `pytest`, and `from nfl4th import pipeline` at the top, and already defines `_synthetic_pbp` and `_synthetic_schedules` further down; this new test uses those same two fixtures, so it must come after their definitions in the file, right after `test_run_saves_report_and_metadata_when_model_dir_is_given`):

```python
def test_evaluate_returns_accuracy_and_log_loss(monkeypatch):
    monkeypatch.setattr(pipeline, "load_pbp", lambda seasons: _synthetic_pbp(seasons))
    monkeypatch.setattr(
        pipeline, "load_schedules", lambda seasons: _synthetic_schedules(seasons)
    )

    metrics = pipeline.evaluate(train_seasons=range(2020, 2021), test_seasons=range(2021, 2022))

    assert set(metrics.keys()) == {"accuracy", "log_loss"}
    assert 0.0 <= metrics["accuracy"] <= 1.0
```

- [ ] **Step 7: Run the test to verify it fails**

Run: `pytest tests/test_pipeline.py::test_evaluate_returns_accuracy_and_log_loss -v`
Expected: FAIL with `AttributeError: module 'nfl4th.pipeline' has no attribute 'evaluate'`.

- [ ] **Step 8: Implement `evaluate()`**

Add to `src/nfl4th/pipeline.py`, directly after the existing `save_run_artifacts` function and before `def run(...)`:

```python
def evaluate(train_seasons: range, test_seasons: range) -> dict[str, float]:
    all_seasons = list(train_seasons) + list(test_seasons)
    pbp = load_pbp(all_seasons)
    schedules = load_schedules(all_seasons)
    features = build_feature_table(pbp, schedules)

    train_df, _, test_df = time_based_split(features, train_seasons, range(0, 0), test_seasons)

    baseline_model = train_baseline(train_df)
    embedding_model, vocab = train_embedding_model(train_df)

    baseline_probs = predict_baseline(baseline_model, test_df)
    embedding_probs = predict_with_coldstart(embedding_model, vocab, test_df)
    combined_probs = ensemble_probs(baseline_probs, embedding_probs)

    return _score(combined_probs, test_df["decision"])
```

This deliberately does not train `baseline_no_coach` or save anything to disk — it exists only to answer "how good is the ensemble on this holdout," which is all the weekly sanity check needs.

- [ ] **Step 9: Run the test to verify it passes**

Run: `pytest tests/test_pipeline.py -v`
Expected: PASS, including the new `test_evaluate_returns_accuracy_and_log_loss` and every pre-existing test in the file (no regressions — `evaluate` is additive, nothing existing was changed).

- [ ] **Step 10: Run the full test suite and lint**

Run: `pytest -q` and `ruff check .`
Expected: all tests pass, ruff reports no issues.

- [ ] **Step 11: Commit**

```bash
git add src/nfl4th/pipeline.py tests/test_pipeline.py
git commit -m "Add an evaluate function for scoring the ensemble on a holdout season"
```

---

## Task 2: Retraining script and scheduled workflow

**Files:**
- Create: `scripts/retrain.py`
- Create: `.github/workflows/retrain.yml`
- Modify: `README.md` (add a short note about automated freshness, near the existing "Deployment" section added in the previous plan)

**Interfaces:**
- Consumes: `nfl4th.pipeline.evaluate(train_seasons, test_seasons) -> dict[str, float]` and `nfl4th.pipeline.run(train_seasons, val_seasons, test_seasons, model_dir=None)` (existing, unchanged) from Task 1 and the pre-existing `pipeline.py`; `nfl4th.data.season.current_target_season(today: date) -> int` from Task 1.
- Produces: nothing further consumes this task's output — it is the end of the pipeline (a runnable script plus the workflow that schedules it).

**A real limitation of this task, not a gap to try to close:** the design spec's success criterion "the retraining workflow runs successfully at least once on its schedule and produces a real commit with updated artifacts" cannot be verified by this task. This repo has no GitHub remote configured yet (the user has said they'll push it themselves later), and GitHub Actions schedules and `workflow_dispatch` both require the workflow file to actually exist on GitHub. What this task CAN and does verify: the script's logic runs correctly against real local data (Step 2), and the YAML is syntactically valid (Step 4). Verifying an actual scheduled or manually-dispatched run on GitHub is out of scope here and should happen after the user connects a remote — do not report this task as having confirmed live execution.

- [ ] **Step 1: Write the retraining script**

Create `scripts/retrain.py`:

```python
import sys
from datetime import date
from pathlib import Path

from nfl4th.data.season import current_target_season
from nfl4th.pipeline import evaluate, run

# Fixed on purpose: this is a known-good, already-played-out season (see the
# real accuracy numbers for it in README.md's Results section), so the sanity
# check compares against the same reference point every week instead of a
# moving target that could itself be affected by a bad retrain.
SANITY_CHECK_TRAIN_SEASONS = range(2010, 2025)
SANITY_CHECK_HOLDOUT_SEASON = range(2025, 2026)

# The real baseline model scores 88.3% accuracy on this exact holdout
# (see README.md). 75% leaves generous room for normal season-to-season
# variation while still catching a genuinely broken retrain: the real bug
# fixed during phase 1 (unnormalized features) collapsed accuracy to 46%,
# and always guessing the most common decision only gets 55-65%.
SANITY_CHECK_ACCURACY_FLOOR = 0.75

TRAIN_START_SEASON = 2010
MODEL_DIR = Path("api/models")


def main() -> None:
    metrics = evaluate(SANITY_CHECK_TRAIN_SEASONS, SANITY_CHECK_HOLDOUT_SEASON)
    print(f"Sanity check: accuracy={metrics['accuracy']:.3f}, log_loss={metrics['log_loss']:.3f}")

    if metrics["accuracy"] < SANITY_CHECK_ACCURACY_FLOOR:
        print(
            f"Sanity check failed: accuracy {metrics['accuracy']:.3f} is below the "
            f"floor of {SANITY_CHECK_ACCURACY_FLOOR}. Leaving api/models untouched."
        )
        sys.exit(1)

    latest_season = current_target_season(date.today())
    print(f"Sanity check passed. Retraining on seasons {TRAIN_START_SEASON} through {latest_season}.")

    run(
        train_seasons=range(TRAIN_START_SEASON, latest_season + 1),
        val_seasons=range(latest_season + 1, latest_season + 1),
        test_seasons=range(latest_season + 1, latest_season + 1),
        model_dir=MODEL_DIR,
    )
    print(f"Wrote refreshed models to {MODEL_DIR}")


if __name__ == "__main__":
    main()
```

Note the empty `val_seasons`/`test_seasons` ranges (`range(latest_season + 1, latest_season + 1)`): this matches exactly how the current committed `api/models/` artifacts were originally generated (all available seasons as training data, no held-out split, since the one-time held-out evaluation already happened and is documented in the README) — see `nfl4th.pipeline.run`'s existing signature and behavior, unchanged by this plan.

- [ ] **Step 2: Verify the script runs against real data**

This script is a thin orchestration script over already-tested library functions (`evaluate`, `run`, `current_target_season`), consistent with how `scripts/run_pipeline.py`'s `main()` is also not separately unit tested — so there is no pytest step here. Instead, verify it directly:

Run: `.venv/Scripts/python scripts/retrain.py` (Windows) or `.venv/bin/python scripts/retrain.py` (Linux/Mac) from the repo root, with a working venv that has the project installed (`pip install -e .`) and access to the same cached data other pipeline runs in this repo have used (`data/raw/`, gitignored).

Expected: prints a sanity-check accuracy near 88% (matching the real baseline test accuracy already documented in the README for the 2010-2024 train / 2025 test split), passes the floor check, then retrains on all available seasons and overwrites `api/models/`. This will take real time (the same order of magnitude as any other full pipeline run in this repo) — that's expected, not a bug.

After it completes, run `pytest -q` to confirm the backend test suite still passes against the freshly regenerated `api/models/` artifacts (they're read by `api/loader.py` at import time), and `git status --short api/models` to see what changed (if the current date's `current_target_season` matches what the artifacts were last generated with, the diff may be empty or near-empty — that's fine, it's exactly the "nothing meaningfully changed, so don't commit" case Step 4 of the workflow below handles).

If `git status --short api/models` shows real changes, they are working-tree changes from this manual verification run, not something to commit as part of this task — Step 6 below covers how to leave the tree clean before committing.

- [ ] **Step 3: Write the scheduled workflow**

Create `.github/workflows/retrain.yml`, following this repo's existing `.github/workflows/ci.yml` for Actions conventions (same checkout/setup-python versions, same CPU-only torch install trick already established there to avoid pulling the multi-gigabyte CUDA build):

```yaml
name: Retrain

on:
  schedule:
    # 09:00 UTC every Tuesday: after Monday Night Football has finished and
    # nflverse has had time to finalize the week's play by play data.
    - cron: "0 9 * * 2"
  workflow_dispatch:

permissions:
  contents: write

jobs:
  retrain:
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
        run: pip install -e .
      - name: Retrain and sanity check
        run: python scripts/retrain.py
      - name: Commit and push updated models
        run: |
          if git status --porcelain api/models | grep -q .; then
            git config user.name "github-actions[bot]"
            git config user.email "github-actions[bot]@users.noreply.github.com"
            git add api/models
            git commit -m "Refresh coach tendency models with the latest season data"
            git push
          else
            echo "No changes to api/models, nothing to commit."
          fi
```

Deliberately does not use `actions/cache` for `data/raw/` (the pipeline's parquet cache): every run starts with a clean checkout and no cached play-by-play data, so `load_pbp`/`load_schedules` always fetch fresh from nflverse, including the just-played week for the in-progress season. Caching that directory would risk silently serving stale data for the current season indefinitely — do not add it.

`workflow_dispatch` is included so a run can be triggered manually (from the GitHub Actions UI or `gh workflow run retrain.yml`) without waiting for the Tuesday schedule, which is the only practical way to verify this workflow actually works end to end once it exists on GitHub.

- [ ] **Step 4: Validate the workflow YAML parses**

Run: `.venv/Scripts/python -c "import yaml; yaml.safe_load(open('.github/workflows/retrain.yml'))"` (installs `pyyaml` if not already present in the dev venv — it's a transitive dependency of several existing tools; if the import itself fails with `ModuleNotFoundError`, run `pip install pyyaml` first, it does not need to be added to `pyproject.toml` since this is a one-off local syntax check, not something the project depends on at runtime or in CI).

Expected: no exception — confirms the YAML is well-formed. This cannot verify GitHub Actions-specific semantics (cron syntax, action versions) beyond basic YAML syntax; there is no local tool in this project for deeper Actions validation, and this plan does not add one.

- [ ] **Step 5: Add a README note about automated freshness**

In `README.md`, in the "Deployment" section added by the previous plan (frontend + deployment), add a short paragraph after the existing deployment steps:

```markdown
Model freshness is automatic: `.github/workflows/retrain.yml` runs every
Tuesday, retrains on all available seasons, and only commits the refreshed
`api/models/` if a sanity check against a fixed historical holdout still
clears a reasonable accuracy floor. A passing run's commit lands on `master`
and both Render and Netlify redeploy on push, so no manual step is needed to
keep coach tendencies current through the season.
```

- [ ] **Step 6: Confirm a clean working tree, then run the full verification suite**

If Step 2's manual run left `api/models/` modified, discard those changes before committing this task's actual deliverable (the script and workflow file, not a fresh set of trained models — regenerating `api/models/` for real is what the scheduled workflow itself is for, not something this task should also do as a side effect):

Run: `git status --short` to see what's modified. If `api/models/` shows changes from Step 2's manual run, run `git checkout -- api/models` to discard them (safe: Step 2's run was verification only, and the currently committed `api/models/` artifacts are still the correct, real, already-reviewed serving artifacts).

Then run: `pytest -q` and `ruff check .`
Expected: all pass, matching the state before Step 2's manual run.

- [ ] **Step 7: Commit**

```bash
git add scripts/retrain.py .github/workflows/retrain.yml README.md
git commit -m "Add the scheduled workflow that keeps model artifacts current"
```
