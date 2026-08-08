# NFL 4th Down Coach Tendency Predictor — Phase 1 Design

## Status
Approved. Phase 1 of a two-phase project. Phase 2 (web app / interactive frontend) is out of scope for this spec and will get its own design once phase 1 is stable.

## Problem

Build a model that predicts what an NFL head coach will actually do on 4th down (go for it, punt, or attempt a field goal) given the game situation, based on that coach's real historical tendencies. This is a tendency predictor, not a "what's optimal" recommender: the target is the coach's real decision, not the win-probability-optimal one.

The end goal is a portfolio-quality project: a modular, tested, documented codebase that demonstrates real data engineering and ML skill, not a notebook dump.

## Goals

- Ingest and clean NFL play-by-play data for the 2010-2024 seasons.
- Build a reliable per-game, per-team, head coach mapping (including in-season coaching changes).
- Engineer a clean feature table of every genuine 4th-down decision point.
- Train and compare two models that predict the coach's decision: an XGBoost baseline and a PyTorch model with learned coach embeddings.
- Handle first-time head coaches (no NFL head-coaching track record) with a principled cold-start approach rather than ignoring or crashing on them.
- Produce per-coach tendency/aggression analysis that phase 2 can build on.
- Ship this as a clean, tested, CI-checked Python package suitable for a recruiter to read.

## Non-goals (phase 1)

- No web app, API, or hosted service. That is phase 2.
- No win-probability-optimal decision model ("what a coach should do"). May be a later addition, not phase 1.
- No incorporation of pre-head-coaching history (e.g. offensive/defensive coordinator tendencies). Play-by-play data does not reliably attribute play-calling to coordinators, so this is excluded entirely, not deferred.
- No Docker, MLflow, or DVC. Standard git plus GitHub Actions CI is enough at this stage.

## Data

- Source: nflverse play-by-play and schedule data, accessed via the `nfl_data_py` Python package.
- Scope: 2010-2024 regular season and playoffs.
- Raw data is cached locally as parquet under `data/raw/` and is gitignored (too large for git). The README documents the exact steps to regenerate it from scratch.
- A dedicated coach-mapping step builds a team -> head coach table per game, using game dates and known coaching change dates, so a coach fired or hired mid-season is attributed correctly rather than defaulting to whoever held the job at season start.

## Architecture

```
src/
  data/
    ingest.py       # pulls and caches raw play-by-play and schedule data
    coaches.py       # builds the per-game team -> head coach mapping
  features/
    build_features.py  # filters to real 4th-down decisions, engineers features and labels
  models/
    baseline.py       # XGBoost multiclass model, coach id as categorical feature
    embedding_model.py  # PyTorch model with learned coach embeddings
  analysis/
    tendencies.py     # per-coach aggression scores and tendency comparisons
tests/
  ...                  # unit tests for data/feature layer
notebooks/
  ...                  # exploration and reporting only, no pipeline logic lives here
.github/workflows/
  ci.yml               # lint (ruff) + pytest on every push
```

### Feature engineering

A row is only included as a "4th down decision" if it is a genuine coaching decision:

- Excluded: plays negated by a pre-snap penalty, spikes, kneel-downs, aborted snaps.
- Excluded: clear garbage-time or end-of-half/end-of-game situations where the decision is forced or meaningless as tendency signal (e.g. final snap of a half with no timeouts and no ability to affect the outcome).
- Included features: down, distance, yard line, score differential, time remaining, quarter, timeouts remaining (both teams), home/away, season, week, coach id, team id, and a running count of career decisions coached (see cold start, below).
- Label: the actual decision taken (go for it, punt, field goal attempt).

### Modeling

Two models trained and compared on the same feature table, split by season (earlier seasons for train/validation, most recent seasons held out for test) to avoid leakage and mimic real forecasting:

1. **Baseline**: XGBoost multiclass classifier. Includes coach id as a plain categorical feature. Interpretable, fast to train, easy to explain.
2. **Embedding model**: PyTorch model with a learned per-coach embedding concatenated with situational features, feeding a classifier head. This is the deep-learning component of the project.

The baseline model, trained without needing per-coach history to generalize, doubles as the fallback/prior used for cold start (below).

### Cold start: coaches with no head-coaching track record

Every coach starts their head-coaching career with zero decisions on record, regardless of background (college hire, first-time coordinator promotion, etc). This is handled uniformly:

- The situational baseline model (no coach identity) is the prediction for any coach with no or minimal decision history. This is the natural fallback since it already represents "what does a coach do in this situation" without needing identity.
- A "career decisions coached" counter is included as an explicit feature. For the embedding model, this lets the network learn to discount a thin, undertrained embedding when the count is low. Any never-before-seen coach's embedding is initialized at the mean embedding vector rather than randomly.
- For the analysis layer, per-coach tendency is reported using an empirical-Bayes-style shrinkage estimate (`weight = n / (n + k)` blending personal rate with the league baseline rate), so early-career small samples are shown appropriately regressed toward the mean rather than as noisy extremes.
- This produces a genuine, showable result: the model visibly reverts to baseline for brand-new head coaches and visibly converges to their personal tendency as their sample size grows.

### Testing

- Unit tests cover the data and feature layer: schema correctness of engineered features, correct exclusion of non-decision plays, correct coach-to-game attribution across in-season coaching changes, no time-based leakage between train and test splits.
- Model accuracy is evaluated, not unit-tested; it belongs in the analysis/reporting layer, not pytest.
- GitHub Actions runs ruff (lint) and pytest on every push.

## Success criteria

- `pytest` passes in CI on a clean checkout with cached data present.
- Feature table and both trained models can be reproduced end to end from the documented setup steps.
- Analysis layer produces a per-coach tendency report demonstrating sensible cold-start behavior for at least one real first-time head coach in the 2010-2024 window.
- README reads as a polished portfolio piece: problem, approach, findings, how to reproduce.

## Future work (explicitly out of scope now)

- Phase 2: web app / API serving live predictions and tendency lookups.
- Win-probability-optimal decision model, for comparison against actual coach behavior.
