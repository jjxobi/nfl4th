# Findings Page and Conversion Model Design

## Status

Approved. Combines two previously separate ideas into one spec, since they overlap: a "findings" page deferred from the phase 2 design (superlative leaderboards, situational splits, league-wide trends over time), and a new conversion-probability model, which is itself one of the findings and also belongs on the live predictor.

## Problem

The site currently answers "what will this coach do" (the predictor) and "what has this coach actually done" (the coach browser), but nothing on the site surfaces the more interesting patterns hiding in the data: how aggressive is the league as a whole, has that changed over time, which coaches are outliers in specific situations, and — a question the predictor can't currently answer at all — if a coach does go for it, how likely is the play to actually work. The tendency models predict a decision; nothing predicts an outcome.

## Goals

- A `/findings` page: superlative leaderboards, situational splits (league-wide and per-coach), and a league-wide aggressiveness trend from 2010 to the present.
- A new conversion-probability model: given a 4th down situation, what's the chance a go-for-it attempt succeeds. Situation-only, not coach-specific.
- The conversion model's prediction shows up in two places from one source of truth: a new field on the predictor's result, and a conversion-by-distance chart on the findings page.
- The conversion model joins the existing weekly automated retrain, so it stays current the same way the tendency models do.

## Non-goals

- No change to the existing tendency models, their ensemble, or the existing `/predict` fields (`predicted`, `coach_career_average`, `league_baseline`) beyond adding one new field.
- No coach-specific conversion modeling. Conversion probability answers "how hard is this situation," not "how good is this coach's offense" — conflating the two would need per-coach sample sizes the data doesn't reliably have and would muddy the project's coach-tendency framing with a different question (execution quality, not decision-making).
- No new frontend charting library. Trend and split visuals are hand-built (SVG/CSS), consistent with the predictor's existing hand-built probability meters, not a generic chart-library look.

## Architecture

### Backend: new aggregation module

`src/nfl4th/analysis/findings.py`, computed once per pipeline run (not per API request, matching every other piece of served data in this project):

- **Situational splits** — league-wide go-for-it rate bucketed by yards to go: `1`, `2`, `3`, `4-6`, `7-10`, `11+`. Six buckets, computed from the same `train_df` the tendency report already uses.
- **Per-coach bucket leaderboards** — within each of the six buckets above, each coach's go-for-it rate, restricted to coaches with at least 10 attempts in that specific bucket (matching the `k=10` shrinkage constant already used elsewhere in this project, for consistency — this is a minimum-sample floor to keep the leaderboard off the noise, not shrinkage itself, since a leaderboard is showing raw rates, not a blended estimate).
- **League trend over time** — go-for-it rate per season, 2010 through the latest available season.

Superlative leaderboards (most/least aggressive coach overall, biggest gap between observed and situational-baseline rate) need no new backend code — they're derivable client-side from the coach list `GET /coaches` already returns in full.

### New conversion model

`src/nfl4th/models/conversion.py`, following the same shape as `src/nfl4th/models/baseline.py`'s `train_baseline`/`predict_baseline`/`save_baseline`/`load_baseline` functions: a single XGBoost classifier (no coach feature, no embedding counterpart, no ensemble — there's nothing to ensemble with since this is answering a different question than the tendency models). Training data is `filter_fourth_down_decisions`'s existing output (already excludes penalty-negated plays) narrowed to just the `go_for_it` decision class, joined against the raw play-by-play's `fourth_down_converted` column (confirmed present and clean on real cached data: no missing values, a real binary outcome for every real go-for-it attempt) as the label. Features: the full situational set (`ydstogo`, `yardline_100`, `score_differential`, `game_seconds_remaining`, `qtr`, `posteam_timeouts_remaining`, `defteam_timeouts_remaining`, `is_home` — the same set the coach-agnostic baseline model already uses, minus `career_decisions` and `season`, neither of which describes play difficulty).

### Persisted artifacts and API

`api/models/conversion/` (new model directory, same save/load pattern as the existing `baseline`/`baseline_no_coach`/`embedding` directories) and `api/models/findings.json` (the precomputed splits, leaderboards, trend, and a conversion-probability grid across the six distance buckets — computed once at pipeline run time, served statically).

Two API changes:
- `POST /predict`'s response gains `conversion_probability: float`, always present regardless of which decision the model predicts (a coach predicted to punt might still be a useful "what if they went for it" data point).
- A new `GET /findings` endpoint serves `findings.json`'s contents directly, following the same zero-live-computation pattern as `/coaches`.

### Frontend

- New `frontend/src/pages/findings.astro`. Sections: superlative leaderboards (small stat cards), situational splits (a bar-style visual per distance bucket, matching the existing meter component's visual language, plus the per-bucket coach leaderboard), league trend over time (a hand-built SVG line, one point per season), and conversion rate by distance (same bar-style treatment as the splits section, sourced from the conversion grid in `findings.json`).
- `Layout.astro`'s nav gains a fifth link: Findings.
- `frontend/src/pages/predict.astro` gains one more result card next to the existing three, showing `conversion_probability`, labeled clearly as "if they go for it" so it doesn't read as conditioned on the predicted decision actually being go-for-it.

### Automated retraining

`scripts/retrain.py` (already runs weekly, already gated by a sanity check before committing) additionally trains the conversion model and regenerates `findings.json` in the same run, using data it has already loaded — no new fetch, no new schedule, same commit-or-nothing guarantee the existing sanity check already provides.

## Testing

- **Backend**: unit tests for each `findings.py` function against synthetic multi-coach, multi-season, multi-bucket data (matching the existing project convention in `tests/analysis/test_tendencies.py`), and for the conversion model's train/predict/save/load functions against synthetic labeled attempts (matching `tests/models/test_baseline.py`'s existing style). `tests/api/test_main.py`-style `TestClient` tests against the real committed artifacts for `GET /findings` and the new `conversion_probability` field on `POST /predict`.
- **Frontend**: manual browser verification, consistent with the rest of this project — no new test framework.

## Success criteria

- `/findings` shows real, correct numbers computed from the actual committed data, not placeholders.
- The predictor's conversion probability and the findings page's conversion-by-distance chart agree with each other for the same distance, since both come from the same model.
- The weekly retrain workflow's next real run trains and commits the conversion model and `findings.json` alongside the existing artifacts, with no separate schedule or manual step.

## Future work (explicitly out of scope now)

- Coach-specific conversion modeling, if a future need for it is clearly established.
- A findings item combining tendency and conversion (e.g. "coaches who go for it more than the model says they should, and whether it pays off") — a genuinely interesting follow-up, but a distinct analysis from what's scoped here.
