# NFL 4th Down Predictor — Phase 2 Design: Web App

## Status
Approved. Phase 2 of a two-phase project. Phase 1 (data pipeline, two models, cold-start handling, tendency report) is complete, merged to `master`, and validated against real 2010-2025 nflverse data (see `README.md` for results). This spec covers turning that pipeline into a live, public web app.

## Problem

Phase 1 produces trained models and a coach tendency report, but only as something you run locally from the command line. Phase 2 makes it a real, deployed product: a public site where a visitor can look up any coach's real tendencies, or plug in a game situation and see what the models predict a coach will actually do. This is the piece that turns the project from "a repo with good tests" into something a recruiter can actually click through.

## Goals

- A situation predictor: pick a down/distance/score/time/etc situation and a coach, get the ensemble model's predicted decision, compared against that coach's real historical tendency and the league baseline.
- A coach browser: a sortable list of all coaches with their tendency profile, click into any coach for their full report.
- Deployed publicly at a subdomain of the user's existing personal site (`4thdown.jesse-obrien.com` as a working default; confirm exact subdomain name at deploy time), not just runnable locally.
- Stays current automatically: coaching changes and updated tendencies should show up within about a week of real games happening, with no manual step required.
- SEO-friendly and shareable: proper metadata, fast static pages, good link previews when shared (LinkedIn, etc), since discoverability is part of the portfolio value.

## Non-goals (phase 2)

- No user accounts, no saved history, no personalization. This is a public lookup tool, not an app with state per visitor.
- No win-probability-optimal decision model. Still tendency-only, per the phase 1 spec's original framing.
- No mobile app. Responsive web only.
- No paid hosting commitment. Backend runs on Render's free tier (accepting the cold-start-after-idle tradeoff); frontend on Netlify's free tier.

## Architecture

Same repository, two new top-level directories alongside the existing `src/nfl4th/`:

```
api/
  main.py            # FastAPI app
  models/             # committed trained model artifacts + tendency report
    baseline/
    baseline_no_coach/
    embedding/
    coach_tendency_report.csv
frontend/
  (Astro project)
.github/workflows/
  retrain.yml          # new: scheduled weekly retraining workflow
  ci.yml               # existing: lint + test on push (already in place)
```

`api/` imports `nfl4th` as a library (already installable via `pyproject.toml`). It does not retrain anything at request time or at startup beyond loading the committed model and report files.

`api/models/` is a **tracked** directory, distinct from the existing gitignored top-level `models/` (that one stays as local-dev scratch output from `scripts/run_pipeline.py --model-dir models`, unchanged). The committed copies in `api/models/` are what the deployed API actually serves from.

### Sequencing

Built as three sequential pieces, each verified before the next starts:

1. **Backend API** — extend `pipeline.py` to also persist the tendency report (not just the models), then build and test the FastAPI app against real committed model artifacts.
2. **Frontend + deployment** — Astro site consuming the API, deployed to Netlify on the chosen subdomain; backend deployed to Render.
3. **Automated retraining** — the scheduled GitHub Actions workflow that keeps everything current.

## Backend API

Three endpoints, all read-only (GET/POST, no auth, no user state):

- `GET /coaches` — list of all coaches with summary stats (name, n_decisions, aggression rate, shrinkage_weight). Backs the coach browser list and the predictor's coach dropdown.
- `GET /coaches/{name}` — one coach's full tendency profile: the same fields already produced by `coach_tendency_report()` (observed/expected/shrunk rate per decision class, n_decisions, shrinkage_weight).
- `POST /predict` — request body: down, distance (ydstogo), field position (yardline_100), score_differential, game_seconds_remaining, qtr, posteam_timeouts_remaining, defteam_timeouts_remaining, is_home, coach name. Response: the ensemble model's predicted probability for each decision class, plus that coach's real observed/shrunk rates from the precomputed report and the league baseline rate, so the frontend can show "model predicts X, this coach's real tendency is Y, league average is Z" side by side.

Coach names in `/predict` are constrained to the dropdown populated from `/coaches`, so there is no need to design cold-start UX for a free-typed unknown coach name in phase 2 — every coach offered by the API is by definition one the models have at least baseline-level information about (known coach, or the coach-agnostic fallback already handles anyone not in the committed vocab, consistent with phase 1's existing cold-start behavior).

### Data the API needs at startup

- The three model artifact directories under `api/models/` (`baseline/`, `baseline_no_coach/`, `embedding/`), loaded via the existing `load_baseline`/`load_embedding_model` functions.
- `api/models/coach_tendency_report.csv` — a new artifact. `pipeline.py`'s `run()` already builds this DataFrame and returns it; it needs to also be saved to disk when `model_dir` is given, alongside the models, so the API can load it directly without recomputing it (recomputing it would require the full feature table, which requires a live nflverse pull — exactly what the deployed API must not do).

No database. The report CSV and the model files are the entire data layer.

## Frontend

Astro, not a plain client-rendered SPA. Reasoning: this is a portfolio piece meant to be discoverable and shareable, and a pure React/Vite SPA renders its content only after JS executes, which is worse for search indexing and for link-preview bots (LinkedIn, Twitter/X) that often don't execute JS at all. Astro ships static HTML by default with proper `<title>`/meta/Open Graph tags and a sitemap, while still allowing interactive "islands" (a React or vanilla component) for the parts that need real interactivity.

Two pages:

- **Predictor** (`/`) — a form (situation inputs + coach dropdown) that calls `POST /predict` and renders the result: predicted decision breakdown, the selected coach's real tendency, and the league baseline, likely as a simple bar/probability comparison.
- **Coach browser** (`/coaches`) — a sortable table backed by `GET /coaches` (sort by aggression rate, sample size, etc), each row linking to a per-coach detail view backed by `GET /coaches/{name}`.

Chart/visualization choices are left to the implementation plan; nothing here mandates a specific charting library.

## Deployment

- **Backend**: Render, free web service tier. Loads `api/models/` at startup. Accepts the free tier's cold-start-after-15-minutes-idle tradeoff; can move to a paid always-on tier later if that matters more than the small cost.
- **Frontend**: Netlify, connected to the user's existing Netlify account, deployed at a subdomain of `jesse-obrien.com` (default: `4thdown.jesse-obrien.com`). Fully independent build/deploy from the user's existing personal site — no changes to that site's repo, just a DNS record for the subdomain.
- Both auto-deploy on push to `master` (standard Render/Netlify behavior), which is what makes the weekly retraining workflow's commits automatically go live with no extra "trigger deploy" step.

## Automated freshness

A new scheduled GitHub Actions workflow (`.github/workflows/retrain.yml`), running weekly (Tuesdays, after Monday Night Football has been played and nflverse data for the week is finalized):

1. Checkout, set up Python 3.11, install the project.
2. Run the pipeline against all available seasons as training data (no held-out val/test split needed for this production refresh — that evaluation already happened once and is documented in the README; this run's only job is to produce the freshest possible models and report).
3. Sanity check the result before committing anything: assert the retrained baseline model's accuracy on a small fixed holdout doesn't collapse (e.g. stays above some reasonable floor). This exists to catch a broken retrain (bad data pull, a code regression) before it goes live, not to re-litigate phase 1's validation.
4. If the sanity check passes, overwrite `api/models/` (models + `coach_tendency_report.csv`) and commit and push to `master` with a plain, human-sounding commit message, no different from any other commit in this repo's history. If it fails, the workflow fails loudly and nothing gets committed, leaving the previous week's known-good artifacts live.

This workflow authors commits the same way any commit in this repo does: plain language, no em dashes, no AI/Claude mentions, consistent with every other constraint already established for this project. It runs under a repo-scoped `GITHUB_TOKEN` or equivalent, not the user's personal account, since it is genuinely automated (this differs from the project's general "no push without explicit confirmation" rule for interactive work; a user-approved, spec'd, scheduled CI job is the documented exception, not an ad hoc push).

## Testing

- **Backend**: FastAPI's `TestClient`, hitting all three endpoints against the real, small, committed model artifacts (not mocks) — matching phase 1's established practice of testing against real behavior wherever the cost of doing so is low, which it is here (models are ~3MB and load fast).
- **Retraining workflow**: the in-workflow sanity check described above is the test; no separate pytest suite for the workflow itself beyond that.
- **Frontend**: kept light and pragmatic given project scope — no mandated test framework here; manual verification in a browser is acceptable, consistent with how much testing weight a portfolio-scale frontend actually needs.

## Success criteria

- The predictor and coach browser pages work end to end against the live deployed API, not just locally.
- A cold visit to `4thdown.jesse-obrien.com` (or whatever subdomain is chosen) loads correctly, is mobile-responsive, and has correct page titles/meta tags.
- The retraining workflow runs successfully at least once on its schedule and produces a real commit with updated artifacts.
- Sharing the URL on LinkedIn (or similar) produces a proper link preview (title, description, not a blank/broken card).

## Future work (explicitly out of scope now)

- User accounts, saved comparisons, or personalized views.
- A win-probability-optimal decision model shown alongside the tendency prediction.
- Historical charts of how a coach's aggression has trended over their career.
- Moving off Render's free tier if the cold-start delay becomes a real problem.
