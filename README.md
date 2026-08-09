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
   home/away, and career decisions coached so far). Career decisions is
   counted from the start of the loaded data window (2010), not from a
   coach's actual first game as a head coach, so a coach whose tenure
   started earlier will show an artificially low count in their early
   loaded seasons.
3. **Modeling**: trains two models side by side on a time-based split
   (train on earlier seasons, test on the most recent ones, no leakage):
   - An XGBoost baseline with coach as a categorical feature.
   - A PyTorch model with a learned per-coach embedding, its situational
     features normalized (raw scale varies wildly, season is ~2010-2024,
     game seconds remaining is 0-3600) and trained in shuffled mini
     batches rather than one full batch step per epoch.
   - Averaging the two models' predicted probabilities (a simple ensemble,
     no extra training) beats either model alone, see Results below.
4. **Cold start**: a coach's first 4th down decision as a head coach comes
   with zero track record, regardless of background. The two models handle
   this differently. The XGBoost model treats an unseen coach as a missing
   categorical value and routes the prediction through whatever default
   split direction it learned during training. The embedding model has no
   learned vector for an unseen coach, so it substitutes the mean of every
   trained coach's embedding in its place. Neither of these is shrinkage;
   shrinkage is a separate step described next.
5. **Analysis**: produces a per-coach tendency report comparing each
   coach's observed decision rate against a coach-agnostic situational
   baseline rate, blended by sample size using empirical Bayes shrinkage
   (`weight = n / (n + k)`), so a coach's early decisions are shown
   appropriately regressed toward the baseline rather than as noisy
   extremes.

## Results

Validated against real nflverse data, not just synthetic test fixtures:
trained on 2010-2023, held out the entire 2025 season as a test set the
models never saw.

| Model | Test accuracy | Test log loss |
|---|---|---|
| XGBoost baseline | 88.3% | 0.295 |
| PyTorch embedding | 87.6% | 0.301 |
| Ensemble (average of both) | 88.9% | 0.271 |

For context, always guessing the most common decision (usually punt, roughly
55-65% of situations) would only get accuracy into the 50s or 60s, so both
models are learning real situational signal, not just the majority class.

The cold start story shows up in the tendency report as designed. A coach
with only 15 decisions on record gets shrunk hard toward the situational
baseline (`shrinkage_weight` around 0.6); a coach with 1,500+ decisions
barely moves off their own observed rate (`shrinkage_weight` above 0.99).

Two things were tested and explicitly rejected along the way: adding the
pregame point spread as a feature made no real difference (in-game context
like score and time already dominates the decision), and the embedding
model was initially badly undertrained on real data (46% accuracy, worse
than random guessing) until its features were normalized, which is why
that normalization step exists.

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
a coach tendency report. Pass `--model-dir models` to also save both trained
models to disk (`baseline/`, `baseline_no_coach/`, `embedding/`) so a later
run, or a future web app, can load them instead of retraining from scratch.

To reproduce the Results numbers above (train 2010-2023, test on 2025):

```bash
.venv/Scripts/python scripts/run_pipeline.py --train-start 2010 --train-end 2024 --val-end 2025 --test-end 2026
```

## Backend API

The trained models are also served behind a small FastAPI backend, so a web
app can get predictions without retraining anything. The committed serving
artifacts in `api/models/` were produced by running the full pipeline with
`--model-dir api/models` against the complete dataset.

Run it locally:

```bash
.venv/Scripts/python -m pip install -e ".[api]"
.venv/Scripts/python scripts/run_api.py
```

Endpoints:
- `GET /` - health check
- `GET /coaches` - every coach's tendency profile
- `GET /coaches/{name}` - one coach's profile
- `POST /predict` - given a situation and a coach, returns the model's
  predicted decision, that coach's career average, and a coach-agnostic
  league baseline

Interactive docs are available at `/docs` once the server is running.

## Frontend

A static Astro site in `frontend/` consumes the backend API: a situation
predictor and a coach tendency browser. Requires Node >= 22.12.

```bash
cd frontend
npm install
cp .env.example .env
npm run dev
```

`frontend/.env` needs `PUBLIC_API_URL` pointing at a running backend (see
`frontend/.env.example`).

## Deployment

Backend deploys to [Render](https://render.com) as a free-tier Python web
service, using the `render.yaml` blueprint at the repo root. Frontend
deploys to [Netlify](https://netlify.com), using `frontend/netlify.toml`,
with `frontend` set as the site's base directory.

To actually deploy:
1. Push this repo to GitHub.
2. On Render, create a new Blueprint from the repo; it reads `render.yaml`
   automatically. Set the `ALLOWED_ORIGINS` environment variable to the
   frontend's real URL once you know it.
3. On Netlify, create a new site from the repo, with `frontend` as the base
   directory. Set `PUBLIC_API_URL` in Netlify's environment variables to the
   Render service's URL. The first deploy happens before this variable
   exists, so trigger a redeploy afterward to pick it up.
4. Point a subdomain (e.g. `4thdown.yourdomain.com`) at the Netlify site via
   a DNS CNAME record, then add it as a custom domain in Netlify's site
   settings.

`frontend/astro.config.mjs` currently sets `site` to a placeholder
(`https://4thdown.example.com`) so Open Graph tags resolve to absolute URLs.
Update it to the real subdomain once DNS is set up.

Model freshness is automatic: `.github/workflows/retrain.yml` runs every
Tuesday, retrains on all available seasons, and only commits the refreshed
`api/models/` if a sanity check against a fixed historical holdout still
clears a reasonable accuracy floor. A passing run's commit lands on `master`
and both Render and Netlify redeploy on push, so no manual step is needed to
keep coach tendencies current through the season.

## What's next

Phase 2 is complete: backend API, frontend, deployment, and automated weekly
retraining are all built and described above. Nothing further is currently
planned. Deliberately out of scope for now: user accounts or saved
comparisons, a win-probability-optimal decision model shown alongside the
tendency prediction, and historical charts of how a coach's aggression has
trended over their career.
