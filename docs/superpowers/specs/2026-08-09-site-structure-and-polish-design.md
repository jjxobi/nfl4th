# Site Structure and Polish Design

## Status

Approved. First of two sub-projects split out from a broader feature request (a dedicated home/about split, a findings page, a current-coaches filter, and a friendlier time input). This spec covers everything except the findings page, which needs its own backend aggregation work and gets its own spec next.

## Problem

The deployed app currently has two pages: `/` (the situation predictor) and `/coaches` (the tendency browser). `/` is doing double duty as both the site's front door and its only interactive feature, so there's nowhere to explain what the project actually is or why the numbers should be trusted. The coach browser also lists every coach who appears anywhere in the 2010-2025 data, which means browsing it mixes current NFL head coaches with people who haven't held the job in over a decade. And the predictor's time-remaining field asks for raw seconds, which nobody thinks in.

## Goals

- A real home page at `/` that orients a first-time visitor and links to the rest of the site.
- A dedicated about page explaining the project, the coach-tendency framing, and the data source.
- A way to see only currently active NFL head coaches on the coach browser, on by default.
- A predictor form that takes time remaining as minutes and seconds instead of raw seconds.
- Every new page matches the existing dark stadium theme and passes the same "doesn't look AI-generated" bar already enforced on the existing pages (real design tokens, no default fonts, no layout-thrashing animations, no side-tab borders) — see the project's design conventions already established in `Layout.astro`.

## Non-goals (this spec)

- The findings page (superlative leaderboards, situational splits, league-wide trends over time) — separate spec, needs new backend aggregation the current report doesn't compute.
- Any change to the prediction model, the ensemble logic, or the existing `/predict`, `/coaches`, `/coaches/{name}` response shapes beyond the one addition described below.
- Actual deployment to Render/Netlify, or updating the placeholder subdomain in `astro.config.mjs`.

## Architecture

### Page structure

Four routes after this ships:

- `/` — new home page. A short pitch (what this predicts and why it's tendency-based, not win-probability-optimal), and three cards linking to Predict, Coaches, and About. Static content, no API calls.
- `/predict` — the existing predictor form, moved from `/` with no logic changes beyond the time-input change below.
- `/coaches` — the existing coach browser, unchanged except for the new filter.
- `/about` — new. Project story, methodology, data source (nflverse, 2010-2025), a link to the GitHub repo. Static content, no API calls, adapted from the README's framing but written for a site visitor rather than someone reading source.

`Layout.astro`'s nav (`frontend/src/layouts/Layout.astro:43-46`) changes from two links (`Predictor`, `Coaches`) to four (`Home`, `Predict`, `Coaches`, `About`), matching the new routes. The existing `isActive()` helper and `aria-current` pattern extend to all four without changes to the helper itself.

### Current head coach filter

The existing coach tendency report (`src/nfl4th/analysis/tendencies.py`'s `coach_tendency_report()`) groups training decisions by coach and never records which seasons each coach appears in — that information exists in the input `decisions` DataFrame (it has a `season` column) but is discarded during the groupby. One new field closes this gap:

```python
row = {"coach": coach, "n_decisions": n, "shrinkage_weight": weight, "last_season": int(group["season"].max())}
```

This flows through the existing pipeline unchanged: `save_run_artifacts` already writes the full report to `coach_tendency_report.csv`, so `last_season` is just a new column that rides along. `api/loader.py` already loads this CSV into a DataFrame indexed by coach; `api/schemas.py`'s `CoachProfile` gets a new `last_season: int` field, and `api/routes_coaches.py`'s `_coach_profile_from_row` helper picks it up from the row like every other field already there.

The frontend doesn't need a separate "is current" boolean from the backend — `GET /` already returns `latest_season` (`api/main.py`'s health endpoint), and `frontend/src/lib/api.ts` already has a typed client for it. `coaches.astro` treats a coach as current if `coach.last_season === latestSeason` fetched from that same endpoint, so there's a single source of truth and nothing to keep in sync by hand. The filter is a checkbox/toggle above the table, checked by default, filtering the already-fetched coach list client-side (no new API call, no new endpoint).

### Time input as minutes and seconds

`index.astro`'s (moving to `predict.astro`) situation form currently has a single numeric field bound to `game_seconds_remaining`. This becomes two fields, minutes and seconds, combined client-side (`minutes * 60 + seconds`) into the same `game_seconds_remaining` value before it's sent to `POST /predict`. No backend change: `api/schemas.py`'s existing `Field(ge=0, le=3600)` bound on `game_seconds_remaining` still validates the combined value exactly as it does today. Minutes is bounded 0-59 and seconds 0-59 in the HTML, so the combined value tops out at 3599, comfortably within the backend's bound. This deliberately makes the single instant of "exactly 3600 seconds remaining" (kickoff, before any time has elapsed) unreachable through the two-field input; a fourth-down decision cannot occur before a single play has been run, so that instant was never a real input anyway.

## Testing

- **Backend:** a new test in `tests/analysis/test_tendencies.py` asserting `coach_tendency_report()` returns the correct `last_season` per coach against synthetic multi-season data (a coach who appears in seasons 2020 and 2022 should report `last_season == 2022`). A new or extended test in `tests/api/test_routes_coaches.py` asserting `GET /coaches` and `GET /coaches/{name}` include `last_season` in their response.
- **Frontend:** kept light and pragmatic, consistent with the rest of this project — no new test framework, manual verification in a browser (all four pages render, the filter toggles correctly against real data, the minutes/seconds inputs produce the same prediction as an equivalent raw-seconds value would have).

## Success criteria

- Visiting `/` explains the project and links to Predict, Coaches, and About without needing to read the README.
- The coach browser, by default, shows only coaches who coached in the most recent season of data — verifiable against the real committed `api/models/coach_tendency_report.csv` once regenerated with the `last_season` field.
- Entering "2 min 15 sec" on the predictor produces an identical prediction to manually computing and entering `135` seconds today.
- No em dashes anywhere in any new page's copy. No default/overused fonts, layout-thrashing CSS, or side-tab borders on any new page — same bar the existing pages were already held to.

## Future work (explicitly out of scope now)

- The findings page (next spec): superlative leaderboards from the existing report, league-wide trend over time (needs a new season-level aggregation), and per-coach-within-situation-bucket leaderboards (needs a new per-coach-per-bucket aggregation) — both require new backend work beyond this spec's single `last_season` addition.
