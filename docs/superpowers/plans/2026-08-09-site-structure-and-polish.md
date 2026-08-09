# Site Structure and Polish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure the site into four pages (home, predict, coaches, about), add a "current head coaches only" filter to the coach browser, and let the predictor accept time remaining as minutes and seconds instead of raw seconds.

**Architecture:** Two independent stages. Stage 1: a small backend addition (`last_season` per coach, threaded through the report, the API, and the real committed model artifacts) runs in parallel with the frontend page restructure (new home page, `/predict` route, new `/about` page, updated nav), since neither touches the other's files. Stage 2: the coach browser's filter (needs stage 1's `last_season` field) runs in parallel with the predictor's minutes/seconds input (needs stage 1's page restructure to have created `predict.astro` first).

**Tech Stack:** Python (pandas, FastAPI, Pydantic), Astro, TypeScript.

## Global Constraints

- No em dashes anywhere in any UI copy, code, or commit message.
- No mention of AI/Claude/Anthropic anywhere in committed content.
- No "Co-Authored-By" trailers in any commit.
- Commit messages must be plain, human-sounding.
- No code comments unless they explain a genuinely non-obvious WHY, never a WHAT.
- New pages must use `Layout.astro`'s existing design tokens (`--color-bg`, `--color-surface`, `--color-surface-raised`, `--color-line`, `--color-text`, `--color-text-dim`, `--color-accent`, `--color-accent-2`, `--font-display`, `--font-mono`, `--font-body`, `--space-2` through `--space-6`) rather than introducing new colors, fonts, or spacing values.
- Backend testing convention already established in this project: `tests/api/*` hits the real, small, committed `api/models/` artifacts through `TestClient(app)`, not mocks.
- The `game_seconds_remaining` field's existing backend bound (`api/schemas.py`'s `Field(ge=0, le=3600)`) does not change. The minutes/seconds inputs are bounded 0-59 each in the HTML, so their combined value tops out at 3599, safely under 3600 without needing any backend change.

---

## Task 1: Add `last_season` to the coach tendency report and API

**Files:**
- Modify: `src/nfl4th/analysis/tendencies.py`
- Modify: `tests/analysis/test_tendencies.py`
- Modify: `api/schemas.py`
- Modify: `api/routes_coaches.py`
- Modify: `tests/api/test_routes_coaches.py`
- Regenerate: `api/models/coach_tendency_report.csv` and the other files in `api/models/` (real committed serving artifacts, not source code)

**Interfaces:**
- Consumes: nothing new (uses `coach_tendency_report`'s existing `decisions` DataFrame, which already has a `season` column from the real feature table).
- Produces: `coach_tendency_report()`'s returned DataFrame gains a `last_season` int column. `CoachProfile` (Pydantic, in `api/schemas.py`) gains a `last_season: int` field, so `GET /coaches` and `GET /coaches/{name}` both return `last_season` in every coach's JSON. Task 2 consumes this exact field name from those two endpoints.

- [ ] **Step 1: Update the existing report tests to include a `season` column**

The two existing tests in `tests/analysis/test_tendencies.py` build synthetic `decisions` DataFrames without a `season` column. The change in Step 3 makes `coach_tendency_report()` read `group["season"]`, so these two tests will raise `KeyError: 'season'` unless updated first. Replace the full contents of `tests/analysis/test_tendencies.py` with:

```python
import pandas as pd

from nfl4th.analysis.tendencies import coach_tendency_report, shrinkage_weight


def test_shrinkage_weight_boundaries():
    assert shrinkage_weight(0, k=10) == 0.0
    assert shrinkage_weight(10, k=10) == 0.5
    assert shrinkage_weight(10_000, k=10) > 0.99


def test_report_blends_toward_baseline_for_small_sample():
    decisions = pd.DataFrame({"coach": ["Rookie"], "decision": ["go_for_it"], "season": [2024]})
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
    decisions = pd.DataFrame(
        {"coach": ["Veteran"] * n, "decision": ["go_for_it"] * n, "season": [2024] * n}
    )
    baseline_probs = pd.DataFrame(
        {"punt": [0.7] * n, "field_goal": [0.2] * n, "go_for_it": [0.1] * n}, index=decisions.index
    )

    report = coach_tendency_report(decisions, baseline_probs, k=10.0)

    row = report.iloc[0]
    assert row["go_for_it_shrunk"] > 0.95


def test_report_records_the_coachs_most_recent_season():
    decisions = pd.DataFrame(
        {
            "coach": ["Multi Year", "Multi Year", "Multi Year"],
            "decision": ["go_for_it", "punt", "field_goal"],
            "season": [2020, 2022, 2021],
        }
    )
    baseline_probs = pd.DataFrame(
        {"punt": [0.7, 0.7, 0.7], "field_goal": [0.2, 0.2, 0.2], "go_for_it": [0.1, 0.1, 0.1]},
        index=decisions.index,
    )

    report = coach_tendency_report(decisions, baseline_probs, k=10.0)

    assert report.iloc[0]["last_season"] == 2022
```

The seasons in the last test are deliberately out of order (2020, 2022, 2021) so the test actually proves `last_season` is the maximum, not just whichever row happens to be first or last.

- [ ] **Step 2: Run the tests to verify the new one fails**

Run: `pytest tests/analysis/test_tendencies.py -v`
Expected: `test_report_records_the_coachs_most_recent_season` FAILS with `KeyError: 'last_season'`. The other three should already pass (they only needed the `season` column added, which doesn't change their existing assertions).

- [ ] **Step 3: Add `last_season` to `coach_tendency_report`**

In `src/nfl4th/analysis/tendencies.py`, change:

```python
        row = {"coach": coach, "n_decisions": n, "shrinkage_weight": weight}
```

to:

```python
        row = {
            "coach": coach,
            "n_decisions": n,
            "shrinkage_weight": weight,
            "last_season": int(group["season"].max()),
        }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/analysis/test_tendencies.py -v`
Expected: PASS, all 4 tests.

- [ ] **Step 5: Commit**

```bash
git add src/nfl4th/analysis/tendencies.py tests/analysis/test_tendencies.py
git commit -m "Track each coach's most recent season in the tendency report"
```

- [ ] **Step 6: Add `last_season` to the API schema**

In `api/schemas.py`, change `CoachProfile` from:

```python
class CoachProfile(BaseModel):
    coach: str
    n_decisions: int
    shrinkage_weight: float
    punt: DecisionRates
    field_goal: DecisionRates
    go_for_it: DecisionRates
```

to:

```python
class CoachProfile(BaseModel):
    coach: str
    n_decisions: int
    shrinkage_weight: float
    last_season: int
    punt: DecisionRates
    field_goal: DecisionRates
    go_for_it: DecisionRates
```

- [ ] **Step 7: Wire `last_season` into the coach routes**

In `api/routes_coaches.py`, change `_coach_profile_from_row` from:

```python
def _coach_profile_from_row(coach: str, row: pd.Series) -> CoachProfile:
    return CoachProfile(
        coach=coach,
        n_decisions=int(row["n_decisions"]),
        shrinkage_weight=float(row["shrinkage_weight"]),
        punt=DecisionRates(
```

to:

```python
def _coach_profile_from_row(coach: str, row: pd.Series) -> CoachProfile:
    return CoachProfile(
        coach=coach,
        n_decisions=int(row["n_decisions"]),
        shrinkage_weight=float(row["shrinkage_weight"]),
        last_season=int(row["last_season"]),
        punt=DecisionRates(
```

(Everything else in the function, and the whole rest of the file, stays unchanged.)

- [ ] **Step 8: Update the API tests**

In `tests/api/test_routes_coaches.py`, add one assertion to each existing test. The full file becomes:

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
    assert isinstance(first["last_season"], int)


def test_get_coach_returns_full_profile_for_a_known_coach():
    coach_name = client.get("/coaches").json()[0]["coach"]

    response = client.get(f"/coaches/{coach_name}")

    assert response.status_code == 200
    body = response.json()
    assert body["coach"] == coach_name
    assert "observed" in body["punt"]
    assert "expected" in body["punt"]
    assert "shrunk" in body["punt"]
    assert isinstance(body["last_season"], int)


def test_get_coach_returns_404_for_unknown_coach():
    response = client.get("/coaches/Definitely Not A Real Coach")

    assert response.status_code == 404
```

- [ ] **Step 9: Regenerate the real committed model artifacts**

`tests/api/test_routes_coaches.py` hits the real, committed `api/models/coach_tendency_report.csv` through `TestClient(app)`, not a mock. That file does not have a `last_season` column yet (it was generated before this task), so `_coach_profile_from_row`'s `row["last_season"]` will raise `KeyError` against the currently-committed data even though the code above is correct. Regenerate the real artifacts with the same command used to produce the currently-committed ones:

```bash
.venv/Scripts/python scripts/run_pipeline.py --train-start 2010 --train-end 2026 --val-end 2026 --test-end 2026 --model-dir api/models
```

This retrains on all available seasons with no held-out split (matching how the currently-committed artifacts were made) and overwrites every file in `api/models/`, including a `coach_tendency_report.csv` that now has the `last_season` column. This is a real training run against real cached data and will take real time and memory; if it fails with a memory error, retry once, and if it fails again, stop and report rather than guessing at a workaround.

After it completes, check `git diff --stat api/models` to confirm the report file changed and skim `coach_tendency_report.csv`'s new `last_season` column for plausible values (should be years like 2024 or 2025, not garbage).

- [ ] **Step 10: Run the full test suite**

Run: `pytest -q` and `ruff check .`
Expected: all tests pass (including the two updated in Step 8, now running against the regenerated real artifacts from Step 9), ruff clean.

- [ ] **Step 11: Commit**

```bash
git add api/schemas.py api/routes_coaches.py tests/api/test_routes_coaches.py api/models
git commit -m "Expose each coach's most recent season through the API"
```

---

## Task 2: Add the current head coaches filter to the coach browser

**Files:**
- Modify: `frontend/src/lib/api.ts`
- Modify: `frontend/src/pages/coaches.astro`

**Interfaces:**
- Consumes: `last_season: int` on `CoachProfile` and `latest_season: int` from `GET /` (Task 1). The health endpoint's exact response shape, already live and unchanged by this plan: `{"name": "nfl4th API", "latest_season": <int>, "n_coaches": <int>}` (`api/main.py`'s `health()` function).
- Produces: nothing further depends on this task.

- [ ] **Step 1: Add `last_season` to the frontend's `CoachProfile` type and add a health client function**

In `frontend/src/lib/api.ts`, change the `CoachProfile` interface from:

```typescript
export interface CoachProfile {
  coach: string;
  n_decisions: number;
  shrinkage_weight: number;
  punt: DecisionRates;
  field_goal: DecisionRates;
  go_for_it: DecisionRates;
}
```

to:

```typescript
export interface CoachProfile {
  coach: string;
  n_decisions: number;
  shrinkage_weight: number;
  last_season: number;
  punt: DecisionRates;
  field_goal: DecisionRates;
  go_for_it: DecisionRates;
}
```

Then add a new interface and function, placed after the existing `getCoach` function and before `predict`:

```typescript
export interface HealthInfo {
  name: string;
  latest_season: number;
  n_coaches: number;
}

export async function getHealth(): Promise<HealthInfo> {
  const response = await fetch(`${API_URL}/`);
  if (!response.ok) throw new Error(`Failed to load health info: ${response.status}`);
  return response.json();
}
```

- [ ] **Step 2: Add the filter checkbox to the page markup**

In `frontend/src/pages/coaches.astro`, add a filter toggle inside the `.coach-page-head` div, right after the existing `<p class="coach-summary" ...>` line:

```astro
  <div class="coach-page-head">
    <h1>Coach Tendencies</h1>
    <p class="coach-lede">Click a column to sort. Click a coach to see their full profile.</p>
    <p class="coach-summary" id="coach-summary" aria-live="polite">Loading coach data&hellip;</p>
    <label class="filter-toggle">
      <input type="checkbox" id="current-only" checked />
      <span>Current head coaches only</span>
    </label>
  </div>
```

- [ ] **Step 3: Wire the filter into the script**

In `frontend/src/pages/coaches.astro`'s `<script>` block, change the import line from:

```typescript
  import { getCoaches, type CoachProfile, type DecisionRates } from "../lib/api";
```

to:

```typescript
  import { getCoaches, getHealth, type CoachProfile, type DecisionRates } from "../lib/api";
```

Add two new module-level variables right after the existing `let sortDescending = true;` line:

```typescript
  let latestSeason: number | null = null;
  let currentOnly = true;
```

Add a new function right before `sortedCoaches()`:

```typescript
  function visibleCoaches(): CoachProfile[] {
    if (currentOnly && latestSeason !== null) {
      return coaches.filter((coach) => coach.last_season === latestSeason);
    }
    return coaches;
  }
```

Change `sortedCoaches()` from:

```typescript
  function sortedCoaches(): CoachProfile[] {
    return [...coaches].sort((a, b) => {
```

to:

```typescript
  function sortedCoaches(): CoachProfile[] {
    return [...visibleCoaches()].sort((a, b) => {
```

Change `updateSummary()` from:

```typescript
  function updateSummary() {
    const summary = document.getElementById("coach-summary")!;
    const direction = sortDescending ? "highest to lowest" : "lowest to highest";
    summary.textContent = `${coaches.length} coaches on file, sorted by ${columnLabels[sortKey]}, ${direction}.`;
  }
```

to:

```typescript
  function updateSummary() {
    const summary = document.getElementById("coach-summary")!;
    const direction = sortDescending ? "highest to lowest" : "lowest to highest";
    const count = visibleCoaches().length;
    summary.textContent = `${count} coaches on file, sorted by ${columnLabels[sortKey]}, ${direction}.`;
  }
```

Add a listener for the new checkbox, right after the existing `updateSortIndicators();` call that precedes the `getCoaches()` call at the bottom of the script:

```typescript
  document.getElementById("current-only")!.addEventListener("change", (event) => {
    currentOnly = (event.target as HTMLInputElement).checked;
    render();
  });
```

Finally, change the data-loading block at the bottom of the script from:

```typescript
  getCoaches()
    .then((data) => {
      coaches = data;
      render();
    })
    .catch((error) => {
      const body = document.getElementById("coach-table-body")!;
      body.innerHTML = `<tr class="table-message-row"><td colspan="5">Could not load coaches: ${error instanceof Error ? error.message : "unknown error"}</td></tr>`;
      const summary = document.getElementById("coach-summary")!;
      summary.textContent = "Coach data is unavailable right now.";
    });
```

to:

```typescript
  Promise.all([getCoaches(), getHealth()])
    .then(([coachData, health]) => {
      coaches = coachData;
      latestSeason = health.latest_season;
      render();
    })
    .catch((error) => {
      const body = document.getElementById("coach-table-body")!;
      body.innerHTML = `<tr class="table-message-row"><td colspan="5">Could not load coaches: ${error instanceof Error ? error.message : "unknown error"}</td></tr>`;
      const summary = document.getElementById("coach-summary")!;
      summary.textContent = "Coach data is unavailable right now.";
    });
```

- [ ] **Step 4: Style the filter toggle**

Add to the `<style is:global>` block in `frontend/src/pages/coaches.astro`, near the existing `.coach-summary` rule:

```css
  .filter-toggle {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    margin-top: var(--space-3);
    font-size: 0.9rem;
    color: var(--color-text);
    cursor: pointer;
  }

  .filter-toggle input[type="checkbox"] {
    width: 1.1rem;
    height: 1.1rem;
    accent-color: var(--color-accent);
  }
```

- [ ] **Step 5: Verify manually**

This project's frontend testing is manual browser verification, no new test framework (matches existing convention). With the backend running against the regenerated `api/models/` from Task 1 (`.venv/Scripts/python scripts/run_api.py`) and the frontend dev server running (`cd frontend && npm run dev`):

- Load `/coaches`. The "Current head coaches only" checkbox should be checked by default, and the table should show fewer coaches than the total (some historical coaches from earlier seasons should be filtered out).
- Uncheck the box. The full historical coach list should reappear, and the summary line's count should increase to match.
- The summary text's coach count should always match however many rows are actually in the table.
- Run `cd frontend && npm run build` and confirm it completes with no errors.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/lib/api.ts frontend/src/pages/coaches.astro
git commit -m "Filter the coach browser to current head coaches by default"
```

---

## Task 3: Restructure into home, predict, coaches, and about pages

**Files:**
- Create: `frontend/src/pages/predict.astro` (the current `frontend/src/pages/index.astro`, moved verbatim)
- Modify: `frontend/src/pages/index.astro` (rewritten as the new home page)
- Create: `frontend/src/pages/about.astro`
- Modify: `frontend/src/layouts/Layout.astro`

**Interfaces:**
- Consumes: nothing from Task 1 or Task 2.
- Produces: `frontend/src/pages/predict.astro` exists with the exact content the current `index.astro` has today. Task 4 modifies this file's time input.

- [ ] **Step 1: Move the predictor page to `/predict`**

Copy the file byte for byte so nothing gets paraphrased or dropped:

```bash
cp frontend/src/pages/index.astro frontend/src/pages/predict.astro
```

Do not edit `predict.astro` as part of this step; Task 4 will modify its time input field later. `index.astro` still exists at this point (Step 2 replaces it).

- [ ] **Step 2: Replace `index.astro` with the new home page**

Replace the full contents of `frontend/src/pages/index.astro` with:

```astro
---
import Layout from "../layouts/Layout.astro";
---

<Layout title="4th Down Coach Tendencies | NFL 4th Down Predictor" description="What will an NFL head coach actually do on 4th down? A tendency model built on real play by play history, not the win probability optimal call.">
  <section class="hero">
    <p class="eyebrow">NFL 4th down coach tendencies</p>
    <h1>What will this coach actually do on 4th down?</h1>
    <p class="lede">
      Most 4th down tools tell you the mathematically optimal call. This one tells you what a
      specific coach is actually likely to do, based on their real history, not what a model
      thinks they should do.
    </p>
  </section>

  <section class="link-cards">
    <a class="link-card" href="/predict">
      <h2>Predict a call</h2>
      <p>Set up a situation and a coach, see the model's real prediction next to that coach's career tendency and the league baseline.</p>
      <span class="link-cta">Try the predictor &rarr;</span>
    </a>
    <a class="link-card" href="/coaches">
      <h2>Browse coaches</h2>
      <p>Every NFL head coach's real 4th down tendency, ranked by aggression, filterable to who's currently on the sideline.</p>
      <span class="link-cta">See the full list &rarr;</span>
    </a>
    <a class="link-card" href="/about">
      <h2>About this project</h2>
      <p>How the model works, where the data comes from, and why it's built around tendency instead of optimal play.</p>
      <span class="link-cta">Read the writeup &rarr;</span>
    </a>
  </section>
</Layout>

<style>
  .hero {
    max-width: 52rem;
    margin-bottom: var(--space-6);
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

  .link-cards {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(16rem, 1fr));
    gap: var(--space-4);
    margin-bottom: var(--space-6);
  }

  .link-card {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
    background: var(--color-surface);
    border: 1px solid var(--color-line);
    border-radius: 6px;
    padding: var(--space-4);
    text-decoration: none;
    transition: border-color 0.15s ease, transform 0.1s ease;
  }

  .link-card:hover {
    border-color: var(--color-accent);
    transform: translateY(-2px);
  }

  .link-card h2 {
    font-size: 1.2rem;
    margin: 0;
  }

  .link-card p {
    font-size: 0.9rem;
    margin: 0;
  }

  .link-cta {
    font-family: var(--font-mono);
    font-size: 0.78rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--color-accent);
    margin-top: auto;
  }

  @media (prefers-reduced-motion: reduce) {
    .link-card {
      transition: none;
    }
  }
</style>
```

- [ ] **Step 3: Create the about page**

Create `frontend/src/pages/about.astro`:

```astro
---
import Layout from "../layouts/Layout.astro";
---

<Layout title="About | NFL 4th Down Predictor" description="How this project predicts NFL coach tendencies on 4th down, and why it models tendency instead of the optimal call.">
  <section class="about">
    <p class="eyebrow">About this project</p>
    <h1>Tendency, not optimal play</h1>
    <p class="lede">
      Plenty of public tools already answer what a coach should do on 4th down. This one
      answers a different question: what will this specific coach actually do, based on their
      real decision history. The two often disagree, and that gap is the whole point.
    </p>

    <h2>How it works</h2>
    <p>
      Every 4th down decision an NFL head coach has made since 2010, going for it, punting, or
      attempting a field goal, is pulled from real play by play data via nflverse, then matched
      to the coach who called it and the situation they called it in: down and distance, field
      position, score, time remaining, timeouts, and how many decisions that coach has already
      made.
    </p>
    <p>
      Two models are trained side by side on that history: a gradient boosted tree with the
      coach as a feature, and a neural network that learns its own embedding for each coach.
      Their predictions are simply averaged together, and that averaging beats either model
      alone.
    </p>
    <p>
      A coach's very first 4th down decision comes with no track record, so early on their
      predicted tendency leans on how similar coaches have called similar situations, then
      shifts toward that coach's own real behavior as more of their decisions come in.
    </p>

    <h2>The data</h2>
    <p>
      Play by play and schedule data comes from nflverse, refreshed automatically every week
      during the season, so coaching changes and new tendencies show up without anyone having
      to manually retrain anything.
    </p>

    <h2>The code</h2>
    <p>
      Full source, the model training pipeline, and the real accuracy numbers behind this are
      on <a href="https://github.com/jjxobi/nfl4th">GitHub</a>.
    </p>
  </section>
</Layout>

<style>
  .about {
    max-width: 46rem;
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
  }

  .about h2 {
    font-size: 1.3rem;
    margin-top: var(--space-5);
  }
</style>
```

- [ ] **Step 4: Update the site navigation**

In `frontend/src/layouts/Layout.astro`, change the `<nav>` block from:

```astro
        <nav aria-label="Primary">
          <a href="/" aria-current={isActive("/") ? "page" : undefined}>Predictor</a>
          <a href="/coaches" aria-current={isActive("/coaches") ? "page" : undefined}>Coaches</a>
        </nav>
```

to:

```astro
        <nav aria-label="Primary">
          <a href="/" aria-current={isActive("/") ? "page" : undefined}>Home</a>
          <a href="/predict" aria-current={isActive("/predict") ? "page" : undefined}>Predict</a>
          <a href="/coaches" aria-current={isActive("/coaches") ? "page" : undefined}>Coaches</a>
          <a href="/about" aria-current={isActive("/about") ? "page" : undefined}>About</a>
        </nav>
```

The existing `isActive` helper (`const isActive = (href: string) => (href === "/" ? path === "/" : path.startsWith(href));`) needs no changes: it already special-cases exact match for `/` so the new home page doesn't get marked active on every other route, and the three other routes don't overlap as prefixes of each other.

- [ ] **Step 5: Verify manually**

Run `cd frontend && npm run build` and confirm it completes with no errors and reports 4 pages. Then with `npm run dev` running, visit `/`, `/predict`, `/coaches`, and `/about` in a browser and confirm: each page renders with the shared header/footer, the nav highlights the correct current page on each, `/predict` still has the exact same predictor form and behavior it had before this task (nothing here changes its logic, only its file location and URL), and none of the new copy contains an em dash.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/pages/index.astro frontend/src/pages/predict.astro frontend/src/pages/about.astro frontend/src/layouts/Layout.astro
git commit -m "Split the site into home, predict, coaches, and about pages"
```

---

## Task 4: Accept time remaining as minutes and seconds

**Files:**
- Modify: `frontend/src/pages/predict.astro`

**Interfaces:**
- Consumes: `frontend/src/pages/predict.astro` as created by Task 3.
- Produces: nothing further depends on this task.

- [ ] **Step 1: Replace the seconds field with minutes and seconds fields**

In `frontend/src/pages/predict.astro`, change:

```astro
        <label class="field">
          <span class="field-label">Seconds remaining</span>
          <span class="field-hint">In the game</span>
          <input type="number" id="game_seconds_remaining" min="0" max="3600" value="1800" required />
        </label>
```

to:

```astro
        <label class="field">
          <span class="field-label">Minutes remaining</span>
          <span class="field-hint">In the game</span>
          <input type="number" id="minutes_remaining" min="0" max="59" value="30" required />
        </label>
        <label class="field">
          <span class="field-label">Seconds</span>
          <span class="field-hint">Within that minute</span>
          <input type="number" id="seconds_remaining" min="0" max="59" value="0" required />
        </label>
```

(30 minutes, 0 seconds matches the previous default of 1800 raw seconds exactly, so the form's default prediction doesn't change.)

- [ ] **Step 2: Combine the two fields into `game_seconds_remaining` before the API call**

In the same file's `<script>` block, change the `situation` object construction from:

```typescript
      const situation = {
        coach: coachSelect.value,
        ydstogo: Number((document.getElementById("ydstogo") as HTMLInputElement).value),
        yardline_100: Number((document.getElementById("yardline_100") as HTMLInputElement).value),
        score_differential: Number((document.getElementById("score_differential") as HTMLInputElement).value),
        game_seconds_remaining: Number((document.getElementById("game_seconds_remaining") as HTMLInputElement).value),
        qtr: Number((document.getElementById("qtr") as HTMLInputElement).value),
        posteam_timeouts_remaining: Number((document.getElementById("posteam_timeouts_remaining") as HTMLInputElement).value),
        defteam_timeouts_remaining: Number((document.getElementById("defteam_timeouts_remaining") as HTMLInputElement).value),
        is_home: (document.getElementById("is_home") as HTMLInputElement).checked,
      };
```

to:

```typescript
      const minutesRemaining = Number((document.getElementById("minutes_remaining") as HTMLInputElement).value);
      const secondsRemaining = Number((document.getElementById("seconds_remaining") as HTMLInputElement).value);

      const situation = {
        coach: coachSelect.value,
        ydstogo: Number((document.getElementById("ydstogo") as HTMLInputElement).value),
        yardline_100: Number((document.getElementById("yardline_100") as HTMLInputElement).value),
        score_differential: Number((document.getElementById("score_differential") as HTMLInputElement).value),
        game_seconds_remaining: minutesRemaining * 60 + secondsRemaining,
        qtr: Number((document.getElementById("qtr") as HTMLInputElement).value),
        posteam_timeouts_remaining: Number((document.getElementById("posteam_timeouts_remaining") as HTMLInputElement).value),
        defteam_timeouts_remaining: Number((document.getElementById("defteam_timeouts_remaining") as HTMLInputElement).value),
        is_home: (document.getElementById("is_home") as HTMLInputElement).checked,
      };
```

- [ ] **Step 3: Verify manually**

With the backend and frontend dev servers running, load `/predict` and confirm the form now shows "Minutes remaining" and "Seconds" instead of "Seconds remaining", defaulting to 30 and 0 (same total as the old default of 1800 raw seconds). Pick any coach, set minutes to 2 and seconds to 15, and submit. Then independently call the backend directly with the equivalent raw value, using the interactive docs at `http://localhost:8000/docs` (or `curl`) to `POST /predict` with the exact same coach and situation fields but `game_seconds_remaining: 135` (2 * 60 + 15). The two predictions should be identical, confirming the minutes/seconds fields are being combined correctly before the API call.

Run `cd frontend && npm run build` and confirm it completes with no errors.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/pages/predict.astro
git commit -m "Accept time remaining as minutes and seconds on the predictor"
```
