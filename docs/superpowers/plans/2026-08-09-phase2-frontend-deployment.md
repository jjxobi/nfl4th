# Phase 2 Frontend and Deployment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a static Astro frontend (a situation predictor page and a coach browser page) that consumes the phase 2 backend API, add CORS and a health endpoint to that API so the frontend can actually call it cross-origin, and produce the deployment configuration (Netlify + Render) needed to put this live at a public URL.

**Architecture:** `frontend/` is a new top-level Astro project (static output, no server-side rendering) that fetches from the FastAPI backend client-side at runtime — nothing about coach data or predictions is baked in at build time. `api/main.py` gets CORS middleware (configurable allowed origins) and a `GET /` health endpoint. Deployment config (`render.yaml`, `frontend/netlify.toml`) is committed as infrastructure-as-code; actually connecting Render/Netlify accounts to this repo and adding the DNS record are manual steps outside this plan's reach (no account access), documented clearly in the README instead.

**Tech Stack:** Astro (static output, TypeScript, no UI framework — vanilla `<script>` islands are enough for this scope), the existing FastAPI backend.

## Global Constraints

- Python 3.11 exactly for anything touching `api/` or `src/nfl4th/`. Node.js >=22.12 for `frontend/` (verified installed: Node 22.17.1, npm 10.9.2).
- Every git commit message is plain, simple, human-sounding language. No em dashes. No "Co-Authored-By" trailer, no mention of Claude or AI anywhere in any commit message. Do not push to any remote.
- Code comments: none by default. Only add a comment where the reasoning is genuinely non-obvious.
- **Astro's scaffolding tool auto-generates `CLAUDE.md` and `AGENTS.md` files in new projects. Delete both immediately after scaffolding, before the first commit.** Verified this happens with `npm create astro@latest`. Neither file serves the application; both directly contradict this project's standing "no trace of AI involvement" requirement.
- The backend's real Pydantic schemas (verified from the merged phase 2 backend API branch, not guessed) are: `CoachProfile` (`coach: str`, `n_decisions: int`, `shrinkage_weight: float`, `punt`/`field_goal`/`go_for_it`: `DecisionRates`), `DecisionRates` (`observed`, `expected`, `shrunk`: all `float`), `PredictResponse` (`predicted`, `coach_career_average`, `league_baseline`: all `DecisionProbabilities`), `DecisionProbabilities` (`punt`, `field_goal`, `go_for_it`: all `float`), `SituationRequest` (`coach: str`, `ydstogo`, `yardline_100`, `score_differential`, `game_seconds_remaining`: `float` with bounds, `qtr`, `posteam_timeouts_remaining`, `defteam_timeouts_remaining`: `int` with bounds, `is_home: bool`, `week: int = 9`).
- `coach_career_average` in `PredictResponse` is a career-wide aggregate, NOT conditional on the requested situation (unlike `predicted` and `league_baseline`, which both are). The frontend must not present these three as directly comparable without that distinction being visible to the user — this was a real, deliberately-fixed naming/framing issue in the backend's final review; don't reintroduce the confusion in the UI layer.
- Render's blueprint YAML schema (verified against Render's own docs, not guessed): `type: web`, `runtime: python`, `plan: free`, `buildCommand`, `startCommand`, and `envVars: [{key: ..., sync: false}]` for a value that must be entered manually in the dashboard rather than hardcoded.
- Astro's static output mode fetches nothing at build time here — every page is static HTML/CSS/JS that calls the API client-side after the page loads in a browser. This is deliberate: it keeps the build decoupled from the API being reachable at build time, and matches the SEO/shareability goals from the design spec (static HTML, not a client-rendered SPA shell).

## Sequencing

Real dependency graph, not a linear default:

- **Task 1** (backend CORS + health endpoint) and **Task 2** (Astro scaffolding + shared layout + API client) touch entirely disjoint files and have no dependency on each other — run them in parallel, each on its own isolated branch/worktree, merge both before starting Task 3.
- **Task 3** (coach browser page) and **Task 4** (predictor page) both depend on Task 2's shared layout and API client existing, but touch entirely disjoint page files from each other (`coaches.astro` vs `index.astro`) — run them in parallel too, once Task 2 is merged in.
- **Task 5** (deployment config + README) depends on Tasks 1, 3, and 4 all being done, since it documents and configures the finished result.

```
Task 1 ─┐
        ├─→ Task 3 ─┐
Task 2 ─┘           ├─→ Task 5
        └─→ Task 4 ─┘
```

---

## Task 1: Backend CORS and health endpoint

**Files:**
- Modify: `api/main.py`
- Create: `tests/api/test_health_and_cors.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `GET /` (health check: `{"name": str, "latest_season": int, "n_coaches": int}`), CORS middleware allowing configurable origins via the `ALLOWED_ORIGINS` environment variable (comma-separated, default `http://localhost:4321` for local Astro dev). Nothing later in this plan depends on this task's code directly (Task 2's frontend calls the API over HTTP, not as a Python import), but the deployed API is unusable from a browser without it.

- [ ] **Step 1: Write the failing tests**

Create `tests/api/test_health_and_cors.py`:

```python
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health_endpoint_returns_basic_info():
    response = client.get("/")

    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "nfl4th API"
    assert isinstance(body["latest_season"], int)
    assert isinstance(body["n_coaches"], int)
    assert body["n_coaches"] > 0


def test_cors_allows_the_configured_origin():
    response = client.get("/", headers={"Origin": "http://localhost:4321"})

    assert response.headers.get("access-control-allow-origin") == "http://localhost:4321"


def test_cors_rejects_an_unlisted_origin():
    response = client.get("/", headers={"Origin": "https://not-allowed.example.com"})

    assert "access-control-allow-origin" not in response.headers
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/api/test_health_and_cors.py -v`
Expected: FAIL, `GET /` currently 404s (no such route), so all three tests fail (the CORS tests fail because a 404 response still carries no CORS header for an unlisted origin by coincidence, but let's confirm the actual failure is the missing route: the health test fails on `response.status_code == 200`).

- [ ] **Step 3: Add CORS middleware and the health route**

Replace the full contents of `api/main.py` with:

```python
from __future__ import annotations

import os

from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.loader import LoadedModels
from api.routes_coaches import register_coach_routes
from api.routes_predict import register_predict_routes

app = FastAPI(title="NFL 4th Down Coach Tendency API")

allowed_origins = os.environ.get("ALLOWED_ORIGINS", "http://localhost:4321").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

loaded = LoadedModels()

router = APIRouter()


@router.get("/")
def health() -> dict:
    return {"name": "nfl4th API", "latest_season": loaded.latest_season, "n_coaches": len(loaded.report)}


register_coach_routes(router, loaded)
register_predict_routes(router, loaded)
app.include_router(router)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/api/test_health_and_cors.py -v`
Expected: 3 passed.

- [ ] **Step 5: Run the full suite and lint, then commit**

Run: `.venv/Scripts/python -m pytest -v` and `.venv/Scripts/python -m ruff check .`
Expected: all tests pass, lint clean.

```bash
git add api/main.py tests/api/test_health_and_cors.py
git commit -m "Add CORS and a health endpoint to the backend api"
```

---

## Task 2: Astro scaffolding, shared layout, and API client

**Files:**
- Create: `frontend/` (Astro project: `package.json`, `astro.config.mjs`, `tsconfig.json`, `public/favicon.svg`, `.gitignore`)
- Create: `frontend/.env.example`
- Create: `frontend/src/layouts/Layout.astro`
- Create: `frontend/src/lib/api.ts`

**Interfaces:**
- Consumes: nothing (this is the frontend project's foundation).
- Produces: the `frontend/` Astro project itself; `Layout.astro` (props: `title: string`, `description: string`; renders a `<slot />` for page content, with nav links to `/` and `/coaches`); `frontend/src/lib/api.ts` exporting `getCoaches()`, `getCoach(name)`, `predict(situation)`, and the TypeScript types `CoachProfile`, `DecisionRates`, `DecisionProbabilities`, `PredictResponse`, `SituationInput`. Used by Tasks 3 and 4.

- [ ] **Step 1: Scaffold the Astro project**

From the repo root:

```bash
npm create astro@latest -- frontend --template minimal --no-install --no-git --typescript strict --yes
```

Expected: creates a `frontend/` directory with `package.json`, `astro.config.mjs`, `tsconfig.json`, `public/`, `src/pages/index.astro`, and (verified during planning) `CLAUDE.md` and `AGENTS.md`.

- [ ] **Step 2: Delete the AI-agent instruction files the scaffold generated**

```bash
rm frontend/CLAUDE.md frontend/AGENTS.md
```

Verify they're gone: `ls frontend/` should not list either file.

- [ ] **Step 3: Install dependencies and verify the default scaffold builds**

```bash
cd frontend && npm install && npm run build && cd ..
```

Expected: installs cleanly, `npm run build` reports `output: "static"` and completes with no errors. This confirms the toolchain works before any real code is written on top of it.

- [ ] **Step 4: Add a `.gitignore` for the frontend project (if the scaffold didn't already create one, or extend it)**

Check if `frontend/.gitignore` exists. If not, create it; if it exists, make sure it contains at least:

```
node_modules/
dist/
.env
```

- [ ] **Step 5: Add environment variable handling for the API base URL**

Create `frontend/.env.example`:

```
PUBLIC_API_URL=http://localhost:8000
```

Create a local `frontend/.env` (gitignored, not committed) with the same content, so local dev works immediately:

```
PUBLIC_API_URL=http://localhost:8000
```

- [ ] **Step 6: Write the API client**

Create `frontend/src/lib/api.ts`:

```typescript
export interface DecisionRates {
  observed: number;
  expected: number;
  shrunk: number;
}

export interface CoachProfile {
  coach: string;
  n_decisions: number;
  shrinkage_weight: number;
  punt: DecisionRates;
  field_goal: DecisionRates;
  go_for_it: DecisionRates;
}

export interface DecisionProbabilities {
  punt: number;
  field_goal: number;
  go_for_it: number;
}

export interface PredictResponse {
  predicted: DecisionProbabilities;
  coach_career_average: DecisionProbabilities;
  league_baseline: DecisionProbabilities;
}

export interface SituationInput {
  coach: string;
  ydstogo: number;
  yardline_100: number;
  score_differential: number;
  game_seconds_remaining: number;
  qtr: number;
  posteam_timeouts_remaining: number;
  defteam_timeouts_remaining: number;
  is_home: boolean;
}

const API_URL = import.meta.env.PUBLIC_API_URL;

export async function getCoaches(): Promise<CoachProfile[]> {
  const response = await fetch(`${API_URL}/coaches`);
  if (!response.ok) throw new Error(`Failed to load coaches: ${response.status}`);
  return response.json();
}

export async function getCoach(name: string): Promise<CoachProfile> {
  const response = await fetch(`${API_URL}/coaches/${encodeURIComponent(name)}`);
  if (!response.ok) throw new Error(`Failed to load coach: ${response.status}`);
  return response.json();
}

export async function predict(situation: SituationInput): Promise<PredictResponse> {
  const response = await fetch(`${API_URL}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(situation),
  });
  if (!response.ok) throw new Error(`Prediction failed: ${response.status}`);
  return response.json();
}
```

- [ ] **Step 7: Write the shared layout**

Create `frontend/src/layouts/Layout.astro`. Apply real, intentional visual design here, not bare unstyled HTML — this is a portfolio piece. Use the `frontend-design` skill's guidance for typography, spacing, and color choices. At minimum the structure must include:

```astro
---
interface Props {
  title: string;
  description: string;
}
const { title, description } = Astro.props;
---
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width" />
    <link rel="icon" type="image/svg+xml" href="/favicon.svg" />
    <meta name="description" content={description} />
    <title>{title}</title>
  </head>
  <body>
    <header>
      <nav>
        <a href="/">Predictor</a>
        <a href="/coaches">Coaches</a>
      </nav>
    </header>
    <main>
      <slot />
    </main>
  </body>
</html>
```

Add real CSS (a `<style>` block or a separate stylesheet) for a clean, legible, intentional look — not the browser default. Keep it minimal enough not to block Tasks 3/4, which will add their own page-specific content inside `<main>`.

- [ ] **Step 8: Replace the placeholder `index.astro` with a minimal stub using the layout**

Replace `frontend/src/pages/index.astro` with:

```astro
---
import Layout from "../layouts/Layout.astro";
---

<Layout title="4th Down Predictor | NFL Coach Tendencies" description="Enter a 4th down situation and see what an NFL coach is actually likely to do, based on real historical tendency.">
  <h1>4th Down Predictor</h1>
  <p>Coming soon.</p>
</Layout>
```

(Task 4 replaces this with the real predictor page. This stub exists so the project builds cleanly and has something real using `Layout.astro` before Task 2 is done.)

- [ ] **Step 9: Verify the build still works and commit**

```bash
cd frontend && npm run build && cd ..
```

Expected: builds cleanly, no errors, `frontend/dist/index.html` exists.

```bash
git add frontend
git commit -m "Scaffold the astro frontend with a shared layout and api client"
```

---

## Task 3: Coach browser page

**Files:**
- Create: `frontend/src/pages/coaches.astro`

**Interfaces:**
- Consumes: `Layout.astro`, `getCoaches()` and `CoachProfile` from `frontend/src/lib/api.ts` (Task 2).
- Produces: the `/coaches` page. Nothing later depends on this task.

- [ ] **Step 1: Write the coach browser page**

Create `frontend/src/pages/coaches.astro`:

```astro
---
import Layout from "../layouts/Layout.astro";
---

<Layout title="Coach Tendencies | NFL 4th Down Predictor" description="Browse every NFL head coach's real 4th down tendency, ranked by aggression.">
  <h1>Coach Tendencies</h1>
  <p>Click a column to sort. Click a coach to see their full profile.</p>
  <table id="coach-table">
    <thead>
      <tr>
        <th data-sort="coach">Coach</th>
        <th data-sort="n_decisions">Decisions</th>
        <th data-sort="go_for_it">Go For It Rate</th>
        <th data-sort="punt">Punt Rate</th>
        <th data-sort="field_goal">Field Goal Rate</th>
      </tr>
    </thead>
    <tbody id="coach-table-body"></tbody>
  </table>
  <div id="coach-detail" hidden></div>
</Layout>

<script>
  import { getCoaches, type CoachProfile } from "../lib/api";

  let coaches: CoachProfile[] = [];
  let sortKey = "go_for_it";
  let sortDescending = true;

  function sortValue(coach: CoachProfile, key: string): number | string {
    if (key === "coach") return coach.coach;
    if (key === "n_decisions") return coach.n_decisions;
    if (key === "go_for_it") return coach.go_for_it.shrunk;
    if (key === "punt") return coach.punt.shrunk;
    if (key === "field_goal") return coach.field_goal.shrunk;
    return "";
  }

  function render() {
    const sorted = [...coaches].sort((a, b) => {
      const av = sortValue(a, sortKey);
      const bv = sortValue(b, sortKey);
      const cmp = av < bv ? -1 : av > bv ? 1 : 0;
      return sortDescending ? -cmp : cmp;
    });

    const body = document.getElementById("coach-table-body")!;
    body.innerHTML = "";
    for (const coach of sorted) {
      const row = document.createElement("tr");
      row.innerHTML = `
        <td>${coach.coach}</td>
        <td>${coach.n_decisions}</td>
        <td>${(coach.go_for_it.shrunk * 100).toFixed(1)}%</td>
        <td>${(coach.punt.shrunk * 100).toFixed(1)}%</td>
        <td>${(coach.field_goal.shrunk * 100).toFixed(1)}%</td>
      `;
      row.addEventListener("click", () => showDetail(coach));
      body.appendChild(row);
    }
  }

  function showDetail(coach: CoachProfile) {
    const detail = document.getElementById("coach-detail")!;
    detail.hidden = false;
    detail.innerHTML = `
      <h2>${coach.coach}</h2>
      <p>${coach.n_decisions} career 4th down decisions.</p>
      <p>Go for it: ${(coach.go_for_it.shrunk * 100).toFixed(1)}% (observed ${(coach.go_for_it.observed * 100).toFixed(1)}%, situational baseline ${(coach.go_for_it.expected * 100).toFixed(1)}%)</p>
      <p>Punt: ${(coach.punt.shrunk * 100).toFixed(1)}% (observed ${(coach.punt.observed * 100).toFixed(1)}%, situational baseline ${(coach.punt.expected * 100).toFixed(1)}%)</p>
      <p>Field goal: ${(coach.field_goal.shrunk * 100).toFixed(1)}% (observed ${(coach.field_goal.observed * 100).toFixed(1)}%, situational baseline ${(coach.field_goal.expected * 100).toFixed(1)}%)</p>
    `;
  }

  document.querySelectorAll("th[data-sort]").forEach((th) => {
    th.addEventListener("click", () => {
      const key = (th as HTMLElement).dataset.sort!;
      if (sortKey === key) {
        sortDescending = !sortDescending;
      } else {
        sortKey = key;
        sortDescending = true;
      }
      render();
    });
  });

  getCoaches()
    .then((data) => {
      coaches = data;
      render();
    })
    .catch((error) => {
      const body = document.getElementById("coach-table-body")!;
      body.innerHTML = `<tr><td colspan="5">Could not load coaches: ${error instanceof Error ? error.message : "unknown error"}</td></tr>`;
    });
</script>
```

Apply real styling consistent with `Layout.astro` — a readable table, sensible spacing, a visibly clickable row/column affordance. Use the `frontend-design` skill's guidance rather than leaving this as bare unstyled HTML.

- [ ] **Step 2: Manually verify in a real browser**

Start the backend in one terminal: `.venv/Scripts/python scripts/run_api.py` (from the repo root; do not use `python -m uvicorn api.main:app` directly — on Windows this reliably crashes with a torch/pandas DLL load-order error, since `-m uvicorn`'s own CLI bootstrapping imports things before `nfl4th` ever gets a chance to run its own torch-first import guard. `scripts/run_api.py` exists specifically to import `nfl4th` first, then hand off to uvicorn — see Task 5).
Start the frontend dev server in another: `cd frontend && npm run dev`.
Open the printed local URL (default `http://localhost:4321/coaches`) in a browser. Confirm: the table populates with real coach data, sorting by clicking a column header works, clicking a coach row shows their detail panel with sensible numbers.

- [ ] **Step 3: Verify the build still works and commit**

```bash
cd frontend && npm run build && cd ..
```

Expected: builds cleanly, no TypeScript errors.

```bash
git add frontend/src/pages/coaches.astro
git commit -m "Add the coach tendency browser page"
```

---

## Task 4: Predictor page

**Files:**
- Modify: `frontend/src/pages/index.astro` (replacing Task 2's stub)

**Interfaces:**
- Consumes: `Layout.astro`, `getCoaches()`, `predict()`, `SituationInput`, `DecisionProbabilities` from `frontend/src/lib/api.ts` (Task 2).
- Produces: the real `/` page. Nothing later depends on this task.

- [ ] **Step 1: Write the predictor page**

Replace the full contents of `frontend/src/pages/index.astro` with:

```astro
---
import Layout from "../layouts/Layout.astro";
---

<Layout title="4th Down Predictor | NFL Coach Tendencies" description="Enter a 4th down situation and see what an NFL coach is actually likely to do, based on real historical tendency.">
  <h1>What will this coach actually do?</h1>
  <form id="predict-form">
    <label>
      Coach
      <select id="coach" required></select>
    </label>
    <label>
      Yards to go
      <input type="number" id="ydstogo" min="1" max="99" value="2" required />
    </label>
    <label>
      Yard line (distance from opponent's end zone)
      <input type="number" id="yardline_100" min="1" max="99" value="40" required />
    </label>
    <label>
      Score differential (positive if leading)
      <input type="number" id="score_differential" min="-60" max="60" value="0" required />
    </label>
    <label>
      Seconds remaining in game
      <input type="number" id="game_seconds_remaining" min="0" max="3600" value="1800" required />
    </label>
    <label>
      Quarter
      <input type="number" id="qtr" min="1" max="5" value="2" required />
    </label>
    <label>
      Offense timeouts remaining
      <input type="number" id="posteam_timeouts_remaining" min="0" max="3" value="3" required />
    </label>
    <label>
      Defense timeouts remaining
      <input type="number" id="defteam_timeouts_remaining" min="0" max="3" value="3" required />
    </label>
    <label>
      <input type="checkbox" id="is_home" checked />
      Home team has the ball
    </label>
    <button type="submit">Predict</button>
  </form>
  <div id="result" hidden></div>
</Layout>

<script>
  import { getCoaches, predict, type DecisionProbabilities } from "../lib/api";

  const coachSelect = document.getElementById("coach") as HTMLSelectElement;
  getCoaches()
    .then((coaches) => {
      for (const coach of coaches) {
        const option = document.createElement("option");
        option.value = coach.coach;
        option.textContent = coach.coach;
        coachSelect.appendChild(option);
      }
    })
    .catch((error) => {
      coachSelect.innerHTML = `<option>Could not load coaches</option>`;
      console.error(error);
    });

  function renderProbabilities(label: string, note: string, probs: DecisionProbabilities): string {
    return `
      <div class="prob-group">
        <h3>${label}</h3>
        <p class="note">${note}</p>
        <p>Go for it: ${(probs.go_for_it * 100).toFixed(1)}%</p>
        <p>Punt: ${(probs.punt * 100).toFixed(1)}%</p>
        <p>Field goal: ${(probs.field_goal * 100).toFixed(1)}%</p>
      </div>
    `;
  }

  const form = document.getElementById("predict-form") as HTMLFormElement;
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const result = document.getElementById("result")!;

    try {
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

      const response = await predict(situation);

      result.hidden = false;
      result.innerHTML = `
        ${renderProbabilities("Model prediction", "For this exact situation.", response.predicted)}
        ${renderProbabilities("This coach's career average", "Across every 4th down they've faced, not specific to this situation.", response.coach_career_average)}
        ${renderProbabilities("League baseline", "A coach-agnostic model, for this exact situation.", response.league_baseline)}
      `;
    } catch (error) {
      result.hidden = false;
      result.innerHTML = `<p>Something went wrong: ${error instanceof Error ? error.message : "unknown error"}</p>`;
    }
  });
</script>
```

Note the explicit "not specific to this situation" note on the career average — this is required, not optional styling, per the Global Constraints section above.

Apply real styling consistent with `Layout.astro` and `coaches.astro` — a clean form layout, visually distinct result cards for the three probability groups. Use the `frontend-design` skill's guidance.

- [ ] **Step 2: Manually verify in a real browser**

With the backend and frontend dev servers both running (see Task 3 Step 2), open `http://localhost:4321/`. Confirm: the coach dropdown populates with real names, submitting the form returns three distinct probability breakdowns, and changing the situation inputs (e.g. very short yardage vs very long yardage) visibly changes the "Model prediction" numbers.

- [ ] **Step 3: Verify the build still works and commit**

```bash
cd frontend && npm run build && cd ..
```

Expected: builds cleanly, no TypeScript errors.

```bash
git add frontend/src/pages/index.astro
git commit -m "Add the situation predictor page"
```

---

## Task 5: Deployment configuration and README

**Files:**
- Create: `render.yaml`
- Create: `frontend/netlify.toml`
- Create: `scripts/run_api.py`
- Modify: `README.md`

**Interfaces:**
- None (configuration and documentation only).

- [ ] **Step 1: Write the Render blueprint**

Create `render.yaml` at the repo root:

```yaml
services:
  - type: web
    name: nfl4th-api
    runtime: python
    plan: free
    buildCommand: pip install -e ".[api]"
    startCommand: uvicorn api.main:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: ALLOWED_ORIGINS
        sync: false
```

- [ ] **Step 2: Write the Netlify build config**

Create `frontend/netlify.toml`:

```toml
[build]
  command = "npm run build"
  publish = "dist"

[build.environment]
  NODE_VERSION = "22"
```

(Note: this file lives inside `frontend/`, and Netlify is expected to be configured with `frontend` as the site's base directory when the site is connected, since that's the actual Astro project root within this repo.)

- [ ] **Step 3: Write a local backend launcher script**

Running `python -m uvicorn api.main:app` directly crashes on Windows with a torch/pandas DLL load-order error (`OSError: [WinError 1114] ... c10.dll`), because `-m uvicorn`'s own CLI bootstrapping does its own imports before `nfl4th` ever gets a chance to run its torch-first import guard (`src/nfl4th/__init__.py`). Verified directly: `python -c "import nfl4th; import uvicorn; uvicorn.run(...)"` (import order controlled explicitly) works fine, so the fix is a tiny launcher script that guarantees that order, rather than the raw `uvicorn` CLI. (`render.yaml`'s `startCommand` in Step 1 does not need this fix — Render runs Linux, where this DLL conflict does not exist; this is a Windows-local-development-only problem.)

Create `scripts/run_api.py`:

```python
import nfl4th  # noqa: F401  (must import before uvicorn, see nfl4th/__init__.py for why)
import uvicorn

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
```

Verify: `.venv/Scripts/python scripts/run_api.py`, then in another terminal `curl http://localhost:8000/` should return the health check JSON. Confirm no `WinError 1114` traceback appears.

- [ ] **Step 4: Write the deployment and setup sections of the README**

Add these sections to `README.md` (place them after the existing "Running the full pipeline" section and before "What's next"):

```markdown
## Backend API

The trained models are also served behind a small FastAPI backend, so a web
app can get predictions without retraining anything.

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
predictor and a coach tendency browser.

```bash
cd frontend
npm install
npm run dev
```

Requires `PUBLIC_API_URL` set in `frontend/.env` (see
`frontend/.env.example`) pointing at a running backend.

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
   Render service's URL.
4. Point a subdomain (e.g. `4thdown.yourdomain.com`) at the Netlify site via
   a DNS CNAME record, then add it as a custom domain in Netlify's site
   settings.
```

- [ ] **Step 5: Verify everything still works together**

Run the backend test suite: `.venv/Scripts/python -m pytest -v` and `.venv/Scripts/python -m ruff check .` — expect all passing, clean.
Run the frontend build: `cd frontend && npm run build && cd ..` — expect it to succeed.

- [ ] **Step 6: Commit**

```bash
git add render.yaml frontend/netlify.toml scripts/run_api.py README.md
git commit -m "Add deployment config and document the backend and frontend"
```
