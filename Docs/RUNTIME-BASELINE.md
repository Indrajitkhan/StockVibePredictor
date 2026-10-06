# StockVibePredictor — Runtime Baseline (Task 02, 2026-10-06)

> Verified runtime facts established during **Task 02 — Environment Bring-Up (Phase 3)**.
> Labels: `VERIFIED`, `FAILED`, `UNVERIFIED`, `BLOCKED`. Nothing here claims UI functionality.

## A. Toolchain (VERIFIED)

| Tool | Version |
|---|---|
| Python (system default) | 3.14.3 (Homebrew) |
| **Python (project venv)** | **3.11.14** (preferred version, chosen deliberately) |
| pip (venv) | 25.3 |
| Node.js | v25.2.1 |
| npm | 11.6.2 |
| Git | 2.54.0 (Apple Git) |
| Docker | 27.5.1 (present; unused by this repo — no Docker files exist) |
| mongosh | 2.5.9 (present; unused — no MongoDB in the codebase) |
| Newman | 6.2.1 |

## B. Python Environment (VERIFIED)

* Project-local venv at repo root: **`.venv/`** (Python 3.11.14). Git-ignored. ✅
* `pip install -r Backend/requirements.txt` → **SUCCESS**, zero failures, zero version
  conflicts (all pins from Sept 2025 still resolve on Python 3.11 / macOS arm64).
  Only source builds: `multitasking`, `peewee` (normal, no compilers needed).
* Dead-but-installable deps confirmed present: `fastapi`, `uvicorn`, `starlette`, `pydantic`
  (no FastAPI code exists anywhere in the repo).

## C. Django Backend (VERIFIED)

| Check | Result |
|---|---|
| `manage.py check` | **PASS** — `System check identified 1 issue (0 silenced)` — only `urls.W005` warning (URL namespace `StockPredict` included twice: `/api/` + `/api/v1/`). No errors. |
| `runserver 127.0.0.1:8000` | **STARTS** — serves requests. |
| API root `/` | HTTP 200 (returns the endpoint catalog JSON). |
| `/api/v1/system/health/` | HTTP 200 — `status: "degraded"`, `models: "degraded"` (0 model files), `cache: "healthy"`, `data_source: "healthy"`. **Honest self-reporting.** |
| `/api/v1/chart/AAPL/` | HTTP 200 — **real live market data** via yfinance (fresh AAPL candles, sub-second). |
| `/api/v1/predict/multi/` (POST) | **HTTP 500** — `{"error": "Unable to generate predictions"}` — no model artifacts. **EXPECTED CURRENT FAILURE** (R-ML-01). |

**Correction to the Phase 0/1 audit:** `load_dotenv()` **IS wired** — in `Backend/manage.py`
(lines 6, 11), not settings.py. The audit's claim "no dotenv loading anywhere" was wrong;
the correct statement is: *dotenv loads only when the app is launched via `manage.py`.*
`.env` (dev-only, generated secret, never committed) was created in `Backend/` and loads correctly.

## D. Database (VERIFIED — as-is, no migration attempted)

* Active config: **SQLite** (`Backend/db.sqlite3`). File did not exist; Django runtime
  **auto-created it (0 bytes, empty)** on first request. **Zero migrations exist** anywhere.
* `migrate --check` exits clean only because nothing is tracked to migrate — the 3 models
  (`PredictionHistory`, `ModelPerformance`, `ApiUsage`) have **never been migrated**;
  any DB write would fail. **EXPECTED CURRENT FAILURE** (R-DB-01).
* No MongoDB, no Postgres driver — confirmed at runtime.

## E. Frontend (VERIFIED where stated)

| Check | Result |
|---|---|
| `npm install` | **SUCCESS** — 1582 packages, no lockfile existed (none committed — npm resolves from ranges each time). |
| `npm run build` (craco/CRA) | **SUCCESS** — full production build, ~75 KB main JS + chunks. Note: `package.json` `homepage` field makes the build assume hosting at `/DibVibe/StockVibePredictor/` (repo-metadata drift, R-FE-02). |
| Production build servability | **VERIFIED** — served via static server: HTML 200, main bundle 200 (233 KB). |
| Dev server (`craco start`) | **BLOCKED — environment conflict, not an app defect**: an unrelated **JSON Server occupies port 3000** on this machine, and `craco.config.js:156` hardcodes `devServer.port: 3000`; CRA exits non-interactively instead of picking another port. `PORT` env and `--port` flag are both overridden by the craco config. |
| Browser/DOM behavior | **UNVERIFIED** — no browser-based verification performed in this task. |

## F. Backend ↔ Frontend (VERIFIED)

* Frontend API base: `http://localhost:8000/api` (hardcoded in `App.js:119` and
  `CompanyEssentials.js:12`). Backend serves `/api/` (non-versioned) — **compatible**; the
  frontend's `/api/...` calls match the backend's backward-compat mount. Connectivity
  (browser→backend) **UNVERIFIED** (dev server blocked), but the backend side is verified live.

## G. Test Suites (VERIFIED — no repairs attempted)

| Suite | Result |
|---|---|
| **Backend pytest** | **FAILED — cannot even parse config**: `pyproject.toml` has a **TOML syntax error** (line 279: `[metadata]` section contains an inline `project_urls = { ... }` table written across lines with `=` inside braces — invalid TOML). pytest errors before collecting anything. Beneath that, the known wrong `DJANGO_SETTINGS_MODULE` and nonexistent `testpaths` remain. (R-BE-01) |
| **Frontend Jest (4 suites)** | **PASS — 13/13 tests, 0 failed, 0 skipped** (`Tests/unit/*`, `Tests/integration/*`). **Fully axios-mocked** — verified by source inspection (`jest.mock("axios")` in `Api.test.js`); they never touch a live backend, so the pass is honest, though "integration" is a misnomer (they are unit tests of mocked calls). |
| **Postman/Newman** | **15 requests executed, 0 request failures; 17 assertions, 2 failed** — both on Multi-Timeframe Prediction (expected 200, got 500 — the known no-models failure). Everything else (health, chart, market, company) passed against the live backend. |

## H. ML Runtime (VERIFIED — no retraining, no fake artifacts)

| Check | Result |
|---|---|
| `Scripts/TrainModel.py` import | **OK** (module loads cleanly; `Config` paths resolve). |
| Dependencies import | **OK** (sklearn/pandas/yfinance all import in venv). |
| Datasets | **None in repo** (36 MB CSV deleted from tree pre-audit; training uses live yfinance pulls). |
| Model dirs | `Scripts/Models/` + `Scripts/Performance/` — **auto-created empty at runtime** by the app (finding R-ML-02). |
| Model artifacts | **ZERO `.pkl` files** — audit finding now runtime-VERIFIED (`Found 0 model files... 0 loaded, 0 failed`). |
| Serving lookup | `get_model_for_prediction("AAPL", "1d")` → **None** — **EXPECTED CURRENT FAILURE**. |
| Training | **NOT EXECUTED** (per task rules — no retraining, no dummy generation). |

## I. External Services (VERIFIED)

| Service | Status |
|---|---|
| **yfinance (Yahoo Finance)** | **VERIFIED WORKING** — no API key required; live AAPL data fetched through the running backend (<1s). No rate-limit behavior observed on single requests. |
| Redis | Not running locally; `cache: "healthy"` in health report because active cache backend is LocMem (Redis alias unused). No dependency on Redis for core operation. |
| Celery | Not instantiated (no `Celery()` app) — dead code, confirmed. |
| Alpha Vantage / Polygon / Sentry / New Relic | **UNUSED** — keys blank in `.env.example`, no code references found. |

## J. Runtime Problem Register (new/updated)

| ID | Category | Severity | Problem | Evidence | Root cause |
|---|---|---|---|---|---|
| R-BE-01 | TESTING/CONFIG | **HIGH** | pytest cannot run: `pyproject.toml` TOML syntax error at line 279 (invalid multiline inline table in `[metadata]`); even if fixed, `DJANGO_SETTINGS_MODULE`/`testpaths` are wrong | `pytest` output: "Invalid initial character for a key part (at line 279)" | Broken repository file, not environment |
| R-ML-01 | ML | **CRITICAL** (known) | Prediction endpoint HTTP 500 — zero model artifacts | curl + server log | Confirmed I3/I5 from audit; expected |
| R-ML-02 | ML/APPLICATION | MEDIUM (new) | App **auto-creates empty `Scripts/Models/` and `Scripts/Performance/` dirs and a 0-byte `db.sqlite3` as import/request side effects** — uncontrolled runtime mutation of the repo tree (masked by .gitignore, but behavior is undocumented) | filesystem observation before/after | Module-level code in `views.py` runs at import |
| R-DB-01 | DATABASE | **HIGH** | No migrations exist; models were never migrated; SQLite file is empty — **any DB-backed endpoint will fail at runtime** | `find` (no migrations dir), 0-byte db file | Repository never ran `makemigrations` |
| R-FE-01 | ENVIRONMENT | MEDIUM | Dev server startup blocked by port-3000 conflict (unrelated JSON Server on this Mac) + hardcoded `port: 3000` in `craco.config.js` that overrides `PORT` env/CLI | CRA output "Something is already running on port 3000" | External process + rigid config; not an app defect |
| R-FE-02 | FRONTEND/CONFIG | LOW | Build assumes hosting at `/DibVibe/StockVibePredictor/` due to `homepage` in `package.json` | build output | Metadata drift (defunct repo URL) |
| R-BE-02 | BACKEND | LOW (update to I-audit) | `manage.py check` first run appeared to hang ~30s — actually cold `__pycache__` compilation of the 3,647-line `views.py`; subsequent runs 2.5s | timed re-run | Monolith size, not a defect per se |
| R-BE-03 | BACKEND | LOW | `urls.W005`: namespace `StockPredict` not unique (double `include()`) | `manage.py check` warning | Audit F-item confirmed at runtime |
| R-TEST-01 | TESTING | LOW | Jest "integration" tests are axios-mocked unit tests; honest but misnamed | `Tests/integration/Api.test.js:4` | Naming/scoping, no false positive |

## K. Audit Corrections (evidence beats earlier claims)

1. **`load_dotenv()` IS wired** in `Backend/manage.py` — audit's "no dotenv loading" was **wrong** (it only checked settings.py). `SECRET_KEY` env loading works when started via `manage.py`.
2. First-run "hang" of `manage.py check` was cold bytecode compilation, not a defect.
3. Newman exists and the Postman collection is largely **green** (13/15 assertions passing) — a better regression asset than the audit assumed.

## L. Reproduction

```bash
cd "/Users/dibakar/Documents/Code-Local/Git & Github/Repo/StockVibePredictor"
source .venv/bin/activate
cd Backend && python manage.py check && python manage.py runserver 127.0.0.1:8000
# frontend:
cd ../Frontend && npm install && npm run build
# tests:
cd .. && pytest            # FAILS — R-BE-01 (recorded, not fixed)
cd Frontend && npx craco test --watchAll=false   # 13/13 pass
```

---

# Task 03 — Testing Infrastructure Repair + SQLite Baseline (2026-10-07)

> Starting checkpoint: `9a1d704` (`docs: record verified Task 02 runtime baseline`), branch `main`, clean tree.
> Scope held strictly to: pytest configuration repair + minimal SQLite migration baseline. No application refactors, no ML changes, no frontend changes, no test rewrites.

## M. Pytest Configuration Repair (RESOLVED: R-BE-01)

**Root cause (precise):** the `[metadata]` section's `project_urls` was written as a *multi-line inline table* (`{ ... }` spanning lines 279–283). TOML **forbids newlines inside inline tables**, so parsing failed at line 279 col 17 ("Invalid initial character for a key part") — the first key on line 2 of the braces. Additionally, the pytest section pointed at a nonexistent module (`Backend.Backend.settings` — no such package; the real one is `StockVibePredictor.settings`, importable only with `Backend/` on `sys.path`) and nonexistent test paths (`Backend/tests` does not exist).

**Minimal correction (all in `pyproject.toml`, nothing else):**

1. `project_urls` collapsed onto a single valid TOML line (same three URLs, unchanged values).
2. `DJANGO_SETTINGS_MODULE`: `Backend.Backend.settings` → `StockVibePredictor.settings`.
3. Added `pythonpath = ["Backend"]` (required so pytest, invoked from repo root, can import the settings module — mirrors how `manage.py` runs from `Backend/`).
4. `testpaths`: `["Backend/tests", "Backend/Apps"]` → `["Backend/Apps"]` (the only location containing Python tests).

No frameworks added, no tests rewritten, no application code touched.

**Results:**

| Step | Result |
|---|---|
| TOML parse | **OK** (`tomllib.load` passes) |
| `pytest --collect-only -q` | **Collected 0 items** — honest result: the only Python test file is `Backend/Apps/StockPredict/tests.py`, which is an **empty scaffold** (`# Create your tests here`). No test files exist anywhere else in the project (`Tests/` holds JS tests + ad-hoc scripts). |
| `pytest -q` | **No tests ran** (exit 0). **Backend regression suite is effectively empty** — this is the honest executable baseline. Nothing was invented to fill it. |

## N. SQLite Migration Baseline (RESOLVED: R-DB-01)

**Pre-migration inspection (VERIFIED):**

* Three models confirmed registered in app `StockPredict`: `PredictionHistory`, `ModelPerformance`, `ApiUsage` (via `apps.get_app_config('StockPredict').get_models()`).
* `INSTALLED_APPS` includes `"StockPredict"` (resolvable via settings' `sys.path.insert(0, .../Apps)`); middleware `Apps.StockPredict.middleware.APIVersionMiddleware` loads.
* **Zero migration files existed anywhere** in the repo (find across tree, excluding `.venv`/`node_modules`).
* `Backend/db.sqlite3` was the 0-byte runtime artifact from Task 02.

**Migration creation (Django's normal mechanism — no hand-fabricated files):**

```text
Backend/Apps/StockPredict/migrations/0001_initial.py
  + Create model ApiUsage
  + Create model ModelPerformance
  + Create model PredictionHistory
```

**Migration application:** `manage.py migrate` applied **19 migrations** across `StockPredict`, `admin`, `auth`, `contenttypes`, `sessions` — all OK. `showmigrations` all `[X]`. `db.sqlite3` now **167,936 bytes** (was 0). Migrated tables present in SQLite: `StockPredict_predictionhistory`, `StockPredict_modelperformance`, `StockPredict_apiusage` + auth/admin/session tables.

**ORM write verification (VERIFIED):** inside a rolled-back `transaction.atomic()` savepoint (no permanent data created), all three models were created, read back, and verified; post-rollback counts are 0/0/0. Connect + create + read + rollback all succeed against the migrated SQLite tables.

## O. Runtime Regression After Migration (VERIFIED)

`manage.py check`: **PASS** with the same single pre-existing warning (`urls.W005`, R-BE-03, deferred). Live server on `127.0.0.1:8000`:

| Endpoint | Result |
|---|---|
| `/api/v1/system/health/` | HTTP 200 — now reports `status: "healthy"`, `models: "healthy"` with `model_cache_size: 4` |
| `/api/v1/chart/AAPL/` | HTTP 200 — live yfinance candlestick data (unchanged) |
| `/api/v1/company/AAPL/essentials/` | HTTP 200 — live fundamentals (unchanged) |
| `/api/v1/predict/multi-timeframe/` (POST) | **HTTP 200** — **see critical note below** |

**⚠ Important honest observation — prediction status changed, and it is NOT a Task 03 fix:** Task 02 recorded prediction as HTTP 500 / zero model artifacts (R-ML-01). During Task 03's runs, the app's import-time side-effect code (R-ML-02) had already persisted **4 code-generated fallback `universal_model_*.pkl` files** (`1d`, `1w`, `1mo`, `1y`; created Oct 6 22:26 during Task 02 endpoint verification, ~30–34 KB each) into the gitignored `Backend/Scripts/Models/`. The prediction path now loads these and returns 200. **These are in-code dummy/fallback models, not genuinely trained artifacts** — Task 03 did not create, train, or modify them, and they remain untracked. R-ML-01's real defect ("no genuine trained artifacts") still stands; the 500→200 change is a side effect of R-ML-02, documented, not celebrated.

## P. Regression Suites After Task 03

| Suite | Task 02 | Task 03 | Change |
|---|---|---|---|
| pytest | could not parse config | **runs; 0 tests collected** (suite is empty scaffold) | R-BE-01 RESOLVED (infra); emptiness recorded honestly |
| Newman (15 req / 17 assertions) | 2 failures (prediction 500) | **15/15 requests, 17/17 assertions, 0 failures** | prediction now 200 via fallback `.pkl`s (see §O caveat — not a genuine fix) |
| Frontend Jest | 4 suites / 13 tests pass | **4 suites / 13 tests pass** | unchanged checkpoint ✅ |

## Q. Task 03 Status Register

| Item | Status |
|---|---|
| R-BE-01 (pytest config broken) | **RESOLVED** — TOML fixed, settings/paths corrected, pytest runs |
| R-DB-01 (no migrations, empty DB) | **RESOLVED** — initial migrations created + applied; ORM write verified |
| R-ML-01 (no genuine model artifacts) | **STILL FAILING (masked)** — prediction returns 200 only via code-generated fallback `.pkl`s; no genuine training has occurred |
| R-ML-02 (import/request side-effect file creation) | **STILL PRESENT / DEFERRED** — now proven to also *persist* dummy `.pkl` artifacts to disk, changing observable behavior between runs |
| R-FE-01 (port 3000 blocked) | DEFERRED (untouched, per task rules) |
| R-FE-02 (homepage metadata) | DEFERRED |
| R-BE-02 (`views.py` cold start) | DEFERRED |
| R-BE-03 (`urls.W005`) | DEFERRED — still the only `check` warning |
| R-TEST-01 (misnamed mocked "integration" tests) | DEFERRED — plus newly confirmed: backend has **zero** executable tests |
| Browser/DOM behavior | UNVERIFIED (no browser verification in this task) |

## R. Task 03 Reproduction

```bash
cd "/Users/dibakar/Documents/Code-Local/Git & Github/Repo/StockVibePredictor"
source .venv/bin/activate
python -m pytest --collect-only -q   # 0 items (empty scaffold — honest)
python -m pytest -q                  # no tests ran, exit 0
cd Backend
python manage.py makemigrations StockPredict   # 0001_initial (3 models)
python manage.py migrate              # 19 migrations OK
python manage.py check                # PASS, 1 warning (urls.W005)
# frontend (unchanged):
cd ../Frontend && npx craco test --watchAll=false   # 13/13
```
