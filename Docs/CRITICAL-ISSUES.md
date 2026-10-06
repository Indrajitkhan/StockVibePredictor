# StockVibePredictor — Critical Issues (from Phase 0/1 audit)

> Derived from the read-only audit (ADR-000). Nothing here has been fixed yet unless marked.
> Status per issue: **OPEN** (unresolved) or **RESOLVED**.

| # | Priority | Issue | Status | Evidence |
|---|----------|-------|--------|----------|
| I1 | CRITICAL | Train/serve feature-schema mismatch — `TrainModel.py` produces 15 features; `views.py` dummy models expect a different list → script-trained models cannot be served by the API. | OPEN | `TrainModel.py:460–473` vs `views.py:976–986` |
| I2 | CRITICAL | ML evaluation invalid for time series — `train_test_split` shuffles the series (future leaks into train); all accuracy numbers are **UNVERIFIED / HISTORICAL**. | OPEN | `TrainModel.py:594–605` |
| I3 | CRITICAL | Prediction capability currently fictional — **zero model artifacts on disk** (`Scripts/Models/` doesn't exist); predictions fall back to dummy models; README accuracies unreproducible. | OPEN | `ls api/Scripts/` (path renamed `Backend/` → `api/` in Task 04) |
| I4 | HIGH | Security misconfig — `CORS_ALLOW_ALL_ORIGINS = True` (overrides whitelist); `DEBUG = True` hardcoded; no env/dotenv loading. | OPEN | `api/StockVibePredictor/settings.py` (path renamed in Task 04) |
| I5 | HIGH | Database identity crisis — SQLite active / Postgres in `.env.example` (no driver) / Postgres in dead `Settings/Base.py` / MongoDB target undecided. No migrations, no db file. | OPEN | `settings.py`, `.env.example`, `Settings/Base.py` |
| I6 | HIGH | `views.py` monolith — 3,647 lines mixing 25+ endpoints, ML, caching, trading, watchlist, monitoring; duplicates `TrainModel.py` logic. | OPEN | `wc -l api/Apps/StockPredict/views.py` (path renamed in Task 04) |
| I7 | HIGH | Git bloat & tracked artifacts — 36 MB CSV + `node_modules` blobs in history; logs/PDFs/`.DS_Store`/`.gitconfig` were tracked. | **PARTIALLY RESOLVED (Task 01)** — logs/`.DS_Store`/`.gitconfig` untracked; PDFs archived. History slimming deferred (no rewrite without explicit approval). | `git log` / Task 01 |
| I8 | MEDIUM | Broken/dead code — `Api/V1/views.py` imports from empty `Services/__init__.py`; dead Celery (no `Celery()` app); docker Makefile targets reference nonexistent files; pytest config points to nonexistent settings module. | OPEN | audit §F |
| I9 | MEDIUM | Frontend hardcoding — API base URL duplicated in ≥2 files; no central API client; no router; `ui/Apps/Dashboard` remnant (path renamed in Task 04). | OPEN | `App.js:119`, `CompanyEssentials.js:12` |
| I10 | MEDIUM | Naming violations repo-wide vs `ProjectNaming.md` — PascalCase folders/files. | OPEN | naming spec |
| I11 | LOW | Metadata drift — `requires-python >=3.8` but Django 5.2 needs ≥3.10; MIT claimed but no LICENSE; `.gitignore` "YourUsername" placeholder; two GitHub repo URLs; ChangeLog dates predate first commit. | OPEN | `pyproject.toml`, root listing |
| I12 | LOW | Error-handling style — bare `except:` clauses; `warnings.filterwarnings("ignore")`; swallowed fallbacks; `ApiUsage` stores client IPs. | OPEN | grep evidence |

## Notes

* **Real-money / brokerage / payment / wallet:** none exist — **intentionally safe** (see ADR-008).
* **`origin/Group` branch:** unaudited and excluded from current architecture (see ADR-009).
* Fixes belong to their scheduled phases (Task 02+); this file is the tracking index.
