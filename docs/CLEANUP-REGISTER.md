# Task 05 — Cleanup Register

Checkpoint: started from `397805b` (`refactor: migrate backend and frontend to api and ui`) on `main`.

Classification key: **ACTIVE**, **REQUIRED INFRASTRUCTURE**, **HISTORICAL**, **REDUNDANT**, **OBSOLETE**, **GENERATED**, **SUSPICIOUS**.

| Item | Decision | Reason | Evidence | Destination |
| ---- | -------- | ------ | -------- | ----------- |
| `Backend/` | Moved (Task 04) | Target architecture | Git R100 rename | `api/` |
| `Frontend/` | Moved (Task 04) | Target architecture | Git R100 rename | `ui/` |
| `Docs/` | Moved | Naming consistency | Git-aware case-only rename via temp path (macOS) | `docs/` |
| `Tests/` | Moved | Naming consistency | Git-aware case-only rename | `tests/` |
| `Postman/` | **KEEP + MOVE** | Newman is the live API regression suite (15 requests / 17 assertions). Django `tests.py` is an empty scaffold. | Task 03/04 Newman baseline; collection still targets live `/api/` routes | `postman/` |
| `postman/Collections/StockVibePredictor_API_v2.postman_collection.json` | Retained | Canonical collection | Consumed by Newman / `run_tests.sh` | `postman/Collections/` |
| `postman/Environments/*` | Retained | Dev/Staging/Production env files; no local filesystem paths | JSON env vars only | `postman/Environments/` |
| `postman/Examples/`, `postman/Scripts/` | Retained | Examples + Newman runner | `run_tests.sh`, `setup.py` | `postman/` |
| `postman/Tests/*.json` | Removed | Unreferenced JSON descriptors; Newman does not consume them | No code/CI references | deleted |
| `docs/API-Endpoints.json` | Removed | Duplicate Postman-format 10-request collection; not Newman’s source | Canonical 15-request collection in `postman/` | deleted |
| `ui/Apps/Dashboard/` | Removed | README + unused `react-emoji-render` package.json only; no source, no imports | Tree + grep | deleted |
| `api/package.json`, `api/package-lock.json` | Removed | Misplaced Node manifest for unused `react-emoji-render` | No JS under `api/` | deleted |
| `api/setup.py` | Removed | Duplicate packaging metadata | Root `pyproject.toml` is authoritative | deleted |
| Makefile Docker targets | Removed | No Dockerfile / docker-compose in repo | `find` / glob | Makefile help + recipes cleaned |
| `.github/CODEOWNERS` | Updated | Own current layout | Paths `/api/` `/ui/` `/docs/` `/tests/` `/postman/` | in place |
| `ui/craco.config.js` | Updated | Jest roots after `Tests/` → `tests/` | `../tests/`, `SetupTests.js`, `../tests/Backend/` | in place |
| `docs/API.md`, `docs/ChangeLog.md`, `docs/Frontend-Integration.md`, `docs/Migration-Notes.md` | Retained | Historical / unverified; not silently deleted | Index flags UNVERIFIED | `docs/` |
| `docs/archive/*.pdf` | Retained | College-era historical evidence | Task 01 archive | `docs/archive/` |
| `tests/Backend/*.py`, `tests/e2e/EnterpriseTest.py` | Retained | Incomplete or live-network scripts; not deleted for low coverage | Inventory | `tests/` |
| Fallback `api/Scripts/Models/universal_model_*.pkl` | Not tracked | GENERATED import-time dummy models (R-ML-02). Not genuine trained artifacts. | gitignored; Task 02–04 notes | local-only |
| `.venv/`, `node_modules/`, `ui/build/`, `api/db.sqlite3`, `api/.env`, logs, `__pycache__` | Not tracked | GENERATED / local | `.gitignore` | local-only |
| Legacy `.gitignore` `Frontend/*` and `Backend/Logs/` rules | Retained (inert) | Harmless compatibility patterns; current `ui/` / `api/Logs/` rules also exist | `.gitignore` | in place |
| Legacy `Frontend` skip/exclude in `pyproject.toml` | Retained + `ui` added | Inert leftover plus current frontend path | coverage/mypy/flake8/bandit | in place |
| Root `LICENSE` | Missing | MIT claimed in badges/`pyproject.toml` | no file | **deferred** |
| Root `requirements.txt` | Not created | Actual manifest is `api/requirements.txt` | tree | n/a |

## Postman conclusion

**KEEP + MOVE** (`Postman/` → `postman/`), then **REDUCE** dead `Tests/` JSON only.

Do not delete the collection because pytest is empty. Newman remains the API regression gate.

## Documentation identity

Root `README.md` was rewritten in place to describe the **then-current** tree (`api/`, `ui/`, `docs/`, `tests/`, `postman/`), strip Next.js / Docker / production-accuracy marketing as **UNVERIFIED**, and point routes at `api/Apps/StockPredict/urls.py` (later normalized to `api/apps/...` in Task 06).

College-era training copy and `docs/API.md` remain as historical text, explicitly labeled.

## Out of scope (not started in Task 05)

ML fallback quarantine (R-ML-02), schema leakage, CRA → Vite, SQLite → MongoDB, dependency upgrades.

---

# Task 06 — Internal Structure, Naming & JS/JSX Normalization

Checkpoint: started from `af2cbb9` (`chore: update repository identity to DibakarArchives`) on `main`. Task 05 remains `199afe9`.

| Item | Decision | Reason | Evidence | Destination |
| ---- | -------- | ------ | -------- | ----------- |
| `api/Apps/` | Moved | Directory casing | Git-aware rename via temp path | `api/apps/` |
| `api/apps/StockPredict/Api/` | Moved | Directory casing | Git-aware rename | `api/apps/StockPredict/api/` |
| `api/apps/StockPredict/Config/` | Moved | Directory casing | Git-aware rename | `api/apps/StockPredict/config/` |
| `api/apps/StockPredict/Monitoring/` | Moved | Directory casing | Git-aware rename | `api/apps/StockPredict/monitoring/` |
| `api/apps/StockPredict/Tasks/` | Moved | Directory casing | Git-aware rename | `api/apps/StockPredict/tasks/` |
| `api/apps/StockPredict/Services/` | **KEEP** | Requested target is PascalCase `Services/` | Task 06 brief; relative imports stay `.Services` | unchanged |
| `api/Logs/` | **KEEP** | Runtime/history logs, already gitignored. Settings, `TrainModel.py`, Makefile use `api/Logs`. No `api/apps/api/` package exists. | settings `Logs/StockPredict.log`; `.gitignore` `api/Logs/` | `api/Logs/` |
| `api/apps/__init__.py` | Added | Make dotted `apps.StockPredict` imports work after `Apps/` → `apps/` | empty package init | `api/apps/__init__.py` |
| `ui/Public/` | Moved | Directory casing | Git-aware rename | `ui/public/` |
| `ui/Src/` | Moved | Directory casing | Git-aware rename | `ui/src/` |
| `ui/src/Components/` | Moved | Directory casing | Git-aware rename | `ui/src/components/` |
| `ui/src/components/CSS/` | Moved | Directory casing | Git-aware rename | `ui/src/components/css/` |
| `ui/src/components/Javascript/` | Moved | Directory casing + shorter name | Git-aware rename | `ui/src/components/js/` |
| `ui/src/Database/` | Moved | Directory casing | Git-aware rename | `ui/src/database/` |
| JSX-containing `.js` files | Renamed `.jsx` | Files that contain JSX markup | `App.jsx`, `index.jsx`, component files, `SetupTests.jsx`, `App.test.jsx` | same folders |
| Pure JS files | **KEEP `.js`** | No JSX | `ReportWebVitals.js`, `StockDatabase.js`, `tests/integration/Api.test.js` | unchanged |
| `ui/src/App.js` | Moved + renamed | Mandatory Task 06 rename | unique `App.jsx` | `ui/src/App.jsx` |

Do not treat historical Task 05 path strings in this register as the live tree. Current routes live at `api/app/StockPredict/urls.py` (moved again in Task 07).

---

# Task 07 — Makefile Architecture + Final API Structure

Checkpoint: started from `20284f7` (`refactor: normalize internal structure and js jsx naming`) on `main`.

Classification key unchanged: **ACTIVE**, **REQUIRED INFRASTRUCTURE**, **HISTORICAL**, **REDUNDANT**, **OBSOLETE**, **GENERATED**, **SUSPICIOUS**.

| Item | Decision | Reason | Evidence | Destination |
| ---- | -------- | ------ | -------- | ----------- |
| Root `Makefile` | Rewritten | Orchestrator delegating to `api/Makefile` + `ui/Makefile` (SHADE-ai/VisionCore pattern) | old root Makefile duplicated recipes and `make clean` failed (`rm: node_modules: Directory not empty`) | root `Makefile` |
| `api/Makefile` | Created | Backend/Django commands; shared root `.venv` (Python 3.11); `--noreload` documented (SIGTERM defect deferred) | new file | `api/Makefile` |
| `ui/Makefile` | Created | CRA/CRACO commands; retry-verified clean (node_modules race proven on macOS) | new file | `ui/Makefile` |
| `make clean` failure | Fixed | Root cause: macOS fails a single recursive `rm` when a directory is touched mid-removal; retry + verification loop, no `\|\| true` | reproduced exit 2 with empty `node_modules` left behind | safe `clean` in all three Makefiles |
| `api/apps/` | Moved | Requested `apps → app` | `git mv` (R100) | `api/app/` |
| `api/app/StockPredict/Services/` | Moved | Requested `Services → services` (supersedes Task 06 keep) | `git mv` via temp path (case-only on APFS) | `api/app/StockPredict/services/` |
| `api/Logs/` | Moved | Requested `Logs → app/api/logs`; runtime logs remain gitignored | plain `mv` (contents untracked), `.gitignore` updated | `api/app/api/logs/` |
| `api/Scripts/` | Moved | Requested `Scripts → app/api/scripts` | `git mv` for `TrainModel.py`; artifacts untracked | `api/app/api/scripts/` |
| Python package identity | Updated | `apps.StockPredict` → `app.StockPredict` in settings/urls/apps.py/views/CompanyEssentials loggers | Django app label remains `StockPredict` | in place |
| `TrainModel.py` paths | Updated | `Config` now resolves Models/Performance/Validation/logs under `api/app/api/` | `MODELS_DIR = BASE_DIR / "app" / "api" / "scripts" / "Models"` | in place |
| `pyproject.toml` | Updated | entry point `api.app.api.scripts.TrainModel:main`, packages `app*`, pytest `api/app` | packaging/isort/pytest sections | in place |
| Fallback `.pkl` artifacts | Preserved (not tracked) | GENERATED import-time dummy models (R-ML-02); not genuine; not retrained | still gitignored | `api/app/api/scripts/Models/` (local-only) |
| `api/app/api/` | NOT a Django app | Structural/support location per Task 07 brief §14; no `apps.py`/`models.py` added | no `INSTALLED_APPS` entry | support dir |

ML fallback quarantine (R-ML-01/02), train/serve schema mismatch, `TimeSeriesSplit`, and the SIGTERM/autoreload defect remain **deferred** to the next engineering phase.

---

# Task 08 — Root-Level Architecture Cleanup

Checkpoint: started from `0dd5da4` (`chore: finalize makefile and api structure`) on `main`.

Classification key unchanged: **ACTIVE**, **REQUIRED INFRASTRUCTURE**, **HISTORICAL**, **REDUNDANT**, **OBSOLETE**, **GENERATED**, **SUSPICIOUS**.

| Item | Decision | Reason | Evidence | Destination |
| ---- | -------- | ------ | -------- | ----------- |
| Root `.venv/` | Moved | Backend-specific environment; root represents the project, not one dev environment (Task 08 §3) | `mv` (same volume, untracked); 30 embedded-path files patched (`bin/activate*`, `pyvenv.cfg`, console scripts); env verified in place (Python 3.11.14, Django 5.2.5, DRF 3.16.1) | `api/.venv/` |
| Root `.pytest_cache/` | Moved | pytest is a backend concern (Task 08 §4); prevented root regeneration via `pyproject.toml` `cache_dir = "api/.pytest_cache"` | `mv`; `git check-ignore` confirms ignored | `api/.pytest_cache/` |
| `.github/` (incl. `CODEOWNERS`) | Removed | Explicit owner request (Task 08 §5); may be recreated later | `git rm -r .github` (tracked deletion); not recreated, no replacement mechanism, no Actions added | removed |
| `tests/Backend/` | Renamed | Sole remaining PascalCase directory; repo-wide lowercase convention (Task 06) | `git mv` via temp path (APFS case-only limitation); `ui/craco.config.js` Jest ignore updated | `tests/backend/` |
| `.DS_Store` (root, `ui/`) | Removed | macOS Finder junk; gitignored; regenerates harmlessly | `rm`; `git check-ignore .DS_Store` | removed |
| `pyproject.toml` | Updated | pytest `cache_dir` pinned to `api/.pytest_cache`; `Repository` URL `https://https://` typo corrected | `[tool.pytest.ini_options]`, `[project.urls]` | in place |
| `api/Makefile` | Updated | `VENV ?= .venv`; venv now created inside `api/` (not at root); help text updated | `$(PY)` rule creates `api/.venv` | in place |
| Root `Makefile` | Updated | Help text no longer claims "shared root .venv" | `make install` description | in place |
| `README.md` | Updated | Tree (no `.github/`, `tests/backend/`, `api/.venv/` artifacts note), install help, startup `source .venv/bin/activate` from `api/` | Task 08 pass; historical sections preserved | in place |
| `docs/README.md` | Updated | Index now covers Task 08 | CLEANUP-REGISTER row | in place |
| `api/app/api/` | Kept (verified) | Structural/support dir, NOT a Django app (Task 08 §7) — no `apps.py`/`models.py`, not in `INSTALLED_APPS` | audit | in place |
| `postman/` | Kept | Referenced by root Makefile `api-regression` + Newman; no structural reason to move | audit | in place |
| Fallback `.pkl` artifacts | Preserved (not tracked) | GENERATED import-time dummy models (R-ML-02); not retrained (Task 08 hard rule) | still gitignored | `api/app/api/scripts/Models/` (local-only) |

Historical documentation (`docs/RUNTIME-BASELINE.md`, earlier CLEANUP-REGISTER/ADR sections) intentionally retains old path strings (e.g. repo-root `.venv`, `tests/Backend/`, `.github/CODEOWNERS`) — provenance is preserved; only **current**-architecture references were updated.
