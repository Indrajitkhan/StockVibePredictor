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

Root `README.md` was rewritten in place to describe the **current** tree (`api/`, `ui/`, `docs/`, `tests/`, `postman/`), strip Next.js / Docker / production-accuracy marketing as **UNVERIFIED**, and point routes at `api/Apps/StockPredict/urls.py`.

College-era training copy and `docs/API.md` remain as historical text, explicitly labeled.

## Out of scope (not started)

ML fallback quarantine (R-ML-02), schema leakage, CRA → Vite, SQLite → MongoDB, dependency upgrades, Task 06.
