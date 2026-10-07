# Architecture Decisions — StockVibePredictor

This file is the project's **Architecture Decision Record (ADR) log**. Each entry is numbered
and dated. Decisions are made deliberately and changed only by a new ADR that explicitly
supersedes the earlier one.

**Status vocabulary used throughout this project:**

```text
IMPLEMENTED   — code exists and has been verified to run
EXPERIMENTAL  — code exists but is not yet validated as correct
UNVERIFIED    — claimed but not yet reproduced or tested
PLANNED       — intended but not yet built
BLOCKED       — cannot proceed until a dependency/gate is resolved
REQUIRES REVIEW — needs business / legal / regulatory sign-off before any implementation
```

---

## ADR-000 — Phase 0/1 Audit Baseline (2026-10-06)

**Status: Implemented (as evidence baseline)**

A full read-only audit of the repository was completed. Key evidence it established:

* Frontend is **React 18 + JavaScript/JSX only** — **zero TypeScript** (0 `.ts`/`.tsx` files,
  0 `tsconfig*`, 0 TypeScript deps). The project's frontend language standard is already met.
* Backend is **Django 5 + DRF**; active DB is **SQLite** (no db file, no migrations).
  `.env.example` implies Postgres (no driver installed); MongoDB is the stated target — undecided.
* ML evaluation is **invalid for time series** (random `train_test_split` shuffles the series);
  all accuracy numbers are **UNVERIFIED / HISTORICAL** (no model artifacts on disk).
* A train/serve **feature-schema mismatch** makes script-trained models unservable by the API.
* No brokerage integration, no payment processing, **no real-money trading path** — safe.

**Decision:** the audit is the accepted evidence baseline. All later work builds on it.

---

## ADR-001 — Long-Term Product Direction: AI-Native Investment Platform

**Status: PLANNED / REQUIRES BUSINESS + LEGAL + REGULATORY REVIEW**

StockVibePredictor will evolve beyond a stock-prediction/research app toward a serious
**AI-native market and investment platform**. The conceptual long-term flow:

```text
Market Data → Data Intelligence → Quantitative Analysis → AI / ML
  → Signals + Forecasts → Strategy Engine → Risk Engine → Backtesting
  → Paper Trading → Portfolio Intelligence → Broker / Custodian Integration
  → Controlled Live Execution → User Investment Account
```

An eventual **investment center** would let eligible users add capital, keep a cash reserve,
have validated strategies manage the investable portion, see clear P&L and full transaction
history, and withdraw available capital subject to settlement/liquidity/risk/regulatory
constraints — with transparent records and disclosures.

> **This is a long-term destination, not authorization to implement real-money functionality now.**

---

## ADR-002 — Money Is a First-Class Domain (Ledger, Not a Balance Field)

**Status: PLANNED**

Future financial functionality must **never** reduce user money to a simple balance field
(`user.balance += amount`). The eventual architecture must conceptually support a proper
financial ledger / domain model, including (at minimum):

```text
User, Investment Account, Cash Ledger, Ledger Transaction, Deposit, Withdrawal,
Reserved Cash, Investable Cash, Order, Execution / Fill, Position, Portfolio Valuation,
Realized P&L, Unrealized P&L, Fees, Settlement, Reconciliation, Audit Event, Risk Limits
```

Financial state must be **auditable, reproducible, idempotent where appropriate, reconciliable,
protected against double-spend/double-credit, resistant to partial-transaction failure, and
traceable** from deposit → ledger → investment → execution → position → P&L → withdrawal.

---

## ADR-003 — AI Must Never Have Unrestricted Authority Over Money

**Status: PLANNED — permanent architectural principle**

The AI recommends within defined strategy constraints; it never moves money. The conceptual
control flow is:

```text
AI / Quant Research → Strategy Decision → Portfolio Construction
  → Risk Engine → Policy / Authorization Layer → Execution Gateway → Broker / Custodian
```

Never architect `AI → direct database money mutation` or `AI → unrestricted broker access`.

---

## ADR-004 — User Cash Reserve Must Be Supported

**Status: PLANNED**

Users must be able to designate cash to remain **uninvested** (reserved). Example:

```text
Total account capital     ₹1,00,000
Reserved cash               ₹20,000
Investable capital          ₹80,000
```

The strategy engine operates only within the capital it is actually authorized to invest,
and users set their own capital/risk constraints.

---

## ADR-005 — Withdrawals Derive From Real Financial State

**Status: PLANNED**

"Withdraw anytime" must never mean "instantly liquidate anything regardless of settlement or
market conditions." Withdrawable amount must be derived from: available cash, reserved cash,
pending orders, open positions, liquidation requirements, realized vs. unrealized P&L,
settlement status, fees, broker constraints, and legal/operational restrictions.

The platform must clearly distinguish:

```text
Account Value · Invested Value · Available Cash · Reserved Cash · Withdrawable Amount
Pending Withdrawal · Unrealized P&L · Realized P&L
```

---

## ADR-006 — Platform/Service Fee (Hypothesis Only, Undecided)

**Status: PLANNED / REQUIRES BUSINESS + LEGAL REVIEW**

A small, **disclosed** platform/service fee of approximately **0.05%–0.10%** is an **initial
business hypothesis only**. The fee model is **NOT finalized** and may eventually be a
deposit fee, execution fee, asset-management fee, performance fee, withdrawal fee, a
combination, or another legally appropriate model.

Rules recorded now:

* Do **not** silently implement a fee.
* Do **not** hard-code a percentage into financial logic.
* Do **not** deduct money without an explicit, visible fee calculation.
* Do **not** call it a "tax" unless a qualified legal/accounting determination says it is one —
  label it **Platform Fee** / **Service Fee**.
* The eventual fee system must be **configurable and auditable**.

> **No real-money fee collection is implemented in Task 01.**

---

## ADR-007 — No Deceptive Financial UX

**Status: PLANNED — permanent product philosophy**

The platform must never hide fees, disguise fees as taxes, manipulate users into depositing,
claim guaranteed returns, imply guaranteed AI performance, hide losses, selectively show
profitable periods, manipulate charts, exaggerate historical performance, use misleading
"AI guaranteed profit" language, or obscure withdrawal/settlement conditions. Every material
financial calculation must be explainable.

```text
Transparency > Conversion tricks
Evidence > Marketing claims
User control > Lock-in
```

---

## ADR-008 — Real-Money Boundary & Approval Gates

**Status: PLANNED**

There is currently no live brokerage execution path, and it stays that way. The development
ladder is gated; each transition requires an **explicit review gate**:

```text
Research → Validated ML → Validated Signals → Backtesting → Paper Trading
  → Portfolio Simulation → Broker Integration → Controlled Live Execution
  → Real-Money Investment Infrastructure
```

---

## ADR-009 — `origin/Group` Branch Is Unaudited and Excluded

**Status: IMPLEMENTED (decision recorded)**

`origin/Group` contains divergent, unmerged work (25 files, +1541/−4774 vs `main`), including a
680-line `trainmodel_accuracy_generator.py` whose purpose is generating **presentation/accuracy
report artifacts** (`model_ranking.csv`, `model_cards`, `comprehensive_performance_analysis.png`,
PPT tables) with "dynamic random state" variation. Until independently validated, that
methodology is treated as **suspicious**.

**Decision:**

* **Do NOT merge** `origin/Group`.
* **Do NOT cherry-pick** from it.
* **Do NOT delete** it — preserve as historical/unaudited work.

It is intentionally **excluded from the current architecture** and will be reviewed separately
in a later phase before any part of it is considered.

---

## ADR-010 — Repository Hygiene Actions (Task 01)

**Status: IMPLEMENTED (2026-10-06)**

Small, controlled, history-preserving cleanup based on the audit:

* **Untracked** (kept on disk, now ignored): `Backend/Logs/*` (11 runtime/training logs),
  `.DS_Store`, `.gitconfig`.
* **Moved** (content unchanged, history preserved via `git mv`): `Docs/Documentation.pdf` and
  `Docs/Presentation.pdf` → `Docs/archive/` (historical college artifacts, not current docs).
* **`.gitignore`** extended with a Task-01 section so these are not re-tracked.
* No history rewritten, no branch merged, nothing pushed, no application logic changed.

---

## ADR-011 — Internal Directory Casing and JS/JSX Naming (Task 06)

**Status: IMPLEMENTED (2026-10-07)**

Normalize internal directory names and React source extensions without changing application behavior.

### Directory naming

Use lowercase for normal directories:

```text
api/
apps/
api/                 # StockPredict HTTP package (api/apps/StockPredict/api/)
config/
monitoring/
tasks/
ui/
public/
src/
components/
css/
js/
database/
docs/
tests/
postman/
```

Keep Python package identity where the framework already requires it:

* Django app label remains `StockPredict` (`INSTALLED_APPS = "StockPredict"`).
* `Services/` stays PascalCase because that was the requested target; no extra rename.
* `api/Logs/` stays as the ignored runtime log directory. Settings, `TrainModel.py`, and the Makefile write/read there. There is no `api/apps/api/` package, so logs were not moved into source.

### React source naming

```text
.jsx  → files containing JSX
.js   → pure JavaScript (API/data/util/config modules, hooks with no JSX)
```

No TypeScript (`.ts` / `.tsx`) is introduced.

Mandatory rename: `ui/src/App.js` → `ui/src/App.jsx`.

---

## ADR-012 — Makefile Architecture + Final API Structure (Task 07)

**Status: IMPLEMENTED (2026-10-08)**

### Makefile system (adopted from SHADE-ai / VisionCore)

```text
Makefile           # orchestrator: delegates via make -C api / make -C ui
api/Makefile       # Django backend (shared root .venv, Python 3.11)
ui/Makefile        # CRA/CRACO frontend
```

* Root delegates rather than duplicating recipes; `make test` runs backend + Jest;
  `make api-regression` gates Newman on a health check.
* No lint/format targets: none exist in this repo (no linters in venv; `ui/package.json`
  defines no lint scripts). No training target by design (ML integrity issues open —
  `docs/CRITICAL-ISSUES.md`).
* `make clean` is safe: removes `__pycache__`, `.pytest_cache`, `ui/node_modules/`,
  `ui/build/` only. Never touches `api/.env`, `api/db.sqlite3`, `api/app/api/logs/`,
  `api/app/api/scripts/` artifacts, sources, or Git metadata. No destructive
  `clean-all` variant was added, so there is nothing extra to document.

### Final API structure

```text
api/
├── app/
│   ├── StockPredict/      # Django app (label unchanged: "StockPredict")
│   │   ├── api/V1/ config/ monitoring/ services/ tasks/
│   └── api/               # structural/support (NOT a Django app)
│       ├── logs/           # runtime logs, gitignored
│       └── scripts/        # TrainModel.py + Models/ Performance/ Validation/
├── Makefile
├── manage.py
├── requirements.txt
└── StockVibePredictor/    # settings, urls, wsgi/asgi
```

Moves (all Git-aware; case-only via temp path on macOS):

* `api/apps/` → `api/app/` (package identity `apps.StockPredict` → `app.StockPredict`)
* `api/app/StockPredict/Services/` → `api/app/StockPredict/services/` (supersedes
  ADR-011's keep-PascalCase decision; requested in Task 07)
* `api/Logs/` → `api/app/api/logs/` (runtime logs, still gitignored)
* `api/Scripts/` → `api/app/api/scripts/`

Deliberately unchanged: `StockPredict`, `V1`, `Models`, `Performance`, `Validation`.
