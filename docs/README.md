# StockVibePredictor — Documentation Index

This index distinguishes **current** documentation from **historical** material. Statuses:
`IMPLEMENTED`, `EXPERIMENTAL`, `UNVERIFIED`, `PLANNED`, `BLOCKED`, `REQUIRES REVIEW`, `HISTORICAL`.

## Current / Authoritative

| Document | Purpose | Status |
|----------|---------|--------|
| [`ARCHITECTURE-DECISIONS.md`](./ARCHITECTURE-DECISIONS.md) | ADR log — audit baseline, investment-platform direction, Group-branch exclusion, hygiene actions. | IMPLEMENTED |
| [`PRODUCT-VISION.md`](./PRODUCT-VISION.md) | Long-term investment-platform direction, principles, gated roadmap. | PLANNED (vision) |
| [`CRITICAL-ISSUES.md`](./CRITICAL-ISSUES.md) | Prioritized open issues from the Phase 0/1 audit. | IMPLEMENTED (tracking) |
| [`RUNTIME-BASELINE.md`](./RUNTIME-BASELINE.md) | Task 02 runtime facts + Task 03 test-infra/SQLite baseline + Task 04 structural migration record (`Backend → api`, `Frontend → ui`). Historical path strings from those tasks are preserved. | IMPLEMENTED (verified 2026-10-06 / 2026-10-07) |
| [`CLEANUP-REGISTER.md`](./CLEANUP-REGISTER.md) | Task 05 classification plus Task 06 internal casing / JS↔JSX decisions plus Task 07 Makefile architecture / final API structure plus Task 08 root-level architecture cleanup. | IMPLEMENTED (Task 05–08) |

## Legacy / Needs Accuracy Pass

| Document | Purpose | Status |
|----------|---------|--------|
| `API.md` | API documentation. | UNVERIFIED — badge claims "Production Ready" (false; debug on, CORS open, dummy models, no DB). |
| `ChangeLog.md` | Changelog. | UNVERIFIED — dates (2024-12-07) predate the repo's first commit (2025-04-23). |
| `Migration-Notes.md` | Alias-endpoint migration notes. | UNVERIFIED |
| `Frontend-Integration.md` | Frontend integration notes. | UNVERIFIED |

## Historical (preserved, not current)

| Path | Purpose | Status |
|------|---------|--------|
| `archive/Documentation.pdf` | College-era documentation. | HISTORICAL — accuracy claims UNVERIFIED. |
| `archive/Presentation.pdf` | College-era presentation. | HISTORICAL — accuracy claims UNVERIFIED. |

---

**Task 01 note:** this index is a minimal landing page. Legacy docs above are **not yet**
rewritten — that is scheduled for the documentation-reconstruction phase. Until then, treat
anything not under "Current / Authoritative" as potentially stale.
