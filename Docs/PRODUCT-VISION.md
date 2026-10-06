# StockVibePredictor — Product Vision & Roadmap

> **Scope of this document:** direction and intent only. It is **not** an authorization to build
> real-money functionality. Every financial capability listed under "Future" is **PLANNED** and
> gated behind explicit review.

## 1. What This Is (Current System)

StockVibePredictor is, today, a **stock-prediction research prototype**:

* **Frontend:** React 18 + JavaScript/JSX (single-page tab UI, no router). **Status: IMPLEMENTED / UNVERIFIED build.**
* **Backend:** Django 5 + DRF, function-based views, yfinance data, scikit-learn models. **Status: EXPERIMENTAL / partially broken.**
* **Database:** SQLite (no db file, no migrations). MongoDB is the intended target — undecided.
* **ML:** next-bar direction classification. **Status: UNVERIFIED** — evaluation methodology is
  invalid for time series (random shuffle); no model artifacts on disk.

There is **no brokerage integration, no payment processing, and no real-money trading path.**
This must remain true until an explicit, gated phase authorizes otherwise.

## 2. Where It's Going (Future Investment Platform)

The long-term destination is a serious **AI-native market and investment platform** with an
"investment center" where eligible users can:

* add capital to an investment account/wallet through supported funding mechanisms — **PLANNED / REQUIRES REVIEW**
* maintain a desired cash reserve — **PLANNED**
* allow validated strategies to manage the **investable** portion — **PLANNED**
* see portfolio value and P&L clearly — **PLANNED**
* withdraw available capital and/or realized returns subject to actual settlement, liquidity,
  risk, operational, and regulatory constraints — **PLANNED / REQUIRES REVIEW**
* view complete transaction history and understand exactly what happened to their money — **PLANNED**
* understand every platform fee or other applicable charge — **PLANNED / REQUIRES REVIEW**
* receive transparent records and disclosures — **PLANNED**

```text
Market Data → Data Intelligence → Quantitative Analysis → AI/ML → Signals + Forecasts
  → Strategy Engine → Risk Engine → Backtesting → Paper Trading → Portfolio Intelligence
  → Broker/Custodian Integration → Controlled Live Execution → User Investment Account
```

## 3. Non-Negotiable Principles

1. **Money is a first-class domain** — a proper ledger/account model, never a balance field (see ADR-002).
2. **AI never has unrestricted money authority** — risk/policy/execution layers gate all actions (ADR-003).
3. **User cash reserve is supported** — strategies touch only authorized capital (ADR-004).
4. **Withdrawals derive from real financial state** — settlement, positions, constraints (ADR-005).
5. **Fees are transparent and auditable** — never hidden, never mislabeled as a "tax" (ADR-006).
6. **No deceptive financial UX** — no guaranteed-return or "AI guaranteed profit" language (ADR-007).
7. **Real-money features stay behind explicit approval gates** (ADR-008).

## 4. Development Ladder (gated)

```text
Research → Validated ML → Validated Signals → Backtesting → Paper Trading
  → Portfolio Simulation → Broker Integration → Controlled Live Execution
  → Real-Money Investment Infrastructure
```

Each `→` is an explicit review gate. We are at **Research**, and the immediate priority is
making the ML evaluation *valid* (chronological splits, baselines) before any capability claims.

## 5. North Star

```text
Evidence → Architecture → Implementation → Tests → Verification → Checkpoint → Next phase
```

We are building toward a serious AI-native investment platform — **not** fake "AI profit"
marketing software, **not** hidden fees, **not** promised returns, and **never** an AI with
unrestricted access to user money. We build the foundations first.
