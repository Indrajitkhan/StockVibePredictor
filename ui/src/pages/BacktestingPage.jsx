/**
 * Backtesting — engine shell only.
 *
 * The backtesting engine does not exist. NO fake results, NO
 * misleading charts, NO strategy performance claims.
 */

import React from "react";
import { PlaceholderView } from "../components/shared/StateViews";

const BacktestingPage = () => (
  <div className="sv-page">
    <div className="sv-page-header">
      <h1 className="sv-page-title">Backtesting</h1>
      <p className="sv-page-subtitle">Evaluate strategies against historical data, once the engine is live.</p>
    </div>

    <section className="sv-card">
      <PlaceholderView
        badge="Engine connection pending"
        title="Backtesting engine connection pending."
        message="This area will let you define a strategy, universe, and execution assumptions, then run rigorous historical simulations with honest metrics."
        items={[
          "Strategy and universe definition",
          "Date range, capital, and position sizing",
          "Transaction costs and slippage modeling",
          "Results: return, CAGR, Sharpe, max drawdown, win rate, turnover",
        ]}
      />
    </section>
  </div>
);

export default BacktestingPage;
