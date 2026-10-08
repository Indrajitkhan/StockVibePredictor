/**
 * Portfolio — future-ready shell.
 *
 * No real portfolio backend exists. No fabricated holdings, values,
 * or P&L. This prepares the architecture without pretending the
 * functionality exists.
 */

import React from "react";
import { PlaceholderView } from "../components/shared/StateViews";

const PortfolioPage = () => (
  <div className="sv-page">
    <div className="sv-page-header">
      <h1 className="sv-page-title">Portfolio</h1>
      <p className="sv-page-subtitle">Portfolio intelligence, once your investment account is enabled.</p>
    </div>

    <section className="sv-card">
      <PlaceholderView
        badge="Not connected"
        title="Portfolio intelligence is not connected yet."
        message="Once your investment account is enabled, this space will show a live picture of your positions and risk."
        items={[
          "Capital allocation",
          "Current positions",
          "Realized P&L",
          "Unrealized P&L",
          "Risk exposure",
          "AI portfolio insights",
        ]}
      />
    </section>
  </div>
);

export default PortfolioPage;
