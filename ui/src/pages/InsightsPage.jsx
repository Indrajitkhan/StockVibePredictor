/**
 * AI Insights — UI foundation only.
 *
 * The backend does not provide market-level AI analysis yet, so this
 * page renders a polished empty state. NO fabricated AI observations,
 * regimes, or risk narratives.
 */

import React from "react";
import { PlaceholderView } from "../components/shared/StateViews";

const InsightsPage = () => (
  <div className="sv-page">
    <div className="sv-page-header">
      <h1 className="sv-page-title">AI Insights</h1>
      <p className="sv-page-subtitle">AI market intelligence, once connected.</p>
    </div>

    <section className="sv-card">
      <PlaceholderView
        badge="In preparation"
        title="AI Insights are being connected to the intelligence engine."
        message="This area will surface market regime detection, key observations, risk factors, and what-to-watch signals — all derived from real data pipelines."
        items={[
          "Market regime classification (bullish / neutral / defensive)",
          "Key observations derived from live market data",
          "Risk factor monitoring across tracked symbols",
          "What-to-watch alerts from the intelligence engine",
        ]}
      />
    </section>
  </div>
);

export default InsightsPage;
