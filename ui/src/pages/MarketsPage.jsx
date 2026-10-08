/**
 * Markets — real index data from /market/overview/.
 * Sectors, Top Movers, and Breadth tables that the backend does not
 * yet provide render as honest unavailable sections, never fabricated.
 */

import React, { useMemo } from "react";
import { Link } from "react-router-dom";
import useApi from "../hooks/useApi";
import { getMarketOverview } from "../api/market";
import { LoadingView, ErrorView } from "../components/shared/StateViews";

const fmt = (value, digits = 2) =>
  value === null || value === undefined || Number.isNaN(Number(value))
    ? "—"
    : Number(value).toLocaleString("en-US", {
        minimumFractionDigits: digits,
        maximumFractionDigits: digits,
      });

const fmtPct = (v) => {
  if (v === null || v === undefined || Number.isNaN(Number(v))) return "—";
  const n = Number(v);
  return `${n > 0 ? "+" : ""}${n.toFixed(2)}%`;
};

const tone = (v) =>
  v === null || v === undefined || Number.isNaN(Number(v))
    ? ""
    : Number(v) > 0
    ? "sv-num-positive"
    : Number(v) < 0
    ? "sv-num-negative"
    : "";

const ALL_KEYS = ["^GSPC", "^IXIC", "^DJI", "^VIX", "^NSEI", "^BSESN"];

const NAME_MAP = {
  "^GSPC": "S&P 500",
  "^IXIC": "NASDAQ Composite",
  "^DJI": "Dow Jones",
  "^VIX": "VIX",
  "^NSEI": "NIFTY 50",
  "^BSESN": "SENSEX",
};

const MarketsPage = () => {
  const overview = useApi(getMarketOverview, { loadOnMount: true });

  const rows = useMemo(() => {
    const marketData = overview.data?.market_data || {};
    return ALL_KEYS.map((key) => ({
      key,
      name: marketData[key]?.name || NAME_MAP[key],
      ...marketData[key],
    }));
  }, [overview.data]);

  return (
    <div className="sv-page">
      <div className="sv-page-header">
        <h1 className="sv-page-title">Markets</h1>
        <p className="sv-page-subtitle">Major indices and current market conditions.</p>
      </div>

      <section className="sv-card" aria-labelledby="mkt-indices">
        <div className="sv-card-header">
          <h2 id="mkt-indices" className="sv-card-title">
            Major Indices
          </h2>
          {overview.data?.trading_session && (
            <span className="sv-badge">
              Session {overview.data.trading_session}
            </span>
          )}
        </div>

        {overview.loading && <LoadingView label="Loading market data..." />}
        {overview.error && (
          <ErrorView message={overview.error.message} onRetry={overview.refetch} />
        )}

        {!overview.loading && !overview.error && (
          <div className="sv-table-wrap">
            <table className="sv-table">
              <thead>
                <tr>
                  <th scope="col">Index</th>
                  <th scope="col">Price</th>
                  <th scope="col">Change</th>
                  <th scope="col">Change %</th>
                  <th scope="col"><span className="sv-visually-hidden">Open analysis</span></th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r) => (
                  <tr key={r.key}>
                    <td>{r.error ? r.name : <Link to={`/stocks/${encodeURIComponent(r.key)}`}>{r.name}</Link>}</td>
                    <td className="sv-num">{r.error ? "—" : fmt(r.price)}</td>
                    <td className={`sv-num ${tone(r.change)}`}>{r.error ? "—" : fmt(r.change)}</td>
                    <td className={`sv-num ${tone(r.change_percent)}`}>
                      {r.error ? "—" : fmtPct(r.change_percent)}
                    </td>
                    <td>
                      {!r.error && (
                        <Link className="sv-btn sv-btn-ghost sv-btn-sm" to={`/stocks/${encodeURIComponent(r.key)}`}>
                          Analyze
                        </Link>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      {/* Sections the backend does not yet serve — honest placeholders */}
      <div className="sv-two-col" style={{ marginTop: "var(--sv-space-6)" }}>
        <section className="sv-card" aria-labelledby="mkt-movers">
          <div className="sv-card-header">
            <h2 id="mkt-movers" className="sv-card-title">Top Movers</h2>
          </div>
          <div className="sv-empty">
            <p className="sv-empty-title">Not available yet</p>
            <p className="sv-empty-text">
              Movers require a market-wide screener data source that isn't
              connected. Values will appear only when real data is available.
            </p>
          </div>
        </section>

        <section className="sv-card" aria-labelledby="mkt-breadth">
          <div className="sv-card-header">
            <h2 id="mkt-breadth" className="sv-card-title">Market Breadth</h2>
          </div>
          <div className="sv-empty">
            <p className="sv-empty-title">Not available yet</p>
            <p className="sv-empty-text">
              Advance/decline breadth needs an exchange-wide feed. We won't
              invent it.
            </p>
          </div>
        </section>
      </div>

      <section className="sv-card" style={{ marginTop: "var(--sv-space-6)" }} aria-labelledby="mkt-vol">
        <div className="sv-card-header">
          <h2 id="mkt-vol" className="sv-card-title">Volatility</h2>
        </div>
        {overview.data?.market_data?.["^VIX"] && !overview.data.market_data["^VIX"].error ? (
          <div className="sv-pulse-grid">
            <div className="sv-metric">
              <span className="sv-metric-label">VIX</span>
              <span className="sv-metric-value">{fmt(overview.data.market_data["^VIX"].price)}</span>
            </div>
            <div className="sv-metric">
              <span className="sv-metric-label">Change %</span>
              <span className={`sv-num ${tone(overview.data.market_data["^VIX"].change_percent)}`}>
                {fmtPct(overview.data.market_data["^VIX"].change_percent)}
              </span>
            </div>
          </div>
        ) : (
          <div className="sv-empty">
            <p className="sv-empty-title">VIX data unavailable</p>
            <p className="sv-empty-text">Volatility data will appear when the market data service returns it.</p>
          </div>
        )}
      </section>
    </div>
  );
};

export default MarketsPage;
