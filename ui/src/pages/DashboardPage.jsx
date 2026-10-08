/**
 * Dashboard — the first real screen.
 *
 * Market Overview (real indices from /market/overview/), Market Pulse
 * (derived from the same real data — no fabricated breadth/momentum),
 * Watchlist preview (local UI state), and an honest AI Insights teaser.
 * Unavailable metrics render as "—", never invented.
 */

import React, { useMemo } from "react";
import { Link } from "react-router-dom";
import useApi from "../hooks/useApi";
import { getMarketOverview } from "../api/market";
import { LoadingView, ErrorView } from "../components/shared/StateViews";
import { useWatchlist } from "./WatchlistPage";

const fmt = (value, digits = 2) =>
  value === null || value === undefined || Number.isNaN(Number(value))
    ? "—"
    : Number(value).toLocaleString("en-US", {
        minimumFractionDigits: digits,
        maximumFractionDigits: digits,
      });

const fmtPct = (value) => {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  const v = Number(value);
  return `${v > 0 ? "+" : ""}${v.toFixed(2)}%`;
};

const tone = (value) =>
  value === null || value === undefined || Number.isNaN(Number(value))
    ? ""
    : Number(value) > 0
    ? "sv-num-positive"
    : Number(value) < 0
    ? "sv-num-negative"
    : "";

const INDEX_ORDER = ["^GSPC", "^IXIC", "^DJI", "^VIX", "^NSEI", "^BSESN"];

const DashboardPage = () => {
  const overview = useApi(getMarketOverview, { loadOnMount: true });
  const { symbols: watchlist } = useWatchlist();

  const indices = useMemo(() => {
    const marketData = overview.data?.market_data || {};
    return INDEX_ORDER.map((key) => ({ key, ...marketData[key] })).filter(
      (i) => i && i.name
    );
  }, [overview.data]);

  const pulse = useMemo(() => {
    if (!overview.data || indices.length === 0) return null;
    const usable = indices.filter((i) => i.change_percent !== undefined);
    if (usable.length === 0) return null;
    const up = usable.filter((i) => i.change_percent > 0).length;
    return {
      up,
      total: usable.length,
      breadth: Math.round((up / usable.length) * 100),
      sentiment: overview.data.market_sentiment || "—",
      session: overview.data.trading_session || "—",
      timestamp: overview.data.timestamp,
    };
  }, [overview.data, indices]);

  return (
    <div className="sv-page">
      <div className="sv-page-header">
        <h1 className="sv-page-title">Dashboard</h1>
        <p className="sv-page-subtitle">Here's what matters in the market today.</p>
      </div>

      {/* ─── Market Overview ─── */}
      <section className="sv-card" aria-labelledby="dash-overview">
        <div className="sv-card-header">
          <div>
            <h2 id="dash-overview" className="sv-card-title">
              Market Overview
            </h2>
            <p className="sv-card-subtitle">Major indices, live from the market data service</p>
          </div>
          {overview.data?.timestamp && (
            <span className="sv-badge">
              Updated{" "}
              {new Date(overview.data.timestamp).toLocaleTimeString([], {
                hour: "2-digit",
                minute: "2-digit",
              })}
            </span>
          )}
        </div>

        {overview.loading && <LoadingView label="Loading market data..." />}

        {overview.error && (
          <ErrorView message={overview.error.message} onRetry={overview.refetch} />
        )}

        {!overview.loading && !overview.error && indices.length === 0 && (
          <p className="sv-empty-text">
            No index data is currently available from the market data service.
          </p>
        )}

        {!overview.loading && indices.length > 0 && (
          <ul className="sv-index-grid" aria-label="Major indices">
            {indices.map((idx) => (
              <li key={idx.key}>
                <Link
                  to={`/stocks/${encodeURIComponent(idx.key)}`}
                  className="sv-index-tile"
                >
                  <span className="sv-metric">
                    <span className="sv-metric-label">{idx.name}</span>
                    <span className="sv-metric-value">
                      {idx.error ? "—" : fmt(idx.price)}
                    </span>
                    {!idx.error && (
                      <span className={`sv-num ${tone(idx.change_percent)}`}>
                        {fmtPct(idx.change_percent)}
                      </span>
                    )}
                  </span>
                </Link>
              </li>
            ))}
          </ul>
        )}
      </section>

      {/* ─── Market Pulse + AI teaser ─── */}
      <div className="sv-two-col">
        <section className="sv-card" aria-labelledby="dash-pulse">
          <div className="sv-card-header">
            <h2 id="dash-pulse" className="sv-card-title">
              Market Pulse
            </h2>
            <span className="sv-badge">{pulse ? `Session ${pulse.session}` : "—"}</span>
          </div>

          {overview.loading && <LoadingView label="Loading market data..." />}
          {overview.error && <ErrorView message={overview.error.message} onRetry={overview.refetch} />}

          {pulse && (
            <div className="sv-pulse-grid">
              <div className="sv-metric">
                <span className="sv-metric-label">Index Breadth</span>
                <span className="sv-metric-value">
                  {pulse.up}/{pulse.total} up
                </span>
              </div>
              <div className="sv-metric">
                <span className="sv-metric-label">Breadth %</span>
                <span className="sv-metric-value">{fmt(pulse.breadth, 0)}%</span>
              </div>
              <div className="sv-metric">
                <span className="sv-metric-label">Market Regime</span>
                <span className="sv-metric-value" style={{ font: "var(--sv-text-subheading)" }}>
                  {pulse.sentiment}
                </span>
              </div>
            </div>
          )}

          {!overview.loading && !overview.error && !pulse && (
            <p className="sv-empty-text">
              Market breadth becomes available once index data loads.
            </p>
          )}
        </section>

        <section className="sv-card" aria-labelledby="dash-ai">
          <div className="sv-card-header">
            <h2 id="dash-ai" className="sv-card-title">
              AI Market View
            </h2>
          </div>
          <p className="sv-card-subtitle" style={{ marginBottom: "var(--sv-space-4)" }}>
            AI Insights are being connected to the intelligence engine.
          </p>
          <Link to="/insights" className="sv-btn sv-btn-sm">
            View AI Insights
          </Link>
        </section>
      </div>

      {/* ─── Watchlist preview ─── */}
      <section className="sv-card" aria-labelledby="dash-watchlist">
        <div className="sv-card-header">
          <div>
            <h2 id="dash-watchlist" className="sv-card-title">
              Watchlist
            </h2>
            <p className="sv-card-subtitle">Symbols you're tracking (stored locally in this browser)</p>
          </div>
          <Link to="/watchlist" className="sv-btn sv-btn-sm">
            Open Watchlist
          </Link>
        </div>

        {watchlist.length === 0 ? (
          <div className="sv-empty">
            <p className="sv-empty-title">Nothing here yet.</p>
            <p className="sv-empty-text">
              Add symbols from the Watchlist page to see them here.
            </p>
          </div>
        ) : (
          <ul className="sv-watch-row-list">
            {watchlist.map((symbol) => (
              <li key={symbol}>
                <Link to={`/stocks/${encodeURIComponent(symbol)}`} className="sv-watch-row">
                  <span className="sv-num sv-watch-symbol">{symbol}</span>
                  <span className="sv-watch-meta">
                    Open analysis
                  </span>
                </Link>
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
};

export default DashboardPage;
