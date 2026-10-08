/**
 * Stock Analysis — /stocks/:symbol
 *
 * Structured analysis hierarchy: price header → chart → analysis
 * sections. Reuses the existing data-backed components
 * (StockChart, AnalysisTab composition, CompanyEssentials,
 * PredictionResult) inside the new product shell.
 *
 * Prediction output is displayed as experimental model output —
 * never as validated or production-grade advice.
 */

import React, { useState, useMemo, lazy, Suspense } from "react";
import { useParams } from "react-router-dom";
import useApi from "../hooks/useApi";
import { predictStock, getChart, getCompanyEssentials } from "../api/stocks";
import { LoadingView, ErrorView, PlaceholderView } from "../components/shared/StateViews";

const StockChart = lazy(() => import("../components/js/StockChart"));
const CompanyEssentials = lazy(() => import("../components/js/CompanyEssentials"));
const PredictionResult = lazy(() => import("../components/js/PredictionResult"));
const SentimentPanel = lazy(() => import("../components/js/SentimentPanel"));
const RiskAssessmentPanel = lazy(() => import("../components/js/RiskAssessmentPanel"));

const TABS = [
  { id: "overview", label: "Overview" },
  { id: "technical", label: "Technical" },
  { id: "fundamentals", label: "Fundamentals" },
  { id: "sentiment", label: "Sentiment" },
  { id: "ai", label: "AI Analysis" },
  { id: "risk", label: "Risk" },
];

const CHART_TIMEFRAMES = [
  { id: "1d", label: "1D" },
  { id: "5d", label: "5D" },
  { id: "1mo", label: "1M" },
  { id: "3mo", label: "3M" },
  { id: "6mo", label: "6M" },
  { id: "1y", label: "1Y" },
];

const fmtPrice = (v) =>
  v === null || v === undefined || Number.isNaN(Number(v))
    ? "—"
    : Number(v).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });

const extractCompanyName = (data, ticker) =>
  data?.market_info?.company_name ||
  data?.market_info?.long_name ||
  data?.market_info?.short_name ||
  data?.company_info?.company_name ||
  data?.company_info?.long_name ||
  ticker.toUpperCase();

const StockAnalysisPage = () => {
  const { symbol } = useParams();
  const ticker = decodeURIComponent(symbol || "").toUpperCase();

  const [activeTab, setActiveTab] = useState("overview");
  const [chartTimeframe, setChartTimeframe] = useState("1mo");

  const prediction = useApi(() => predictStock(ticker), {
    loadOnMount: Boolean(ticker),
    deps: [ticker],
  });
  const chart = useApi(() => getChart(ticker, { timeframe: chartTimeframe }), {
    loadOnMount: Boolean(ticker),
    deps: [ticker, chartTimeframe],
  });
  const company = useApi(() => getCompanyEssentials(ticker), {
    loadOnMount: Boolean(ticker),
    deps: [ticker],
  });

  const analysis = prediction.data?.analysis || null;
  const predictions = prediction.data?.predictions || null;
  const companyName = extractCompanyName(prediction.data, ticker);

  const priceHeader = useMemo(() => {
    const summary = chart.data?.summary;
    if (summary?.latest_price !== undefined) {
      return {
        price: summary.latest_price,
        changePct: summary.change_percent,
        change: summary.change,
      };
    }
    const mi = prediction.data?.market_info;
    if (mi?.current_price !== undefined) {
      return { price: mi.current_price, changePct: mi.change_percent, change: mi.change };
    }
    return null;
  }, [chart.data, prediction.data]);

  const sentimentData =
    analysis?.sentiment || analysis?.sentiment_analysis || null;

  const riskData = analysis
    ? {
        risk_level: analysis.risk_level || analysis.recommendation?.risk_level,
        risk_factors: analysis.risk_factors,
        risk_mitigation: analysis.risk_mitigation,
      }
    : null;

  return (
    <div className="sv-page">
      {/* ─── Price header ─── */}
      <div className="sv-page-header">
        <div className="sv-stock-header">
          <div>
            <h1 className="sv-page-title sv-num">{ticker}</h1>
            <p className="sv-page-subtitle">{companyName !== ticker ? companyName : ""}</p>
          </div>

          {priceHeader && (
            <div className="sv-stock-pricing">
              <span className="sv-num sv-stock-price">${fmtPrice(priceHeader.price)}</span>
              {priceHeader.changePct !== undefined && priceHeader.changePct !== null && (
                <span
                  className={`sv-num sv-num-${Number(priceHeader.changePct) >= 0 ? "positive" : "negative"}`}
                >
                  {Number(priceHeader.changePct) >= 0 ? "▲" : "▼"}{" "}
                  {Number(priceHeader.changePct).toFixed(2)}%
                </span>
              )}
            </div>
          )}
        </div>

        {/* Timeframe selector for the chart */}
        <div className="sv-tf-row" role="group" aria-label="Chart timeframe">
          {CHART_TIMEFRAMES.map((tf) => (
            <button
              key={tf.id}
              type="button"
              className={`sv-tf-btn${chartTimeframe === tf.id ? " active" : ""}`}
              aria-pressed={chartTimeframe === tf.id}
              onClick={() => setChartTimeframe(tf.id)}
            >
              {tf.label}
            </button>
          ))}
        </div>
      </div>

      {/* ─── Price chart (reused component) ─── */}
      <section className="sv-card" aria-label={`${ticker} price chart`}>
        {chart.loading && <LoadingView label="Loading chart data..." />}
        {chart.error && (
          <ErrorView message={chart.error.message} onRetry={chart.refetch} />
        )}
        {!chart.loading && !chart.error && chart.data?.data?.length > 0 && (
          <Suspense fallback={<LoadingView label="Preparing chart..." />}>
            <StockChart data={chart.data.data} ticker={ticker} />
          </Suspense>
        )}
        {!chart.loading && !chart.error && !chart.data?.data?.length && (
          <p className="sv-empty-text">
            No chart data is available for this symbol yet.
          </p>
        )}
      </section>

      {/* ─── Analysis tabs ─── */}
      <div className="sv-tabs" role="tablist" aria-label="Analysis sections" style={{ marginTop: "var(--sv-space-6)" }}>
        {TABS.map((tab) => (
          <button
            key={tab.id}
            type="button"
            role="tab"
            aria-selected={activeTab === tab.id}
            className={`sv-tab-btn${activeTab === tab.id ? " active" : ""}`}
            onClick={() => setActiveTab(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </div>

      <div className="sv-card" style={{ marginTop: "var(--sv-space-4)" }}>
        {/* Overview */}
        {activeTab === "overview" && (
          prediction.loading ? (
            <LoadingView label={`Analyzing ${ticker}...`} />
          ) : prediction.error ? (
            <ErrorView message={prediction.error.message} onRetry={prediction.refetch} />
          ) : (
            <div className="sv-analysis-stack">
              {predictions && (
                <Suspense fallback={<LoadingView label="Loading predictions..." />}>
                  <PredictionResult predictions={predictions} ticker={ticker} selectedTimeframes={["1d", "1w", "1mo"]} />
                </Suspense>
              )}
              {!predictions && (
                <p className="sv-empty-text">
                  Model output is unavailable for this symbol right now.
                </p>
              )}
              <p className="sv-card-subtitle">
                Model outputs are experimental and shown for research only. They
                are not validated financial signals or investment advice.
              </p>
            </div>
          )
        )}

        {/* Technical */}
        {activeTab === "technical" && (
          analysis ? (
            <div className="sv-tech-grid">
              <div className="sv-metric">
                <span className="sv-metric-label">Trend</span>
                <span className="sv-metric-value" style={{ font: "var(--sv-text-subheading)" }}>
                  {analysis.technical?.trend || analysis.trend || "—"}
                </span>
              </div>
              <div className="sv-metric">
                <span className="sv-metric-label">RSI</span>
                <span className="sv-metric-value">
                  {analysis.technical?.rsi !== undefined
                    ? analysis.technical.rsi.toFixed(2)
                    : analysis.technical_indicators?.rsi !== undefined
                    ? Number(analysis.technical_indicators.rsi).toFixed(2)
                    : "—"}
                </span>
              </div>
              <div className="sv-metric">
                <span className="sv-metric-label">MACD</span>
                <span className="sv-metric-value">
                  {analysis.technical_indicators?.macd !== undefined
                    ? Number(analysis.technical_indicators.macd).toFixed(4)
                    : "—"}
                </span>
              </div>
              <div className="sv-metric">
                <span className="sv-metric-label">SMA 20</span>
                <span className="sv-metric-value">
                  {analysis.technical_indicators?.sma_20 !== undefined
                    ? `$${Number(analysis.technical_indicators.sma_20).toFixed(2)}`
                    : "—"}
                </span>
              </div>
              <div className="sv-metric">
                <span className="sv-metric-label">SMA 50</span>
                <span className="sv-metric-value">
                  {analysis.technical_indicators?.sma_50 !== undefined
                    ? `$${Number(analysis.technical_indicators.sma_50).toFixed(2)}`
                    : "—"}
                </span>
              </div>
              <div className="sv-metric">
                <span className="sv-metric-label">Volatility regime</span>
                <span className="sv-metric-value" style={{ font: "var(--sv-text-subheading)" }}>
                  {analysis.technical?.volatility_regime || "—"}
                </span>
              </div>
            </div>
          ) : prediction.loading ? (
            <LoadingView label="Analyzing technicals..." />
          ) : (
            <ErrorView message={prediction.error?.message || "Technical analysis is unavailable."} onRetry={prediction.refetch} />
          )
        )}

        {/* Fundamentals — reused presentation component, data from the
            centralized API layer (not its legacy direct-axios fetch) */}
        {activeTab === "fundamentals" &&
          (company.loading ? (
            <LoadingView label="Loading company fundamentals..." />
          ) : company.error ? (
            <ErrorView message={company.error.message} onRetry={company.refetch} />
          ) : (
            <Suspense fallback={<LoadingView label="Preparing fundamentals..." />}>
              <CompanyEssentials data={company.data} ticker={ticker} />
            </Suspense>
          ))}

        {/* Sentiment — only real analysis.sentiment values; the legacy
            SentimentAnalysis.jsx fabricates mock data and is never used. */}
        {activeTab === "sentiment" &&
          (prediction.loading ? (
            <LoadingView label="Analyzing sentiment..." />
          ) : sentimentData ? (
            <Suspense fallback={<LoadingView label="Preparing sentiment..." />}>
              <SentimentPanel sentiment={sentimentData} expanded={true} />
            </Suspense>
          ) : (
            <PlaceholderView
              title="Sentiment data is unavailable."
              message="Sentiment analysis is being connected to the intelligence engine. No sentiment data is currently provided for this symbol."
            />
          ))}

        {/* AI Analysis — honest not-connected state (no fabricated AI) */}
        {activeTab === "ai" && (
          <PlaceholderView
            title="AI Insights are being connected to the intelligence engine."
            message="Individual-asset AI interpretation will appear here once the intelligence engine is connected. Model outputs shown in Overview are experimental research output, not validated signals."
          />
        )}

        {/* Risk — only real analysis risk fields */}
        {activeTab === "risk" &&
          (prediction.loading ? (
            <LoadingView label="Analyzing risk..." />
          ) : riskData && riskData.risk_level ? (
            <Suspense fallback={<LoadingView label="Preparing risk assessment..." />}>
              <RiskAssessmentPanel riskData={riskData} expanded={true} />
            </Suspense>
          ) : (
            <PlaceholderView
              title="Risk assessment is unavailable."
              message="No risk assessment data is currently provided for this symbol."
            />
          ))}
      </div>
    </div>
  );
};

export default StockAnalysisPage;
