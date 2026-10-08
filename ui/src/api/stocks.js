/**
 * Stock-level API surface (analysis, predictions, charts, company data).
 */

import api from "./client";

/**
 * POST /predict/multi/ — multi-timeframe predictions with market info,
 * analysis, and history. NOTE: predictions may come from fallback
 * models; never present them as validated/production-grade.
 */
export const predictStock = (ticker, timeframes = ["1d", "1w", "1mo"]) =>
  api.post(
    "/predict/multi/",
    { ticker, timeframes, include_analysis: true },
    { fallbackMessage: "Stock analysis is unavailable right now.", timeout: 45000 }
  );

/** GET /chart/<ticker>/ — price/volume chart data. */
export const getChart = (ticker, { timeframe = "1mo", chartType = "line", indicators = [], limit = 150 } = {}) =>
  api.get(`/chart/${encodeURIComponent(ticker)}/`, {
    params: {
      timeframe,
      chart_type: chartType,
      ...(indicators.length ? { indicators: indicators.join(",") } : {}),
      optimize: true,
      points_limit: limit,
    },
    fallbackMessage: "Chart data is unavailable.",
  });

/** GET /company/<ticker>/essentials/ — company fundamentals. */
export const getCompanyEssentials = (ticker) =>
  api.get(`/company/${encodeURIComponent(ticker)}/essentials/`, {
    fallbackMessage: "Company fundamentals are unavailable.",
  });

const stocksApi = { predictStock, getChart, getCompanyEssentials };

export default stocksApi;
