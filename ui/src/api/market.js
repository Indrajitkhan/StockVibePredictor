/**
 * Market-level API surface (indices, market overview).
 */

import api from "./client";

/** GET /market/overview/ — real index data (S&P 500, NASDAQ, DOW, VIX, NIFTY, SENSEX). */
export const getMarketOverview = () =>
  api.get("/market/overview/", { fallbackMessage: "Market overview is unavailable." });

/** GET /system/health/ — backend health/status. */
export const getSystemHealth = () =>
  api.get("/system/health/", { fallbackMessage: "System status is unavailable." });

const marketApi = { getMarketOverview, getSystemHealth };

export default marketApi;
