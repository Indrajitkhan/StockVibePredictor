/**
 * StockVibe API Client — single centralized HTTP layer.
 *
 * All components must go through this module (or the /api/* wrappers).
 * No component may import axios directly or hardcode a base URL.
 *
 * Base URL resolution order:
 *   1. REACT_APP_API_BASE_URL (environment configuration)
 *   2. http://localhost:8000/api (local Django dev default)
 */

import axios from "axios";

const resolveBaseUrl = () => {
  const fromEnv = process.env.REACT_APP_API_BASE_URL;
  if (fromEnv && fromEnv.trim()) {
    return fromEnv.trim().replace(/\/+$/, "");
  }
  return "http://localhost:8000/api";
};

export const API_BASE_URL = resolveBaseUrl();

const client = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: { "Content-Type": "application/json" },
});

/**
 * Normalize any axios/Django error into a safe, user-facing message.
 * Never exposes raw stack traces, Django error pages, or Python
 * exception text to the user.
 */
export const toUserMessage = (error, fallback = "We couldn't load this data.") => {
  if (!error) return fallback;

  // Network-level failures
  if (!error.response) {
    if (error.code === "ECONNABORTED" || error.name === "AbortError") {
      return "The request timed out. Please try again.";
    }
    return "Cannot reach the StockVibe API. Please check your connection.";
  }

  const { status, data } = error.response;

  // Prefer a server-provided human message when present, short, and
  // clearly user-facing — never pass through technical/server text
  // (tracebacks, framework names, file paths) to the UI.
  const serverMessage = data?.error;
  const looksTechnical = (text) =>
    /traceback|exception|\braise[d]?\b|\.py\b|django|stack |trace |\n/i.test(text);
  if (
    typeof serverMessage === "string" &&
    serverMessage.trim().length > 0 &&
    serverMessage.length < 200 &&
    !looksTechnical(serverMessage)
  ) {
    return serverMessage;
  }

  if (status === 400) return "This request was rejected by the server.";
  if (status === 404) return "We couldn't find data for this symbol.";
  if (status === 429) return "Too many requests. Please wait a moment and retry.";
  if (status >= 500) return "The server encountered an error. Please try again.";

  return fallback;
};

/** Extract the typed status for UI state machines. */
export const errorStatus = (error) =>
  error?.response?.status ?? (error?.code === "ECONNABORTED" ? "timeout" : "network");

const request = async (config) => {
  try {
    const response = await client.request(config);
    return { ok: true, data: response.data };
  } catch (error) {
    return {
      ok: false,
      error: {
        status: errorStatus(error),
        message: toUserMessage(error, config.fallbackMessage),
      },
    };
  }
};

const apiService = {
  get: (url, config = {}) => request({ ...config, method: "GET", url }),
  post: (url, body, config = {}) =>
    request({ ...config, method: "POST", url, data: body }),
};

export default apiService;
