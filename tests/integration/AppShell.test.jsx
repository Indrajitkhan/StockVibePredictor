/**
 * App shell + routing behavior tests.
 * Renders the real App with mocked API responses.
 */

import React from "react";
import { render, screen, waitFor, act } from "@testing-library/react";
import "@testing-library/jest-dom";

// The centralized client (src/api/client.js) creates its axios instance at
// module load. The created instance's methods are the SAME jest.fn objects
// as the top-level mock's, so per-test behavior is configured by mocking
// axios.request (client.get ultimately calls instance.request).
import axios from "axios";

import App from "../../ui/src/App";

// ─── Shared mock payloads (real endpoint shapes) ───

const healthResponse = {
  data: {
    status: "healthy",
    services: {},
    metrics: { model_cache_size: 4 },
  },
};

const marketOverviewResponse = {
  data: {
    market_data: {
      "^GSPC": {
        name: "S&P 500",
        price: 5900.12,
        change: 21.34,
        change_percent: 0.36,
        direction: "up",
      },
      "^IXIC": {
        name: "NASDAQ",
        price: 19200.55,
        change: -40.2,
        change_percent: -0.21,
        direction: "down",
      },
    },
    market_sentiment: "neutral",
    timestamp: "2026-10-08T10:30:00Z",
    trading_session: "open",
  },
};

const renderAt = async (route) => {
  window.history.pushState({}, "Test", route);
  const result = render(<App />);
  // Flush the Topbar health poll + dashboard fetches inside act()
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 0));
  });
  return result;
};

describe("Application shell", () => {
  beforeEach(() => {
    // All app traffic flows through client.request → the shared mock fn.
    axios.request.mockImplementation((config) => {
      const url = config?.url || "";
      if (url.includes("/system/health/")) return Promise.resolve(healthResponse);
      if (url.includes("/market/overview/")) return Promise.resolve(marketOverviewResponse);
      return Promise.resolve({ data: {} });
    });
  });

  test("renders the shell with brand, search, and navigation", async () => {
    await renderAt("/");

    // Accessible name computes as "Stock Vibe" (text split across elements)
    expect(
      screen.getByRole("link", { name: /Stock\s*Vibe/i })
    ).toBeInTheDocument();

    // Sidebar navigation
    expect(screen.getAllByRole("link", { name: "Dashboard" }).length).toBeGreaterThan(0);
    expect(screen.getAllByRole("link", { name: "Markets" }).length).toBeGreaterThan(0);
    expect(screen.getAllByRole("link", { name: "Watchlist" }).length).toBeGreaterThan(0);
    expect(screen.getAllByRole("link", { name: "Portfolio" }).length).toBeGreaterThan(0);
    expect(screen.getAllByRole("link", { name: "AI Insights" }).length).toBeGreaterThan(0);
    expect(screen.getAllByRole("link", { name: "Backtesting" }).length).toBeGreaterThan(0);
    expect(screen.getAllByRole("link", { name: "Settings" }).length).toBeGreaterThan(0);
  });

  test("dashboard renders real market overview data from the API", async () => {
    await renderAt("/");

    await waitFor(() => {
      expect(screen.getByText("S&P 500")).toBeInTheDocument();
    });
    expect(screen.getByText("NASDAQ")).toBeInTheDocument();
    // Real values from the mocked response — never invented
    expect(screen.getByText("5,900.12")).toBeInTheDocument();
  });

  test("dashboard shows loading state before data arrives", async () => {
    // Never-resolving promise keeps the page in the loading state.
    // renderAt flushes the microtask queue; the synchronous loading UI
    // must be visible before any promise resolution.
    axios.request.mockImplementation(() => new Promise(() => {}));

    await renderAt("/");
    // Loading appears in multiple dashboard cards — assert presence, not uniqueness
    expect(screen.getAllByText("Loading market data...").length).toBeGreaterThan(0);
  });

  test("dashboard shows error state with retry when API fails", async () => {
    axios.request.mockImplementation(() =>
      Promise.reject(new Error("Network Error"))
    );

    await renderAt("/");
    // Error state appears in multiple dashboard cards
    await waitFor(() => {
      expect(screen.getAllByText("We couldn't load this data.").length).toBeGreaterThan(
        0
      );
    });
    expect(screen.getAllByRole("button", { name: "Try again" }).length).toBeGreaterThan(0);
  });

  test("stock analysis route renders with symbol from URL", async () => {
    await renderAt("/stocks/AAPL");

    await waitFor(() => {
      expect(screen.getByText("AAPL")).toBeInTheDocument();
    });

    // Analysis tabs render
    expect(screen.getByRole("tab", { name: "Overview" })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "Technical" })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "Fundamentals" })).toBeInTheDocument();
  });

  test("future-capability pages render honest placeholder states", async () => {
    await renderAt("/portfolio");
    expect(
      screen.getByText("Portfolio intelligence is not connected yet.")
    ).toBeInTheDocument();

    await renderAt("/backtesting");
    expect(
      screen.getByText("Backtesting engine connection pending.")
    ).toBeInTheDocument();
  });

  test("unknown route shows the 404 page, not a silent re-render", async () => {
    await renderAt("/this-route-does-not-exist");
    expect(screen.getByText("Page not found")).toBeInTheDocument();
  });
});
