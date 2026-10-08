/**
 * StockVibe — application root.
 *
 * Routes every product area through the AppShell. The legacy
 * monolithic prediction-first App has been replaced by the
 * market-intelligence product shell; reusable analysis components
 * live on under /stocks/:symbol.
 */

import React from "react";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import AppShell from "./components/shell/AppShell";
import { WatchlistProvider } from "./pages/WatchlistPage";
import DashboardPage from "./pages/DashboardPage";
import MarketsPage from "./pages/MarketsPage";
import WatchlistPage from "./pages/WatchlistPage";
import StockAnalysisPage from "./pages/StockAnalysisPage";
import InsightsPage from "./pages/InsightsPage";
import PortfolioPage from "./pages/PortfolioPage";
import BacktestingPage from "./pages/BacktestingPage";
import SettingsPage from "./pages/SettingsPage";
import NotFoundPage from "./pages/NotFoundPage";

import "./styles/tokens.css";
import "./styles/base.css";
import "./styles/components.css";
import "./styles/pages.css";
import "./components/shell/shell.css";

function App() {
  return (
    <BrowserRouter>
      <WatchlistProvider>
        <Routes>
          <Route element={<AppShell />}>
            <Route index element={<DashboardPage />} />
            <Route path="/markets" element={<MarketsPage />} />
            <Route path="/watchlist" element={<WatchlistPage />} />
            <Route path="/portfolio" element={<PortfolioPage />} />
            <Route path="/insights" element={<InsightsPage />} />
            <Route path="/backtesting" element={<BacktestingPage />} />
            <Route path="/stocks/:symbol" element={<StockAnalysisPage />} />
            <Route path="/settings" element={<SettingsPage />} />
            <Route path="*" element={<NotFoundPage />} />
          </Route>
        </Routes>
      </WatchlistProvider>
    </BrowserRouter>
  );
}

export default App;
