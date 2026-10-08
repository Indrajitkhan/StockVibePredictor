/**
 * AppShell — the persistent product chrome: topbar, sidebar (desktop),
 * bottom tabs (mobile), routed content, and footer disclaimer.
 */

import React from "react";
import { Outlet } from "react-router-dom";
import Topbar from "./Topbar";
import Sidebar from "./Sidebar";
import BottomTabs from "./BottomTabs";

const AppShell = () => (
  <div className="sv-shell">
    <Topbar />
    <div className="sv-shell-body">
      <Sidebar />
      <main className="sv-shell-main" id="main-content">
        <Outlet />
      </main>
    </div>
    <BottomTabs />
    <footer className="sv-disclaimer" role="contentinfo">
      StockVibe provides market data and analytical tools for research and
      education. Nothing here is investment advice. Model outputs are
      experimental and must not be treated as validated financial signals.
    </footer>
  </div>
);

export default AppShell;
