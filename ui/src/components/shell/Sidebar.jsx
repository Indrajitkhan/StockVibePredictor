/**
 * Application sidebar — persistent desktop navigation.
 * Mobile uses a bottom tab bar (separate component).
 */

import React from "react";
import { NavLink } from "react-router-dom";

export const NAV_ITEMS = [
  { to: "/", label: "Dashboard", end: true },
  { to: "/markets", label: "Markets" },
  { to: "/watchlist", label: "Watchlist" },
  { to: "/portfolio", label: "Portfolio" },
  { to: "/insights", label: "AI Insights" },
  { to: "/backtesting", label: "Backtesting" },
];

const Sidebar = () => (
  <aside className="sv-sidebar" aria-label="Primary">
    <nav className="sv-sidebar-nav">
      <ul>
        {NAV_ITEMS.map((item) => (
          <li key={item.to}>
            <NavLink
              to={item.to}
              end={item.end}
              className={({ isActive }) =>
                `sv-sidebar-link${isActive ? " active" : ""}`
              }
            >
              {item.label}
            </NavLink>
          </li>
        ))}
      </ul>
    </nav>

    <div className="sv-sidebar-footer">
      <NavLink
        to="/settings"
        className={({ isActive }) =>
          `sv-sidebar-link${isActive ? " active" : ""}`
        }
      >
        Settings
      </NavLink>
    </div>
  </aside>
);

export default Sidebar;
