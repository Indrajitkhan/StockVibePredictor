/**
 * Mobile bottom tab bar — persistent navigation on small screens.
 */

import React from "react";
import { NavLink } from "react-router-dom";

const TABS = [
  { to: "/", label: "Home", end: true },
  { to: "/markets", label: "Markets" },
  { to: "/watchlist", label: "Watchlist" },
  { to: "/insights", label: "Insights" },
];

const BottomTabs = () => (
  <nav className="sv-bottom-tabs" aria-label="Primary mobile">
    {TABS.map((tab) => (
      <NavLink
        key={tab.to}
        to={tab.to}
        end={tab.end}
        className={({ isActive }) => `sv-bottom-tab${isActive ? " active" : ""}`}
      >
        {tab.label}
      </NavLink>
    ))}
  </nav>
);

export default BottomTabs;
