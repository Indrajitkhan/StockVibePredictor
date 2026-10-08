/**
 * Topbar — brand, global search, system health, profile affordance.
 */

import React, { useState, useEffect, useCallback } from "react";
import { Link } from "react-router-dom";
import GlobalSearch from "./GlobalSearch";
import { getSystemHealth } from "../../api/market";

const HEALTH_POLL_MS = 60000;

const Topbar = () => {
  const [health, setHealth] = useState({ status: "checking", label: "Checking..." });
  const [healthError, setHealthError] = useState(false);

  const checkHealth = useCallback(async () => {
    const result = await getSystemHealth();
    if (result.ok) {
      const raw = (result.data?.status || "unknown").toLowerCase();
      const map = {
        healthy: "API online",
        degraded: "API degraded",
        warning: "API degraded",
        unhealthy: "API offline",
      };
      setHealth({ status: raw, label: map[raw] || "Status unknown" });
      setHealthError(false);
    } else {
      setHealth({ status: "unreachable", label: "API unreachable" });
      setHealthError(true);
    }
  }, []);

  useEffect(() => {
    checkHealth();
    const t = setInterval(checkHealth, HEALTH_POLL_MS);
    return () => clearInterval(t);
  }, [checkHealth]);

  const tone =
    health.status === "healthy"
      ? "positive"
      : healthError || health.status === "unhealthy"
      ? "negative"
      : "warning";

  return (
    <header className="sv-topbar">
      <div className="sv-topbar-brand">
        {/* Mobile-only hamburger is unnecessary: mobile uses bottom tabs. */}
        <Link to="/" className="sv-brand">
          Stock<span className="sv-brand-accent">Vibe</span>
        </Link>
      </div>

      <div className="sv-topbar-search">
        <GlobalSearch />
      </div>

      <div className="sv-topbar-right">
        <span
          className={`sv-badge sv-badge-${tone}`}
          title={`StockVibe API: ${health.label}`}
        >
          <span className="sv-badge-dot" aria-hidden="true"></span>
          {health.label}
        </span>

        {/* Profile is a future capability (no auth backend yet) — rendered
            as a disabled affordance, never as a fake working account. */}
        <button
          type="button"
          className="sv-btn sv-btn-sm"
          disabled
          title="Accounts arrive with a future release"
        >
          Profile
        </button>
      </div>
    </header>
  );
};

export default Topbar;
