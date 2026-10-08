/**
 * Settings — current environment facts only.
 * Preferences (theme, defaults) arrive with future releases.
 */

import React from "react";
import { API_BASE_URL } from "../api/client";

const SettingsPage = () => (
  <div className="sv-page">
    <div className="sv-page-header">
      <h1 className="sv-page-title">Settings</h1>
      <p className="sv-page-subtitle">Environment and connection information.</p>
    </div>

    <section className="sv-card">
      <div className="sv-pulse-grid">
        <div className="sv-metric">
          <span className="sv-metric-label">API endpoint</span>
          <span className="sv-metric-value" style={{ font: "var(--sv-text-body)", wordBreak: "break-all" }}>
            {API_BASE_URL}
          </span>
        </div>
        <div className="sv-metric">
          <span className="sv-metric-label">Configuration</span>
          <span className="sv-metric-value" style={{ font: "var(--sv-text-body)" }}>
            {API_BASE_URL.startsWith("http://localhost")
              ? "Default local development"
              : "Environment-provided (REACT_APP_API_BASE_URL)"}
          </span>
        </div>
        <div className="sv-metric">
          <span className="sv-metric-label">Theme</span>
          <span className="sv-metric-value" style={{ font: "var(--sv-text-body)" }}>
            Dark (system)
          </span>
        </div>
      </div>
      <p className="sv-card-subtitle" style={{ marginTop: "var(--sv-space-4)" }}>
        User preferences and profile settings arrive with account support.
      </p>
    </section>
  </div>
);

export default SettingsPage;
