/**
 * Shared, accessible loading / error / empty / placeholder states.
 * Every data-driven surface must render one of these rather than
 * ad-hoc markup, so states stay consistent across the product.
 */

import React from "react";

export const LoadingView = ({ label = "Loading market data..." }) => (
  <div className="sv-loading" role="status" aria-live="polite">
    <div className="sv-loading-spinner" aria-hidden="true"></div>
    <p className="sv-loading-label">{label}</p>
  </div>
);

export const ErrorView = ({ message = "We couldn't load this data.", onRetry, retryLabel = "Try again" }) => (
  <div className="sv-error" role="alert">
    <p className="sv-error-title">We couldn't load this data.</p>
    <p className="sv-error-text">{message}</p>
    {onRetry && (
      <button type="button" className="sv-btn sv-btn-sm" onClick={onRetry}>
        {retryLabel}
      </button>
    )}
  </div>
);

export const EmptyView = ({ title = "Nothing here yet.", message }) => (
  <div className="sv-empty">
    <p className="sv-empty-title">{title}</p>
    {message && <p className="sv-empty-text">{message}</p>}
  </div>
);

/**
 * Future-capability placeholder. Used for areas whose backing
 * engines do not exist yet (portfolio, backtesting, AI insights).
 * Never filled with fabricated numbers.
 */
export const PlaceholderView = ({ title, message, items, badge = "Coming soon" }) => (
  <div className="sv-placeholder">
    <span className="sv-badge">{badge}</span>
    {title && <h2 className="sv-placeholder-title">{title}</h2>}
    {message && <p className="sv-placeholder-text">{message}</p>}
    {items && items.length > 0 && (
      <ul className="sv-placeholder-list">
        {items.map((item) => (
          <li key={item}>{item}</li>
        ))}
      </ul>
    )}
  </div>
);
