/**
 * 404 — intentional route miss, never a silent re-render.
 */

import React from "react";
import { Link } from "react-router-dom";

const NotFoundPage = () => (
  <div className="sv-page">
    <div className="sv-card">
      <div className="sv-placeholder">
        <span className="sv-badge">404</span>
        <h1 className="sv-placeholder-title">Page not found</h1>
        <p className="sv-placeholder-text">
          The page you're looking for doesn't exist.
        </p>
        <Link to="/" className="sv-btn sv-btn-primary">
          Back to Dashboard
        </Link>
      </div>
    </div>
  </div>
);

export default NotFoundPage;
