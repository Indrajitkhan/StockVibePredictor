import { useState, useEffect } from "react";
import "../css/CompanyEssentials.css";

/**
 * CompanyEssentials — presentation component for company fundamentals.
 *
 * Data now comes from the centralized API layer (src/api/stocks.js) via the
 * `data` prop. Legacy direct-axios fetching with a hardcoded localhost URL
 * was removed so all traffic flows through src/api/client.js.
 *
 * Backward compatibility: if `data` is not provided (legacy callers),
 * the component renders nothing rather than fetching on its own.
 */
const CompanyEssentials = ({ ticker, data = null, onCompanyDataReceived = null }) => {
  const [showAdditionalMetrics, setShowAdditionalMetrics] = useState(false);

  // Legacy callback support — notify parent when centralized data arrives.
  // (Kept for legacy callers that rely on it; new shell pages don't use it.)
  useEffect(() => {
    if (data && onCompanyDataReceived) {
      onCompanyDataReceived(data);
    }
  }, [data, onCompanyDataReceived]);

  const companyData = data;

  if (!companyData) {
    return null;
  }

  const getValueClass = (value) => {
    if (value === null || value === undefined || value === "N/A") return "";
    const numValue = parseFloat(value);
    if (isNaN(numValue)) return "";
    return numValue >= 0 ? "positive" : "negative";
  };

  const {
    essentials,
    additional_metrics,
    company_name,
    current_price,
    price_summary,
    currency,
    company_info,
  } = companyData;
  const currencySymbol = currency?.symbol || "$";

  return (
    <div className="company-essentials">
      {/* Header Section */}
      <div className="essentials-header">
        <div className="company-title">
          <h3>📊 Company Essentials</h3>
          <h2>{company_name || ticker}</h2>
          <div className="current-price">
            <span className="price">
              {currencySymbol}
              {current_price?.toFixed(2) || "N/A"}
            </span>
            {price_summary?.price_change !== undefined && (
              <span
                className={`price-change ${
                  price_summary.price_change >= 0 ? "positive" : "negative"
                }`}
              >
                {price_summary.price_change >= 0 ? "▲" : "▼"}
                {currencySymbol}
                {Math.abs(price_summary.price_change).toFixed(2)}(
                {price_summary.price_change_percent?.toFixed(2) || "0.00"}%)
              </span>
            )}
          </div>
        </div>
      </div>

      {/* Essentials Grid - 3 Column Layout */}
      <div className="essentials-grid">
        {/* Left Column */}
        <div className="essentials-column">
          <div className="essential-item">
            <label>{essentials.market_cap?.label || "MARKET CAP"}</label>
            <span className="value">
              {essentials.market_cap?.formatted || "N/A"}
            </span>
          </div>

          <div className="essential-item">
            <label>{essentials.pe_ratio?.label || "P/E"}</label>
            <span className="value">
              {essentials.pe_ratio?.formatted || "N/A"}
            </span>
          </div>

          <div className="essential-item">
            <label>{essentials.dividend_yield?.label || "DIV. YIELD"}</label>
            <span className="value">
              {essentials.dividend_yield?.formatted || "N/A"}
            </span>
          </div>

          <div className="essential-item">
            <label>{essentials.debt?.label || "DEBT"}</label>
            <span className="value">{essentials.debt?.formatted || "N/A"}</span>
          </div>

          <div className="essential-item">
            <label>{essentials.sales_growth?.label || "SALES GROWTH"}</label>
            <span
              className={`value ${getValueClass(
                essentials.sales_growth?.value
              )}`}
            >
              {essentials.sales_growth?.formatted || "N/A"}
            </span>
          </div>

          <div className="essential-item">
            <label>{essentials.profit_growth?.label || "PROFIT GROWTH"}</label>
            <span
              className={`value ${getValueClass(
                essentials.profit_growth?.value
              )}`}
            >
              {essentials.profit_growth?.formatted || "N/A"}
            </span>
          </div>
        </div>

        {/* Middle Column */}
        <div className="essentials-column">
          <div className="essential-item">
            <label>
              {essentials.enterprise_value?.label || "ENTERPRISE VALUE"}
            </label>
            <span className="value">
              {essentials.enterprise_value?.formatted || "N/A"}
            </span>
          </div>

          <div className="essential-item">
            <label>{essentials.pb_ratio?.label || "P/B"}</label>
            <span className="value">
              {essentials.pb_ratio?.formatted || "N/A"}
            </span>
          </div>

          <div className="essential-item">
            <label>
              {essentials.book_value_ttm?.label || "BOOK VALUE (TTM)"}
            </label>
            <span className="value">
              {essentials.book_value_ttm?.formatted || "N/A"}
            </span>
          </div>

          <div className="essential-item">
            <label>
              {essentials.promoter_holding?.label || "PROMOTER HOLDING"}
            </label>
            <span className="value">
              {essentials.promoter_holding?.formatted || "N/A"}
            </span>
          </div>

          <div className="essential-item">
            <label>{essentials.roe?.label || "ROE"}</label>
            <span className={`value ${getValueClass(essentials.roe?.value)}`}>
              {essentials.roe?.formatted || "N/A"}
            </span>
          </div>

          <div className="essential-item">
            <label>Add Your Ratio</label>
            <button
              className="add-ratio-btn"
              onClick={() => setShowAdditionalMetrics(!showAdditionalMetrics)}
              title="View additional metrics"
            >
              {showAdditionalMetrics ? "−" : "+"}
            </button>
          </div>
        </div>

        {/* Right Column */}
        <div className="essentials-column">
          <div className="essential-item">
            <label>{essentials.num_shares?.label || "NO. OF SHARES"}</label>
            <span className="value">
              {essentials.num_shares?.formatted || "N/A"}
            </span>
          </div>

          <div className="essential-item">
            <label>{essentials.face_value?.label || "FACE VALUE"}</label>
            <span className="value">
              {essentials.face_value?.formatted || "N/A"}
            </span>
          </div>

          <div className="essential-item">
            <label>{essentials.cash?.label || "CASH"}</label>
            <span className="value">{essentials.cash?.formatted || "N/A"}</span>
          </div>

          <div className="essential-item">
            <label>{essentials.eps_ttm?.label || "EPS (TTM)"}</label>
            <span className="value">
              {essentials.eps_ttm?.formatted || "N/A"}
            </span>
          </div>

          <div className="essential-item">
            <label>{essentials.roa?.label || "ROA"}</label>
            <span className={`value ${getValueClass(essentials.roa?.value)}`}>
              {essentials.roa?.formatted || "N/A"}
            </span>
          </div>

          <div className="essential-item empty-slot">
            <label></label>
            <span className="value"></span>
          </div>
        </div>
      </div>

      {/* Additional Metrics Section (Expandable) */}
      {showAdditionalMetrics && additional_metrics && (
        <div className="additional-metrics">
          <h4>Additional Metrics</h4>
          <div className="metrics-grid">
            {additional_metrics.debt_to_equity && (
              <div className="metric-item">
                <label>{additional_metrics.debt_to_equity.label}</label>
                <span className="value">
                  {additional_metrics.debt_to_equity.formatted}
                </span>
              </div>
            )}
            {additional_metrics.profit_margins && (
              <div className="metric-item">
                <label>{additional_metrics.profit_margins.label}</label>
                <span className="value">
                  {additional_metrics.profit_margins.formatted}
                </span>
              </div>
            )}
            {additional_metrics.forward_pe && (
              <div className="metric-item">
                <label>{additional_metrics.forward_pe.label}</label>
                <span className="value">
                  {additional_metrics.forward_pe.formatted}
                </span>
              </div>
            )}
            {additional_metrics.peg_ratio && (
              <div className="metric-item">
                <label>{additional_metrics.peg_ratio.label}</label>
                <span className="value">
                  {additional_metrics.peg_ratio.formatted}
                </span>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Price Summary Section */}
      {price_summary && (
        <div className="price-summary">
          <h4>Price Summary</h4>
          <div className="price-stats">
            <div className="price-stat">
              <label>Today's High</label>
              <span>
                {currencySymbol}
                {price_summary.day_high?.toFixed(2) || "N/A"}
              </span>
            </div>
            <div className="price-stat">
              <label>Today's Low</label>
              <span>
                {currencySymbol}
                {price_summary.day_low?.toFixed(2) || "N/A"}
              </span>
            </div>
            <div className="price-stat">
              <label>52W High</label>
              <span className="high-price">
                {currencySymbol}
                {price_summary.week_52_high?.toFixed(2) || "N/A"}
              </span>
            </div>
            <div className="price-stat">
              <label>52W Low</label>
              <span className="low-price">
                {currencySymbol}
                {price_summary.week_52_low?.toFixed(2) || "N/A"}
              </span>
            </div>
          </div>
          {/* Price range indicator */}
          {price_summary.week_52_high &&
            price_summary.week_52_low &&
            current_price && (
              <div className="price-range-indicator">
                <div className="range-bar">
                  <div
                    className="current-position"
                    style={{
                      left: `${
                        ((current_price - price_summary.week_52_low) /
                          (price_summary.week_52_high -
                            price_summary.week_52_low)) *
                        100
                      }%`,
                    }}
                    title={`Current: ${currencySymbol}${current_price.toFixed(
                      2
                    )}`}
                  />
                </div>
                <div className="range-labels">
                  <span>52W Low</span>
                  <span>52W High</span>
                </div>
              </div>
            )}
        </div>
      )}

      {/* Company Information Footer */}
      {company_info && (
        <div className="company-info-summary">
          <div className="info-item">
            <strong>Sector:</strong> {company_info.sector || "N/A"}
          </div>
          <div className="info-item">
            <strong>Industry:</strong> {company_info.industry || "N/A"}
          </div>
          <div className="info-item">
            <strong>Exchange:</strong> {company_info.exchange || "N/A"}
          </div>
          {company_info.full_time_employees > 0 && (
            <div className="info-item">
              <strong>Employees:</strong>{" "}
              {company_info.full_time_employees.toLocaleString()}
            </div>
          )}
          {company_info.website && company_info.website !== "N/A" && (
            <div className="info-item">
              <strong>Website:</strong>
              <a
                href={company_info.website}
                target="_blank"
                rel="noopener noreferrer"
              >
                {new URL(company_info.website).hostname}
              </a>
            </div>
          )}
        </div>
      )}

      {/* Last Updated Timestamp */}
      {companyData.last_updated && (
        <div className="last-updated">
          Last updated: {new Date(companyData.last_updated).toLocaleString()}
        </div>
      )}
    </div>
  );
};

export default CompanyEssentials;
