import React from "react";
import "./PredictionResult.css";

const PredictionResult = ({
  predictions,
  analysis,
  marketInfo,
  ticker,
  selectedTimeframes,
}) => {
  if (!predictions || Object.keys(predictions).length === 0) {
    return null;
  }

  const getDirectionIcon = (direction) => {
    return direction === "UP" ? "📈" : "📉";
  };

  const getDirectionColor = (direction) => {
    return direction === "UP" ? "green" : "red";
  };

  const getConfidenceLevel = (confidence) => {
    if (confidence >= 80) return "high";
    if (confidence >= 60) return "medium";
    return "low";
  };

  const formatTimeframeName = (timeframe) => {
    const names = {
      "1d": "1 Day",
      "1w": "1 Week",
      "1mo": "1 Month",
      "1y": "1 Year",
    };
    return names[timeframe] || timeframe;
  };

  // Sort predictions by timeframe order
  const timeframeOrder = ["1d", "1w", "1mo", "1y"];
  const sortedPredictions = Object.entries(predictions).sort(
    ([a], [b]) => timeframeOrder.indexOf(a) - timeframeOrder.indexOf(b)
  );

  return (
    <div className="prediction-result">
      <div className="prediction-header">
        <h2>🎯 Predictions for {ticker}</h2>
        <div className="prediction-summary">
          {sortedPredictions.length} timeframe
          {sortedPredictions.length !== 1 ? "s" : ""} analyzed
        </div>
      </div>

      <div className="predictions-grid">
        {sortedPredictions.map(([timeframe, prediction]) => (
          <div key={timeframe} className="prediction-card">
            <div className="card-header">
              <h3 className="timeframe-title">
                {formatTimeframeName(timeframe)}
              </h3>
              <div className={`model-badge ${prediction.model_type}`}>
                {prediction.model_type}
              </div>
            </div>

            <div className="prediction-main">
              <div
                className={`direction-indicator ${prediction.direction.toLowerCase()}`}
              >
                <span className="direction-icon">
                  {getDirectionIcon(prediction.direction)}
                </span>
                <span className="direction-text">{prediction.direction}</span>
              </div>

              <div className="confidence-section">
                <div className="confidence-bar-container">
                  <div className="confidence-label">
                    Confidence: {prediction.confidence}%
                  </div>
                  <div className="confidence-bar">
                    <div
                      className={`confidence-fill ${getConfidenceLevel(
                        prediction.confidence
                      )}`}
                      style={{ width: `${prediction.confidence}%` }}
                    ></div>
                  </div>
                </div>
              </div>
            </div>

            <div className="prediction-details">
              <div className="price-info">
                <div className="price-row">
                  <span className="label">Current Price:</span>
                  <span className="value">${prediction.current_price}</span>
                </div>
                <div className="price-row">
                  <span className="label">Target Price:</span>
                  <span
                    className="value"
                    style={{ color: getDirectionColor(prediction.direction) }}
                  >
                    ${prediction.price_target}
                  </span>
                </div>
                <div className="price-row">
                  <span className="label">Expected Return:</span>
                  <span
                    className={`value ${
                      prediction.expected_return >= 0 ? "positive" : "negative"
                    }`}
                  >
                    {prediction.expected_return > 0 ? "+" : ""}
                    {prediction.expected_return}%
                  </span>
                </div>
              </div>

              <div className="model-info">
                <div className="accuracy-badge">
                  Model Accuracy: {prediction.model_accuracy}%
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Overall Summary */}
      {analysis?.recommendation && (
        <div className="overall-summary">
          <h3>📊 Overall Assessment</h3>
          <div className="summary-content">
            <div
              className={`overall-recommendation ${analysis.recommendation.overall.toLowerCase()}`}
            >
              <strong>{analysis.recommendation.overall}</strong>
              <span className="overall-confidence">
                ({analysis.recommendation.confidence}% confidence)
              </span>
            </div>
            <div className="summary-details">
              <span>Risk: {analysis.recommendation.risk_level}</span>
              <span>•</span>
              <span>
                Strategy: {analysis.recommendation.holding_period} term
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Consensus View */}
      <div className="consensus-view">
        <h4>🎭 Consensus Across Timeframes</h4>
        <div className="consensus-bars">
          {(() => {
            const upCount = sortedPredictions.filter(
              ([, p]) => p.direction === "UP"
            ).length;
            const totalCount = sortedPredictions.length;
            const bullishPercent = (upCount / totalCount) * 100;

            return (
              <div className="consensus-container">
                <div className="consensus-label">
                  {upCount}/{totalCount} timeframes bullish (
                  {bullishPercent.toFixed(0)}%)
                </div>
                <div className="consensus-bar">
                  <div
                    className="bullish-portion"
                    style={{ width: `${bullishPercent}%` }}
                  ></div>
                  <div
                    className="bearish-portion"
                    style={{ width: `${100 - bullishPercent}%` }}
                  ></div>
                </div>
                <div className="consensus-legend">
                  <span className="bullish-legend">📈 Bullish</span>
                  <span className="bearish-legend">📉 Bearish</span>
                </div>
              </div>
            );
          })()}
        </div>
      </div>

      {/* Risk Disclaimer */}
      <div className="disclaimer">
        <p>
          ⚠️ <strong>Disclaimer:</strong> These predictions are based on
          technical analysis and machine learning models. Past performance does
          not guarantee future results. Always do your own research and consider
          consulting with a financial advisor before making investment
          decisions.
        </p>
      </div>
    </div>
  );
};

export default PredictionResult;
