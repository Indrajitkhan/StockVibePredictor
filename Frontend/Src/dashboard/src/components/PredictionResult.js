import React from "react";
import "./PredictionResult.css";

const PredictionResult = ({ prediction, ticker }) => {
  const getPredictionIcon = () => {
    return prediction?.direction === "UP" ? "📈" : "📉";
  };

  const getPredictionColor = () => {
    return prediction?.direction === "UP" ? "#4ade80" : "#f87171";
  };

  const getConfidenceColor = (confidence) => {
    if (confidence >= 80) return "#22c55e";
    if (confidence >= 60) return "#f59e0b";
    return "#ef4444";
  };

  return (
    <div className="prediction-result">
      <div className="prediction-header">
        <h2 className="prediction-title">🤖 AI Prediction for {ticker}</h2>
      </div>

      <div className="prediction-content">
        <div
          className="prediction-main"
          style={{ backgroundColor: getPredictionColor() }}
        >
          <div className="prediction-icon">{getPredictionIcon()}</div>
          <div className="prediction-text">
            <h3 className="prediction-direction">
              {prediction?.direction === "UP" ? "BULLISH" : "BEARISH"}
            </h3>
            <p className="prediction-subtitle">
              Price Expected to go{" "}
              <strong>{prediction?.direction === "UP" ? "UP" : "DOWN"}</strong>{" "}
              tomorrow
            </p>
          </div>
        </div>

        <div className="prediction-details">
          <div className="detail-card">
            <div className="detail-label">Confidence</div>
            <div
              className="detail-value"
              style={{ color: getConfidenceColor(prediction?.confidence || 0) }}
            >
              {prediction?.confidence
                ? `${prediction.confidence.toFixed(1)}%`
                : "N/A"}
            </div>
            <div className="confidence-bar">
              <div
                className="confidence-fill"
                style={{
                  width: `${prediction?.confidence || 0}%`,
                  backgroundColor: getConfidenceColor(
                    prediction?.confidence || 0
                  ),
                }}
              />
            </div>
          </div>

          <div className="detail-card">
            <div className="detail-label">Current Price</div>
            <div className="detail-value">
              $
              {prediction?.current_price
                ? prediction.current_price.toFixed(2)
                : "N/A"}
            </div>
          </div>

          <div className="detail-card">
            <div className="detail-label">Predicted Change</div>
            <div
              className="detail-value"
              style={{ color: getPredictionColor() }}
            >
              {prediction?.predicted_change
                ? `${
                    prediction.predicted_change > 0 ? "+" : ""
                  }${prediction.predicted_change.toFixed(2)}%`
                : "N/A"}
            </div>
          </div>
        </div>

        {prediction?.features && (
          <div className="features-summary">
            <h4>📊 Key Technical Indicators</h4>
            <div className="features-grid">
              {Object.entries(prediction.features).map(([key, value]) => (
                <div key={key} className="feature-item">
                  <span className="feature-name">
                    {key.replace(/_/g, " ").toUpperCase()}
                  </span>
                  <span className="feature-value">
                    {typeof value === "number" ? value.toFixed(2) : value}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

        <div className="disclaimer">
          <p>
            ⚠️ <strong>Disclaimer:</strong>Stock market investments carry
            inherent risks. Always do your own research and consider consulting
            with a financial advisor before making investment decisions.
          </p>
        </div>
      </div>
    </div>
  );
};

export default PredictionResult;
