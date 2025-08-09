import React, { useState } from 'react';
import './StockInput.css';

const StockInput = ({ onSubmit, loading, onReset }) => {
  const [ticker, setTicker] = useState('');
  const [suggestions] = useState([
    'AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'META', 'NFLX', 'NVDA'
  ]);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (ticker.trim() && !loading) {
      onSubmit(ticker.trim().toUpperCase());
    }
  };

  const handleSuggestionClick = (suggestedTicker) => {
    setTicker(suggestedTicker);
    onSubmit(suggestedTicker);
  };

  const handleReset = () => {
    setTicker('');
    onReset();
  };

  return (
    <div className="stock-input-container">
      <form onSubmit={handleSubmit} className="stock-form">
        <div className="input-group">
          <input
            type="text"
            value={ticker}
            onChange={(e) => setTicker(e.target.value.toUpperCase())}
            placeholder="Enter stock ticker (e.g., AAPL, TSLA)"
            className="stock-input"
            disabled={loading}
            maxLength={10}
          />
          <button 
            type="submit" 
            className="predict-button"
            disabled={!ticker.trim() || loading}
          >
            {loading ? '🔄 Analyzing...' : '🔮 Predict'}
          </button>
          {(ticker || loading) && (
            <button 
              type="button" 
              onClick={handleReset}
              className="reset-button"
              disabled={loading}
            >
              🔄 Reset
            </button>
          )}
        </div>
      </form>

      <div className="suggestions">
        <p className="suggestions-label">Popular Stocks:</p>
        <div className="suggestions-grid">
          {suggestions.map((stock) => (
            <button
              key={stock}
              onClick={() => handleSuggestionClick(stock)}
              className="suggestion-chip"
              disabled={loading}
            >
              {stock}
            </button>
          ))}
        </div>
      </div>

      <div className="info-text">
        <p>📈 Enter any valid stock ticker symbol to get AI-powered predictions!</p>
        <p>🤖 Our model analyzes historical data, technical indicators, and market trends</p>
      </div>
    </div>
  );
};

export default StockInput;
