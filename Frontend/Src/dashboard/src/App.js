import React, { useState } from "react";
import axios from "axios";
import "./App.css";
import StockChart from "./Components/StockChart";
import StockInput from "./Components/StockInput";
import PredictionResult from "./Components/PredictionResult";
import LoadingSpinner from "./Components/LoadingSpinner";

// TODO: Note : Need to make more pictures and emojis for the app

function App() {
  const [stockData, setStockData] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [currentTicker, setCurrentTicker] = useState("");

  // Base URL for Django backend API
  const API_BASE_URL = "http://localhost:8000/api";

  const fetchStockData = async (ticker) => {
    if (!ticker.trim()) {
      setError("Please enter a valid stock ticker");
      return;
    }

    setLoading(true);
    setError(null);
    setCurrentTicker(ticker.toUpperCase());

    try {
      // Call Django backend API for stock data and prediction
      const response = await axios.post(`${API_BASE_URL}/predict/`, {
        ticker: ticker.toUpperCase(),
      });

      setStockData(response.data.stock_data);
      setPrediction(response.data.prediction);
    } catch (err) {
      console.error("Error fetching stock data:", err);
      setError(
        err.response?.data?.error ||
          "Failed to fetch stock data. Please check if the backend server is running."
      );
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setStockData(null);
    setPrediction(null);
    setError(null);
    setCurrentTicker("");
  };

  return (
    <div className="App">
      <header className="app-header">
        <h1 className="app-title">StockVibePredictor</h1>
        <p className="app-subtitle">AI-Powered Stock Market Predictions</p>
      </header>

      <main className="app-main">
        <div className="container">
          <StockInput
            onSubmit={fetchStockData}
            loading={loading}
            onReset={handleReset}
          />

          {error && (
            <div className="error-message">
              <span className="error-icon">⚠️</span>
              {error}
            </div>
          )}

          {loading && <LoadingSpinner />}

          {prediction && !loading && (
            <PredictionResult prediction={prediction} ticker={currentTicker} />
          )}

          {stockData && !loading && (
            <StockChart data={stockData} ticker={currentTicker} />
          )}
        </div>
      </main>

      <footer className="app-footer">
        <p>Built with React + Django + Machine Learning</p>
      </footer>
    </div>
  );
}

export default App;
