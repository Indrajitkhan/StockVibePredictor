import React, { useState } from "react";
import axios from "axios";
import "./App.css";
import StockInput from "./Components/StockInput";
import PredictionResult from "./Components/PredictionResult";
import StockChart from "./Components/StockChart";
import LoadingSpinner from "./Components/LoadingSpinner";

function App() {
  const [stockData, setStockData] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [currentTicker, setCurrentTicker] = useState("");

  const API_BASE_URL = "http://localhost:8000/api";

  const fetchStockData = async (ticker) => {
    if (!ticker.trim()) {
      setError("Gimme a ticker, bro! 😎");
      setStockData(null);
      setPrediction(null);
      return;
    }

    setLoading(true);
    setError(null);
    setCurrentTicker(ticker.toUpperCase());

    try {
      const response = await axios.post(`${API_BASE_URL}/predict/`, {
        ticker: ticker.toUpperCase(),
      });

      setStockData(response.data.history);
      setPrediction(response.data.prediction);
    } catch (err) {
      console.error("Error fetching stock data, yo:", err);
      setError(
        err.response?.data?.error ||
          "Sh*t broke, fam! Check if the backend’s running. 🛠️"
      );
      setStockData(null);
      setPrediction(null);
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
        <h1 className="app-title">StockVibePredictor 🚀</h1>
        <p className="app-subtitle">AI-Powered Stock Market Predictions 💸</p>
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
        <p>Built with React + Django + Machine Learning 🔥</p>
      </footer>
    </div>
  );
}

export default App;
