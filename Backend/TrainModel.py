import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
import pickle
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(asctime)s %(message)s",
    handlers=[logging.FileHandler("Logs/stockpredict.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def compute_macd(data, fast=12, slow=26, signal=9):
    exp1 = data["Close"].ewm(span=fast, adjust=False).mean()
    exp2 = data["Close"].ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line


def prepare_data(ticker="AAPL", period="2y"):
    try:
        logger.info(f"Fetching data for {ticker}")
        data = yf.download(ticker, period=period, interval="1d", progress=False)
        if data.empty:
            logger.error(f"No data for {ticker}")
            raise ValueError("No data available")

        data["Return"] = data["Close"].pct_change()
        data["MA5"] = data["Close"].rolling(window=5).mean()
        data["MA20"] = data["Close"].rolling(window=20).mean()
        data["Volatility"] = data["Return"].rolling(window=20).std()
        data["RSI"] = compute_rsi(data["Close"], 14)
        data["Volume_Change"] = data["Volume"].pct_change()
        data["MACD"], data["MACD_Signal"] = compute_macd(data)
        data["Target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)
        data = data.dropna()

        return data
    except Exception as e:
        logger.error(f"Error preparing data: {str(e)}")
        raise


def train_model():
    try:
        data = prepare_data()
        features = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "Return",
            "MA5",
            "MA20",
            "Volatility",
            "RSI",
            "Volume_Change",
            "MACD",
            "MACD_Signal",
        ]
        X = data[features]
        y = data["Target"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_split=5, random_state=42
        )
        model.fit(X_train, y_train)

        cv_scores = cross_val_score(model, X, y, cv=5)
        logger.info(f"Cross-validation scores: {cv_scores}")
        logger.info(f"Mean CV accuracy: {cv_scores.mean():.2%}")

        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        logger.info(f"Test accuracy: {accuracy:.2%}")

        with open("stock_model.pkl", "wb") as f:
            pickle.dump(model, f)
        logger.info("Model saved as stock_model.pkl")
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        raise


if __name__ == "__main__":
    train_model()
