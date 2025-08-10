import os
import re
import asyncio
import aiohttp
from django.http import JsonResponse
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
import yfinance as yf
import pandas as pd
import numpy as np
import pickle
import logging
from pathlib import Path
from django.core.cache import cache
from asgiref.sync import sync_to_async, async_to_sync

BASE_DIR = Path(__file__).resolve().parent.parent.parent

logger = logging.getLogger("apps.stockpredict")

try:
    with open(os.path.join(BASE_DIR, "Scripts", "stock_model.pkl"), "rb") as f:
        model = pickle.load(f)
    logger.info("Stock prediction model loaded successfully")
    logger.info(f"Model type: {type(model)}")
    if hasattr(model, "feature_names_in_"):
        logger.info(f"Expected features: {model.feature_names_in_}")
except Exception as e:
    logger.error(f"Failed to load stock prediction model: {str(e)}")
    raise


def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def compute_macd(data, fast=12, slow=26, signal=9):
    exp1 = data.ewm(span=fast, adjust=False).mean()
    exp2 = data.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line


def compute_atr(data, period=14):
    high_low = data["High"] - data["Low"]
    high_close = np.abs(data["High"] - data["Close"].shift())
    low_close = np.abs(data["Low"] - data["Close"].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(window=period).mean()


def compute_bollinger_bands(data, period=20, std=2):
    ma = data["Close"].rolling(window=period).mean()
    std_dev = data["Close"].rolling(window=period).std()
    upper = ma + (std_dev * std)
    lower = ma - (std_dev * std)
    return upper, lower


def validate_ticker(ticker):
    if not ticker or not isinstance(ticker, str):
        return False
    if not re.match(r"^[A-Z0-9]{1,10}$", ticker):
        return False
    return True


async def fetch_stock_data(ticker, period="1y", interval="1d"):
    try:
        async with aiohttp.ClientSession() as session:
            data = await sync_to_async(yf.download)(
                ticker, period=period, interval=interval, progress=False, timeout=30
            )
            if data.empty:
                logger.error(f"No data found for ticker: {ticker}")
                return None
            logger.info(f"Fetched {len(data)} rows for {ticker}")
            return data
    except asyncio.TimeoutError:
        logger.error(f"Timeout fetching data for {ticker}")
        return None
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {str(e)}")
        return None


@api_view(["GET"])
def redis_check(request):
    try:
        cache.set("test_key", "test_value", timeout=60)
        value = cache.get("test_key")
        if value == "test_value":
            logger.info("Redis connection successful")
            return Response({"status": "Redis is connected"}, status=status.HTTP_200_OK)
        else:
            logger.error("Redis test failed")
            return Response(
                {"error": "Redis test failed"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
    except Exception as e:
        logger.error(f"Redis connection failed: {str(e)}")
        return Response(
            {"error": f"Redis connection failed: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@api_view(["POST"])
def predict_stock_trend(request):
    ticker = request.data.get("ticker")
    if not ticker:
        logger.error("No ticker provided ...")
        return Response(
            {"error": "Gimme a ticker symbol"}, status=status.HTTP_400_BAD_REQUEST
        )

    if not validate_ticker(ticker):
        logger.error(f"Sketchy ticker format: {ticker}")
        return Response(
            {"error": "Ticker’s gotta be uppercase letters/numbers, 1-10 chars ..."},
            status=status.HTTP_400_BAD_REQUEST,
        )
    ticker = ticker.upper()

    cache_key = f"stock_data_{ticker}"
    cache.delete(cache_key)
    cached_data = cache.get(cache_key)
    if cached_data:
        logger.info(f"Got cached data for {ticker}, sweet!")
        data = pd.DataFrame(cached_data)
    else:
        try:
            logger.info(f"Fetching stock data for {ticker}")
            data = async_to_sync(fetch_stock_data)(ticker)
            if data is None or data.empty:
                logger.error(f"Couldn’t grab stock data for {ticker}")
                return Response(
                    {"error": "No stock data found, try another ticker ..."},
                    status=status.HTTP_404_NOT_FOUND,
                )
            logger.debug(f"DataFrame columns: {list(data.columns)}")
            logger.debug(f"DataFrame index: {data.index}")
            logger.debug(f"DataFrame head: {data.head().to_dict()}")
            cache.set(cache_key, data.to_dict(), timeout=3600)
            logger.info(f"Cached stock data for {ticker}, good to go")
        except Exception as e:
            logger.error(f"Sh*t hit the fan fetching data for {ticker}: {str(e)}")
            return Response(
                {"error": f"Something broke: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    try:
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] for col in data.columns]
        else:
            data.columns = [str(col) for col in data.columns]
        if "Close" not in data.columns:
            logger.error(
                f"No 'Close' column in data for {ticker}. Columns: {data.columns}"
            )
            return Response(
                {"error": "Stock data missing 'Close' column, yo"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
    except Exception as e:
        logger.error(f"Error fixing columns for {ticker}: {str(e)}")
        return Response(
            {"error": "Failed to process stock data columns"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    try:
        logger.info(f"Data rows for {ticker}: {len(data)}")
        if data.empty or len(data) < 50:
            logger.error(f"Not enough data for {ticker}: {len(data)} rows")
            return Response(
                {
                    "error": f"Not enough data to compute indicators, got {len(data)} rows"
                },
                status=status.HTTP_400_BAD_REQUEST,
            )
        for col in ["Close", "Open", "High", "Low", "Volume"]:
            data[col] = pd.to_numeric(data[col], errors="coerce")
        data = data.dropna(subset=["Close", "Open", "High", "Low", "Volume"])
        if data["Close"].isnull().all():
            logger.error(f"All 'Close' values are null for {ticker}")
            return Response(
                {"error": "Invalid 'Close' data, all values are null"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
        data["RSI"] = compute_rsi(data["Close"])
        data["MACD"], data["MACD_Signal"] = compute_macd(data["Close"])
        data["MA20"] = data["Close"].rolling(window=20).mean()
        data["MA50"] = data["Close"].rolling(window=50).mean()
        data["MA200"] = (
            data["Close"].rolling(window=200).mean() if len(data) >= 200 else None
        )
        data["ATR"] = compute_atr(data)
        data["BB_upper"], data["BB_lower"] = compute_bollinger_bands(data)
        logger.debug(
            f"Indicators computed: RSI={data['RSI'].iloc[-1]}, MA50={data['MA50'].iloc[-1]}, MA200={data['MA200'].iloc[-1] if len(data) >= 200 else 'None'}"
        )
    except Exception as e:
        logger.error(f"Error computing indicators for {ticker}: {str(e)}")
        logger.error(f"Data sample: {data.head().to_dict()}")
        return Response(
            {"error": f"Failed to compute technical indicators: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    try:
        latest_features = pd.DataFrame(
            {
                "Close": [data["Close"].iloc[-1]],
                "Open": [data["Open"].iloc[-1]],
                "High": [data["High"].iloc[-1]],
                "Low": [data["Low"].iloc[-1]],
                "Volume": [data["Volume"].iloc[-1]],
                "RSI": [data["RSI"].iloc[-1]],
                "MACD": [data["MACD"].iloc[-1]],
                "MACD_Signal": [data["MACD_Signal"].iloc[-1]],
                "MA20": [data["MA20"].iloc[-1]],
                "MA50": [data["MA50"].iloc[-1]],
                "MA200": [
                    data["MA200"].iloc[-1] if data["MA200"].iloc[-1] is not None else 0
                ],
                "ATR": [data["ATR"].iloc[-1]],
                "BB_upper": [data["BB_upper"].iloc[-1]],
            }
        )
        logger.debug(f"Feature values: {latest_features.to_dict(orient='records')}")
        if latest_features.isnull().values.any():
            logger.error(f"Invalid feature values for {ticker}: {latest_features}")
            return Response(
                {"error": "Invalid feature values for prediction"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
        if hasattr(model, "feature_names_in_"):
            expected_features = set(model.feature_names_in_)
            provided_features = set(latest_features.columns)
            if expected_features != provided_features:
                logger.error(
                    f"Feature mismatch for {ticker}. Expected: {expected_features}, Got: {provided_features}"
                )
                return Response(
                    {"error": f"Model expects features: {expected_features}"},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR,
                )
        prediction = model.predict(latest_features)
        prediction = "UP" if prediction[0] == 1 else "DOWN"
        confidence = None
        if hasattr(model, "predict_proba"):
            confidence = model.predict_proba(latest_features)[0][
                model.predict(latest_features)[0]
            ]
        else:
            logger.warning(
                f"Model {type(model)} does not support predict_proba for {ticker}"
            )
            confidence = 0.5
        logger.info(f"Prediction for {ticker}: {prediction}, Confidence: {confidence}")
    except Exception as e:
        logger.error(f"Prediction failed for {ticker}: {str(e)}")
        logger.error(f"Feature values: {latest_features.to_dict(orient='records')}")
        return Response(
            {"error": f"Prediction went sideways, check the model: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    try:
        data["Date"] = data.index.strftime("%Y-%m-%d")
        data = data.reset_index(drop=True)
        data = data.replace([np.nan, np.inf, -np.inf], None)
        for col in data.columns:
            if data[col].dtype == "datetime64[ns]":
                data[col] = data[col].astype(str)
        history = data.to_dict(orient="records")
        for record in history:
            for key, value in record.items():
                if not isinstance(value, (str, int, float, bool, type(None))):
                    logger.warning(f"Non-JSON-serializable value in {key}: {value}")
                    record[key] = str(value)
    except Exception as e:
        logger.error(f"Error converting DataFrame to dict: {str(e)}")
        logger.error(f"DataFrame dtypes: {data.dtypes}")
        logger.error(f"DataFrame sample: {data.head().to_dict()}")
        return Response(
            {"error": "Failed to process stock history, data’s too wild"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    features = {
        "rsi": latest_features["RSI"].iloc[0],
        "macd": latest_features["MACD"].iloc[0],
        "macd_signal": latest_features["MACD_Signal"].iloc[0],
        "ma50": latest_features["MA50"].iloc[0],
        "ma200": latest_features["MA200"].iloc[0],
    }

    return Response(
        {
            "ticker": ticker,
            "prediction": {
                "direction": prediction,
                "confidence": confidence * 100 if confidence is not None else 50.0,
                "current_price": data["Close"].iloc[-1],
                "predicted_change": (1.05 if prediction == "UP" else -0.05)
                * data["Close"].iloc[-1],
                "features": features,
            },
            "history": history,
        },
        status=status.HTTP_200_OK,
    )
