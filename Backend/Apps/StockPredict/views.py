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
from asgiref.sync import sync_to_async

BASE_DIR = Path(__file__).resolve().parent.parent.parent

logger = logging.getLogger("apps.stockpredict")

try:
    with open(os.path.join(BASE_DIR, "stock_model.pkl"), "rb") as f:
        model = pickle.load(f)
    logger.info("Stock prediction model loaded successfully")
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
    exp1 = data["Close"].ewm(span=fast, adjust=False).mean()
    exp2 = data["Close"].ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line


def validate_ticker(ticker):
    """Validate ticker format (uppercase, alphanumeric, 1-10 chars)."""
    if not ticker or not isinstance(ticker, str):
        return False
    if not re.match(r"^[A-Z0-9]{1,10}$", ticker):
        return False
    return True


async def fetch_stock_data(ticker, period="1mo", interval="1d"):
    """Fetch stock data asynchronously with timeout."""
    try:
        async with aiohttp.ClientSession() as session:
            data = await sync_to_async(yf.download)(
                ticker, period=period, interval=interval, progress=False, timeout=10
            )
            if data.empty:
                logger.error(f"No data found for ticker: {ticker}")
                return None
            return data
    except asyncio.TimeoutError:
        logger.error(f"Timeout fetching data for {ticker}")
        return None
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {str(e)}")
        return None


@api_view(["POST"])
def predict_stock_trend(request):
    ticker = request.data.get("ticker")

    if not validate_ticker(ticker):
        logger.warning(f"Invalid ticker format: {ticker}")
        return Response(
            {
                "error": "Invalid ticker format (use uppercase letters/numbers, 1-10 chars)"
            },
            status=status.HTTP_400_BAD_REQUEST,
        )

    if hasattr(request, "throttle_scope") and request.throttled:
        logger.warning(
            f"Rate limit exceeded for client IP: {request.META.get('REMOTE_ADDR')}"
        )
        return Response(
            {"error": "Rate limit exceeded (100 requests/day)"},
            status=status.HTTP_429_TOO_MANY_REQUESTS,
        )

    cache_key = f"stock_data_{ticker}"
    cached_data = cache.get(cache_key)
    if cached_data:
        logger.info(f"Using cached data for {ticker}")
        data = pd.DataFrame(cached_data)
    else:
        loop = asyncio.get_event_loop()
        data = loop.run_until_complete(fetch_stock_data(ticker))
        if data is None:
            return Response(
                {"error": "Failed to fetch stock data"},
                status=status.HTTP_404_NOT_FOUND,
            )
        cache.set(cache_key, data.to_dict(), timeout=3600)
        logger.info(f"Cached stock data for {ticker}")

    try:
        data["Return"] = data["Close"].pct_change()
        data["MA5"] = data["Close"].rolling(window=5).mean()
        data["MA20"] = data["Close"].rolling(window=20).mean()
        data["Volatility"] = data["Return"].rolling(window=20).std()
        data["RSI"] = compute_rsi(data["Close"], 14)
        data["Volume_Change"] = data["Volume"].pct_change()
        data["MACD"], data["MACD_Signal"] = compute_macd(data)
        data = data.dropna()

        if data.empty:
            logger.error(f"No valid data after processing for {ticker}")
            return Response(
                {"error": "Insufficient data for prediction"},
                status=status.HTTP_404_NOT_FOUND,
            )

        latest = data.iloc[-1][
            [
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
        ]
        prediction = model.predict([latest])[0]
        confidence = model.predict_proba([latest])[0].max()
        trend = "Up" if prediction == 1 else "Down"

        logger.info(f"Prediction for {ticker}: {trend} (Confidence: {confidence:.2%})")

        history = data[["Close"]].reset_index().to_dict(orient="records")
        history = [{"date": str(row["Date"]), "close": row["Close"]} for row in history]

        return Response(
            {
                "ticker": ticker,
                "prediction": trend,
                "confidence": round(confidence, 4),
                "history": history,
            },
            status=status.HTTP_200_OK,
        )

    except Exception as e:
        logger.error(f"Error processing {ticker}: {str(e)}")
        return Response(
            {"error": f"Failed to process request: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
