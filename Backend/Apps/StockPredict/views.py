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

    if not isinstance(ticker, str) or not ticker.isalnum() or len(ticker) > 10:
        logger.error(f"Sketchy ticker format: {ticker}")
        return Response(
            {"error": "Ticker’s gotta be uppercase letters/numbers, 1-10 chars ..."},
            status=status.HTTP_400_BAD_REQUEST,
        )
    ticker = ticker.upper()

    cache_key = f"stock_data_{ticker}"
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

    prediction = "Up"
    confidence = 0.82

    try:
        data.columns = [str(col) for col in data.columns]
        if isinstance(data.index, pd.MultiIndex):
            data.index = [str(idx) for idx in data.index]
        else:
            data.index = data.index.astype(str)
        data = data.replace([np.nan, np.inf, -np.inf], None)
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

    return Response(
        {
            "ticker": ticker,
            "prediction": prediction,
            "confidence": confidence,
            "history": history,
        },
        status=status.HTTP_200_OK,
    )
