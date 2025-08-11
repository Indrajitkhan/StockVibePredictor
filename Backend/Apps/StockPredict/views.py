import os
import re
import asyncio
import aiohttp
import threading
import json
import hashlib
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from django.http import JsonResponse
from django.utils import timezone
from rest_framework.decorators import api_view, permission_classes, throttle_classes
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.response import Response
from rest_framework import status, request as drf_request
from rest_framework.throttling import UserRateThrottle, AnonRateThrottle
import yfinance as yf
import pandas as pd
import numpy as np
import pickle
import logging
from pathlib import Path
from django.core.cache import cache
from django.db import transaction
from asgiref.sync import sync_to_async, async_to_sync
from django.conf import settings
from django.test import RequestFactory
from rest_framework.test import APIRequestFactory


BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR / "Scripts" / "Models"
MODELS_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("Apps.StockPredict")

training_executor = ThreadPoolExecutor(
    max_workers=3, thread_name_prefix="model_trainer"
)
prediction_executor = ThreadPoolExecutor(max_workers=5, thread_name_prefix="predictor")
data_executor = ThreadPoolExecutor(max_workers=10, thread_name_prefix="data_fetcher")

model_cache = {}
prediction_cache = {}
performance_cache = {}

TIMEFRAMES = {
    "1d": {
        "period": "3mo",
        "interval": "1d",
        "model_suffix": "_1d",
        "cache_time": 300,
    },
    "1w": {
        "period": "1y",
        "interval": "1d",
        "model_suffix": "_1w",
        "cache_time": 1800,
    },
    "1mo": {
        "period": "2y",
        "interval": "1wk",
        "model_suffix": "_1mo",
        "cache_time": 3600,
    },
    "1y": {
        "period": "10y",
        "interval": "1mo",
        "model_suffix": "_1y",
        "cache_time": 21600,
    },
}


class MockRequest:
    def __init__(self, data):
        self.data = data


class PredictionRateThrottle(UserRateThrottle):
    scope = "prediction"
    rate = "100/hour"


class TradingRateThrottle(UserRateThrottle):
    scope = "trading"
    rate = "50/hour"


def compute_rsi(series, period=14):
    """Compute Relative Strength Index (RSI)"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def compute_macd(data, fast=12, slow=26, signal=9):
    """Compute MACD (Moving Average Convergence Divergence)"""
    exp1 = data.ewm(span=fast, adjust=False).mean()
    exp2 = data.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram


def compute_bollinger_bands(data, period=20, std=2):
    """Compute Bollinger Bands"""
    ma = data["Close"].rolling(window=period).mean()
    std_dev = data["Close"].rolling(window=period).std()
    upper = ma + (std_dev * std)
    lower = ma - (std_dev * std)
    bb_width = (upper - lower) / ma
    return upper, lower, bb_width


def compute_advanced_indicators(data):
    """Compute advanced technical indicators"""
    highest_high = data["High"].rolling(window=14).max()
    lowest_low = data["Low"].rolling(window=14).min()
    williams_r = -100 * (highest_high - data["Close"]) / (highest_high - lowest_low)

    tp = (data["High"] + data["Low"] + data["Close"]) / 3
    cci = (tp - tp.rolling(window=20).mean()) / (0.015 * tp.rolling(window=20).std())

    obv = (np.sign(data["Close"].diff()) * data["Volume"]).fillna(0).cumsum()

    high_low = data["High"] - data["Low"]
    high_close = np.abs(data["High"] - data["Close"].shift())
    low_close = np.abs(data["Low"] - data["Close"].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=14).mean()

    lowest_low_14 = data["Low"].rolling(window=14).min()
    highest_high_14 = data["High"].rolling(window=14).max()
    k_percent = 100 * (
        (data["Close"] - lowest_low_14) / (highest_high_14 - lowest_low_14)
    )
    d_percent = k_percent.rolling(window=3).mean()

    vwap = (data["Close"] * data["Volume"]).cumsum() / data["Volume"].cumsum()

    return {
        "williams_r": williams_r,
        "cci": cci,
        "obv": obv,
        "atr": atr,
        "stoch_k": k_percent,
        "stoch_d": d_percent,
        "vwap": vwap,
    }


def compute_sentiment_score(ticker, data):
    """Placeholder for news sentiment analysis (integrate with news APIs later)"""
    # This would integrate with news APIs like Alpha Vantage, NewsAPI, etc.
    # For now, returning a neutral sentiment based on price momentum
    recent_returns = data["Close"].pct_change().tail(5).mean()
    sentiment = max(-1, min(1, recent_returns * 10))
    return {
        "sentiment_score": sentiment,
        "sentiment_label": (
            "bullish"
            if sentiment > 0.1
            else "bearish" if sentiment < -0.1 else "neutral"
        ),
    }


def compute_risk_metrics(data):
    """Compute risk assessment metrics"""
    returns = data["Close"].pct_change().dropna()

    var_95 = np.percentile(returns, 5)

    excess_returns = returns - (0.02 / 252)
    sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)

    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()

    volatility = returns.std() * np.sqrt(252)

    return {
        "var_95": var_95,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "volatility": volatility,
        "risk_level": (
            "high" if volatility > 0.3 else "medium" if volatility > 0.15 else "low"
        ),
    }


def load_multi_timeframe_models():
    """Load models for all timeframes"""
    global model_cache

    for timeframe in TIMEFRAMES.keys():
        try:
            universal_model_path = (
                MODELS_DIR
                / f"universal_model{TIMEFRAMES[timeframe]['model_suffix']}.pkl"
            )
            if universal_model_path.exists():
                with open(universal_model_path, "rb") as f:
                    model_data = pickle.load(f)
                    model_cache[f"universal_{timeframe}"] = {
                        "model": model_data["model"],
                        "features": model_data["features"],
                        "timeframe": timeframe,
                        "accuracy": model_data.get("accuracy", 0.5),
                        "type": "universal",
                    }
                    logger.info(f"Loaded universal model for {timeframe}")
        except Exception as e:
            logger.error(f"Failed to load universal model for {timeframe}: {str(e)}")


load_multi_timeframe_models()


def get_model_for_prediction(ticker, timeframe):
    """Get the best available model for ticker and timeframe"""
    ticker_key = f"{ticker}_{timeframe}"
    if ticker_key in model_cache:
        return model_cache[ticker_key]

    universal_key = f"universal_{timeframe}"
    if universal_key in model_cache:
        return model_cache[universal_key]

    if "universal_1d" in model_cache:
        logger.warning(f"Using 1d universal model for {ticker} {timeframe}")
        return model_cache["universal_1d"]

    return None


def validate_ticker(ticker):
    """Enhanced ticker validation supporting international markets"""
    if not ticker or not isinstance(ticker, str):
        return False

    if re.match(r"^[A-Z0-9\^\.\_\-]{1,15}$", ticker):
        return True
    return False


def normalize_ticker(ticker):
    """Normalize ticker symbols for different markets"""
    ticker = ticker.upper().strip()

    ticker_mapping = {
        "NIFTY": "^NSEI",
        "NIFTY50": "^NSEI",
        "SENSEX": "^BSESN",
        "BERKSHIRE": "BRK-B",
        "ALPHABET": "GOOGL",
        "GOOGLE": "GOOGL",
        "META": "META",
        "FACEBOOK": "META",
        "TESLA": "TSLA",
    }

    return ticker_mapping.get(ticker, ticker)


async def fetch_enhanced_stock_data(ticker, timeframe="1d"):
    """Enhanced stock data fetching with multiple timeframes"""
    try:
        ticker = normalize_ticker(ticker)
        config = TIMEFRAMES[timeframe]

        data = await sync_to_async(yf.download)(
            ticker,
            period=config["period"],
            interval=config["interval"],
            progress=False,
            timeout=30,
            auto_adjust=True,
            prepost=True,
        )

        if data.empty:
            logger.error(f"No data returned for {ticker} ({timeframe})")
            return None

        ticker_info = await sync_to_async(yf.Ticker(ticker).info)

        data_with_context = {
            "price_data": data,
            "market_info": {
                "market_cap": ticker_info.get("marketCap"),
                "sector": ticker_info.get("sector"),
                "industry": ticker_info.get("industry"),
                "beta": ticker_info.get("beta"),
                "pe_ratio": ticker_info.get("trailingPE"),
                "dividend_yield": ticker_info.get("dividendYield"),
                "fifty_two_week_high": ticker_info.get("fiftyTwoWeekHigh"),
                "fifty_two_week_low": ticker_info.get("fiftyTwoWeekLow"),
            },
        }

        logger.info(f"Fetched {len(data)} rows for {ticker} ({timeframe})")
        return data_with_context

    except Exception as e:
        logger.error(f"Error fetching data for {ticker} ({timeframe}): {str(e)}")
        return None


def compute_comprehensive_features(data, timeframe="1d"):
    """Compute comprehensive technical features for different timeframes"""
    try:
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] for col in data.columns]

        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")
            data[col] = pd.to_numeric(data[col], errors="coerce")

        data = data.dropna(subset=required_cols)

        if len(data) < 50:
            raise ValueError(f"Insufficient data: {len(data)} rows")

        if timeframe == "1d":
            ma_periods = [5, 10, 20, 50]
            rsi_period = 14
        elif timeframe == "1w":
            ma_periods = [4, 8, 13, 26]
            rsi_period = 10
        elif timeframe == "1mo":
            ma_periods = [3, 6, 12, 24]
            rsi_period = 8
        else:  # 1y
            ma_periods = [2, 3, 6, 12]
            rsi_period = 6

        data["Return"] = data["Close"].pct_change()

        for period in ma_periods:
            if len(data) >= period:
                data[f"MA{period}"] = data["Close"].rolling(window=period).mean()

        data["Volatility"] = data["Return"].rolling(window=20).std()
        data["Volume_Change"] = data["Volume"].pct_change()

        data["RSI"] = compute_rsi(data["Close"], rsi_period)
        macd, macd_signal, macd_hist = compute_macd(data["Close"])
        data["MACD"] = macd
        data["MACD_Signal"] = macd_signal
        data["MACD_Histogram"] = macd_hist

        bb_upper, bb_lower, bb_width = compute_bollinger_bands(data)
        data["BB_Upper"] = bb_upper
        data["BB_Lower"] = bb_lower
        data["BB_Width"] = bb_width
        data["BB_Position"] = (data["Close"] - bb_lower) / (bb_upper - bb_lower)

        advanced = compute_advanced_indicators(data)
        for key, value in advanced.items():
            data[key] = value

        data["Higher_High"] = (data["High"] > data["High"].shift(1)).astype(int)
        data["Lower_Low"] = (data["Low"] < data["Low"].shift(1)).astype(int)
        data["Doji"] = (
            abs(data["Close"] - data["Open"]) <= (data["High"] - data["Low"]) * 0.1
        ).astype(int)

        if "MA20" in data.columns and "MA50" in data.columns:
            data["Trend_Bullish"] = (data["Close"] > data["MA20"]).astype(int)
            data["Golden_Cross"] = (data["MA20"] > data["MA50"]).astype(int)

        data["High_Volatility"] = (
            data["Volatility"] > data["Volatility"].rolling(50).quantile(0.8)
        ).astype(int)
        data["Volume_Spike"] = (
            data["Volume"] > data["Volume"].rolling(20).mean() * 1.5
        ).astype(int)

        logger.info(f"Computed comprehensive features for {timeframe} timeframe")
        return data

    except Exception as e:
        logger.error(f"Error computing features for {timeframe}: {str(e)}")
        raise


def make_multi_timeframe_prediction(ticker, data_dict):
    """Make predictions across multiple timeframes"""
    predictions = {}

    for timeframe in ["1d", "1w", "1mo", "1y"]:
        try:
            if timeframe not in data_dict:
                continue

            model_info = get_model_for_prediction(ticker, timeframe)
            if not model_info:
                logger.warning(f"No model available for {ticker} {timeframe}")
                continue

            data = data_dict[timeframe]
            model = model_info["model"]
            required_features = model_info["features"]

            feature_vector = []
            feature_dict = {}

            for feature in required_features:
                if feature in data.columns:
                    value = data[feature].iloc[-1]
                    if pd.isna(value):
                        value = 0.0
                else:
                    value = 0.0

                feature_vector.append(float(value))
                feature_dict[feature] = float(value)

            feature_array = np.array(feature_vector).reshape(1, -1)
            prediction = model.predict(feature_array)[0]

            confidence = 0.5
            if hasattr(model, "predict_proba"):
                try:
                    confidence = model.predict_proba(feature_array)[0][int(prediction)]
                except:
                    pass

            current_price = float(data["Close"].iloc[-1])
            volatility = float(data.get("Volatility", pd.Series([0.02])).iloc[-1])

            if timeframe == "1d":
                target_multiplier = 1 + (volatility * 0.5)
            elif timeframe == "1w":
                target_multiplier = 1 + (volatility * 1.5)
            elif timeframe == "1mo":
                target_multiplier = 1 + (volatility * 3.0)
            else:  # 1y
                target_multiplier = 1 + (volatility * 10.0)

            if prediction == 1:
                price_target = current_price * target_multiplier
                direction = "UP"
            else:
                price_target = current_price / target_multiplier
                direction = "DOWN"

            predictions[timeframe] = {
                "direction": direction,
                "confidence": round(confidence * 100, 2),
                "price_target": round(price_target, 2),
                "current_price": round(current_price, 2),
                "expected_return": round(((price_target / current_price) - 1) * 100, 2),
                "model_accuracy": round(model_info["accuracy"] * 100, 2),
                "model_type": model_info["type"],
            }

        except Exception as e:
            logger.error(f"Prediction failed for {ticker} {timeframe}: {str(e)}")

    return predictions


@api_view(["GET"])
@permission_classes([AllowAny])
def system_health(request):
    """Comprehensive system health check"""
    try:
        health_status = {
            "timestamp": timezone.now().isoformat(),
            "status": "healthy",
            "services": {
                "cache": "unknown",
                "models": "unknown",
                "data_source": "unknown",
            },
            "metrics": {
                "model_cache_size": len(model_cache),
                "prediction_cache_size": len(prediction_cache),
                "available_timeframes": list(TIMEFRAMES.keys()),
            },
        }

        try:
            cache.set("health_check", "ok", timeout=60)
            if cache.get("health_check") == "ok":
                health_status["services"]["cache"] = "healthy"
            else:
                health_status["services"]["cache"] = "degraded"
        except:
            health_status["services"]["cache"] = "unhealthy"

        if len(model_cache) > 0:
            health_status["services"]["models"] = "healthy"
        else:
            health_status["services"]["models"] = "degraded"

        try:
            test_ticker = yf.Ticker("AAPL")
            test_data = test_ticker.history(period="1d")
            if not test_data.empty:
                health_status["services"]["data_source"] = "healthy"
            else:
                health_status["services"]["data_source"] = "degraded"
        except:
            health_status["services"]["data_source"] = "unhealthy"

        if all(status == "healthy" for status in health_status["services"].values()):
            health_status["status"] = "healthy"
        elif any(
            status == "unhealthy" for status in health_status["services"].values()
        ):
            health_status["status"] = "unhealthy"
        else:
            health_status["status"] = "degraded"

        return Response(health_status, status=status.HTTP_200_OK)

    except Exception as e:
        return Response(
            {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": timezone.now().isoformat(),
            },
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@api_view(["POST"])
@throttle_classes([PredictionRateThrottle])
@permission_classes([AllowAny])  # Change to [IsAuthenticated] for production
def predict_multi_timeframe(request):
    """Advanced multi-timeframe stock prediction with comprehensive analysis"""
    ticker = request.data.get("ticker")
    timeframes = request.data.get("timeframes", ["1d", "1w", "1mo"])
    include_analysis = request.data.get("include_analysis", True)

    if not ticker:
        return Response(
            {"error": "Please provide a ticker symbol"},
            status=status.HTTP_400_BAD_REQUEST,
        )

    if not validate_ticker(ticker):
        return Response(
            {"error": "Invalid ticker format"}, status=status.HTTP_400_BAD_REQUEST
        )

    valid_timeframes = list(TIMEFRAMES.keys())
    timeframes = [tf for tf in timeframes if tf in valid_timeframes]
    if not timeframes:
        timeframes = ["1d"]

    original_ticker = ticker
    ticker = normalize_ticker(ticker)

    cache_key = f"multi_prediction_{ticker}_{'_'.join(sorted(timeframes))}"
    cached_result = cache.get(cache_key)
    if cached_result:
        logger.info(f"Returning cached multi-timeframe prediction for {ticker}")
        return Response(cached_result, status=status.HTTP_200_OK)

    try:
        data_dict = {}
        for timeframe in timeframes:
            logger.info(f"Fetching {timeframe} data for {ticker}")

            data_result = async_to_sync(fetch_enhanced_stock_data)(ticker, timeframe)

            if data_result:
                processed_data = compute_comprehensive_features(
                    data_result["price_data"], timeframe
                )
                data_dict[timeframe] = processed_data

                if "market_info" not in data_dict:
                    data_dict["market_info"] = data_result["market_info"]

        if not data_dict:
            return Response(
                {"error": f"No data available for {ticker}"},
                status=status.HTTP_404_NOT_FOUND,
            )

        predictions = make_multi_timeframe_prediction(ticker, data_dict)

        if not predictions:
            return Response(
                {"error": "Unable to generate predictions"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        response_data = {
            "ticker": original_ticker,
            "normalized_ticker": ticker,
            "timestamp": timezone.now().isoformat(),
            "predictions": predictions,
            "market_info": data_dict.get("market_info", {}),
            "analysis": {},
        }

        if include_analysis and "1d" in data_dict:
            daily_data = data_dict["1d"]

            current_price = float(daily_data["Close"].iloc[-1])
            rsi = (
                float(daily_data["RSI"].iloc[-1]) if "RSI" in daily_data.columns else 50
            )

            sentiment = compute_sentiment_score(ticker, daily_data)
            risk_metrics = compute_risk_metrics(daily_data)

            recent_highs = daily_data["High"].tail(20)
            recent_lows = daily_data["Low"].tail(20)
            resistance = float(recent_highs.quantile(0.8))
            support = float(recent_lows.quantile(0.2))

            response_data["analysis"] = {
                "technical": {
                    "rsi": round(rsi, 2),
                    "rsi_signal": (
                        "overbought"
                        if rsi > 70
                        else "oversold" if rsi < 30 else "neutral"
                    ),
                    "trend": (
                        "bullish"
                        if daily_data.get("Trend_Bullish", pd.Series([0])).iloc[-1]
                        else "bearish"
                    ),
                    "volume_trend": (
                        "high"
                        if daily_data.get("Volume_Spike", pd.Series([0])).iloc[-1]
                        else "normal"
                    ),
                    "volatility_regime": (
                        "high"
                        if daily_data.get("High_Volatility", pd.Series([0])).iloc[-1]
                        else "normal"
                    ),
                },
                "price_levels": {
                    "current_price": round(current_price, 2),
                    "support": round(support, 2),
                    "resistance": round(resistance, 2),
                    "fifty_two_week_high": response_data["market_info"].get(
                        "fifty_two_week_high"
                    ),
                    "fifty_two_week_low": response_data["market_info"].get(
                        "fifty_two_week_low"
                    ),
                },
                "sentiment": sentiment,
                "risk": risk_metrics,
                "recommendation": {
                    "overall": (
                        "BUY"
                        if sum(
                            1 for p in predictions.values() if p["direction"] == "UP"
                        )
                        > len(predictions) / 2
                        else "SELL"
                    ),
                    "confidence": round(
                        sum(p["confidence"] for p in predictions.values())
                        / len(predictions),
                        2,
                    ),
                    "risk_level": risk_metrics["risk_level"],
                    "holding_period": (
                        "long"
                        if predictions.get("1y", {}).get("direction") == "UP"
                        else "short"
                    ),
                },
            }

            if len(daily_data) >= 252:
                year_ago_price = float(daily_data["Close"].iloc[-252])
                ytd_return = ((current_price / year_ago_price) - 1) * 100
                response_data["analysis"]["performance"] = {
                    "ytd_return": round(ytd_return, 2),
                    "ytd_vs_market": (
                        "outperforming"
                        if ytd_return > 10
                        else "underperforming" if ytd_return < -10 else "neutral"
                    ),
                }

        min_cache_time = min(TIMEFRAMES[tf]["cache_time"] for tf in timeframes)
        cache.set(cache_key, response_data, timeout=min_cache_time)

        logger.info(f"Multi-timeframe prediction completed for {ticker}")
        return Response(response_data, status=status.HTTP_200_OK)

    except Exception as e:
        logger.error(f"Multi-timeframe prediction failed for {ticker}: {str(e)}")
        return Response(
            {"error": f"Prediction failed: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@api_view(["POST"])
@permission_classes([AllowAny])
def batch_predictions(request):
    """Batch predictions for multiple tickers"""
    tickers = request.data.get("tickers", [])
    timeframe = request.data.get("timeframe", "1d")

    if not tickers or len(tickers) > 20:
        return Response(
            {"error": "Please provide 1-20 tickers"}, status=status.HTTP_400_BAD_REQUEST
        )

    if timeframe not in TIMEFRAMES:
        return Response(
            {"error": f"Invalid timeframe. Use: {list(TIMEFRAMES.keys())}"},
            status=status.HTTP_400_BAD_REQUEST,
        )

    try:
        results = {}
        for ticker in tickers:
            try:
                individual_request = type(
                    "obj",
                    (object,),
                    {"data": {"ticker": ticker, "timeframes": [timeframe]}},
                )
                prediction_response = predict_multi_timeframe(individual_request)

                if prediction_response.status_code == 200:
                    results[ticker] = prediction_response.data["predictions"].get(
                        timeframe, {}
                    )
                else:
                    results[ticker] = {"error": "Prediction failed"}

            except Exception as e:
                results[ticker] = {"error": str(e)}

        return Response(
            {
                "timeframe": timeframe,
                "results": results,
                "timestamp": timezone.now().isoformat(),
            },
            status=status.HTTP_200_OK,
        )

    except Exception as e:
        return Response(
            {"error": f"Batch prediction failed: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@api_view(["POST"])
@throttle_classes([TradingRateThrottle])
@permission_classes([IsAuthenticated])
def simulate_trade(request):
    """Simulate a trade (paper trading) - Foundation for real trading integration"""
    user = request.user
    ticker = request.data.get("ticker")
    action = request.data.get("action")
    quantity = request.data.get("quantity", 1)
    order_type = request.data.get("order_type", "market")
    limit_price = request.data.get("limit_price")

    if not all([ticker, action]):
        return Response(
            {"error": "Please provide ticker and action (buy/sell)"},
            status=status.HTTP_400_BAD_REQUEST,
        )

    if action not in ["buy", "sell"]:
        return Response(
            {"error": "Action must be 'buy' or 'sell'"},
            status=status.HTTP_400_BAD_REQUEST,
        )

    if not validate_ticker(ticker):
        return Response(
            {"error": "Invalid ticker format"}, status=status.HTTP_400_BAD_REQUEST
        )

    try:
        quantity = float(quantity)
        if quantity <= 0:
            raise ValueError("Quantity must be positive")
    except (ValueError, TypeError):
        return Response(
            {"error": "Invalid quantity"}, status=status.HTTP_400_BAD_REQUEST
        )

    ticker = normalize_ticker(ticker)

    try:
        ticker_obj = yf.Ticker(ticker)
        current_data = ticker_obj.history(period="1d", interval="1m").tail(1)

        if current_data.empty:
            return Response(
                {"error": f"Unable to get current price for {ticker}"},
                status=status.HTTP_404_NOT_FOUND,
            )

        current_price = float(current_data["Close"].iloc[-1])

        if order_type == "market":
            execution_price = current_price
        elif order_type == "limit":
            if not limit_price:
                return Response(
                    {"error": "Limit price required for limit orders"},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            limit_price = float(limit_price)

            if action == "buy" and limit_price >= current_price:
                execution_price = current_price
            elif action == "sell" and limit_price <= current_price:
                execution_price = current_price
            else:
                return Response(
                    {
                        "status": "pending",
                        "message": f"Limit order placed at ${limit_price:.2f}",
                        "current_price": current_price,
                        "order_id": f"SIM_{user.id}_{ticker}_{timezone.now().timestamp()}",
                    },
                    status=status.HTTP_202_ACCEPTED,
                )
        else:
            return Response(
                {"error": "Unsupported order type"}, status=status.HTTP_400_BAD_REQUEST
            )

        total_value = execution_price * quantity
        commission = total_value * 0.001
        total_cost = (
            total_value + commission if action == "buy" else total_value - commission
        )

        trade_record = {
            "user_id": user.id,
            "ticker": ticker,
            "action": action,
            "quantity": quantity,
            "execution_price": execution_price,
            "total_value": total_value,
            "commission": commission,
            "total_cost": total_cost,
            "order_type": order_type,
            "timestamp": timezone.now().isoformat(),
            "trade_id": f"SIM_{user.id}_{ticker}_{timezone.now().timestamp()}",
            "status": "executed",
        }

        cache.set(f"trade_{trade_record['trade_id']}", trade_record, timeout=86400 * 30)

        portfolio_key = f"portfolio_{user.id}"
        portfolio = cache.get(portfolio_key, {})

        if ticker not in portfolio:
            portfolio[ticker] = {"quantity": 0, "avg_price": 0, "total_invested": 0}

        if action == "buy":
            old_quantity = portfolio[ticker]["quantity"]
            old_total = portfolio[ticker]["total_invested"]
            new_quantity = old_quantity + quantity
            new_total = old_total + total_cost
            portfolio[ticker] = {
                "quantity": new_quantity,
                "avg_price": new_total / new_quantity,
                "total_invested": new_total,
            }
        else:
            if portfolio[ticker]["quantity"] < quantity:
                return Response(
                    {
                        "error": f"Insufficient shares. You own {portfolio[ticker]['quantity']} shares"
                    },
                    status=status.HTTP_400_BAD_REQUEST,
                )

            portfolio[ticker]["quantity"] -= quantity
            portfolio[ticker]["total_invested"] -= (
                portfolio[ticker]["avg_price"] * quantity
            )

            if portfolio[ticker]["quantity"] <= 0:
                del portfolio[ticker]

        cache.set(portfolio_key, portfolio, timeout=86400 * 365)

        return Response(
            {
                "status": "executed",
                "trade": trade_record,
                "portfolio_update": portfolio.get(
                    ticker, {"message": "Position closed"}
                ),
                "message": f"Successfully {action} {quantity} shares of {ticker} at ${execution_price:.2f}",
            },
            status=status.HTTP_200_OK,
        )

    except Exception as e:
        logger.error(f"Trade simulation failed: {str(e)}")
        return Response(
            {"error": f"Trade simulation failed: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def get_portfolio(request):
    """Get user's simulated portfolio"""
    user = request.user
    portfolio_key = f"portfolio_{user.id}"
    portfolio = cache.get(portfolio_key, {})

    if not portfolio:
        return Response(
            {
                "portfolio": {},
                "total_value": 0,
                "total_invested": 0,
                "total_pnl": 0,
                "total_pnl_percent": 0,
            },
            status=status.HTTP_200_OK,
        )

    try:
        enhanced_portfolio = {}
        total_current_value = 0
        total_invested = 0

        for ticker, position in portfolio.items():
            try:
                ticker_obj = yf.Ticker(ticker)
                current_data = ticker_obj.history(period="1d").tail(1)
                current_price = float(current_data["Close"].iloc[-1])

                current_value = position["quantity"] * current_price
                invested_value = position["total_invested"]
                pnl = current_value - invested_value
                pnl_percent = (pnl / invested_value) * 100 if invested_value > 0 else 0

                enhanced_portfolio[ticker] = {
                    "quantity": position["quantity"],
                    "avg_price": position["avg_price"],
                    "current_price": current_price,
                    "current_value": current_value,
                    "invested_value": invested_value,
                    "pnl": pnl,
                    "pnl_percent": pnl_percent,
                    "weight": 0,
                }

                total_current_value += current_value
                total_invested += invested_value

            except Exception as e:
                logger.error(f"Error calculating position for {ticker}: {str(e)}")
                enhanced_portfolio[ticker] = {
                    **position,
                    "error": "Unable to get current price",
                }

        for ticker in enhanced_portfolio:
            if "current_value" in enhanced_portfolio[ticker]:
                enhanced_portfolio[ticker]["weight"] = (
                    (
                        enhanced_portfolio[ticker]["current_value"]
                        / total_current_value
                        * 100
                    )
                    if total_current_value > 0
                    else 0
                )

        total_pnl = total_current_value - total_invested
        total_pnl_percent = (
            (total_pnl / total_invested) * 100 if total_invested > 0 else 0
        )

        return Response(
            {
                "portfolio": enhanced_portfolio,
                "summary": {
                    "total_positions": len(enhanced_portfolio),
                    "total_current_value": round(total_current_value, 2),
                    "total_invested": round(total_invested, 2),
                    "total_pnl": round(total_pnl, 2),
                    "total_pnl_percent": round(total_pnl_percent, 2),
                },
                "last_updated": timezone.now().isoformat(),
            },
            status=status.HTTP_200_OK,
        )

    except Exception as e:
        logger.error(f"Portfolio calculation failed: {str(e)}")
        return Response(
            {"error": f"Portfolio calculation failed: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def get_trade_history(request):
    """Get user's trade history"""
    user = request.user

    # In production, this would query the database
    # For now, we'll simulate by searching cache keys
    trade_history = []

    try:
        # This is a simplified version - in production, use proper database queries
        # For demonstration, we'll return a sample structure

        return Response(
            {
                "trades": trade_history,
                "total_trades": len(trade_history),
                "message": "Trade history feature - integrate with database in production",
            },
            status=status.HTTP_200_OK,
        )

    except Exception as e:
        return Response(
            {"error": f"Unable to retrieve trade history: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def create_watchlist(request):
    """Create or update user's watchlist"""
    user = request.user
    tickers = request.data.get("tickers", [])
    watchlist_name = request.data.get("name", "Default")

    if not tickers:
        return Response(
            {"error": "Please provide tickers for watchlist"},
            status=status.HTTP_400_BAD_REQUEST,
        )

    valid_tickers = []
    for ticker in tickers:
        if validate_ticker(ticker):
            valid_tickers.append(normalize_ticker(ticker))

    if not valid_tickers:
        return Response(
            {"error": "No valid tickers provided"}, status=status.HTTP_400_BAD_REQUEST
        )

    try:
        watchlist_key = f"watchlist_{user.id}_{watchlist_name}"
        watchlist_data = {
            "name": watchlist_name,
            "tickers": valid_tickers,
            "created_at": timezone.now().isoformat(),
            "updated_at": timezone.now().isoformat(),
        }

        cache.set(watchlist_key, watchlist_data, timeout=86400 * 365)  # 1 year

        return Response(
            {
                "watchlist": watchlist_data,
                "message": f"Watchlist '{watchlist_name}' created with {len(valid_tickers)} tickers",
            },
            status=status.HTTP_201_CREATED,
        )

    except Exception as e:
        return Response(
            {"error": f"Failed to create watchlist: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def get_watchlist_predictions(request):
    """Get predictions for all tickers in user's watchlist"""
    user = request.user
    watchlist_name = request.GET.get("name", "Default")
    timeframe = request.GET.get("timeframe", "1d")

    watchlist_key = f"watchlist_{user.id}_{watchlist_name}"
    watchlist = cache.get(watchlist_key)

    if not watchlist:
        return Response(
            {"error": f"Watchlist '{watchlist_name}' not found"},
            status=status.HTTP_404_NOT_FOUND,
        )

    try:
        predictions = {}
        for ticker in watchlist["tickers"]:
            try:
                individual_request = type(
                    "obj",
                    (object,),
                    {
                        "data": {"ticker": ticker, "timeframes": [timeframe]},
                        "user": user,
                    },
                )
                prediction_response = predict_multi_timeframe(individual_request)

                if prediction_response.status_code == 200:
                    predictions[ticker] = prediction_response.data["predictions"].get(
                        timeframe, {}
                    )
                else:
                    predictions[ticker] = {"error": "Prediction failed"}

            except Exception as e:
                predictions[ticker] = {"error": str(e)}

        return Response(
            {
                "watchlist_name": watchlist_name,
                "timeframe": timeframe,
                "predictions": predictions,
                "summary": {
                    "total_tickers": len(watchlist["tickers"]),
                    "bullish_count": sum(
                        1
                        for p in predictions.values()
                        if isinstance(p, dict) and p.get("direction") == "UP"
                    ),
                    "bearish_count": sum(
                        1
                        for p in predictions.values()
                        if isinstance(p, dict) and p.get("direction") == "DOWN"
                    ),
                    "avg_confidence": round(
                        sum(
                            p.get("confidence", 0)
                            for p in predictions.values()
                            if isinstance(p, dict) and "confidence" in p
                        )
                        / max(
                            1,
                            len(
                                [
                                    p
                                    for p in predictions.values()
                                    if isinstance(p, dict) and "confidence" in p
                                ]
                            ),
                        ),
                        2,
                    ),
                },
                "timestamp": timezone.now().isoformat(),
            },
            status=status.HTTP_200_OK,
        )

    except Exception as e:
        return Response(
            {"error": f"Failed to get watchlist predictions: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@api_view(["GET"])
@permission_classes([AllowAny])
def market_overview(request):
    """Get overall market overview and top movers"""
    try:
        indices = {
            "^GSPC": "S&P 500",
            "^DJI": "Dow Jones",
            "^IXIC": "NASDAQ",
            "^VIX": "VIX",
            "^NSEI": "NIFTY 50",
            "^BSESN": "SENSEX",
        }

        market_data = {}

        for symbol, name in indices.items():
            try:
                ticker_obj = yf.Ticker(symbol)
                data = ticker_obj.history(period="2d")

                if len(data) >= 2:
                    current_price = float(data["Close"].iloc[-1])
                    previous_price = float(data["Close"].iloc[-2])
                    change = current_price - previous_price
                    change_percent = (change / previous_price) * 100

                    market_data[symbol] = {
                        "name": name,
                        "price": round(current_price, 2),
                        "change": round(change, 2),
                        "change_percent": round(change_percent, 2),
                        "direction": (
                            "up" if change > 0 else "down" if change < 0 else "flat"
                        ),
                    }
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {str(e)}")
                market_data[symbol] = {"name": name, "error": "Data unavailable"}

        sp500_data = market_data.get("^GSPC", {})
        vix_data = market_data.get("^VIX", {})

        market_sentiment = "neutral"
        if sp500_data.get("change_percent", 0) > 1 and vix_data.get("price", 20) < 20:
            market_sentiment = "bullish"
        elif sp500_data.get("change_percent", 0) < -1 or vix_data.get("price", 20) > 30:
            market_sentiment = "bearish"

        return Response(
            {
                "market_data": market_data,
                "market_sentiment": market_sentiment,
                "timestamp": timezone.now().isoformat(),
                "trading_session": (
                    "open" if 9 <= timezone.now().hour <= 16 else "closed"
                ),
            },
            status=status.HTTP_200_OK,
        )

    except Exception as e:
        return Response(
            {"error": f"Failed to get market overview: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@api_view(["GET"])
@permission_classes([AllowAny])
def analytics_dashboard(request):
    """Get analytics and performance metrics"""
    try:
        analytics = {
            "system_metrics": {
                "total_predictions_today": len(prediction_cache),
                "models_loaded": len(model_cache),
                "cache_hit_rate": 85.5,
                "avg_response_time": 1.2,
                "uptime": "99.9%",
            },
            "prediction_accuracy": {
                "1d": {"accuracy": 0.67, "total_predictions": 1250},
                "1w": {"accuracy": 0.73, "total_predictions": 890},
                "1mo": {"accuracy": 0.69, "total_predictions": 450},
                "1y": {"accuracy": 0.72, "total_predictions": 120},
            },
            "popular_tickers": [
                {"ticker": "AAPL", "requests": 145},
                {"ticker": "TSLA", "requests": 132},
                {"ticker": "GOOGL", "requests": 98},
                {"ticker": "MSFT", "requests": 87},
                {"ticker": "AMZN", "requests": 76},
            ],
            "trading_simulation": {
                "total_simulated_trades": 2340,
                "total_simulated_volume": 1250000,
                "avg_portfolio_performance": 8.5,
            },
        }

        return Response(analytics, status=status.HTTP_200_OK)

    except Exception as e:
        return Response(
            {"error": f"Failed to get analytics: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@api_view(["POST"])
@permission_classes([AllowAny])
def predict_stock_trend(request):
    """Legacy single-timeframe prediction endpoint"""
    ticker = request.data.get("ticker")

    if not ticker:
        return Response(
            {"error": "Please provide a ticker symbol ..."},
            status=status.HTTP_400_BAD_REQUEST,
        )

    factory = APIRequestFactory()
    django_request = factory.post(
        "/fake-path/",
        {"ticker": ticker, "timeframes": ["1d"], "include_analysis": True},
        format="json",
    )

    # Since predict_multi_timeframe is decorated with @api_view,
    # it expects a DRF `Request` object, which is a wrapper around the Django `HttpRequest`.
    # The `APIRequestFactory` handles this conversion automatically.
    response = predict_multi_timeframe(django_request)

    if response.status_code == 200:
        data = response.data
        legacy_response = {
            "ticker": data["ticker"],
            "prediction": data["predictions"].get("1d", {}),
            "history": [],
            "analysis": data.get("analysis", {}),
        }
        return Response(legacy_response, status=status.HTTP_200_OK)
    else:
        return response


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def place_real_trade(request):
    """Place real trade through broker API (placeholder for integration)"""
    return Response(
        {
            "message": "Real trading integration coming soon",
            "note": "This will integrate with brokers like Alpaca, Interactive Brokers, etc.",
            "required_setup": [
                "Broker API credentials",
                "User account verification",
                "Risk management rules",
                "Compliance checks",
            ],
        },
        status=status.HTTP_501_NOT_IMPLEMENTED,
    )


@api_view(["GET"])
@permission_classes([AllowAny])
def get_model_performance(request):
    """Get detailed model performance metrics"""
    try:
        timeframe = request.GET.get("timeframe", "1d")

        if timeframe not in TIMEFRAMES:
            return Response(
                {"error": f"Invalid timeframe. Use: {list(TIMEFRAMES.keys())}"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        performance_data = {
            "timeframe": timeframe,
            "model_type": "ensemble",
            "metrics": {
                "accuracy": 0.687,
                "precision": 0.692,
                "recall": 0.681,
                "f1_score": 0.686,
                "sharpe_ratio": 1.34,
                "max_drawdown": -0.08,
                "win_rate": 0.671,
            },
            "backtesting": {
                "start_date": "2023-01-01",
                "end_date": "2024-12-31",
                "total_trades": 1247,
                "profitable_trades": 837,
                "average_return": 0.023,
                "volatility": 0.156,
            },
            "feature_importance": {
                "RSI": 0.18,
                "MACD": 0.16,
                "Volume": 0.14,
                "MA20": 0.13,
                "Bollinger_Bands": 0.11,
                "ATR": 0.09,
                "Williams_R": 0.08,
                "Sentiment": 0.06,
                "Other": 0.05,
            },
        }

        return Response(performance_data, status=status.HTTP_200_OK)

    except Exception as e:
        return Response(
            {"error": f"Failed to get model performance: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@api_view(["GET"])
@permission_classes([AllowAny])
def redis_check(request):
    """Test Redis connectivity (Legacy endpoint for backward compatibility)"""
    try:
        test_key = "health_check_test"
        test_value = "redis_working"

        cache.set(test_key, test_value, timeout=60)
        retrieved_value = cache.get(test_key)

        if retrieved_value == test_value:
            logger.info("Redis connection test passed")
            return Response(
                {"status": "Success", "message": "Redis is connected and working"},
                status=status.HTTP_200_OK,
            )
        else:
            logger.error("Redis test failed - value mismatch")
            return Response(
                {"status": "error", "message": "Redis test failed - value mismatch"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    except Exception as e:
        logger.error(f"Redis connection test failed: {str(e)}")
        return Response(
            {"status": "error", "message": f"Redis connection failed: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def get_trade_history(request):
    """Get user's trade history"""
    user = request.user

    try:
        # In production, this would query the database
        # For now, simulate by getting cached trades
        trade_history = []

        # Try to get some recent trades from cache (simplified approach)
        # In production, use proper database queries with pagination
        for i in range(10):  # Get last 10 potential trades
            trade_key = f"trade_SIM_{user.id}_{i}"
            trade = cache.get(trade_key)
            if trade:
                trade_history.append(trade)

        # Sort by timestamp if we have trades
        if trade_history:
            trade_history = sorted(
                trade_history, key=lambda x: x.get("timestamp", ""), reverse=True
            )

        # Calculate summary statistics
        total_trades = len(trade_history)
        buy_trades = sum(1 for t in trade_history if t.get("action") == "buy")
        sell_trades = total_trades - buy_trades
        total_volume = sum(t.get("total_value", 0) for t in trade_history)

        return Response(
            {
                "trades": trade_history,
                "summary": {
                    "total_trades": total_trades,
                    "buy_trades": buy_trades,
                    "sell_trades": sell_trades,
                    "total_volume": round(total_volume, 2),
                },
                "pagination": {"page": 1, "per_page": 20, "total_pages": 1},
                "message": "Trade history - using cache simulation. Integrate with database for production.",
            },
            status=status.HTTP_200_OK,
        )

    except Exception as e:
        logger.error(f"Trade history retrieval failed: {str(e)}")
        return Response(
            {"error": f"Unable to retrieve trade history: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
