import os
import re
import asyncio
import aiohttp
import threading
from concurrent.futures import ThreadPoolExecutor
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

# Define fallback implementations for technical indicators
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
    return macd, signal_line

def compute_bollinger_bands(data, period=20, std=2):
    """Compute Bollinger Bands"""
    ma = data["Close"].rolling(window=period).mean()
    std_dev = data["Close"].rolling(window=period).std()
    upper = ma + (std_dev * std)
    lower = ma - (std_dev * std)
    return upper, lower

def compute_stochastic(data, k_period=14, d_period=3):
    """Compute Stochastic Oscillator"""
    lowest_low = data["Low"].rolling(window=k_period).min()
    highest_high = data["High"].rolling(window=k_period).max()
    k_percent = 100 * ((data["Close"] - lowest_low) / (highest_high - lowest_low))
    d_percent = k_percent.rolling(window=d_period).mean()
    return k_percent, d_percent

# Fallback implementation for training (will use universal model)
def train_model(ticker, save_model=True):
    """Fallback training function - returns None to use universal model"""
    return None, None

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR / "Models"
MODELS_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("apps.stockpredict")

# Thread pool for background training
training_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix='model_trainer')

# Cache for loaded models
model_cache = {}
universal_model = None
universal_features = None

def load_universal_model():
    """Load the universal fallback model"""
    global universal_model, universal_features
    
    try:
        universal_model_path = MODELS_DIR / "universal_model.pkl"
        if universal_model_path.exists():
            with open(universal_model_path, "rb") as f:
                model_data = pickle.load(f)
                universal_model = model_data['model']
                universal_features = model_data['features']
                logger.info("Universal fallback model loaded successfully")
                return True
    except Exception as e:
        logger.error(f"Failed to load universal model: {str(e)}")
    
    # Fallback to old model if universal model doesn't exist
    try:
        old_model_path = BASE_DIR / "Scripts" / "stock_model.pkl"
        if old_model_path.exists():
            with open(old_model_path, "rb") as f:
                universal_model = pickle.load(f)
                # Define features for old model
                universal_features = [
                    "Open", "High", "Low", "Close", "Volume",
                    "Return", "MA5", "MA20", "Volatility", "RSI",
                    "Volume_Change", "MACD", "MACD_Signal"
                ]
                logger.info("Loaded fallback model from old training system")
                return True
    except Exception as e:
        logger.error(f"Failed to load fallback model: {str(e)}")
    
    return False

# Load universal model on startup
load_universal_model()


def get_model_filename(ticker):
    """Get the standardized model filename for a ticker"""
    clean_ticker = ticker.replace('^', 'INDEX_').replace('.', '_').replace('-', '_')
    return MODELS_DIR / f"{clean_ticker}_model.pkl"


def load_model_for_ticker(ticker):
    """Load a specific model for a ticker, with caching"""
    if ticker in model_cache:
        return model_cache[ticker]
    
    model_file = get_model_filename(ticker)
    
    if model_file.exists():
        try:
            with open(model_file, "rb") as f:
                model_data = pickle.load(f)
                model_info = {
                    'model': model_data['model'],
                    'features': model_data['features'],
                    'ticker': model_data.get('ticker', ticker),
                    'accuracy': model_data.get('accuracy', 0.0),
                    'type': 'specific'
                }
                model_cache[ticker] = model_info
                logger.info(f"Loaded specific model for {ticker} with accuracy {model_info['accuracy']:.2%}")
                return model_info
        except Exception as e:
            logger.error(f"Error loading model for {ticker}: {str(e)}")
    
    # Return universal model as fallback
    if universal_model is not None:
        logger.info(f"Using universal model for {ticker}")
        return {
            'model': universal_model,
            'features': universal_features,
            'ticker': 'universal',
            'accuracy': 0.5,
            'type': 'universal'
        }
    
    return None


def train_model_background(ticker):
    """Train a model for a ticker in the background"""
    try:
        logger.info(f"Starting background training for {ticker}")
        model, features = train_model(ticker, save_model=True)
        
        if model is not None:
            # Update cache
            model_info = {
                'model': model,
                'features': features,
                'ticker': ticker,
                'accuracy': 0.6,  # Default accuracy
                'type': 'specific'
            }
            model_cache[ticker] = model_info
            logger.info(f"Background training completed for {ticker}")
        else:
            logger.warning(f"Background training failed for {ticker}")
            
    except Exception as e:
        logger.error(f"Error in background training for {ticker}: {str(e)}")


def validate_ticker(ticker):
    """Enhanced ticker validation supporting international markets"""
    if not ticker or not isinstance(ticker, str):
        return False
    
    # Allow various ticker formats
    # US stocks: AAPL, BRK.B, BRK-A
    # Indices: ^NSEI, ^DJI
    # International: TSM, ASML, SONY
    # NSE/BSE: RELIANCE.NS, HDFCBANK.BO
    
    if re.match(r"^[A-Z0-9\^\.\_\-]{1,15}$", ticker):
        return True
        
    return False


def normalize_ticker(ticker):
    """Normalize ticker symbols for different markets"""
    ticker = ticker.upper().strip()
    
    # Handle common ticker variations
    ticker_mapping = {
        'NIFTY': '^NSEI',
        'NIFTY50': '^NSEI',
        'SENSEX': '^BSESN',
        'BERKSHIRE': 'BRK-B',  # Default to Class B
        'ALPHABET': 'GOOGL',   # Default to Class A
    }
    
    if ticker in ticker_mapping:
        return ticker_mapping[ticker]
    
    return ticker


def compute_enhanced_features(data):
    """Compute enhanced technical indicators for prediction"""
    try:
        # Handle MultiIndex columns
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] for col in data.columns]
        
        # Ensure we have required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Convert to numeric
        for col in required_cols:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # Remove NaN rows
        data = data.dropna(subset=required_cols)
        
        if len(data) < 50:
            raise ValueError("Insufficient data for feature computation")
        
        # Basic features
        data["Return"] = data["Close"].pct_change()
        data["MA5"] = data["Close"].rolling(window=5).mean()
        data["MA20"] = data["Close"].rolling(window=20).mean()
        data["MA50"] = data["Close"].rolling(window=50).mean()
        data["Volatility"] = data["Return"].rolling(window=20).std()
        data["Volume_Change"] = data["Volume"].pct_change()
        
        # Technical indicators
        data["RSI"] = compute_rsi(data["Close"], 14)
        data["MACD"], data["MACD_Signal"] = compute_macd(data["Close"])
        data["BB_Upper"], data["BB_Lower"] = compute_bollinger_bands(data)
        
        # Additional features
        data["BB_Position"] = (data["Close"] - data["BB_Lower"]) / (data["BB_Upper"] - data["BB_Lower"])
        data["Price_Above_MA20"] = (data["Close"] > data["MA20"]).astype(int)
        data["Price_Above_MA50"] = (data["Close"] > data["MA50"]).astype(int)
        data["RSI_Oversold"] = (data["RSI"] < 30).astype(int)
        data["RSI_Overbought"] = (data["RSI"] > 70).astype(int)
        data["MACD_Bullish"] = (data["MACD"] > data["MACD_Signal"]).astype(int)
        
        # Trend indicators
        data["Uptrend"] = ((data["Close"] > data["MA20"]) & (data["MA20"] > data["MA50"])).astype(int)
        data["Downtrend"] = ((data["Close"] < data["MA20"]) & (data["MA20"] < data["MA50"])).astype(int)
        
        return data
        
    except Exception as e:
        logger.error(f"Error computing features: {str(e)}")
        raise


async def fetch_stock_data(ticker, period="1y", interval="1d"):
    """Enhanced stock data fetching with better error handling"""
    try:
        # Normalize ticker
        ticker = normalize_ticker(ticker)
        
        # Use sync_to_async for yfinance
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
    """Redis connectivity check"""
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
    """Enhanced stock prediction with dynamic model loading and training"""
    ticker = request.data.get("ticker")
    
    if not ticker:
        logger.error("No ticker provided")
        return Response(
            {"error": "Please provide a ticker symbol"}, 
            status=status.HTTP_400_BAD_REQUEST
        )

    if not validate_ticker(ticker):
        logger.error(f"Invalid ticker format: {ticker}")
        return Response(
            {"error": "Invalid ticker format. Use uppercase letters, numbers, and symbols like ^, ., -"}, 
            status=status.HTTP_400_BAD_REQUEST
        )
    
    # Normalize ticker
    original_ticker = ticker
    ticker = normalize_ticker(ticker)
    
    # Check cache first
    cache_key = f"stock_prediction_{ticker}"
    cached_result = cache.get(cache_key)
    if cached_result:
        logger.info(f"Returning cached prediction for {ticker}")
        return Response(cached_result, status=status.HTTP_200_OK)
    
    try:
        # Fetch stock data
        logger.info(f"Fetching stock data for {ticker}")
        data = async_to_sync(fetch_stock_data)(ticker)
        
        if data is None or data.empty:
            logger.error(f"No stock data found for {ticker}")
            return Response(
                {"error": f"No stock data found for {ticker}. Please check the ticker symbol."}, 
                status=status.HTTP_404_NOT_FOUND
            )
        
        # Compute enhanced features
        data = compute_enhanced_features(data)
        
        # Load appropriate model
        model_info = load_model_for_ticker(ticker)
        
        if model_info is None:
            logger.error(f"No model available for {ticker}")
            return Response(
                {"error": "No prediction model available. Please try again later."}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        
        # If using universal model and no specific model exists, start background training
        if model_info['type'] == 'universal' and ticker not in model_cache:
            logger.info(f"Starting background training for {ticker}")
            training_executor.submit(train_model_background, ticker)
        
        # Prepare features for prediction
        model = model_info['model']
        required_features = model_info['features']
        
        # Get latest data point for prediction
        latest_data = data.iloc[-1]
        
        # Create feature vector
        feature_vector = []
        feature_dict = {}
        
        for feature in required_features:
            if feature in data.columns:
                value = data[feature].iloc[-1]
                if pd.isna(value):
                    value = 0  # Handle NaN values
                feature_vector.append(value)
                feature_dict[feature] = value
            else:
                # Handle missing features
                if feature == 'Ticker_Hash':
                    value = hash(ticker) % 1000  # For universal model
                else:
                    value = 0
                feature_vector.append(value)
                feature_dict[feature] = value
        
        # Make prediction
        feature_array = np.array(feature_vector).reshape(1, -1)
        prediction = model.predict(feature_array)[0]
        
        # Get confidence if available
        confidence = 0.5
        if hasattr(model, 'predict_proba'):
            try:
                confidence = model.predict_proba(feature_array)[0][prediction]
            except Exception as e:
                logger.warning(f"Failed to get confidence for {ticker}: {str(e)}")
        
        # Convert prediction to human-readable format
        prediction_text = "UP" if prediction == 1 else "DOWN"
        
        # Calculate additional metrics
        current_price = float(data["Close"].iloc[-1])
        price_change_pct = float(data["Return"].iloc[-1] * 100) if not pd.isna(data["Return"].iloc[-1]) else 0.0
        
        # Prepare response data
        response_data = {
            "ticker": original_ticker,
            "normalized_ticker": ticker,
            "prediction": {
                "direction": prediction_text,
                "confidence": float(confidence * 100),
                "current_price": current_price,
                "price_change_24h": price_change_pct,
                "model_type": model_info['type'],
                "model_accuracy": float(model_info['accuracy'] * 100),
                "features": {
                    "rsi": float(feature_dict.get("RSI", 0)),
                    "macd": float(feature_dict.get("MACD", 0)),
                    "macd_signal": float(feature_dict.get("MACD_Signal", 0)),
                    "ma20": float(feature_dict.get("MA20", 0)),
                    "ma50": float(feature_dict.get("MA50", 0)),
                    "volume_change": float(feature_dict.get("Volume_Change", 0)),
                    "volatility": float(feature_dict.get("Volatility", 0))
                }
            },
            "history": []  # We'll add this if needed
        }
        
        # Add historical data if requested (last 100 points to avoid huge responses)
        try:
            historical_data = data.tail(100).copy()
            historical_data["Date"] = historical_data.index.strftime("%Y-%m-%d")
            historical_data = historical_data.reset_index(drop=True)
            historical_data = historical_data.replace([np.nan, np.inf, -np.inf], None)
            
            # Select only essential columns for history
            history_columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
            available_history_columns = [col for col in history_columns if col in historical_data.columns]
            
            response_data["history"] = historical_data[available_history_columns].to_dict(orient="records")
            
        except Exception as e:
            logger.warning(f"Error preparing historical data for {ticker}: {str(e)}")
            response_data["history"] = []
        
        # Cache the result for 5 minutes
        cache.set(cache_key, response_data, timeout=300)
        
        logger.info(f"Prediction completed for {ticker}: {prediction_text} ({confidence*100:.1f}%)")
        return Response(response_data, status=status.HTTP_200_OK)
        
    except Exception as e:
        logger.error(f"Prediction failed for {ticker}: {str(e)}")
        return Response(
            {"error": f"Prediction failed: {str(e)}"}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(["GET"])
def model_status(request):
    """Get status of available models"""
    try:
        models_status = {}
        
        # Check universal model
        models_status['universal'] = {
            'available': universal_model is not None,
            'features_count': len(universal_features) if universal_features else 0
        }
        
        # Check specific models
        specific_models = {}
        for model_file in MODELS_DIR.glob("*_model.pkl"):
            ticker = model_file.stem.replace("_model", "").replace("INDEX_", "^").replace("_", ".")
            try:
                with open(model_file, "rb") as f:
                    model_data = pickle.load(f)
                    specific_models[ticker] = {
                        'accuracy': model_data.get('accuracy', 0.0),
                        'features_count': len(model_data.get('features', [])),
                        'trained_samples': model_data.get('trained_samples', 0)
                    }
            except Exception as e:
                logger.warning(f"Error reading model {model_file}: {str(e)}")
        
        models_status['specific'] = specific_models
        models_status['cache_size'] = len(model_cache)
        
        return Response(models_status, status=status.HTTP_200_OK)
        
    except Exception as e:
        logger.error(f"Error getting model status: {str(e)}")
        return Response(
            {"error": "Failed to get model status"}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
