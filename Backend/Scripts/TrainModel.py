import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import pickle
import logging
import os
import json
from pathlib import Path
from datetime import datetime, timedelta
import warnings
import joblib
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time
import hashlib

warnings.filterwarnings("ignore")


def setup_logging():
    """Setup comprehensive logging system"""
    log_dir = Path("../Logs")
    log_dir.mkdir(exist_ok=True)

    # Create handlers
    file_handler = logging.FileHandler(log_dir / "ModelTraining.log")
    error_handler = logging.FileHandler(log_dir / "TrainingErrors.log")
    console_handler = logging.StreamHandler()

    # Set levels for handlers
    file_handler.setLevel(logging.INFO)
    console_handler.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter("%(levelname)s %(asctime)s [%(name)s] %(message)s")

    # Apply formatter to all handlers
    file_handler.setFormatter(formatter)
    error_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Configure root logger
    logging.basicConfig(
        level=logging.INFO, handlers=[file_handler, error_handler, console_handler]
    )

    return logging.getLogger(__name__)


logger = setup_logging()

# Configuration
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "Scripts" / "Models"
MODELS_DIR.mkdir(exist_ok=True)

PERFORMANCE_DIR = BASE_DIR / "Scripts" / "Performance"
PERFORMANCE_DIR.mkdir(exist_ok=True)

# Comprehensive stock database - organized by categories
STOCK_DATABASE = {
    "mega_cap_tech": ["AAPL", "GOOGL", "GOOG", "MSFT", "AMZN", "META", "TSLA", "NVDA"],
    "mega_cap_other": ["BRK-A", "BRK-B", "JPM", "JNJ", "V", "WMT", "PG", "UNH"],
    "large_cap_tech": [
        "NFLX",
        "CRM",
        "ORCL",
        "ADBE",
        "INTC",
        "AMD",
        "IBM",
        "CSCO",
        "AVGO",
    ],
    "electric_vehicles": ["RIVN", "LCID", "NIO", "XPEV", "LI", "F", "GM"],
    "financial": ["BAC", "WFC", "GS", "MS", "C", "MA", "PYPL", "AXP"],
    "healthcare": [
        "PFE",
        "MRNA",
        "BNTX",
        "ABBV",
        "TMO",
        "ABT",
        "DHR",
        "BMY",
        "LLY",
        "MRK",
    ],
    "entertainment": ["DIS", "CMCSA", "T", "VZ", "SPOT", "NFLX"],
    "retail": ["HD", "COST", "TGT", "LOW", "EBAY"],
    "aerospace": ["BA", "LMT", "RTX", "NOC", "GD"],
    "energy": ["XOM", "CVX", "COP", "SLB", "EOG"],
    "industrial": ["CAT", "DE", "GE", "MMM", "HON"],
    "consumer": ["KO", "PEP", "NKE", "SBUX"],
    "crypto_fintech": ["COIN", "SQ", "HOOD", "PLTR"],
    "gaming": ["ATVI", "EA", "RBLX", "TTWO"],
    "cloud_saas": ["SNOW", "ZM", "DOCU", "TWLO", "OKTA"],
    "etfs": ["SPY", "QQQ", "ARKK", "IWM", "VTI"],
    "international": ["BABA", "JD", "PDD", "TSM", "ASML", "SAP", "TM", "SONY"],
    "indian_market": [
        "^NSEI",
        "RELIANCE.NS",
        "TCS.NS",
        "HDFCBANK.NS",
        "INFY.NS",
        "HINDUNILVR.NS",
        "ITC.NS",
        "SBIN.NS",
        "BHARTIARTL.NS",
    ],
    "meme_stocks": ["GME", "AMC", "BB", "NOK"],
    "recent_ipos": [
        "SPCE",
        "SOFI",
        "UPST",
        "AFRM",
        "DKNG",
        "UBER",
        "LYFT",
        "DASH",
        "ABNB",
    ],
}

# Flatten stock database for easy access
ALL_STOCKS = []
for category, stocks in STOCK_DATABASE.items():
    ALL_STOCKS.extend(stocks)

# Timeframe configurations matching your views.py
TIMEFRAMES = {
    "1d": {"period": "3mo", "interval": "1d", "model_suffix": "_1d"},
    "1w": {"period": "1y", "interval": "1d", "model_suffix": "_1w"},
    "1mo": {"period": "2y", "interval": "1wk", "model_suffix": "_1mo"},
    "1y": {"period": "10y", "interval": "1mo", "model_suffix": "_1y"},
}


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


def compute_stochastic(data, k_period=14, d_period=3):
    """Compute Stochastic Oscillator"""
    lowest_low = data["Low"].rolling(window=k_period).min()
    highest_high = data["High"].rolling(window=k_period).max()
    k_percent = 100 * ((data["Close"] - lowest_low) / (highest_high - lowest_low))
    d_percent = k_percent.rolling(window=d_period).mean()
    return k_percent, d_percent


def compute_advanced_indicators(data):
    """Compute advanced technical indicators"""
    # Williams %R
    highest_high = data["High"].rolling(window=14).max()
    lowest_low = data["Low"].rolling(window=14).min()
    williams_r = -100 * (highest_high - data["Close"]) / (highest_high - lowest_low)

    # On-Balance Volume (OBV)
    obv = (np.sign(data["Close"].diff()) * data["Volume"]).fillna(0).cumsum()

    # Average True Range (ATR)
    high_low = data["High"] - data["Low"]
    high_close = np.abs(data["High"] - data["Close"].shift())
    low_close = np.abs(data["Low"] - data["Close"].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=14).mean()

    # VWAP (Volume Weighted Average Price)
    vwap = (data["Close"] * data["Volume"]).cumsum() / data["Volume"].cumsum()

    return {"williams_r": williams_r, "obv": obv, "atr": atr, "vwap": vwap}


def prepare_data(ticker, timeframe="1d", period=None):
    """
    Prepare comprehensive stock data with all technical indicators
    Supports multiple timeframes and international markets
    """
    try:
        logger.info(f"Fetching {timeframe} data for {ticker}")

        # Handle special ticker cases
        ticker_mapping = {
            "NIFTY": "^NSEI",
            "NIFTY50": "^NSEI",
            "SENSEX": "^BSESN",
            "BERKSHIRE": "BRK-B",
            "ALPHABET": "GOOGL",
            "GOOGLE": "GOOGL",
        }
        ticker = ticker_mapping.get(ticker.upper(), ticker)

        # Get timeframe configuration
        if period is None:
            config = TIMEFRAMES.get(timeframe, TIMEFRAMES["1d"])
            period = config["period"]
            interval = config["interval"]
        else:
            interval = "1d"

        # Download data
        data = yf.download(ticker, period=period, interval=interval, progress=False)

        if data.empty:
            logger.error(f"No data available for {ticker}")
            raise ValueError(f"No data available for ticker: {ticker}")

        # Handle MultiIndex columns
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] for col in data.columns]

        # Ensure required columns exist
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns for {ticker}: {missing_cols}")

        # Convert to numeric
        for col in required_cols:
            data[col] = pd.to_numeric(data[col], errors="coerce")

        # Basic price features
        data["Return"] = data["Close"].pct_change()
        data["Log_Return"] = np.log(data["Close"] / data["Close"].shift(1))
        data["Price_Change"] = data["Close"] - data["Close"].shift(1)

        # Timeframe-specific moving averages
        if timeframe == "1d":
            ma_periods = [5, 10, 20, 50, 200]
        elif timeframe == "1w":
            ma_periods = [4, 8, 13, 26, 52]
        elif timeframe == "1mo":
            ma_periods = [3, 6, 12, 24]
        else:  # 1y
            ma_periods = [2, 3, 6, 12]

        # Moving averages
        for period in ma_periods:
            if len(data) >= period:
                data[f"MA{period}"] = data["Close"].rolling(window=period).mean()

        # Volatility measures
        data["Volatility"] = data["Return"].rolling(window=20).std()
        data["High_Low_Pct"] = (data["High"] - data["Low"]) / data["Close"]
        data["Open_Close_Pct"] = (data["Close"] - data["Open"]) / data["Open"]

        # Volume indicators
        data["Volume_Change"] = data["Volume"].pct_change()
        data["Volume_MA"] = data["Volume"].rolling(window=20).mean()
        data["Volume_Ratio"] = data["Volume"] / data["Volume_MA"]
        data["Volume_Spike"] = (data["Volume"] > data["Volume_MA"] * 2).astype(int)

        # Technical indicators
        rsi_period = 14 if timeframe == "1d" else (10 if timeframe == "1w" else 8)
        data["RSI"] = compute_rsi(data["Close"], rsi_period)
        data["RSI_Oversold"] = (data["RSI"] < 30).astype(int)
        data["RSI_Overbought"] = (data["RSI"] > 70).astype(int)

        macd, macd_signal, macd_hist = compute_macd(data["Close"])
        data["MACD"] = macd
        data["MACD_Signal"] = macd_signal
        data["MACD_Histogram"] = macd_hist
        data["MACD_Bullish"] = (data["MACD"] > data["MACD_Signal"]).astype(int)

        bb_upper, bb_lower, bb_width = compute_bollinger_bands(data)
        data["BB_Upper"] = bb_upper
        data["BB_Lower"] = bb_lower
        data["BB_Width"] = bb_width
        data["BB_Position"] = (data["Close"] - bb_lower) / (bb_upper - bb_lower)
        data["BB_Squeeze"] = (bb_width < bb_width.rolling(20).mean() * 0.8).astype(int)

        stoch_k, stoch_d = compute_stochastic(data)
        data["Stoch_K"] = stoch_k
        data["Stoch_D"] = stoch_d
        data["Stoch_Overbought"] = (stoch_k > 80).astype(int)
        data["Stoch_Oversold"] = (stoch_k < 20).astype(int)

        # Advanced indicators
        advanced = compute_advanced_indicators(data)
        for key, value in advanced.items():
            data[key] = value

        # Price position relative to moving averages
        if "MA20" in data.columns:
            data["Price_Above_MA20"] = (data["Close"] > data["MA20"]).astype(int)
        if "MA50" in data.columns:
            data["Price_Above_MA50"] = (data["Close"] > data["MA50"]).astype(int)
            if "MA20" in data.columns:
                data["MA20_Above_MA50"] = (data["MA20"] > data["MA50"]).astype(int)

        # Trend indicators
        if "MA20" in data.columns and "MA50" in data.columns:
            data["Uptrend"] = (
                (data["Close"] > data["MA20"]) & (data["MA20"] > data["MA50"])
            ).astype(int)
            data["Downtrend"] = (
                (data["Close"] < data["MA20"]) & (data["MA20"] < data["MA50"])
            ).astype(int)

        # Market regime indicators
        data["High_Volatility"] = (
            data["Volatility"] > data["Volatility"].rolling(50).quantile(0.8)
        ).astype(int)
        data["Trending_Market"] = (
            abs(data["Return"].rolling(10).mean()) > data["Volatility"]
        ).astype(int)

        # Momentum indicators
        data["Price_Momentum"] = data["Close"] / data["Close"].shift(10) - 1
        data["Volume_Momentum"] = data["Volume"] / data["Volume"].rolling(20).mean() - 1

        # Target variables for different timeframes
        if timeframe == "1d":
            data["Target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)
        elif timeframe == "1w":
            data["Target"] = (data["Close"].shift(-5) > data["Close"]).astype(int)
        elif timeframe == "1mo":
            data["Target"] = (data["Close"].shift(-1) > data["Close"]).astype(
                int
            )  # Next month
        else:  # 1y
            data["Target"] = (data["Close"].shift(-1) > data["Close"]).astype(
                int
            )  # Next year

        # Additional target variations for ensemble learning
        data["Target_Strong"] = (data["Close"].shift(-1) > data["Close"] * 1.02).astype(
            int
        )  # 2% gain
        data["Target_Weak"] = (data["Close"].shift(-1) > data["Close"] * 0.98).astype(
            int
        )  # Avoid 2% loss

        # Add ticker hash for universal model
        data["Ticker_Hash"] = hash(ticker) % 1000

        # Remove rows with NaN values
        data = data.dropna()

        logger.info(f"Prepared {len(data)} rows of {timeframe} data for {ticker}")
        return data

    except Exception as e:
        logger.error(f"Error preparing {timeframe} data for {ticker}: {str(e)}")
        raise


def get_feature_set(timeframe="1d"):
    """Get optimal feature set for each timeframe"""
    base_features = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Return",
        "Volatility",
        "RSI",
        "Volume_Change",
        "MACD",
        "MACD_Signal",
        "MACD_Histogram",
        "BB_Position",
        "BB_Width",
        "High_Low_Pct",
        "Open_Close_Pct",
    ]

    timeframe_features = {
        "1d": base_features
        + [
            "MA5",
            "MA20",
            "MA50",
            "Stoch_K",
            "Stoch_D",
            "Price_Above_MA20",
            "RSI_Oversold",
            "RSI_Overbought",
            "MACD_Bullish",
            "Volume_Spike",
            "williams_r",
            "atr",
        ],
        "1w": base_features
        + [
            "MA4",
            "MA8",
            "MA13",
            "MA26",
            "Price_Momentum",
            "Volume_Momentum",
            "High_Volatility",
            "Trending_Market",
        ],
        "1mo": base_features
        + ["MA3", "MA6", "MA12", "BB_Squeeze", "Volume_Ratio", "Uptrend", "Downtrend"],
        "1y": base_features
        + [
            "MA2",
            "MA3",
            "MA6",
            "MA12",
            "obv",
            "vwap",
            "Price_Momentum",
            "Volume_Momentum",
        ],
    }

    return timeframe_features.get(timeframe, base_features)


def create_ensemble_model(timeframe="1d"):
    """Create ensemble model optimized for each timeframe"""
    if timeframe == "1d":
        # Fast, responsive models for daily predictions
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
        gb = GradientBoostingClassifier(
            n_estimators=80, max_depth=8, learning_rate=0.1, random_state=42
        )
    elif timeframe == "1w":
        # Medium-term trend models
        rf = RandomForestClassifier(
            n_estimators=150,
            max_depth=15,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
        gb = GradientBoostingClassifier(
            n_estimators=100, max_depth=10, learning_rate=0.08, random_state=42
        )
    elif timeframe == "1mo":
        # Monthly trend models
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=18,
            min_samples_split=3,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1,
        )
        gb = GradientBoostingClassifier(
            n_estimators=120, max_depth=12, learning_rate=0.05, random_state=42
        )
    else:  # 1y
        # Long-term trend models
        rf = RandomForestClassifier(
            n_estimators=250,
            max_depth=20,
            min_samples_split=3,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1,
        )
        gb = GradientBoostingClassifier(
            n_estimators=150, max_depth=15, learning_rate=0.03, random_state=42
        )

    # Create voting ensemble
    ensemble = VotingClassifier(estimators=[("rf", rf), ("gb", gb)], voting="soft")

    return ensemble


def train_model_for_timeframe(ticker, timeframe="1d", save_model=True):
    """Train optimized model for specific ticker and timeframe"""
    try:
        logger.info(f"Training {timeframe} model for {ticker}")

        # Prepare data
        data = prepare_data(ticker, timeframe)

        if len(data) < 100:
            logger.warning(
                f"Insufficient {timeframe} data for {ticker}: {len(data)} rows"
            )
            return None, None, None

        # Get optimal features for timeframe
        available_features = []
        feature_set = get_feature_set(timeframe)

        for feature in feature_set:
            if feature in data.columns and not data[feature].isna().all():
                available_features.append(feature)

        if len(available_features) < 10:
            logger.error(
                f"Insufficient features for {ticker} {timeframe}: {len(available_features)}"
            )
            return None, None, None

        # Prepare training data
        X = data[available_features]
        y = data["Target"]

        # Remove any remaining NaN values
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]

        if len(X) < 50:
            logger.error(
                f"Insufficient clean {timeframe} data for {ticker}: {len(X)} rows"
            )
            return None, None, None

        # Feature scaling for ensemble models
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X), columns=X.columns, index=X.index
        )

        # Split data with temporal awareness (no shuffling for time series)
        split_idx = int(len(X_scaled) * 0.8)
        X_train = X_scaled.iloc[:split_idx]
        X_test = X_scaled.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]

        # Create and train ensemble model
        model = create_ensemble_model(timeframe)
        model.fit(X_train, y_train)

        # Evaluate model
        train_accuracy = model.score(X_train, y_train)
        test_accuracy = model.score(X_test, y_test)

        # Cross-validation with time series splits
        cv_scores = cross_val_score(model, X_scaled, y, cv=3, scoring="accuracy")

        # Feature importance (from Random Forest component)
        rf_model = model.named_estimators_["rf"]
        feature_importance = dict(
            zip(available_features, rf_model.feature_importances_)
        )

        logger.info(f"{timeframe} model training completed for {ticker}")
        logger.info(f"Train accuracy: {train_accuracy:.2%}")
        logger.info(f"Test accuracy: {test_accuracy:.2%}")
        logger.info(f"CV mean accuracy: {cv_scores.mean():.2%}")

        # Save model if requested
        if save_model:
            clean_ticker = (
                ticker.replace("^", "INDEX_").replace(".", "_").replace("-", "_")
            )
            model_filename = (
                MODELS_DIR
                / f"{clean_ticker}_model{TIMEFRAMES[timeframe]['model_suffix']}.pkl"
            )

            model_data = {
                "model": model,
                "scaler": scaler,
                "features": available_features,
                "ticker": ticker,
                "timeframe": timeframe,
                "accuracy": test_accuracy,
                "cv_mean": cv_scores.mean(),
                "cv_std": cv_scores.std(),
                "trained_samples": len(X_train),
                "feature_importance": feature_importance,
                "training_date": datetime.now().isoformat(),
            }

            with open(model_filename, "wb") as f:
                pickle.dump(model_data, f)

            logger.info(f"Model saved: {model_filename}")

            # Save performance metrics
            performance_data = {
                "ticker": ticker,
                "timeframe": timeframe,
                "test_accuracy": test_accuracy,
                "cv_mean": cv_scores.mean(),
                "cv_std": cv_scores.std(),
                "training_samples": len(X_train),
                "features_count": len(available_features),
                "training_date": datetime.now().isoformat(),
            }

            performance_file = (
                PERFORMANCE_DIR / f"{clean_ticker}_{timeframe}_performance.json"
            )
            with open(performance_file, "w") as f:
                json.dump(performance_data, f, indent=2)

        return model, available_features, test_accuracy

    except Exception as e:
        logger.error(f"Error training {timeframe} model for {ticker}: {str(e)}")
        return None, None, None


def train_universal_model_for_timeframe(timeframe="1d"):
    """Train universal model for specific timeframe using top stocks"""
    try:
        logger.info(f"Training universal {timeframe} model...")

        # Select diverse, high-quality stocks for universal model
        universal_stocks = [
            "AAPL",
            "GOOGL",
            "MSFT",
            "TSLA",
            "AMZN",
            "META",  # Tech giants
            "JPM",
            "V",
            "JNJ",
            "WMT",
            "PG",  # Blue chips
            "SPY",
            "QQQ",  # ETFs
            "^NSEI",  # International
        ]

        all_data = []
        common_features = None

        for ticker in universal_stocks:
            try:
                logger.info(f"Adding {ticker} to universal {timeframe} model...")
                data = prepare_data(ticker, timeframe)

                if len(data) > 100:
                    all_data.append(data)

                    # Track common features
                    if common_features is None:
                        common_features = set(data.columns)
                    else:
                        common_features = common_features.intersection(
                            set(data.columns)
                        )

            except Exception as e:
                logger.warning(
                    f"Failed to add {ticker} to universal {timeframe} model: {str(e)}"
                )
                continue

        if len(all_data) < 5:
            logger.error(f"Insufficient stocks for universal {timeframe} model")
            return None, None

        # Combine all data
        combined_data = pd.concat(all_data, ignore_index=True)
        logger.info(
            f"Universal {timeframe} model data: {len(combined_data)} samples from {len(all_data)} stocks"
        )

        # Get features
        feature_set = get_feature_set(timeframe)
        available_features = [
            feature
            for feature in feature_set
            if feature in common_features and not combined_data[feature].isna().all()
        ]

        # Add ticker hash for diversity
        if "Ticker_Hash" in combined_data.columns:
            available_features.append("Ticker_Hash")

        X = combined_data[available_features]
        y = combined_data["Target"]

        # Clean data
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]

        # Feature scaling
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

        # Train universal ensemble model
        split_idx = int(len(X_scaled) * 0.8)
        X_train = X_scaled.iloc[:split_idx]
        X_test = X_scaled.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]

        universal_model = create_ensemble_model(timeframe)
        universal_model.fit(X_train, y_train)

        test_accuracy = universal_model.score(X_test, y_test)
        cv_scores = cross_val_score(universal_model, X_scaled, y, cv=3)

        logger.info(f"Universal {timeframe} model accuracy: {test_accuracy:.2%}")
        logger.info(f"Universal {timeframe} model CV: {cv_scores.mean():.2%}")

        # Save universal model
        model_filename = (
            MODELS_DIR / f"universal_model{TIMEFRAMES[timeframe]['model_suffix']}.pkl"
        )

        model_data = {
            "model": universal_model,
            "scaler": scaler,
            "features": available_features,
            "timeframe": timeframe,
            "accuracy": test_accuracy,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "stocks_used": universal_stocks,
            "trained_samples": len(X_train),
            "training_date": datetime.now().isoformat(),
        }

        with open(model_filename, "wb") as f:
            pickle.dump(model_data, f)

        logger.info(f"Universal {timeframe} model saved: {model_filename}")
        return universal_model, available_features

    except Exception as e:
        logger.error(f"Error training universal {timeframe} model: {str(e)}")
        return None, None


def batch_train_by_category(category_name, timeframes=["1d"], max_workers=4):
    """Train models for a specific category of stocks"""
    if category_name not in STOCK_DATABASE:
        logger.error(f"Unknown category: {category_name}")
        return

    stocks = STOCK_DATABASE[category_name]
    logger.info(
        f"Training {category_name} models for {len(stocks)} stocks across {timeframes}"
    )

    def train_stock_timeframes(ticker):
        results = {}
        for timeframe in timeframes:
            try:
                model, features, accuracy = train_model_for_timeframe(ticker, timeframe)
                if model is not None:
                    results[f"{ticker}_{timeframe}"] = {
                        "status": "Success",
                        "accuracy": accuracy,
                        "features_count": len(features),
                    }
                else:
                    results[f"{ticker}_{timeframe}"] = {"status": "failed"}
            except Exception as e:
                results[f"{ticker}_{timeframe}"] = {
                    "status": "error",
                    "message": str(e),
                }
        return results

    # Use ThreadPoolExecutor for parallel training
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ticker = {
            executor.submit(train_stock_timeframes, ticker): ticker for ticker in stocks
        }

        all_results = {}
        for future in future_to_ticker:
            ticker = future_to_ticker[future]
            try:
                result = future.result(timeout=300)  # 5 minute timeout per stock
                all_results.update(result)
                logger.info(f"Completed training for {ticker}")
            except Exception as e:
                logger.error(f"Training failed for {ticker}: {str(e)}")
                for timeframe in timeframes:
                    all_results[f"{ticker}_{timeframe}"] = {"status": "timeout"}

    # Summary statistics
    successful = sum(1 for r in all_results.values() if r.get("status") == "Success")
    failed = len(all_results) - successful

    logger.info(f"Category {category_name} training completed:")
    logger.info(f"  Successful: {successful}")
    logger.info(f"  Failed: {failed}")
    logger.info(f"  Success rate: {successful/(successful+failed)*100:.1f}%")

    return all_results


def comprehensive_training_pipeline(
    priority_categories=None, timeframes=["1d", "1w"], max_workers=4
):
    """
    Comprehensive training pipeline for enterprise deployment
    """
    logger.info("=" * 80)
    logger.info("STARTING COMPREHENSIVE STOCK TRAINING PIPELINE")
    logger.info("=" * 80)

    start_time = time.time()

    # Default priority categories for large-scale deployment
    if priority_categories is None:
        priority_categories = [
            "mega_cap_tech",  # Highest priority - most traded
            "mega_cap_other",  # Blue chip stocks
            "etfs",  # Popular ETFs
            "financial",  # Financial sector
            "indian_market",  # International markets
            "crypto_fintech",  # Emerging sectors
            "meme_stocks",  # High volatility, popular stocks
        ]

    training_summary = {
        "start_time": datetime.now().isoformat(),
        "timeframes": timeframes,
        "categories": {},
        "universal_models": {},
        "total_models": 0,
        "successful_models": 0,
        "failed_models": 0,
    }

    # Phase 1: Train universal models for all timeframes
    logger.info("Phase 1: Training Universal Models")
    logger.info("-" * 50)

    for timeframe in timeframes:
        try:
            model, features = train_universal_model_for_timeframe(timeframe)
            if model is not None:
                training_summary["universal_models"][timeframe] = {
                    "status": "Success",
                    "features_count": len(features),
                }
                logger.info(f"✓ Universal {timeframe} model trained successfully")
            else:
                training_summary["universal_models"][timeframe] = {"status": "failed"}
                logger.error(f"✗ Universal {timeframe} model training failed")
        except Exception as e:
            training_summary["universal_models"][timeframe] = {
                "status": "error",
                "message": str(e),
            }
            logger.error(f"✗ Universal {timeframe} model error: {str(e)}")

    # Phase 2: Train category-specific models
    logger.info("\nPhase 2: Training Category-Specific Models")
    logger.info("-" * 50)

    for category in priority_categories:
        logger.info(f"Training category: {category}")
        try:
            category_results = batch_train_by_category(
                category, timeframes, max_workers
            )
            training_summary["categories"][category] = category_results

            # Update totals
            category_successful = sum(
                1 for r in category_results.values() if r.get("status") == "Success"
            )
            category_total = len(category_results)
            training_summary["successful_models"] += category_successful
            training_summary["total_models"] += category_total

            logger.info(
                f"Category {category}: {category_successful}/{category_total} successful"
            )

        except Exception as e:
            logger.error(f"Category {category} training failed: {str(e)}")
            training_summary["categories"][category] = {"error": str(e)}

    # Phase 3: Train additional popular individual stocks
    logger.info("\nPhase 3: Training Additional Popular Stocks")
    logger.info("-" * 50)

    additional_stocks = [
        "NVDA",
        "AMD",
        "COIN",
        "PLTR",
        "RBLX",
        "SNOW",
        "ZM",
        "DOCU",  # High-interest stocks
        "GME",
        "AMC",
        "DKNG",
        "ROKU",  # Meme/popular stocks
    ]

    additional_results = {}
    for ticker in additional_stocks:
        logger.info(f"Training additional stock: {ticker}")
        try:
            for timeframe in timeframes:
                model, features, accuracy = train_model_for_timeframe(ticker, timeframe)
                key = f"{ticker}_{timeframe}"
                if model is not None:
                    additional_results[key] = {
                        "status": "Success",
                        "accuracy": accuracy,
                    }
                    training_summary["successful_models"] += 1
                else:
                    additional_results[key] = {"status": "failed"}
                training_summary["total_models"] += 1
        except Exception as e:
            logger.error(f"Additional stock {ticker} failed: {str(e)}")

    training_summary["additional_stocks"] = additional_results
    training_summary["failed_models"] = (
        training_summary["total_models"] - training_summary["successful_models"]
    )

    # Final summary
    end_time = time.time()
    duration = end_time - start_time
    training_summary["end_time"] = datetime.now().isoformat()
    training_summary["duration_seconds"] = duration
    training_summary["duration_formatted"] = (
        f"{duration//3600:.0f}h {(duration%3600)//60:.0f}m {duration%60:.0f}s"
    )

    logger.info("=" * 80)
    logger.info("TRAINING PIPELINE COMPLETED")
    logger.info("=" * 80)
    logger.info(f"Total Models: {training_summary['total_models']}")
    logger.info(f"Successful: {training_summary['successful_models']}")
    logger.info(f"Failed: {training_summary['failed_models']}")
    logger.info(
        f"Success Rate: {training_summary['successful_models']/training_summary['total_models']*100:.1f}%"
    )
    logger.info(f"Duration: {training_summary['duration_formatted']}")
    logger.info(f"Models saved to: {MODELS_DIR}")

    # Save training summary
    summary_file = (
        PERFORMANCE_DIR
        / f"training_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(summary_file, "w") as f:
        json.dump(training_summary, f, indent=2)
    logger.info(f"Training summary saved: {summary_file}")

    return training_summary


def update_existing_models(max_age_days=7, timeframes=["1d"]):
    """
    Update models that are older than specified days
    """
    logger.info(f"Checking for models older than {max_age_days} days...")

    cutoff_date = datetime.now() - timedelta(days=max_age_days)
    models_to_update = []

    # Check existing models
    for model_file in MODELS_DIR.glob("*_model*.pkl"):
        try:
            with open(model_file, "rb") as f:
                model_data = pickle.load(f)

            training_date_str = model_data.get("training_date")
            if training_date_str:
                training_date = datetime.fromisoformat(training_date_str)
                if training_date < cutoff_date:
                    # Extract ticker and timeframe from filename
                    filename_parts = model_file.stem.replace("_model", "").split("_")
                    if len(filename_parts) >= 2:
                        ticker = (
                            filename_parts[0].replace("INDEX_", "^").replace("_", ".")
                        )
                        timeframe = (
                            filename_parts[-1]
                            if filename_parts[-1] in timeframes
                            else "1d"
                        )
                        models_to_update.append((ticker, timeframe, model_file))

        except Exception as e:
            logger.warning(f"Could not check age of {model_file}: {str(e)}")

    logger.info(f"Found {len(models_to_update)} models to update")

    # Update old models
    updated_count = 0
    for ticker, timeframe, old_file in models_to_update:
        try:
            logger.info(f"Updating {timeframe} model for {ticker}")
            model, features, accuracy = train_model_for_timeframe(ticker, timeframe)
            if model is not None:
                updated_count += 1
                logger.info(f"✓ Updated {ticker} {timeframe} model")
            else:
                logger.warning(f"✗ Failed to update {ticker} {timeframe} model")
        except Exception as e:
            logger.error(f"Error updating {ticker} {timeframe}: {str(e)}")

    logger.info(f"Updated {updated_count}/{len(models_to_update)} models")
    return updated_count


def get_model_performance_report():
    """
    Generate comprehensive performance report for all trained models
    """
    logger.info("Generating model performance report...")

    performance_data = []

    # Scan all model files
    for model_file in MODELS_DIR.glob("*_model*.pkl"):
        try:
            with open(model_file, "rb") as f:
                model_data = pickle.load(f)

            performance_data.append(
                {
                    "filename": model_file.name,
                    "ticker": model_data.get("ticker", "unknown"),
                    "timeframe": model_data.get("timeframe", "unknown"),
                    "accuracy": model_data.get("accuracy", 0),
                    "cv_mean": model_data.get("cv_mean", 0),
                    "cv_std": model_data.get("cv_std", 0),
                    "trained_samples": model_data.get("trained_samples", 0),
                    "features_count": len(model_data.get("features", [])),
                    "training_date": model_data.get("training_date", "unknown"),
                }
            )

        except Exception as e:
            logger.warning(f"Could not read {model_file}: {str(e)}")

    # Create performance DataFrame
    df = pd.DataFrame(performance_data)

    if len(df) > 0:
        # Generate summary statistics
        report = {
            "total_models": len(df),
            "timeframes": df["timeframe"].value_counts().to_dict(),
            "average_accuracy": df["accuracy"].mean(),
            "best_models": df.nlargest(10, "accuracy")[
                ["ticker", "timeframe", "accuracy"]
            ].to_dict("records"),
            "worst_models": df.nsmallest(5, "accuracy")[
                ["ticker", "timeframe", "accuracy"]
            ].to_dict("records"),
            "accuracy_by_timeframe": df.groupby("timeframe")["accuracy"]
            .agg(["mean", "std", "count"])
            .to_dict(),
            "models_by_date": df.groupby(
                df["training_date"].str[:10]
                if "training_date" in df.columns
                else "unknown"
            )
            .size()
            .to_dict(),
        }

        # Save report
        report_file = (
            PERFORMANCE_DIR
            / f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Performance report saved: {report_file}")
        logger.info(f"Total models: {report['total_models']}")
        logger.info(f"Average accuracy: {report['average_accuracy']:.2%}")

        return report
    else:
        logger.warning("No model performance data found")
        return None


def cleanup_old_models(max_age_days=30):
    """
    Remove model files older than specified days to save disk space
    """
    logger.info(f"Cleaning up models older than {max_age_days} days...")

    cutoff_date = datetime.now() - timedelta(days=max_age_days)
    removed_count = 0

    for model_file in MODELS_DIR.glob("*_model*.pkl"):
        try:
            file_mtime = datetime.fromtimestamp(model_file.stat().st_mtime)
            if file_mtime < cutoff_date:
                model_file.unlink()
                removed_count += 1
                logger.info(f"Removed old model: {model_file.name}")
        except Exception as e:
            logger.warning(f"Could not remove {model_file}: {str(e)}")

    logger.info(f"Cleaned up {removed_count} old model files")
    return removed_count


if __name__ == "__main__":
    """
    Main execution for enterprise-grade training system
    """

    # Parse command line arguments (you can extend this)
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "full"

    if mode == "full":
        # Full training pipeline for initial deployment
        logger.info("Running full training pipeline...")
        result = comprehensive_training_pipeline(
            timeframes=["1d", "1w", "1mo"], max_workers=6
        )

    elif mode == "daily":
        # Daily update mode - update 1d models only
        logger.info("Running daily update mode...")
        update_existing_models(max_age_days=1, timeframes=["1d"])

    elif mode == "weekly":
        # Weekly update mode - update weekly and monthly models
        logger.info("Running weekly update mode...")
        update_existing_models(max_age_days=7, timeframes=["1w", "1mo"])

    elif mode == "report":
        # Generate performance report
        logger.info("Generating performance report...")
        report = get_model_performance_report()

    elif mode == "cleanup":
        # Cleanup old models
        logger.info("Cleaning up old models...")
        cleanup_old_models(max_age_days=30)

    elif mode == "category":
        # Train specific category
        category = sys.argv[2] if len(sys.argv) > 2 else "mega_cap_tech"
        timeframes = sys.argv[3].split(",") if len(sys.argv) > 3 else ["1d"]
        logger.info(f"Training category: {category}")
        batch_train_by_category(category, timeframes)

    else:
        logger.info("Available modes: full, daily, weekly, report, cleanup, category")
        logger.info("Usage examples:")
        logger.info("  python TrainModel.py full")
        logger.info("  python TrainModel.py daily")
        logger.info("  python TrainModel.py category mega_cap_tech 1d,1w")
        logger.info("  python TrainModel.py report")
        logger.info("  python TrainModel.py cleanup")

    logger.info("Training system execution completed!")
