import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
import pickle
import logging
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(asctime)s %(message)s",
    handlers=[
        logging.FileHandler("../Logs/universal_training.log"),
        logging.StreamHandler()
    ],
)
logger = logging.getLogger(__name__)

# Create models directory if it doesn't exist
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "Models"
MODELS_DIR.mkdir(exist_ok=True)

# Comprehensive stock database - all stocks from frontend
STOCK_DATABASE = [
    # Technology - FAANG & Major Tech
    'AAPL', 'GOOGL', 'GOOG', 'MSFT', 'AMZN', 'META', 'TSLA', 'NFLX', 'NVDA',
    'CRM', 'ORCL', 'ADBE', 'INTC', 'AMD', 'IBM', 'CSCO', 'AVGO', 'TXN', 'QCOM',
    'AMAT', 'MU', 'LRCX', 'ADI', 'KLAC', 'MRVL',

    # Electric Vehicles & Transportation
    'RIVN', 'LCID', 'NIO', 'XPEV', 'LI', 'F', 'GM',

    # Financial Services
    'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'V', 'MA', 'PYPL', 'AXP',
    'BRK-A', 'BRK-B',

    # Healthcare & Biotech
    'JNJ', 'PFE', 'UNH', 'MRNA', 'BNTX', 'ABBV', 'TMO', 'ABT', 'DHR',
    'BMY', 'LLY', 'MRK',

    # Entertainment & Media
    'DIS', 'CMCSA', 'T', 'VZ', 'SPOT',

    # Retail & E-commerce
    'WMT', 'HD', 'COST', 'TGT', 'LOW', 'EBAY',

    # Aerospace & Defense
    'BA', 'LMT', 'RTX', 'NOC', 'GD',

    # Energy
    'XOM', 'CVX', 'COP', 'SLB', 'EOG',

    # Industrial & Manufacturing
    'CAT', 'DE', 'GE', 'MMM', 'HON',

    # Consumer Goods
    'KO', 'PEP', 'PG', 'UL', 'NKE', 'SBUX',

    # Cryptocurrency & Fintech
    'COIN', 'SQ', 'HOOD',

    # Real Estate
    'AMT', 'PLD', 'CCI',

    # Utilities
    'NEE', 'SO', 'D',

    # Gaming
    'ATVI', 'EA', 'RBLX', 'TTWO',

    # Cloud & SaaS
    'SNOW', 'PLTR', 'ZM', 'DOCU', 'TWLO', 'OKTA', 'WORK',

    # ETFs & Emerging Tech
    'ARKK', 'QQQ', 'SPY', 'IWM', 'VTI',

    # International Stocks (ADRs)
    'BABA', 'JD', 'PDD', 'TME', 'BIDU', 'TSM', 'ASML', 'SAP', 'TM', 'SONY',

    # Indian Market (NSE/BSE symbols with .NS/.BO suffix)
    '^NSEI',  # NIFTY 50 Index
    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'HINDUNILVR.NS',
    'ITC.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'KOTAKBANK.NS', 'LT.NS',
    'HCLTECH.NS', 'MARUTI.NS', 'ASIANPAINT.NS', 'BAJFINANCE.NS',
    'WIPRO.NS', 'ULTRACEMCO.NS', 'NESTLEIND.NS', 'ONGC.NS', 'TATASTEEL.NS',

    # Meme Stocks & Popular Retail
    'GME', 'AMC', 'BB', 'NOK', 'WISH', 'CLOV',

    # SPACs & Recent IPOs
    'SPCE', 'OPEN', 'SOFI', 'UPST', 'AFRM',

    # Additional Popular Stocks
    'DKNG', 'PENN', 'ROKU', 'UBER', 'LYFT', 'DASH', 'ABNB'
]


def compute_rsi(series, period=14):
    """Compute Relative Strength Index (RSI)"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def compute_macd(data, fast=12, slow=26, signal=9):
    """Compute MACD (Moving Average Convergence Divergence)"""
    exp1 = data["Close"].ewm(span=fast, adjust=False).mean()
    exp2 = data["Close"].ewm(span=slow, adjust=False).mean()
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


def prepare_data(ticker, period="2y"):
    """
    Prepare stock data with comprehensive technical indicators
    Supports both US and international markets
    """
    try:
        logger.info(f"Fetching data for {ticker}")
        
        # Handle different ticker formats for international markets
        if ticker.upper() == 'NIFTY':
            ticker = '^NSEI'  # NIFTY 50 Index
        elif 'NIFTY' in ticker.upper():
            ticker = '^NSEI'
        
        # Download data with extended period for better training
        data = yf.download(ticker, period=period, interval="1d", progress=False)
        
        if data.empty:
            logger.error(f"No data available for {ticker}")
            raise ValueError(f"No data available for ticker: {ticker}")

        # Handle MultiIndex columns (common with yfinance)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] for col in data.columns]
        
        # Ensure required columns exist
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns for {ticker}: {missing_cols}")

        # Basic price features
        data["Return"] = data["Close"].pct_change()
        data["Log_Return"] = np.log(data["Close"] / data["Close"].shift(1))
        data["Price_Change"] = data["Close"] - data["Close"].shift(1)
        
        # Moving averages
        data["MA5"] = data["Close"].rolling(window=5).mean()
        data["MA10"] = data["Close"].rolling(window=10).mean()
        data["MA20"] = data["Close"].rolling(window=20).mean()
        data["MA50"] = data["Close"].rolling(window=50).mean()
        
        # Volatility measures
        data["Volatility"] = data["Return"].rolling(window=20).std()
        data["High_Low_Pct"] = (data["High"] - data["Low"]) / data["Close"]
        data["Open_Close_Pct"] = (data["Close"] - data["Open"]) / data["Open"]
        
        # Volume indicators
        data["Volume_Change"] = data["Volume"].pct_change()
        data["Volume_MA"] = data["Volume"].rolling(window=20).mean()
        data["Volume_Ratio"] = data["Volume"] / data["Volume_MA"]
        
        # Technical indicators
        data["RSI"] = compute_rsi(data["Close"], 14)
        data["RSI_Oversold"] = (data["RSI"] < 30).astype(int)
        data["RSI_Overbought"] = (data["RSI"] > 70).astype(int)
        
        data["MACD"], data["MACD_Signal"] = compute_macd(data)
        data["MACD_Diff"] = data["MACD"] - data["MACD_Signal"]
        data["MACD_Bullish"] = (data["MACD"] > data["MACD_Signal"]).astype(int)
        
        data["BB_Upper"], data["BB_Lower"] = compute_bollinger_bands(data)
        data["BB_Width"] = (data["BB_Upper"] - data["BB_Lower"]) / data["Close"]
        data["BB_Position"] = (data["Close"] - data["BB_Lower"]) / (data["BB_Upper"] - data["BB_Lower"])
        
        data["Stoch_K"], data["Stoch_D"] = compute_stochastic(data)
        
        # Price position relative to moving averages
        data["Price_Above_MA20"] = (data["Close"] > data["MA20"]).astype(int)
        data["Price_Above_MA50"] = (data["Close"] > data["MA50"]).astype(int)
        data["MA20_Above_MA50"] = (data["MA20"] > data["MA50"]).astype(int)
        
        # Trend indicators
        data["Uptrend"] = ((data["Close"] > data["MA20"]) & 
                          (data["MA20"] > data["MA50"])).astype(int)
        data["Downtrend"] = ((data["Close"] < data["MA20"]) & 
                            (data["MA20"] < data["MA50"])).astype(int)
        
        # Target variable - predict if price will be higher tomorrow
        data["Target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)
        
        # Additional target variations for better training
        data["Target_3d"] = (data["Close"].shift(-3) > data["Close"]).astype(int)
        data["Target_5d"] = (data["Close"].shift(-5) > data["Close"]).astype(int)
        
        # Remove rows with NaN values
        data = data.dropna()
        
        logger.info(f"Prepared {len(data)} rows of data for {ticker}")
        return data
        
    except Exception as e:
        logger.error(f"Error preparing data for {ticker}: {str(e)}")
        raise


def train_model(ticker, save_model=True):
    """
    Train a Random Forest model for a specific stock ticker
    """
    try:
        logger.info(f"Starting training for {ticker}")
        
        # Prepare data
        data = prepare_data(ticker)
        
        if len(data) < 100:
            logger.warning(f"Insufficient data for {ticker}: {len(data)} rows")
            return None, None
        
        # Feature selection - using most important indicators
        feature_columns = [
            "Open", "High", "Low", "Close", "Volume",
            "Return", "MA5", "MA20", "MA50", "Volatility", "RSI",
            "Volume_Change", "MACD", "MACD_Signal", "MACD_Diff",
            "BB_Width", "BB_Position", "Stoch_K", "Stoch_D",
            "High_Low_Pct", "Open_Close_Pct", "Volume_Ratio",
            "Price_Above_MA20", "Price_Above_MA50", "MA20_Above_MA50",
            "RSI_Oversold", "RSI_Overbought", "MACD_Bullish",
            "Uptrend", "Downtrend"
        ]
        
        # Filter out columns that don't exist or have all NaN values
        available_features = []
        for col in feature_columns:
            if col in data.columns and not data[col].isna().all():
                available_features.append(col)
        
        if len(available_features) < 10:
            logger.error(f"Insufficient features for {ticker}: {len(available_features)}")
            return None, None
        
        X = data[available_features]
        y = data["Target"]
        
        # Remove any remaining NaN values
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        
        if len(X) < 50:
            logger.error(f"Insufficient clean data for {ticker}: {len(X)} rows")
            return None, None
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train Random Forest model with optimized parameters
        model = RandomForestClassifier(
            n_estimators=100,  # Reduced for faster training
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1  # Use all available cores
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate model
        cv_scores = cross_val_score(model, X, y, cv=3, scoring='accuracy')  # Reduced CV folds
        train_accuracy = model.score(X_train, y_train)
        test_predictions = model.predict(X_test)
        test_accuracy = accuracy_score(y_test, test_predictions)
        
        logger.info(f"Model training completed for {ticker}")
        logger.info(f"Cross-validation scores: {cv_scores}")
        logger.info(f"Mean CV accuracy: {cv_scores.mean():.2%}")
        logger.info(f"Train accuracy: {train_accuracy:.2%}")
        logger.info(f"Test accuracy: {test_accuracy:.2%}")
        
        # Save model if requested
        if save_model:
            model_filename = MODELS_DIR / f"{ticker.replace('^', 'INDEX_').replace('.', '_')}_model.pkl"
            with open(model_filename, "wb") as f:
                pickle.dump({
                    'model': model,
                    'features': available_features,
                    'ticker': ticker,
                    'accuracy': test_accuracy,
                    'cv_mean': cv_scores.mean(),
                    'trained_samples': len(X_train)
                }, f)
            logger.info(f"Model saved: {model_filename}")
        
        return model, available_features
        
    except Exception as e:
        logger.error(f"Error training model for {ticker}: {str(e)}")
        return None, None


def train_universal_model():
    """
    Train a universal model using multiple popular stocks
    This serves as a fallback for stocks without specific models
    """
    try:
        logger.info("Training universal fallback model...")
        
        # Use top performing stocks for universal model
        universal_stocks = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'META', 
                           'JPM', 'V', 'JNJ', 'WMT', 'SPY', 'QQQ']
        
        all_data = []
        common_features = None
        
        for ticker in universal_stocks:
            try:
                logger.info(f"Adding {ticker} to universal model...")
                data = prepare_data(ticker, period="1y")  # Shorter period for universal
                
                if len(data) > 50:
                    # Add ticker identifier as feature
                    data['Ticker_Hash'] = hash(ticker) % 1000  # Simple ticker encoding
                    all_data.append(data)
                    
                    # Track common features
                    if common_features is None:
                        common_features = set(data.columns)
                    else:
                        common_features = common_features.intersection(set(data.columns))
                        
            except Exception as e:
                logger.warning(f"Failed to add {ticker} to universal model: {str(e)}")
                continue
        
        if len(all_data) < 3:
            logger.error("Insufficient stocks for universal model")
            return None, None
        
        # Combine all data
        combined_data = pd.concat(all_data, ignore_index=True)
        logger.info(f"Universal model data: {len(combined_data)} total samples from {len(all_data)} stocks")
        
        # Feature selection for universal model
        feature_columns = [
            "Open", "High", "Low", "Close", "Volume", "Return", 
            "MA5", "MA20", "Volatility", "RSI", "MACD", "MACD_Signal",
            "Volume_Change", "BB_Position", "Price_Above_MA20",
            "RSI_Oversold", "RSI_Overbought", "Ticker_Hash"
        ]
        
        available_features = [col for col in feature_columns 
                             if col in common_features and not combined_data[col].isna().all()]
        
        X = combined_data[available_features]
        y = combined_data["Target"]
        
        # Clean data
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        
        # Train universal model
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        universal_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            random_state=42,
            n_jobs=-1
        )
        
        universal_model.fit(X_train, y_train)
        
        test_accuracy = universal_model.score(X_test, y_test)
        logger.info(f"Universal model accuracy: {test_accuracy:.2%}")
        
        # Save universal model
        universal_model_path = MODELS_DIR / "universal_model.pkl"
        with open(universal_model_path, "wb") as f:
            pickle.dump({
                'model': universal_model,
                'features': available_features,
                'accuracy': test_accuracy,
                'stocks_used': [stock for stock in universal_stocks if any(ticker in str(data.columns) for data in all_data for ticker in [stock])],
                'trained_samples': len(X_train)
            }, f)
        
        logger.info(f"Universal model saved: {universal_model_path}")
        return universal_model, available_features
        
    except Exception as e:
        logger.error(f"Error training universal model: {str(e)}")
        return None, None


def batch_train_popular_stocks():
    """
    Train models for the most popular stocks in batches
    """
    # Most popular stocks to pre-train
    popular_stocks = [
        'AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'META', 'NFLX', 'NVDA',
        'JPM', 'V', 'JNJ', 'WMT', 'PG', 'HD', 'DIS', 'PYPL',
        'SPY', 'QQQ', 'ARKK', '^NSEI',  # Including NIFTY
        'GME', 'AMC', 'COIN', 'RBLX'
    ]
    
    successful_models = 0
    failed_models = 0
    
    logger.info(f"Starting batch training for {len(popular_stocks)} popular stocks...")
    
    for ticker in popular_stocks:
        try:
            logger.info(f"Training model for {ticker}...")
            model, features = train_model(ticker)
            
            if model is not None:
                successful_models += 1
                logger.info(f"✓ Successfully trained model for {ticker}")
            else:
                failed_models += 1
                logger.warning(f"✗ Failed to train model for {ticker}")
                
        except Exception as e:
            failed_models += 1
            logger.error(f"✗ Error training {ticker}: {str(e)}")
    
    logger.info(f"Batch training completed: {successful_models} successful, {failed_models} failed")
    
    # Train universal fallback model
    train_universal_model()
    
    return successful_models, failed_models


if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    log_dir = Path("../Logs")
    log_dir.mkdir(exist_ok=True)
    
    logger.info("Starting Universal Stock Training System...")
    logger.info(f"Models will be saved to: {MODELS_DIR}")
    
    # Train popular stocks
    successful, failed = batch_train_popular_stocks()
    
    logger.info("Universal Stock Training System completed!")
    logger.info(f"Results: {successful} successful, {failed} failed")
    logger.info(f"Models available in: {MODELS_DIR}")
