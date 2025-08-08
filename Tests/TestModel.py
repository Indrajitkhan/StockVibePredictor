# Tests/test_model.py
import pickle
import pandas as pd
import os
from pathlib import Path

# Adjust path to point to stock_model.pkl
BASE_DIR = Path(__file__).resolve().parent.parent  # Gets to StockVibePredictor
MODEL_PATH = os.path.join(BASE_DIR, "Backend", "Scripts", "stock_model.pkl")

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print(f"Model type: {type(model)}")
    if hasattr(model, "feature_names_in_"):
        print(f"Expected features: {model.feature_names_in_}")
    else:
        print("Model does not store feature names. Check TrainModel.py.")

    # Dummy feature set (replace with actual features after checking TrainModel.py)
    dummy_features = pd.DataFrame(
        {
            "Close": [322.27],
            "RSI": [65.43],
            "MACD": [2.34],
            "MACD_Signal": [1.89],
            "MA50": [310.50],
            "MA200": [305.20],
            # Add placeholders for missing 7 features
            "Open": [320.0],
            "High": [325.0],
            "Low": [318.0],
            "Volume": [103246700],
            "MA20": [315.0],
            "ATR": [10.0],
            "BB_upper": [330.0],
        }
    )
    print(f"Dummy features: {dummy_features.columns.tolist()}")
    print(f"Dummy values: {dummy_features.to_dict(orient='records')}")

    # Test prediction
    prediction = model.predict(dummy_features)
    print(f"Prediction: {'UP' if prediction[0] == 1 else 'DOWN'}")
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(dummy_features)
        print(f"Prediction probabilities: {probs}")
except Exception as e:
    print(f"Error testing model: {str(e)}")
