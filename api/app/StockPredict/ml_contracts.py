"""StockVibe ML Integrity Module - Canonical Contracts v1.0.0"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import hashlib, json
import numpy as np, pandas as pd

# =============================================================================
# WORKSTREAM B: FORECAST CONTRACT
# =============================================================================
class ForecastHorizon(Enum): NEXT_PERIOD = "next_period"

@dataclass
class ForecastContract:
    horizon: ForecastHorizon = ForecastHorizon.NEXT_PERIOD
    target_column: str = "Target"
    target_definition: str = "(Next Close > Current Close) ? 1 : 0"
    drop_terminal_rows: bool = True
    missing_target_value: str = "drop"
    
    def validate(self) -> Tuple[bool, List[str]]:
        errors = []
        if self.horizon not in ForecastHorizon: errors.append(f"Invalid horizon: {self.horizon}")
        return len(errors) == 0, errors

def construct_target(df: pd.DataFrame, price_col: str = "Close", contract: Optional[ForecastContract] = None) -> pd.DataFrame:
    """Construct target: 1 if next close > current close."""
    if contract is None: contract = ForecastContract()
    if price_col not in df.columns: raise ValueError(f"Missing column: {price_col}")
    result = df.copy()
    result[contract.target_column] = (result[price_col].shift(-1) > result[price_col]).astype(int)
    if contract.drop_terminal_rows: result = result[:-1]
    nan_mask = result[contract.target_column].isna()
    if nan_mask.any():
        if contract.missing_target_value == "drop": result = result[~nan_mask]
        elif contract.missing_target_value == "error": raise ValueError(f"Missing target")
    return result

# =============================================================================
# WORKSTREAM C: CANONICAL FEATURES
# =============================================================================
CANONICAL_FEATURES = ("Return","Log_Return","Volatility","Volatility_20","Volume_Ratio","Volume_Change","RSI","High_Low_Ratio","Close_Position","Spread","Gap","Price_Above_MA5","Price_Above_MA20","BB_Position","MACD_Histogram")

FEATURE_METADATA = {
    "Return": {"type": "float", "range": (-1, 1), "description": "Simple daily return"},
    "Log_Return": {"type": "float", "range": (-1, 1), "description": "Log return"},
    "Volatility": {"type": "float", "range": (0, 10), "description": "20-period rolling std"},
    "Volatility_20": {"type": "float", "range": (0, 10), "description": "Alternative volatility"},
    "Volume_Ratio": {"type": "float", "range": (0, 10), "description": "Volume/20-period avg"},
    "Volume_Change": {"type": "float", "range": (-1, 1), "description": "Volume % change"},
    "RSI": {"type": "float", "range": (0, 100), "description": "Relative Strength Index"},
    "High_Low_Ratio": {"type": "float", "range": (0.5, 2), "description": "High/Low ratio"},
    "Close_Position": {"type": "float", "range": (0, 1), "description": "Close position in range"},
    "Spread": {"type": "float", "range": (0, None), "description": "High - Low"},
    "Gap": {"type": "float", "range": (-1, 1), "description": "Open - Prev Close"},
    "Price_Above_MA5": {"type": "int", "range": (0, 1), "description": "Close > MA5"},
    "Price_Above_MA20": {"type": "int", "range": (0, 1), "description": "Close > MA20"},
    "BB_Position": {"type": "float", "range": (0, 1), "description": "BB position"},
    "MACD_Histogram": {"type": "float", "range": (-10, 10), "description": "MACD histogram"},
}

@dataclass
class FeatureSchema:
    features: Tuple = CANONICAL_FEATURES
    metadata: Dict = field(default_factory=lambda: FEATURE_METADATA)
    version: str = "1.0.0"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def get_feature_names(self) -> List[str]: return list(self.features)
    def get_feature_count(self) -> int: return len(self.features)
    def validate_features(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        missing = [f for f in self.features if f not in df.columns]
        return len(missing) == 0, missing
    def get_feature_vector(self, df: pd.DataFrame, fill_value: float = 0.0) -> np.ndarray:
        feats = []
        for name in self.features:
            if name in df.columns:
                v = df[name].iloc[-1]
                feats.append(float(v if not pd.isna(v) else fill_value))
            else: feats.append(fill_value)
        return np.array(feats).reshape(1, -1)
    def to_dict(self) -> Dict[str, Any]: return {"features": list(self.features), "metadata": self.metadata, "version": self.version, "created_at": self.created_at, "feature_count": len(self.features)}

CANONICAL_FEATURE_SCHEMA = FeatureSchema()

def compute_canonical_features(data: pd.DataFrame, timeframe: str = "1d") -> pd.DataFrame:
    """Compute canonical features - only past data, no look-ahead."""
    for col in ["Open","High","Low","Close","Volume"]:
        if col not in data.columns: raise ValueError(f"Missing: {col}")
    r = data.copy()
    for col in ["Open","High","Low","Close","Volume"]: r[col] = pd.to_numeric(r[col], errors="coerce")
    r = r.dropna(subset=["Open","High","Low","Close","Volume"])
    if len(r) < 50: raise ValueError(f"Insufficient: {len(r)} rows")
    r["Return"] = r["Close"].pct_change()
    r["Log_Return"] = np.log(r["Close"] / r["Close"].shift(1))
    r["Volatility"] = r["Return"].rolling(20).std()
    r["Volatility_20"] = r["Return"].rolling(20).std()
    r["Volume_Ratio"] = r["Volume"] / r["Volume"].rolling(20).mean()
    r["Volume_Change"] = r["Volume"].pct_change()
    d = r["Close"].diff()
    g = d.where(d > 0, 0).rolling(14).mean()
    l = (-d.where(d < 0, 0)).rolling(14).mean()
    rs = g / l.replace(0, np.nan)
    r["RSI"] = (100 - (100 / (1 + rs))).fillna(50)
    r["High_Low_Ratio"] = r["High"] / r["Low"]
    r["Close_Position"] = (r["Close"] - r["Low"]) / (r["High"] - r["Low"])
    r["Spread"] = r["High"] - r["Low"]
    r["Gap"] = r["Open"] - r["Close"].shift(1)
    ma5, ma20 = r["Close"].rolling(5).mean(), r["Close"].rolling(20).mean()
    r["Price_Above_MA5"] = (r["Close"] > ma5).astype(int)
    r["Price_Above_MA20"] = (r["Close"] > ma20).astype(int)
    bb_ma, bb_std = r["Close"].rolling(20).mean(), r["Close"].rolling(20).std()
    r["BB_Position"] = (r["Close"] - (bb_ma - 2*bb_std)) / (4*bb_std)
    ema12, ema26 = r["Close"].ewm(span=12, adjust=False).mean(), r["Close"].ewm(span=26, adjust=False).mean()
    r["MACD_Histogram"] = ema12 - ema26 - (ema12 - ema26).ewm(span=9, adjust=False).mean()
    return r

def validate_feature_quality(df: pd.DataFrame, feature_names: List[str]) -> Dict[str, Any]:
    issues, X = [], df[feature_names].values
    nan_c, inf_c = int(np.isnan(X).sum()), int(np.isinf(X).sum())
    if nan_c: issues.append(f"{nan_c} NaN values")
    if inf_c: issues.append(f"{inf_c} Inf values")
    for i,n in enumerate(feature_names):
        if np.nanstd(X[:,i]) < 1e-10: issues.append(f"Constant: {n}")
    return {"is_healthy": len(issues)==0, "issues": issues, "metrics": {"total_samples":len(df), "total_features":len(feature_names), "nan_count":nan_c, "inf_count":inf_c}}

# =============================================================================
# WORKSTREAM D: TEMPORAL EVALUATION
# =============================================================================
@dataclass
class TemporalSplit:
    train_end: str
    val_end: str
    test_end: str
    def __post_init__(self):
        td = lambda x: pd.to_datetime(x)
        if not (td(self.train_end) < td(self.val_end) < td(self.test_end)): raise ValueError("Invalid temporal split")
    def split_dataframe(self, df: pd.DataFrame, date_column: str = "Date") -> Dict[str, pd.DataFrame]:
        if date_column in df.columns: df = df.set_index(date_column)
        return {"train": df[df.index < self.train_end], "val": df[(df.index>=self.train_end)&(df.index<self.val_end)], "test": df[(df.index>=self.val_end)&(df.index<self.test_end)]}

DEFAULT_TEMPORAL_SPLITS = {"standard": TemporalSplit("2023-01-01","2023-07-01","2024-01-01"), "recent": TemporalSplit("2024-01-01","2024-07-01","2025-01-01")}

def compute_baseline_metrics(y_true: np.ndarray) -> Dict[str, float]:
    u,c = np.unique(y_true, return_counts=True)
    return {"majority_class": float(max(c)/len(y_true)), "random_baseline": 0.5}

def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score
    if len(y_true)==0: return {"error":"No samples"}
    return {"accuracy":float(accuracy_score(y_true,y_pred)),"precision":float(precision_score(y_true,y_pred,zero_division=0)),"recall":float(recall_score(y_true,y_pred,zero_division=0)),"f1":float(f1_score(y_true,y_pred,zero_division=0)),"balanced_accuracy":float(balanced_accuracy_score(y_true,y_pred))}

# =============================================================================
# WORKSTREAM E: MODEL ARTIFACT CONTRACT
# =============================================================================
@dataclass
class ModelArtifactMetadata:
    schema_version: str = "1.0.0"
    model_type: str = ""
    model_config: Dict = field(default_factory=dict)
    feature_schema: Dict = field(default_factory=dict)
    feature_order: List = field(default_factory=list)
    target_column: str = "Target"
    target_definition: str = "(Next Close > Current Close) ? 1 : 0"
    forecast_horizon: str = "next_period"
    training_timestamp: str = ""
    training_sample_count: int = 0
    train_end_date: str = ""
    val_end_date: str = ""
    test_end_date: str = ""
    evaluation_metrics: Dict = field(default_factory=dict)
    baseline_metrics: Dict = field(default_factory=dict)
    random_seed: int = 42
    artifact_checksum: str = ""
    def to_dict(self) -> Dict[str, Any]: return {"schema_version":self.schema_version,"model_type":self.model_type,"model_config":self.model_config,"feature_schema":self.feature_schema,"feature_order":self.feature_order,"target_column":self.target_column,"target_definition":self.target_definition,"forecast_horizon":self.forecast_horizon,"training_timestamp":self.training_timestamp,"training_sample_count":self.training_sample_count,"train_end_date":self.train_end_date,"val_end_date":self.val_end_date,"test_end_date":self.test_end_date,"evaluation_metrics":self.evaluation_metrics,"baseline_metrics":self.baseline_metrics,"random_seed":self.random_seed,"artifact_checksum":self.artifact_checksum}

def compute_artifact_checksum(model_data: Dict[str, Any]) -> str:
    s = {"model_type":model_data.get("model_type",""),"feature_order":model_data.get("feature_order",[]),"target_definition":model_data.get("target_definition","")}
    return hashlib.sha256(json.dumps(s,sort_keys=True).encode()).hexdigest()[:16]

def validate_artifact_metadata(metadata: Dict[str, Any]) -> Tuple[bool, List[str]]:
    issues = []
    for f in ["schema_version","model_type","feature_order","target_column"]:
        if f not in metadata: issues.append(f"Missing: {f}")
    if metadata.get("schema_version") != "1.0.0": issues.append("Unsupported schema")
    return len(issues)==0, issues

# =============================================================================
# WORKSTREAM F: API FAILURE SEMANTICS
# =============================================================================
class PredictionError(Exception): pass
class ModelNotAvailableError(PredictionError): pass
class InvalidFeatureVectorError(PredictionError): pass
class ArtifactValidationError(PredictionError): pass

class MLServiceStatus(Enum): HEALTHY="healthy"; DEGRADED="degraded"; UNAVAILABLE="unavailable"; NO_MODEL="no_model"; INVALID_INPUT="invalid_input"; PREDICTION_FAILED="prediction_failed"

def create_error_response(error_type: MLServiceStatus, message: str, details: Optional[Dict]=None) -> Dict[str, Any]:
    r = {"status":error_type.value,"error":message,"timestamp":datetime.now().isoformat()}
    if details: r["details"] = details
    return r

def create_prediction_response(ticker: str, predictions: Dict[str, Any], metadata: Optional[Dict]=None) -> Dict[str, Any]:
    r = {"status":"success","ticker":ticker,"predictions":predictions,"timestamp":datetime.now().isoformat()}
    if metadata: r["metadata"] = metadata
    return r

__all__ = ["ForecastContract","construct_target","ForecastHorizon","CANONICAL_FEATURES","FEATURE_METADATA","FeatureSchema","CANONICAL_FEATURE_SCHEMA","compute_canonical_features","validate_feature_quality","TemporalSplit","DEFAULT_TEMPORAL_SPLITS","compute_baseline_metrics","compute_classification_metrics","ModelArtifactMetadata","compute_artifact_checksum","validate_artifact_metadata","PredictionError","ModelNotAvailableError","InvalidFeatureVectorError","ArtifactValidationError","MLServiceStatus","create_error_response","create_prediction_response"]
