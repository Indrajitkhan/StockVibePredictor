"""Backend tests for ML contracts module."""

import sys, os, pytest, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'api'))

from app.StockPredict.ml_contracts import (
    ForecastContract, construct_target, ForecastHorizon,
    CANONICAL_FEATURES, FeatureSchema, CANONICAL_FEATURE_SCHEMA,
    compute_canonical_features, validate_feature_quality,
    TemporalSplit, DEFAULT_TEMPORAL_SPLITS, compute_baseline_metrics,
    compute_classification_metrics, ModelArtifactMetadata,
    compute_artifact_checksum, validate_artifact_metadata,
    MLServiceStatus, create_error_response, create_prediction_response
)

# =============================================================================
# WORKSTREAM B: FORECAST CONTRACT TESTS
# =============================================================================

class TestForecastContract:
    def test_contract_validation(self):
        contract = ForecastContract()
        is_valid, errors = contract.validate()
        assert is_valid is True
        
    def test_target_construction_basic(self):
        data = pd.DataFrame({'Close': [100, 102, 101, 103, 105, 104, 106]})
        result = construct_target(data)
        assert 'Target' in result.columns
        assert len(result) == 6  # Dropped last row
        expected = [1, 0, 1, 1, 0, 1]
        assert list(result['Target']) == expected
    
    def test_target_missing_column_raises(self):
        data = pd.DataFrame({'Open': [100]})
        with pytest.raises(ValueError, match="Missing"):
            construct_target(data)


class TestCanonicalFeatures:
    def test_canonical_features_count(self):
        assert len(CANONICAL_FEATURES) == 15
    
    def test_canonical_features_expected(self):
        expected = ("Return","Log_Return","Volatility","Volatility_20","Volume_Ratio","Volume_Change","RSI","High_Low_Ratio","Close_Position","Spread","Gap","Price_Above_MA5","Price_Above_MA20","BB_Position","MACD_Histogram")
        assert CANONICAL_FEATURES == expected
    
    def test_feature_schema_singleton(self):
        assert CANONICAL_FEATURE_SCHEMA.get_feature_count() == 15
    
    def test_compute_canonical_features(self):
        dates = pd.date_range(start='2024-01-01', periods=60, freq='D')
        data = pd.DataFrame({
            'Open': np.random.uniform(95, 105, 60),
            'High': np.random.uniform(100, 110, 60),
            'Low': np.random.uniform(90, 100, 60),
            'Close': np.random.uniform(95, 105, 60),
            'Volume': np.random.uniform(1000000, 5000000, 60)
        }, index=dates)
        result = compute_canonical_features(data)
        for feat in CANONICAL_FEATURES:
            assert feat in result.columns
    
    def test_compute_canonical_features_min_rows(self):
        data = pd.DataFrame({
            'Open': [100, 101], 'High': [105, 106],
            'Low': [95, 96], 'Close': [102, 103],
            'Volume': [1000000, 1100000]
        })
        with pytest.raises(ValueError, match="Insufficient"):
            compute_canonical_features(data)
    
    def test_validate_feature_quality_healthy(self):
        dates = pd.date_range(start='2024-01-01', periods=60, freq='D')
        data = pd.DataFrame({f: np.random.randn(60) for f in CANONICAL_FEATURES}, index=dates)
        result = validate_feature_quality(data, list(CANONICAL_FEATURES))
        assert result['is_healthy'] is True


class TestTemporalEvaluation:
    def test_temporal_split_validation(self):
        split = TemporalSplit("2023-01-01", "2023-07-01", "2024-01-01")
        assert split is not None
        with pytest.raises(ValueError):
            TemporalSplit("2024-01-01", "2023-01-01", "2025-01-01")
    
    def test_default_temporal_splits(self):
        assert 'standard' in DEFAULT_TEMPORAL_SPLITS
        assert 'recent' in DEFAULT_TEMPORAL_SPLITS
    
    def test_baseline_metrics(self):
        y_true = np.array([1, 1, 1, 0, 0, 0, 0, 1])
        result = compute_baseline_metrics(y_true)
        assert 'majority_class' in result
        assert result['random_baseline'] == 0.5
    
    def test_classification_metrics(self):
        y_true = np.array([1, 1, 1, 0, 0, 0])
        y_pred = np.array([1, 1, 0, 0, 0, 0])
        result = compute_classification_metrics(y_true, y_pred)
        assert 'accuracy' in result
        assert 'f1' in result


class TestModelArtifact:
    def test_artifact_metadata_defaults(self):
        m = ModelArtifactMetadata()
        assert m.schema_version == "1.0.0"
        assert m.random_seed == 42
    
    def test_artifact_checksum(self):
        data = {"model_type": "RF", "feature_order": list(CANONICAL_FEATURES), "target_definition": "test", "schema_version": "1.0.0"}
        cs = compute_artifact_checksum(data)
        assert isinstance(cs, str)
        assert len(cs) == 16
    
    def test_validate_artifact_metadata_valid(self):
        m = {"schema_version": "1.0.0", "model_type": "RF", "feature_order": list(CANONICAL_FEATURES), "target_column": "Target"}
        is_valid, issues = validate_artifact_metadata(m)
        assert is_valid is True
    
    def test_validate_artifact_metadata_missing(self):
        m = {"schema_version": "1.0.0", "model_type": "RF", "target_column": "Target"}
        is_valid, issues = validate_artifact_metadata(m)
        assert is_valid is False


class TestAPIResponseSemantics:
    def test_create_error_response(self):
        r = create_error_response(MLServiceStatus.NO_MODEL, "No model", {"detail": 1})
        assert r['status'] == "no_model"
        assert 'timestamp' in r
    
    def test_create_prediction_response(self):
        preds = {"1d": {"direction": "UP"}}
        r = create_prediction_response("AAPL", preds, {"acc": 0.6})
        assert r['status'] == 'success'
        assert r['ticker'] == 'AAPL'


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
