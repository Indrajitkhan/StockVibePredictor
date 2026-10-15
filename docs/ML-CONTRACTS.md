# StockVibe ML Contracts Documentation

> Version 1.0.0 | Part of Task 12: ML Integrity & Prediction Reliability

---

## 1. Forecast Contract (Workstream B)

- **Observation time**: Trading period t (the row we are predicting for)
- **Features**: Information available at or before prediction cutoff at t
- **Target**: Whether the next trading period close > current period close
- **Forecast horizon**: One next trading period

Target = 1 if (Next Close > Current Close) else 0
The shift(-1) is intentional for supervised learning target construction.

## 2. Canonical Features (15 total)

Return, Log_Return, Volatility, Volatility_20, Volume_Ratio, Volume_Change, RSI, High_Low_Ratio, Close_Position, Spread, Gap, Price_Above_MA5, Price_Above_MA20, BB_Position, MACD_Histogram

All features use only past data - no look-ahead. Rolling windows are backward-looking only.

## 3. Temporal Evaluation (Workstream D)

Uses chronological split: train_end < val_end < test_end
Random splits cause data leakage - future leaks into train.

Baselines tracked: majority_class, naive_direction, random_baseline (0.5)

## 4. Model Artifact Contract (Workstream E)

Required metadata: schema_version, model_type, feature_order, target_column, target_definition, training_sample_count, train_end_date, evaluation_metrics, baseline_metrics, random_seed, artifact_checksum

## 5. API Failure Semantics (Workstream F)

Status codes: healthy, degraded, unhealthy, no_model, invalid_input, prediction_failed

No silent fallback - failures return honest errors.

## 6. Health Endpoint

Reports ml_service status with model_count, feature_schema_version, note.

## 7. NOT Validated

- No model trained or produced
- No real data evaluation
- No real-money trading enabled

## 8. Tests

Backend: cd api; python -m pytest ../tests/backend/test_ml_contracts.py -v
Frontend: cd ui; npm test -- --watchAll=false

*Task 12: ML Integrity & Prediction Reliability Program*
