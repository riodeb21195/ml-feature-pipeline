import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.extend([str(ROOT / "features"), str(ROOT / "data_generation"), str(ROOT / "monitoring"), str(ROOT / "scoring")])

def _load_test_modules():
    from build_features import build_features
    from check_data_quality import validate_orders
    from evaluate_predictions import evaluate_predictions
    from generate_new_orders import generate_new_orders
    from predict import score_orders
    return build_features, validate_orders, evaluate_predictions, generate_new_orders, score_orders

build_features, validate_orders, evaluate_predictions, generate_new_orders, score_orders = _load_test_modules()


def _history():
    return pd.DataFrame({
        "order_id": ["old1", "old2", "old3"],
        "customer_id": [1, 1, 2],
        "product_category": ["books", "books", "toys"],
        "customer_state": ["SP", "SP", "RJ"],
        "order_purchase_timestamp": pd.to_datetime(["2025-01-01", "2025-01-10", "2025-01-12"]),
        "order_estimated_delivery_days": [3, 5, 3],
        "actual_delivery_days": [5, 3, 2],
        "is_late_delivery": [1, 0, 0],
    })


def test_new_orders_have_no_outcome_or_target():
    orders = generate_new_orders(20, seed=7, start="2026-01-01")
    assert len(orders) == 20
    assert "is_late_delivery" not in orders
    assert "actual_delivery_days" not in orders
    assert orders["order_id"].is_unique


def test_unlabelled_orders_get_point_in_time_features_from_history():
    history = _history()
    new = generate_new_orders(3, seed=8, start="2026-02-01")
    featured = build_features(pd.concat([history, new], ignore_index=True, sort=False))
    fresh = featured[featured["order_id"].str.startswith("new_")]
    assert len(fresh) == 3
    assert fresh["is_late_delivery"].isna().all()
    assert fresh["category_late_rate_30d"].between(0, 1).all()
    assert fresh["customer_prior_order_count"].ge(0).all()


def test_data_quality_passes_valid_historical_orders():
    results = validate_orders(_history())
    assert not any(item["status"] == "fail" for item in results)


def test_prediction_evaluation_reports_metrics():
    predictions = pd.DataFrame({"order_id": ["a", "b", "c", "d"], "predicted_late_risk": [0.9, 0.7, 0.2, 0.1]})
    labels = pd.DataFrame({"order_id": ["a", "b", "c", "d"], "is_late_delivery": [1, 0, 0, 1]})
    result = evaluate_predictions(predictions, labels)
    assert result["evaluated_rows"] == 4
    assert 0 <= result["brier_score"] <= 1
    assert result["roc_auc"] is not None


def test_prediction_evaluation_rejects_duplicate_label_keys():
    predictions = pd.DataFrame({"order_id": ["a"], "predicted_late_risk": [0.5]})
    labels = pd.DataFrame({"order_id": ["a", "a"], "is_late_delivery": [0, 1]})
    with pytest.raises(pd.errors.MergeError):
        evaluate_predictions(predictions, labels)


def test_scoring_rejects_empty_batch_and_bad_probability():
    class BadModel:
        def predict_proba(self, X):
            return np.array([[0.0, 1.2]])

    meta = {"feature_columns": ["x"], "run_id": "test"}
    with pytest.raises(ValueError, match="empty"):
        score_orders(BadModel(), meta, pd.DataFrame({"x": []}))
    with pytest.raises(ValueError, match="outside"):
        score_orders(BadModel(), meta, pd.DataFrame({"x": [1]}))
