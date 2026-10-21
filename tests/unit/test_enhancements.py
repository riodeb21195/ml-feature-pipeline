"""
Tests fuer die vier Ergaenzungen: Baseline-Vergleich, Kalibrierung,
Batch-Scoring (predict.py) und den Retraining-Trigger-Schwellenwert.
Reine Logik-Tests mit kleinen synthetischen Daten, keine externen Services.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).parents[2] / "training"))
sys.path.append(str(Path(__file__).parents[2] / "monitoring"))
sys.path.append(str(Path(__file__).parents[2] / "scoring"))

from check_drift import (
    FEATURE_COLUMNS as DRIFT_FEATURES,
)
from check_drift import (
    RETRAIN_TRIGGER_MIN_DRIFTED_FEATURES,
    check_drift,
    maybe_trigger_retraining,
)
from predict import score_orders
from train_model import (
    FEATURE_COLUMNS,
    evaluate_baseline,
    evaluate_calibration,
    time_based_split,
)


def _make_train_test():
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({
        "category_late_rate_30d": rng.uniform(0, 1, n),
        "state_late_rate_30d": rng.uniform(0, 1, n),
        "customer_prior_order_count": rng.integers(0, 10, n),
        "customer_prior_late_rate": rng.uniform(0, 1, n),
        "order_estimated_delivery_days": rng.choice([3, 5, 7], n),
        "is_late_delivery": rng.integers(0, 2, n),
    })
    return df.iloc[:150], df.iloc[150:]


def test_baseline_returns_fixed_auc_of_half():
    train, test = _make_train_test()
    baseline = evaluate_baseline(train, test)
    assert baseline["roc_auc"] == 0.5
    assert "accuracy" in baseline


def test_calibration_returns_error_between_zero_and_one():
    from sklearn.ensemble import GradientBoostingClassifier

    train, test = _make_train_test()
    model = GradientBoostingClassifier(random_state=42)
    model.fit(train[FEATURE_COLUMNS], train["is_late_delivery"])

    calibration = evaluate_calibration(model, test[FEATURE_COLUMNS], test["is_late_delivery"], n_bins=3)
    assert 0 <= calibration["mean_absolute_calibration_error"] <= 1
    assert "interpretation" in calibration


def test_score_orders_adds_risk_columns():
    class FakeModel:
        def predict_proba(self, X):
            # deterministisch: erste Zeile hoch, zweite niedrig
            return np.array([[0.2, 0.8], [0.9, 0.1]])

    df = pd.DataFrame({col: [0.1, 0.2] for col in FEATURE_COLUMNS})
    meta = {"feature_columns": FEATURE_COLUMNS, "run_id": "test_run"}

    scored = score_orders(FakeModel(), meta, df)
    assert scored.loc[0, "predicted_late_risk"] == 0.8
    assert scored.loc[0, "risk_category"] == "high"
    assert scored.loc[1, "risk_category"] == "low"


def test_score_orders_raises_on_missing_feature_columns():
    class FakeModel:
        def predict_proba(self, X):
            return np.array([[0.5, 0.5]])

    df = pd.DataFrame({"only_one_column": [1]})
    meta = {"feature_columns": FEATURE_COLUMNS, "run_id": "test_run"}

    with pytest.raises(ValueError):
        score_orders(FakeModel(), meta, df)


def test_retrain_not_triggered_below_threshold():
    drifted = ["feature_a"]  # weniger als RETRAIN_TRIGGER_MIN_DRIFTED_FEATURES
    assert len(drifted) < RETRAIN_TRIGGER_MIN_DRIFTED_FEATURES
    result = maybe_trigger_retraining(drifted, auto_retrain=False)
    assert result is False


def test_retrain_criterion_met_but_not_executed_without_auto_flag():
    drifted = ["feature_a", "feature_b", "feature_c"]
    assert len(drifted) >= RETRAIN_TRIGGER_MIN_DRIFTED_FEATURES
    result = maybe_trigger_retraining(drifted, auto_retrain=False)
    assert result is False  # Kriterium erfuellt, aber ohne --auto-retrain nicht ausgefuehrt


def test_split_has_embargo_so_train_labels_are_known_before_test_starts():
    ts = pd.date_range("2025-01-01", periods=100, freq="D")
    df = pd.DataFrame({
        "order_purchase_timestamp": ts,
        "label_available_at": ts + pd.Timedelta(days=5),
        "is_late_delivery": 0,
    })
    train, test, split_ts = time_based_split(df)

    assert (train["label_available_at"] < split_ts).all()   # kein Label aus der Test-Zukunft
    assert (test["order_purchase_timestamp"] >= split_ts).all()
    assert len(train) + len(test) < len(df)                   # Embargo-Zeilen wurden entfernt
    assert train["order_purchase_timestamp"].max() < test["order_purchase_timestamp"].min()


def _drift_frame(shift=0.0, n=3000, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "order_purchase_timestamp": pd.date_range("2025-01-01", periods=n, freq="h"),
        "category_late_rate_30d": rng.normal(0.35, 0.08, n),
        "state_late_rate_30d": rng.normal(0.35, 0.08, n),
        "order_estimated_delivery_days": rng.choice([3, 5, 7], n),
    })
    cut = int(n * 0.8)
    df.loc[cut:, "category_late_rate_30d"] += shift   # nur der neueste Zeitraum verschiebt sich
    return df


def test_drift_check_stays_quiet_on_stationary_data():
    results = check_drift(_drift_frame(shift=0.0))
    assert not any(r["drift_detected"] for r in results)


def test_drift_check_detects_real_shift():
    results = check_drift(_drift_frame(shift=0.15))
    flagged = {r["feature"] for r in results if r["drift_detected"]}
    assert flagged == {"category_late_rate_30d"}


def test_features_that_grow_by_construction_are_not_monitored():
    assert "customer_prior_order_count" not in DRIFT_FEATURES
    assert "customer_prior_late_rate" not in DRIFT_FEATURES


def test_zero_probability_gets_a_risk_category():
    class ZeroModel:
        def predict_proba(self, X):
            return np.array([[1.0, 0.0]])

    df = pd.DataFrame({col: [0.1] for col in FEATURE_COLUMNS})
    scored = score_orders(ZeroModel(), {"feature_columns": FEATURE_COLUMNS, "run_id": "r"}, df)
    assert scored.loc[0, "risk_category"] == "low"
