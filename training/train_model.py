"""
Trainiert ein Gradient-Boosting-Modell auf den Point-in-Time-korrekten Features.

Deckt ab:
- Zeitbasierter statt zufaelliger Train/Test-Split: ein Random-Split wuerde
  Bestellungen aus der "Zukunft" relativ zu anderen Trainingszeilen ins
  Training mischen -- das waere eine weitere, subtilere Leakage-Quelle als
  die Feature-Berechnung selbst.
- Embargo an der Split-Grenze: Trainiert wird nur auf Bestellungen, deren Ergebnis
  (label_available_at) zum Split-Zeitpunkt schon bekannt war. Sonst waeren Labels
  im Training, die erst NACH Beginn des Test-Zeitraums feststehen.
- Schlankes Experiment-Tracking: eine JSON-Datei pro Run statt MLflow-Server,
  enthaelt Metriken, Feature-Version, Random Seed, Split-Grenze -- genug fuer
  Reproduzierbarkeit und Vergleichbarkeit ueber mehrere Laeufe.
"""

import json
import os
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

FEATURES_PATH = os.path.join(os.path.dirname(__file__), "..", "data_generation", "features.parquet")
RUNS_DIR = os.path.join(os.path.dirname(__file__), "runs")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

FEATURE_COLUMNS = [
    "category_late_rate_30d",
    "state_late_rate_30d",
    "customer_prior_order_count",
    "customer_prior_late_rate",
    "order_estimated_delivery_days",
]
TARGET_COLUMN = "is_late_delivery"
RANDOM_SEED = 42
TEST_SPLIT_FRACTION = 0.2  # letzte 20% zeitlich als Test-Set


def time_based_split(df: pd.DataFrame):
    df = df.sort_values("order_purchase_timestamp").reset_index(drop=True)
    split_index = int(len(df) * (1 - TEST_SPLIT_FRACTION))
    split_timestamp = df.loc[split_index, "order_purchase_timestamp"]
    test = df[df["order_purchase_timestamp"] >= split_timestamp]
    # Embargo: nur Trainingszeilen, deren Label zum Split-Zeitpunkt bereits bekannt ist
    train = df[df["label_available_at"] < split_timestamp]
    return train, test, split_timestamp


def evaluate_baseline(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """
    Baseline-Vergleich: ein DummyClassifier, der immer die haeufigste Klasse
    vorhersagt ("es wird nie spaet"). Ohne diesen Vergleich ist ein AUC-Wert
    wie 0.64 schwer einzuordnen -- ist das gut oder schlecht? Die Baseline
    zeigt: ohne jedes Modell/Feature waere die Trefferquote X, das echte
    Modell liegt Y darueber -- DAS ist die eigentliche Aussage, nicht der
    AUC-Wert isoliert.
    """
    X_train, y_train = train[FEATURE_COLUMNS], train[TARGET_COLUMN]
    X_test, y_test = test[FEATURE_COLUMNS], test[TARGET_COLUMN]

    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    baseline_scores = baseline.predict_proba(X_test)[:, 1]

    # Konstante Scores -> ROC-AUC ist 0.5 (kein Rangordnungs-Signal) und die
    # Average Precision entspricht der Verspaetungsquote im Test-Set.
    return {
        "strategy": "most_frequent (always predict majority class)",
        "accuracy": round(float(baseline.score(X_test, y_test)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, baseline_scores)), 4),
        "average_precision": round(float(average_precision_score(y_test, baseline_scores)), 4),
    }


def evaluate_calibration(model, X_test, y_test, n_bins: int = 10) -> dict:
    """
    Prueft, ob eine vom Modell vorhergesagte Wahrscheinlichkeit auch wirklich
    der beobachteten Trefferquote entspricht (z.B. bei allen Bestellungen mit
    vorhergesagten 70% Verspaetungsrisiko sollten tatsaechlich ca. 70% spaet sein).
    Fuer die Business-Frage ("Risiko-Score beim Checkout anzeigen") ist das
    wichtiger als der reine AUC-Wert: ein Score, dem Ops nicht trauen kann,
    ist nutzlos, selbst wenn das Modell insgesamt gut zwischen den Klassen trennt.
    """
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    prob_true, prob_pred = calibration_curve(y_test, y_pred_proba, n_bins=n_bins, strategy="quantile")

    calibration_error = float(np.mean(np.abs(prob_true - prob_pred)))

    return {
        "mean_absolute_calibration_error": round(calibration_error, 4),
        "interpretation": (
            "niedrig (<0.05) = Wahrscheinlichkeiten sind vertrauenswuerdig"
            if calibration_error < 0.05
            else "moderat/hoch = vorhergesagte Wahrscheinlichkeiten sollten nicht "
                 "woertlich als 'X% Risiko' kommuniziert werden, nur als Ranking nutzen"
        ),
        "bins": [
            {"predicted_probability": round(float(p), 4), "observed_frequency": round(float(t), 4)}
            for p, t in zip(prob_pred, prob_true)
        ],
    }


def train_and_evaluate(train: pd.DataFrame, test: pd.DataFrame):
    if train.empty or test.empty:
        raise ValueError("Train and test datasets must both contain rows")
    if train[TARGET_COLUMN].nunique() < 2:
        raise ValueError("Training data must contain both target classes")
    if test[TARGET_COLUMN].nunique() < 2:
        raise ValueError("Test data must contain both target classes for ROC-AUC evaluation")
    missing = set(FEATURE_COLUMNS + [TARGET_COLUMN]) - set(train.columns) | (set(FEATURE_COLUMNS + [TARGET_COLUMN]) - set(test.columns))
    if missing:
        raise ValueError(f"Train/test data missing columns: {sorted(missing)}")
    X_train, y_train = train[FEATURE_COLUMNS], train[TARGET_COLUMN].astype(int)
    X_test, y_test = test[FEATURE_COLUMNS], test[TARGET_COLUMN].astype(int)

    model = GradientBoostingClassifier(random_state=RANDOM_SEED)
    model.fit(X_train, y_train)

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    metrics = {
        "roc_auc": round(roc_auc_score(y_test, y_pred_proba), 4),
        "average_precision": round(average_precision_score(y_test, y_pred_proba), 4),
        "brier_score": round(float(brier_score_loss(y_test, y_pred_proba)), 4),
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision_at_default_threshold": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall_at_default_threshold": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "train_rows": len(train),
        "test_rows": len(test),
        "train_late_rate": round(y_train.mean(), 4),
        "test_late_rate": round(y_test.mean(), 4),
    }

    feature_importance = dict(zip(FEATURE_COLUMNS, [round(v, 4) for v in model.feature_importances_]))
    baseline = evaluate_baseline(train, test)
    calibration = evaluate_calibration(model, X_test, y_test)

    return model, metrics, feature_importance, baseline, calibration


def save_run(
    model, metrics: dict, feature_importance: dict, baseline: dict,
    calibration: dict, split_timestamp, feature_version: str,
) -> tuple[str, str]:
    os.makedirs(RUNS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")

    run_record = {
        "run_id": run_id,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "feature_version": feature_version,
        "feature_columns": FEATURE_COLUMNS,
        "random_seed": RANDOM_SEED,
        "split_strategy": "time_based",
        "split_timestamp": str(split_timestamp),
        "test_split_fraction": TEST_SPLIT_FRACTION,
        "metrics": metrics,
        "baseline_comparison": baseline,
        "calibration": calibration,
        "feature_importance": feature_importance,
        "model_path": f"models/model_{run_id}.joblib",
    }

    run_path = os.path.join(RUNS_DIR, f"run_{run_id}.json")
    with open(run_path, "w", encoding="utf-8") as f:
        json.dump(run_record, f, indent=2)

    # Modell selbst speichern, damit predict.py es fuer Batch-Scoring laden kann.
    # Zusaetzlich als "latest" symlink-aehnlich nochmal unter festem Namen,
    # damit predict.py nicht wissen muss, welcher run_id der neueste ist.
    model_path = os.path.join(MODELS_DIR, f"model_{run_id}.joblib")
    latest_path = os.path.join(MODELS_DIR, "latest_model.joblib")
    latest_meta_path = os.path.join(MODELS_DIR, "latest_model_meta.json")
    joblib.dump(model, model_path)
    joblib.dump(model, latest_path)
    with open(latest_meta_path, "w", encoding="utf-8") as f:
        json.dump({"run_id": run_id, "feature_version": feature_version,
                   "feature_columns": FEATURE_COLUMNS,
                   "base_rate": metrics["train_late_rate"]}, f, indent=2)

    return run_path, model_path


def main():
    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError("Features missing. Run data generation and features/build_features.py first.")
    df = pd.read_parquet(FEATURES_PATH)
    missing = set(FEATURE_COLUMNS + [TARGET_COLUMN, "label_available_at", "order_purchase_timestamp", "feature_version"]) - set(df.columns)
    if missing:
        raise ValueError(f"Feature dataset missing columns: {sorted(missing)}")
    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN, "label_available_at"]).copy()
    feature_version = df["feature_version"].iloc[0]

    train, test, split_timestamp = time_based_split(df)
    print(
        f"[train] Train: {len(train)} rows (Label bekannt vor {split_timestamp}), "
        f"Test: {len(test)} rows, Embargo-Zeilen: {len(df) - len(train) - len(test)}"
    )

    model, metrics, feature_importance, baseline, calibration = train_and_evaluate(train, test)
    print(f"[train] Metrics: {json.dumps(metrics, indent=2)}")
    print(f"[train] Baseline (majority-class) comparison: {json.dumps(baseline, indent=2)}")
    print(f"[train] AUC uplift vs. baseline: {metrics['roc_auc'] - baseline['roc_auc']:.4f}")
    print(f"[train] Average-Precision: Modell {metrics['average_precision']} vs. Baseline {baseline['average_precision']}")
    print(f"[train] Calibration: {json.dumps(calibration['interpretation'])}")
    print(f"[train] Feature importance: {json.dumps(feature_importance, indent=2)}")

    run_path, model_path = save_run(
        model, metrics, feature_importance, baseline, calibration, split_timestamp, feature_version
    )
    print(f"[train] Run saved to {run_path}")
    print(f"[train] Model saved to {model_path} (and models/latest_model.joblib)")


if __name__ == "__main__":
    main()
