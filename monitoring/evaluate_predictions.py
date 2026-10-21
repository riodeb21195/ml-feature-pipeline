"""Bewertet gespeicherte Vorhersagen, sobald echte/simulierte Labels vorliegen.

Labels werden bewusst separat eingelesen: beim Scoring dürfen sie nicht vorhanden sein.
"""
import argparse
import json
import os

import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "data_generation")
PREDICTIONS_PATH = os.path.join(BASE_DIR, "scored_orders.parquet")
LABELS_PATH = os.path.join(BASE_DIR, "resolved_orders.parquet")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "evaluation_latest.json")


def evaluate_predictions(predictions: pd.DataFrame, labels: pd.DataFrame, threshold: float = 0.5) -> dict:
    required_predictions = {"order_id", "predicted_late_risk"}
    required_labels = {"order_id", "is_late_delivery"}
    if required_predictions - set(predictions.columns):
        raise ValueError(f"Predictions missing columns: {sorted(required_predictions - set(predictions.columns))}")
    if required_labels - set(labels.columns):
        raise ValueError(f"Labels missing columns: {sorted(required_labels - set(labels.columns))}")
    joined = predictions[["order_id", "predicted_late_risk"]].merge(
        labels[["order_id", "is_late_delivery"]], on="order_id", how="inner", validate="one_to_one"
    )
    joined = joined.dropna(subset=["predicted_late_risk", "is_late_delivery"])
    if joined.empty:
        raise ValueError("No matched predictions and resolved labels found")
    if not joined["is_late_delivery"].isin([0, 1]).all():
        raise ValueError("Labels must contain only 0/1")
    if not joined["predicted_late_risk"].between(0, 1).all():
        raise ValueError("Predicted risks must be in [0, 1]")
    y = joined["is_late_delivery"].astype(int)
    p = joined["predicted_late_risk"].astype(float)
    metrics = {
        "evaluated_rows": len(joined),
        "label_late_rate": round(float(y.mean()), 4),
        "brier_score": round(float(brier_score_loss(y, p)), 4),
        "precision_at_threshold": round(float(precision_score(y, p >= threshold, zero_division=0)), 4),
        "recall_at_threshold": round(float(recall_score(y, p >= threshold, zero_division=0)), 4),
        "threshold": threshold,
        "roc_auc": round(float(roc_auc_score(y, p)), 4) if y.nunique() == 2 else None,
        "average_precision": round(float(average_precision_score(y, p)), 4),
    }
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", default=PREDICTIONS_PATH)
    parser.add_argument("--labels", default=LABELS_PATH)
    parser.add_argument("--output", default=OUTPUT_PATH)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    predictions, labels = pd.read_parquet(args.predictions), pd.read_parquet(args.labels)
    metrics = evaluate_predictions(predictions, labels, args.threshold)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
