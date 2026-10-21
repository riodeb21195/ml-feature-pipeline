"""
Batch-Scoring: laedt das zuletzt trainierte Modell und wendet es auf neue
Bestellungen an, um ein Verspaetungsrisiko vorherzusagen.

Das ist der Schritt, der in den anderen Skripten bisher fehlte: Training
alleine erzeugt kein Business-Value -- erst wenn das Modell tatsaechlich auf
neue Daten angewendet wird (hier: die neuesten Bestellungen aus dem
Test-Zeitraum, um zu simulieren "so wuerde ein taeglicher Scoring-Batch-Job
aussehen"), schliesst sich der Kreis Feature -> Training -> Nutzung.

In einem echten Setup wuerde dieses Skript taeglich per Orchestrator laufen
und die Scores in eine Tabelle schreiben, die Ops im Checkout-Prozess abfragt.
"""

import argparse
import json
import os

import joblib
import pandas as pd

NEW_ORDERS_PATH = os.path.join(os.path.dirname(__file__), "..", "data_generation", "new_orders.parquet")
RAW_ORDERS_PATH = os.path.join(os.path.dirname(__file__), "..", "data_generation", "raw_orders.parquet")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data_generation", "scored_orders.parquet")

RISK_THRESHOLD_HIGH = 0.5


def load_latest_model():
    model_path = os.path.join(MODELS_DIR, "latest_model.joblib")
    meta_path = os.path.join(MODELS_DIR, "latest_model_meta.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            "Kein trainiertes Modell gefunden. Erst `python training/train_model.py` ausfuehren."
        )

    model = joblib.load(model_path)
    with open(meta_path) as f:
        meta = json.load(f)
    return model, meta


def score_orders(model, meta: dict, df: pd.DataFrame) -> pd.DataFrame:
    feature_columns = meta["feature_columns"]
    missing = set(feature_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Input-Daten fehlen Feature-Spalten: {missing}")

    df = df.copy()
    if df.empty:
        raise ValueError("Cannot score an empty batch")
    if not hasattr(model, "predict_proba"):
        raise TypeError("Loaded model must implement predict_proba")
    probabilities = model.predict_proba(df[feature_columns])[:, 1]
    if not ((probabilities >= 0) & (probabilities <= 1)).all():
        raise ValueError("Model returned probabilities outside [0, 1]")
    df["predicted_late_risk"] = probabilities
    df["risk_category"] = pd.cut(
        df["predicted_late_risk"],
        bins=[0, 0.3, RISK_THRESHOLD_HIGH, 1.0],
        labels=["low", "medium", "high"],
        include_lowest=True,
    )
    df["scored_with_model_run_id"] = meta["run_id"]
    return df


def main():
    parser = argparse.ArgumentParser(description="Batch-Scoring fuer Delivery-Risk")
    parser.add_argument("--input", default=NEW_ORDERS_PATH,
                        help="Parquet-Datei mit ungelabelten neuen Bestellungen")
    parser.add_argument("--history", default=RAW_ORDERS_PATH,
                        help="Historische Bestellungen mit bekannten Lieferergebnissen")
    parser.add_argument("--output", default=OUTPUT_PATH, help="Zieldatei für Scores")
    args = parser.parse_args()

    model, meta = load_latest_model()
    print(f"[predict] Loaded model from run {meta['run_id']} (feature_version={meta['feature_version']})")

    if not os.path.exists(args.input):
        raise FileNotFoundError(
            f"Neue Bestellungen fehlen: {args.input}. Zuerst "
            "`python data_generation/generate_new_orders.py` ausführen."
        )
    if not os.path.exists(args.history):
        raise FileNotFoundError(f"Historische Daten fehlen: {args.history}")

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "features"))
    from build_features import build_features

    history = pd.read_parquet(args.history)
    new_orders = pd.read_parquet(args.input)
    if "actual_delivery_days" in new_orders.columns or "is_late_delivery" in new_orders.columns:
        raise ValueError("Scoring input must not contain actual delivery outcomes or target labels")
    new_orders = new_orders.copy()
    new_orders["_is_scoring_order"] = True
    history = history.copy()
    history["_is_scoring_order"] = False
    combined = pd.concat([history, new_orders], ignore_index=True, sort=False)
    featured = build_features(combined)
    scoring_batch = featured.loc[featured["_is_scoring_order"]].drop(columns="_is_scoring_order")
    # Outcomes are neither needed nor allowed in the inference payload.
    scoring_batch = scoring_batch.drop(
        columns=["actual_delivery_days", "is_late_delivery", "label_available_at"],
        errors="ignore",
    )
    from build_features import FEATURE_VERSION
    if meta.get("feature_version") != FEATURE_VERSION:
        raise ValueError(
            f"Model feature_version={meta.get('feature_version')} does not match "
            f"current feature code={FEATURE_VERSION}; retrain or restore matching code."
        )
    scored = score_orders(model, meta, scoring_batch)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    scored.to_parquet(args.output, index=False)

    high_risk_count = (scored["risk_category"] == "high").sum()
    print(f"[predict] Scored {len(scored)} new orders -> {args.output}")
    print(f"[predict] {high_risk_count} orders flagged as HIGH risk (>{RISK_THRESHOLD_HIGH:.0%})")

    top_risk = scored.nlargest(5, "predicted_late_risk")[
        ["order_id", "product_category", "customer_state", "predicted_late_risk", "risk_category"]
    ]
    print("\n[predict] Top 5 highest-risk orders:")
    print(top_risk.to_string(index=False))


if __name__ == "__main__":
    main()
