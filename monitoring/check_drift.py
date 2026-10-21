"""
Schlanker Drift-Check: vergleicht die Feature-Verteilung des Trainings-Zeitraums
gegen die des Test-/neuesten Zeitraums mit dem Kolmogorov-Smirnov-Test.

Statt eines vollen Monitoring-Tools (Evidently etc.) reicht hier ein einzelner
statistischer Test pro Feature -- fuer ein Portfolio-Projekt mit einem Modell
und wenigen Features ist das der richtige Scope, zeigt aber das Prinzip:
Feature-Drift erkennen, BEVOR das Modell in Produktion stillschweigend
schlechter wird.

Zwei Entscheidungen, damit der Check keinen Dauer-Alarm ausloest:
- Drift gilt nur, wenn der Unterschied statistisch signifikant (p < Schwelle) UND
  praktisch relevant ist (KS-Statistik >= Mindestwert). Bei tausenden Zeilen wird
  sonst jeder winzige Unterschied "signifikant".
- Features, die per Konstruktion mit der Zeit wachsen (Kundenhistorie baut sich erst
  auf: customer_prior_order_count, customer_prior_late_rate), werden nicht geprueft.
  Sie wuerden immer "driften", ohne dass sich das Verhalten der Daten aendert.
"""

import json
import os
import sys
from datetime import datetime, timezone

import pandas as pd
from scipy.stats import ks_2samp

FEATURES_PATH = os.path.join(os.path.dirname(__file__), "..", "data_generation", "features.parquet")

# Nur Features mit zeitlich stabiler Verteilung (siehe Modul-Docstring).
FEATURE_COLUMNS = [
    "category_late_rate_30d",
    "state_late_rate_30d",
    "order_estimated_delivery_days",
]
DRIFT_P_VALUE_THRESHOLD = 0.01
DRIFT_MIN_KS_STATISTIC = 0.15  # Mindest-Effektgroesse, ab der ein Unterschied relevant ist
REFERENCE_TEST_SPLIT_FRACTION = 0.2  # gleiche Grenze wie im Training

# Ab wie vielen gedrifteten Features automatisch retrainiert wird. Bewusst > 1
# gesetzt: ein einzelnes leicht signifikantes Feature (p knapp unter 0.05)
# soll nicht bei jedem Lauf ein Retraining ausloesen (Alert-Fatigue-Vermeidung,
# gleiches Prinzip wie beim Run-Failure-Sensor in Projekt 2).
RETRAIN_TRIGGER_MIN_DRIFTED_FEATURES = 2
DRIFT_LOG_PATH = os.path.join(os.path.dirname(__file__), "drift_history.jsonl")
MIN_SAMPLE_SIZE = 30


def check_drift(df: pd.DataFrame) -> list[dict]:
    df = df.sort_values("order_purchase_timestamp").reset_index(drop=True)
    split_index = int(len(df) * (1 - REFERENCE_TEST_SPLIT_FRACTION))
    reference = df.iloc[:split_index]  # "Trainingsverteilung"
    current = df.iloc[split_index:]    # "neue" Daten, analog zu frischem Inferenz-Traffic

    missing = set(FEATURE_COLUMNS + ["order_purchase_timestamp"]) - set(df.columns)
    if missing:
        raise ValueError(f"Drift input missing columns: {sorted(missing)}")
    if split_index < MIN_SAMPLE_SIZE or len(df) - split_index < MIN_SAMPLE_SIZE:
        raise ValueError(f"Need at least {MIN_SAMPLE_SIZE} rows in both reference and current windows")

    results = []
    for col in FEATURE_COLUMNS:
        ref_values = reference[col].dropna()
        current_values = current[col].dropna()
        if len(ref_values) < MIN_SAMPLE_SIZE or len(current_values) < MIN_SAMPLE_SIZE:
            results.append({"feature": col, "ks_statistic": None, "p_value": None,
                            "drift_detected": False, "status": "insufficient_data"})
            continue
        statistic, p_value = ks_2samp(ref_values, current_values)
        drift_detected = p_value < DRIFT_P_VALUE_THRESHOLD and statistic >= DRIFT_MIN_KS_STATISTIC
        results.append({
            "feature": col,
            "ks_statistic": round(float(statistic), 4),
            "p_value": round(float(p_value), 4),
            "drift_detected": bool(drift_detected),
            "status": "ok",
        })
    return results


def log_drift_result(results: list[dict], retrain_triggered: bool) -> None:
    """Audit-Trail fuer Drift-Checks ueber die Zeit, analog zu den Run-Metadaten
    in Projekt 1/2 -- ohne Historie kann man nicht sehen, ob Drift zu- oder abnimmt."""
    record = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "results": results,
        "retrain_triggered": retrain_triggered,
    }
    os.makedirs(os.path.dirname(DRIFT_LOG_PATH), exist_ok=True)
    with open(DRIFT_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def maybe_trigger_retraining(drifted_features: list[str], auto_retrain: bool) -> bool:
    should_retrain = len(drifted_features) >= RETRAIN_TRIGGER_MIN_DRIFTED_FEATURES

    if not should_retrain:
        return False

    print(
        f"\n[drift] {len(drifted_features)} Features gedriftet (Schwelle: "
        f"{RETRAIN_TRIGGER_MIN_DRIFTED_FEATURES}) -> Retraining-Kriterium erfuellt."
    )

    if auto_retrain:
        print("[drift] Automatisches Überschreiben des aktiven Modells ist absichtlich deaktiviert. "
              "Drift allein beweist keinen Performanceverlust. Bitte Labels auswerten und "
              "einen Trainingskandidaten separat gegen das aktuelle Modell prüfen.")
    else:
        print("[drift] Kein Retraining ausgeführt. Erst Drift und gelabelte Modellleistung fachlich prüfen.")
    return False


def main():
    parser_auto_retrain = "--auto-retrain" in sys.argv

    df = pd.read_parquet(FEATURES_PATH)
    results = check_drift(df)

    print(json.dumps(results, indent=2))

    drifted = [r for r in results if r["drift_detected"]]
    retrain_triggered = False

    if drifted:
        print(f"\n[drift] WARNING: {len(drifted)} feature(s) show significant drift: "
              f"{[r['feature'] for r in drifted]}")
        retrain_triggered = maybe_trigger_retraining(
            [r["feature"] for r in drifted], auto_retrain=parser_auto_retrain
        )
    else:
        print("\n[drift] No significant drift detected. No retraining needed.")

    log_drift_result(results, retrain_triggered)


if __name__ == "__main__":
    main()
