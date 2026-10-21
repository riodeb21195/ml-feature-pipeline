"""
Feature-Engineering mit expliziter Point-in-Time-Correctness.

Zwei Zeitpunkte pro Bestellung sind wichtig:
  1. order_purchase_timestamp: wann gekauft wurde (zu diesem Zeitpunkt soll das Modell scoren)
  2. label_available_at:       wann das Ergebnis "verspaetet ja/nein" bekannt wird
                               (Kaufzeitpunkt + tatsaechliche Lieferdauer)

Ein Feature wie "Verspaetungsrate der Kategorie in den letzten 30 Tagen" darf fuer
eine Bestellung nur Bestellungen nutzen, deren Ergebnis ZUM KAUFZEITPUNKT DIESER
BESTELLUNG BEREITS BEKANNT war. Es reicht NICHT, nur frueher gekaufte Bestellungen zu
nehmen: eine vor zwei Tagen gekaufte Bestellung ist noch unterwegs, ihr Label ist
noch unbekannt. Wuerde man sie einrechnen, sieht das Modell im Training Information,
die es beim echten Einsatz (Checkout) nie haben kann (Label-Leakage).

Implementierung: pro Gruppe die Ergebnisse nach label_available_at sortieren,
kumulativ aufsummieren und pro Bestellung per searchsorted die Summe ueber das
Intervall [Kauf - Fenster, Kauf) ablesen. Bestellungen mit label_available_at >= Kauf
(inklusive der Bestellung selbst) werden dadurch automatisch ausgeschlossen.
"""

import os

import numpy as np
import pandas as pd

INPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data_generation", "raw_orders.parquet")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data_generation", "features.parquet")

FEATURE_VERSION = "v3"
ROLLING_WINDOW_DAYS = 30
# Fester Prior fuer fehlende Werte (Anfang der Historie, Erstbesteller). Bewusst eine
# Konstante, NICHT der Mittelwert der Daten: der wuerde Zukunftswissen einschliessen.
PRIOR_LATE_RATE = 0.15


def add_label_availability(df: pd.DataFrame) -> pd.DataFrame:
    """Setze label_available_at nur fuer Zeilen mit bekanntem Lieferergebnis.

    Unbewertete neue Bestellungen duerfen actual_delivery_days/is_late_delivery
    nicht enthalten. Fuer sie bleibt label_available_at NaT.
    """
    df = df.copy()
    if "actual_delivery_days" not in df.columns:
        df["actual_delivery_days"] = np.nan
    if "is_late_delivery" not in df.columns:
        df["is_late_delivery"] = np.nan
    known = df["actual_delivery_days"].notna() & df["is_late_delivery"].notna()
    df["label_available_at"] = pd.NaT
    df.loc[known, "label_available_at"] = (
        df.loc[known, "order_purchase_timestamp"]
        + pd.to_timedelta(df.loc[known, "actual_delivery_days"], unit="D")
    )
    return df


def known_outcomes(df: pd.DataFrame, group_col: str, window_days: int | None):
    """Zähle bekannte Labels je Gruppe zum Kaufzeitpunkt jeder Zeile.

    Funktioniert sowohl für historische Zeilen mit Label als auch für neue,
    noch nicht zugestellte Bestellungen ohne Label.
    """
    counts = np.zeros(len(df), dtype=float)
    late_sums = np.zeros(len(df), dtype=float)
    valid_mask = df["label_available_at"].notna() & df["is_late_delivery"].notna()

    for positions in df.groupby(group_col, dropna=False).indices.values():
        group_positions = np.asarray(positions)
        valid_positions = group_positions[valid_mask.iloc[group_positions].to_numpy()]
        if len(valid_positions) == 0:
            continue
        outcomes = df.iloc[valid_positions].sort_values("label_available_at")
        label_times = outcomes["label_available_at"].to_numpy(dtype="datetime64[ns]")
        late_values = outcomes["is_late_delivery"].to_numpy(dtype=float)
        cum_late = np.concatenate([[0.0], np.cumsum(late_values)])

        query_times = df.iloc[group_positions]["order_purchase_timestamp"].to_numpy(dtype="datetime64[ns]")
        hi = np.searchsorted(label_times, query_times, side="left")
        if window_days is None:
            lo = np.zeros(len(group_positions), dtype=int)
        else:
            lower_bound = query_times - np.timedelta64(window_days, "D")
            lo = np.searchsorted(label_times, lower_bound, side="left")
        counts[group_positions] = hi - lo
        late_sums[group_positions] = cum_late[hi] - cum_late[lo]

    return counts, late_sums


def _rate(counts: np.ndarray, late_sums: np.ndarray) -> np.ndarray:
    return np.where(counts > 0, late_sums / np.maximum(counts, 1), np.nan)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    required = {"order_id", "customer_id", "product_category", "customer_state",
                "order_purchase_timestamp", "order_estimated_delivery_days"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required input columns: {sorted(missing)}")
    df = df.copy()
    df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"], errors="raise")
    df = add_label_availability(
        df.sort_values("order_purchase_timestamp").reset_index(drop=True)
    )

    df["category_late_rate_30d"] = _rate(*known_outcomes(df, "product_category", ROLLING_WINDOW_DAYS))
    df["state_late_rate_30d"] = _rate(*known_outcomes(df, "customer_state", ROLLING_WINDOW_DAYS))
    df["customer_prior_late_rate"] = _rate(*known_outcomes(df, "customer_id", None))

    # Anzahl fruehere Bestellungen: der Kauf selbst ist zum Kaufzeitpunkt bekannt
    # (nur das Lieferergebnis nicht), deshalb zaehlt hier jede fruehere Bestellung.
    df["customer_prior_order_count"] = df.groupby("customer_id").cumcount()

    for col in ["category_late_rate_30d", "state_late_rate_30d", "customer_prior_late_rate"]:
        df[col] = df[col].fillna(PRIOR_LATE_RATE)

    df["feature_version"] = FEATURE_VERSION
    return df


if __name__ == "__main__":
    raw = pd.read_parquet(INPUT_PATH)
    featured = build_features(raw)
    featured.to_parquet(OUTPUT_PATH, index=False)
    print(f"[features] Wrote {len(featured)} rows with features to {OUTPUT_PATH}")
    print(f"[features] Feature version: {FEATURE_VERSION}")
