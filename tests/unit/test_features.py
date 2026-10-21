"""
Tests fuer die Point-in-Time-Correctness der Features. Der wichtigste Test dieses
Projekts: beweist, dass keine Information einfliesst, die zum Kaufzeitpunkt noch nicht
bekannt war. Dazu gehoert nicht nur "frueher gekauft", sondern auch "Ergebnis schon
bekannt" (Bestellungen, die noch unterwegs sind, zaehlen nicht).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parents[2] / "features"))
sys.path.append(str(Path(__file__).parents[2] / "data_generation"))
from build_features import (
    FEATURE_VERSION,
    PRIOR_LATE_RATE,
    add_label_availability,
    build_features,
    known_outcomes,
)
from generate_raw_orders import generate


def _make_df(rows):
    df = pd.DataFrame(rows)
    df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])
    return add_label_availability(df)


def test_order_still_in_transit_is_not_used():
    df = _make_df([
        # A: Kauf 01.01., 5 Tage unterwegs -> Ergebnis (verspaetet) erst am 06.01. bekannt
        {"product_category": "a", "order_purchase_timestamp": "2025-01-01", "actual_delivery_days": 5, "is_late_delivery": 1},
        # B: Kauf 02.01. -> A ist zu diesem Zeitpunkt noch unterwegs und darf NICHT zaehlen
        {"product_category": "a", "order_purchase_timestamp": "2025-01-02", "actual_delivery_days": 2, "is_late_delivery": 0},
        # C: Kauf 10.01. -> A (06.01.) und B (04.01.) sind bekannt
        {"product_category": "a", "order_purchase_timestamp": "2025-01-10", "actual_delivery_days": 3, "is_late_delivery": 0},
    ])
    counts, late = known_outcomes(df, "product_category", window_days=30)

    assert counts[0] == 0
    assert counts[1] == 0          # sieht A noch nicht (A ist unterwegs)
    assert counts[2] == 2 and late[2] == 1


def test_window_drops_outcomes_older_than_window():
    df = _make_df([
        {"product_category": "a", "order_purchase_timestamp": "2025-01-01", "actual_delivery_days": 5, "is_late_delivery": 1},
        {"product_category": "a", "order_purchase_timestamp": "2025-02-20", "actual_delivery_days": 3, "is_late_delivery": 0},
    ])
    counts, _ = known_outcomes(df, "product_category", window_days=30)
    assert counts[1] == 0  # A war am 06.01. bekannt, liegt aber ausserhalb der 30 Tage vor dem 20.02.


def test_future_changes_do_not_affect_past_features():
    """Leakage-Fingerabdruck: spaetere Zeilen aendern -> Features frueherer Zeilen bleiben gleich."""
    rows = [
        {"product_category": "a", "order_purchase_timestamp": f"2025-01-{d:02d}",
         "actual_delivery_days": 2, "is_late_delivery": d % 2}
        for d in (1, 3, 6, 9, 12, 15)
    ]
    original = _make_df(rows)

    modified_rows = [dict(r) for r in rows]
    for r in modified_rows[-2:]:
        r["is_late_delivery"] = 1 - r["is_late_delivery"]
    modified = _make_df(modified_rows)

    c1, l1 = known_outcomes(original, "product_category", 30)
    c2, l2 = known_outcomes(modified, "product_category", 30)
    assert np.array_equal(c1[:-2], c2[:-2])
    assert np.array_equal(l1[:-2], l2[:-2])


def test_customer_history_ignores_previous_order_that_is_still_in_transit():
    df = _make_df([
        {"customer_id": 1, "order_purchase_timestamp": "2025-01-01", "actual_delivery_days": 5, "is_late_delivery": 1},
        {"customer_id": 1, "order_purchase_timestamp": "2025-01-02", "actual_delivery_days": 2, "is_late_delivery": 0},
        {"customer_id": 1, "order_purchase_timestamp": "2025-01-20", "actual_delivery_days": 2, "is_late_delivery": 0},
    ])
    counts, late = known_outcomes(df, "customer_id", window_days=None)
    assert counts[1] == 0                      # erste Bestellung noch unterwegs
    assert counts[2] == 2 and late[2] == 1     # beide bekannt, eine davon verspaetet


def test_matches_brute_force_reference_implementation():
    df = add_label_availability(generate(n_orders=500, seed=1))
    window = np.timedelta64(30, "D")
    counts, late = known_outcomes(df, "product_category", window_days=30)

    for i in range(0, len(df), 7):
        row = df.iloc[i]
        t = row["order_purchase_timestamp"]
        mask = (
            (df["product_category"] == row["product_category"])
            & (df["label_available_at"] < t)
            & (df["label_available_at"] >= t - pd.Timedelta(window))
        )
        assert counts[i] == mask.sum()
        assert late[i] == df.loc[mask, "is_late_delivery"].sum()


def test_build_features_has_no_missing_values_and_sets_version():
    featured = build_features(generate(n_orders=300, seed=3))
    feature_cols = [
        "category_late_rate_30d", "state_late_rate_30d",
        "customer_prior_order_count", "customer_prior_late_rate",
    ]
    assert not featured[feature_cols].isna().any().any()
    assert (featured["feature_version"] == FEATURE_VERSION).all()
    # erste Bestellung insgesamt hat keine Historie -> fester Prior
    assert featured.loc[0, "category_late_rate_30d"] == PRIOR_LATE_RATE
    assert featured.loc[0, "customer_prior_order_count"] == 0
