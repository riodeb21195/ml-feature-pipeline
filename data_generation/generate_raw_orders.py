"""
Erzeugt eine synthetische Order-Historie (12 Monate), die absichtlich echte
Muster enthaelt, die ein Modell lernen KANN (z.B. bestimmte Kategorien/Regionen
sind systematisch langsamer) -- sonst waere ein "das Modell lernt nichts"-Ergebnis
kein Zeichen von Leakage-Vermeidung, sondern einfach fehlendes Signal in den Daten.
"""

import os

import numpy as np
import pandas as pd

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data_generation", "raw_orders.parquet")

CATEGORIES = ["electronics", "furniture", "toys", "beauty", "sports", "books"]
STATES = ["SP", "RJ", "MG", "BA", "RS", "PR"]

# eingebautes Signal: manche Kategorien/Staaten sind systematisch langsamer
CATEGORY_DELAY_BIAS = {"furniture": 0.35, "sports": 0.10, "electronics": 0.05,
                        "toys": 0.15, "beauty": 0.08, "books": 0.03}
STATE_DELAY_BIAS = {"SP": 0.05, "RJ": 0.10, "MG": 0.15, "BA": 0.25, "RS": 0.20, "PR": 0.12}


def generate(n_orders: int = 8000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    start = pd.Timestamp("2025-01-01")
    purchase_offsets_hours = np.sort(rng.uniform(0, 24 * 365, n_orders))
    purchase_timestamps = [start + pd.Timedelta(hours=h) for h in purchase_offsets_hours]

    categories = rng.choice(CATEGORIES, size=n_orders)
    states = rng.choice(STATES, size=n_orders)
    customer_ids = rng.integers(1, 2500, size=n_orders)

    estimated_days = rng.choice([3, 5, 7], size=n_orders)

    late_prob = np.array([
        0.08 + CATEGORY_DELAY_BIAS[c] + STATE_DELAY_BIAS[s]
        for c, s in zip(categories, states)
    ])
    late_prob = np.clip(late_prob, 0.02, 0.85)
    is_late = rng.random(n_orders) < late_prob

    actual_days = np.where(
        is_late,
        estimated_days + rng.integers(1, 8, size=n_orders),
        rng.integers(1, np.maximum(estimated_days, 2)),
    )

    df = pd.DataFrame({
        "order_id": [f"ord_{i:06d}" for i in range(n_orders)],
        "customer_id": customer_ids,
        "product_category": categories,
        "customer_state": states,
        "order_purchase_timestamp": purchase_timestamps,
        "order_estimated_delivery_days": estimated_days,
        "actual_delivery_days": actual_days,
        "is_late_delivery": is_late.astype(int),
    })
    return df.sort_values("order_purchase_timestamp").reset_index(drop=True)


if __name__ == "__main__":
    df = generate()
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"[generate] Wrote {len(df)} orders to {OUTPUT_PATH}")
    print(f"[generate] Overall late rate: {df['is_late_delivery'].mean():.1%}")
