"""Erzeugt neue, noch nicht zugestellte Bestellungen für den Inferenzlauf."""
import argparse
import os

import numpy as np
import pandas as pd

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "new_orders.parquet")
CATEGORIES = ["electronics", "furniture", "toys", "beauty", "sports", "books"]
STATES = ["SP", "RJ", "MG", "BA", "RS", "PR"]


def generate_new_orders(n_orders: int = 200, seed: int = 2026, start: str | None = None) -> pd.DataFrame:
    if n_orders < 1:
        raise ValueError("n_orders must be >= 1")
    rng = np.random.default_rng(seed)
    start_ts = pd.Timestamp(start) if start else pd.Timestamp.now(tz="UTC").tz_localize(None).floor("h")
    offsets = np.sort(rng.integers(0, 24 * 7, size=n_orders))
    purchase_ts = start_ts + pd.to_timedelta(offsets, unit="h")
    return pd.DataFrame({
        "order_id": [f"new_{seed}_{i:05d}" for i in range(n_orders)],
        "customer_id": rng.integers(1, 3000, size=n_orders),
        "product_category": rng.choice(CATEGORIES, size=n_orders),
        "customer_state": rng.choice(STATES, size=n_orders),
        "order_purchase_timestamp": purchase_ts,
        "order_estimated_delivery_days": rng.choice([3, 5, 7], size=n_orders),
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-orders", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", default=OUTPUT_PATH)
    args = parser.parse_args()
    orders = generate_new_orders(args.n_orders, args.seed)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    orders.to_parquet(args.output, index=False)
    print(f"[generate-new] Wrote {len(orders)} unlabeled orders to {args.output}")
