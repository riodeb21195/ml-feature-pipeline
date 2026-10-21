"""Validiert Schema, Schlüssel, Zeitstempel und Wertebereiche vor Feature-Build/Training."""
import argparse
import json
import os

import pandas as pd

DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "..", "data_generation", "raw_orders.parquet")
REQUIRED = {
    "order_id", "customer_id", "product_category", "customer_state",
    "order_purchase_timestamp", "order_estimated_delivery_days",
    "actual_delivery_days", "is_late_delivery",
}


def validate_orders(df: pd.DataFrame, require_labels: bool = True) -> list[dict]:
    issues: list[dict] = []
    missing = sorted(REQUIRED - set(df.columns))
    if missing:
        issues.append({"check": "required_columns", "status": "fail", "details": missing})
        return issues
    if df.empty:
        issues.append({"check": "non_empty", "status": "fail", "details": "Dataset is empty"})
    if df["order_id"].isna().any() or df["order_id"].duplicated().any():
        issues.append({"check": "unique_order_id", "status": "fail", "details": "Null or duplicate order_id"})
    timestamps = pd.to_datetime(df["order_purchase_timestamp"], errors="coerce")
    if timestamps.isna().any():
        issues.append({"check": "purchase_timestamp", "status": "fail", "details": "Invalid or missing timestamps"})
    estimated = pd.to_numeric(df["order_estimated_delivery_days"], errors="coerce")
    if estimated.isna().any() or (estimated <= 0).any():
        issues.append({"check": "estimated_delivery_days", "status": "fail", "details": "Must be positive numeric values"})
    if require_labels:
        actual = pd.to_numeric(df["actual_delivery_days"], errors="coerce")
        labels = pd.to_numeric(df["is_late_delivery"], errors="coerce")
        if actual.isna().any() or (actual <= 0).any():
            issues.append({"check": "actual_delivery_days", "status": "fail", "details": "Missing or non-positive actual delivery days"})
        if labels.isna().any() or not labels.isin([0, 1]).all():
            issues.append({"check": "binary_target", "status": "fail", "details": "is_late_delivery must contain only 0/1"})
        if actual.notna().all() and labels.notna().all():
            mismatch = ((actual > estimated) != labels.astype(bool)).sum()
            if mismatch:
                issues.append({"check": "target_consistency", "status": "warning", "details": f"{int(mismatch)} rows disagree with actual > estimated"})
    if not issues:
        issues.append({"check": "all_checks", "status": "pass", "details": f"Validated {len(df)} rows"})
    return issues


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_PATH)
    args = parser.parse_args()
    df = pd.read_parquet(args.input)
    results = validate_orders(df)
    print(json.dumps(results, indent=2))
    if any(item["status"] == "fail" for item in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
