"""Synchronize new activities and retrain all existing models."""

import argparse
import json
import sys
from datetime import date
from pathlib import Path

# Also support: python scripts/update_models.py
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import MODEL_DIR
from src.feature_eng import load_cleaned_data
from src.model_trainer import BASE_FEATURES, train_models_for_sport
from src.strava_client import StravaClient


def update_model(metadata_path: Path) -> bool:
    """Retrain one model using the filters saved with its metadata."""
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    filters = metadata.get("filters", {}) or {}
    date_filters = {
        key: date.fromisoformat(value) if value else None
        for key, value in (
            ("start_date", filters.get("start_date")),
            ("end_date", filters.get("end_date")),
        )
    }
    load_filters = {
        "activity_types": filters.get("activity_types"),
        "commute": filters.get("commute"),
        "equipment": filters.get("equipment"),
        "power_data": filters.get("power_data"),
        "min_distance_km": filters.get("min_distance_km", 3.0),
        "max_distance_km": filters.get("max_distance_km"),
        "min_elevation_m": filters.get("min_elevation_m"),
        "max_elevation_m": filters.get("max_elevation_m"),
        **date_filters,
    }
    data = load_cleaned_data(metadata["sport_type"], **load_filters)
    if data.empty:
        print(f"No data for '{metadata_path.stem}'. Skipping.")
        return False
    updated = train_models_for_sport(
        data,
        metadata["sport_type"],
        metadata.get("model_name", metadata_path.stem),
        distance_elevation_only=metadata.get("features") == BASE_FEATURES,
        filters=filters,
    )
    if not updated:
        return False

    new_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    old_count = metadata.get("sample_count")
    new_count = new_metadata.get("sample_count", len(data))
    if old_count is None:
        print(f"  Activities: unknown -> {new_count}")
    else:
        print(f"  Activities: {old_count} -> {new_count} ({new_count - old_count:+d})")
    print("  Changes:")
    for target, new_regression in new_metadata.get("regressions", {}).items():
        old_regression = metadata.get("regressions", {}).get(target)
        if not old_regression:
            print(f"    {target}: new")
            continue
        intercept_delta = new_regression["intercept"] - old_regression["intercept"]
        mae_delta = new_regression["mae"] - old_regression["mae"]
        coefficient_deltas = ", ".join(
            f"{feature}={new_regression['coefficients'][feature] - old_regression['coefficients'].get(feature, 0):+.6g}"
            for feature in new_regression["coefficients"]
        )
        print(
            f"    {target}: intercept={intercept_delta:+.6g}, "
            f"MAE={mae_delta:+.6g}, {coefficient_deltas}"
        )
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-sync", action="store_true", help="Do not fetch new Strava activities"
    )
    args = parser.parse_args()

    if not args.no_sync:
        print("Synchronizing Strava activities...")
        added = StravaClient().sync_activities()
        print(f"Done. {added} activities were added or updated.")

    metadata_files = sorted(MODEL_DIR.glob("*.txt"))
    if not metadata_files:
        print(f"No model metadata found in {MODEL_DIR}.")
        return 1

    updated = sum(update_model(path) for path in metadata_files)
    print(f"Updated {updated} of {len(metadata_files)} models.")
    return 0 if updated else 1


if __name__ == "__main__":
    raise SystemExit(main())
