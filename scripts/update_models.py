"""Retrain all existing models from the local activity cache."""

import json
import sys
from datetime import date
from pathlib import Path

# Also support: python scripts/update_models.py
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import MODEL_DIR
from src.feature_eng import load_cleaned_data
from src.model_trainer import BASE_FEATURES, train_models_for_sport


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
    # Reuse exactly the filters that were used to train the saved model.
    # Missing keys are kept compatible with older metadata files.
    load_filters = {
        key: filters.get(key)
        for key in (
            "activity_types", "commute", "equipment", "power_data",
            "min_distance_km", "max_distance_km",
            "min_elevation_m", "max_elevation_m",
            "min_elevation_up_m", "max_elevation_up_m",
            "min_elevation_down_m", "max_elevation_down_m",
            "min_moving_time_s", "max_moving_time_s",
            "min_elapsed_time_s", "max_elapsed_time_s",
            "heart_rate_data",
        )
    }
    # Models created before filters were persisted used the historical 3 km
    # default. Newer metadata contains the value explicitly (possibly None).
    if "min_distance_km" not in filters:
        load_filters["min_distance_km"] = 3.0
    load_filters.update(date_filters)
    default_elevation_mode: str = (
        "separate" if metadata.get("elevation_mode") == "separate" else "up"
    )
    stored_elevation_mode = filters.get("elevation_mode")
    elevation_mode: str = (
        stored_elevation_mode
        if isinstance(stored_elevation_mode, str)
        and stored_elevation_mode in {"up", "separate"}
        else default_elevation_mode
    )
    print(f"Updating model '{metadata.get('model_name', metadata_path.stem)}'...")
    data = load_cleaned_data(
        metadata["sport_type"], **load_filters, elevation_mode=elevation_mode
    )
    if data.empty:
        print(f"No data for '{metadata_path.stem}'. Skipping.")
        return False
    updated = train_models_for_sport(
        data,
        metadata["sport_type"],
        metadata.get("model_name", metadata_path.stem),
        distance_elevation_only=metadata.get("features") == BASE_FEATURES,
        filters=filters,
        separate_elevation=metadata.get("elevation_mode") == "separate" or
        "elevation_down_m" in metadata.get("features", []),
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
    metadata_files = sorted(MODEL_DIR.glob("*.txt"))
    if not metadata_files:
        print(f"No model metadata found in {MODEL_DIR}.")
        return 1

    print(f"Retraining {len(metadata_files)} saved models...")
    updated = sum(update_model(path) for path in metadata_files)
    print(f"Updated {updated} of {len(metadata_files)} models.")
    return 0 if updated else 1


if __name__ == "__main__":
    raise SystemExit(main())
