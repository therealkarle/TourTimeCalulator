"""Retrain all existing models from the local activity cache."""

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path

# Also support: python scripts/update_models.py
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import MODEL_DIR
from src.feature_eng import load_cleaned_data
from src.model_trainer import BASE_FEATURES, train_models_for_sport


def _model_id(model_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", model_name.strip()).strip("_-").lower()


def update_model(
    metadata_path: Path,
    regression_type_override: str | None = None,
    model_name_override: str | None = None,
) -> bool:
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
                "heart_rate_data", "activity_ids",
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
    stored_regression_type = regression_type_override or filters.get(
        "regression_type", metadata.get("regression_type", "linear")
    )
    regression_type = (
        stored_regression_type
        if stored_regression_type in {"linear", "ridge"}
        else "linear"
    )
    model_name = model_name_override or metadata.get("model_name", metadata_path.stem)
    print(f"Updating model '{model_name}' as {regression_type}...")
    data = load_cleaned_data(
        metadata["sport_type"], **load_filters, elevation_mode=elevation_mode
    )
    if data.empty:
        print(f"No data for '{metadata_path.stem}'. Skipping.")
        return False
    updated = train_models_for_sport(
        data,
        metadata["sport_type"],
        model_name,
        distance_elevation_only=metadata.get("features") == BASE_FEATURES,
        filters={**filters, "regression_type": regression_type},
        separate_elevation=metadata.get("elevation_mode") == "separate" or
        "elevation_down_m" in metadata.get("features", []),
        regression_type=regression_type,
    )
    if not updated:
        return False

    new_metadata_path = (
        MODEL_DIR / f"{_model_id(model_name)}.txt"
        if model_name_override is not None
        else metadata_path
    )
    new_metadata = json.loads(new_metadata_path.read_text(encoding="utf-8"))
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


def create_model_variants(metadata_path: Path) -> bool:
    """Retrain one legacy model as exactly one Linear and one Ridge variant."""
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    source_name = metadata.get("model_name", metadata_path.stem)
    base_name = re.sub(
        r"\s+\((?:linear|ridge)\)$", "", source_name, flags=re.IGNORECASE
    )
    linear_name = f"{base_name} (Linear)"
    ridge_name = f"{base_name} (Ridge)"

    linear_updated = update_model(
        metadata_path,
        regression_type_override="linear",
        model_name_override=linear_name,
    )
    ridge_updated = update_model(
        metadata_path,
        regression_type_override="ridge",
        model_name_override=ridge_name,
    )
    if not (linear_updated and ridge_updated):
        print(f"Could not create both variants for '{base_name}'; source retained.")
        return False

    expected_files = [
        MODEL_DIR / f"{_model_id(linear_name)}.txt",
        MODEL_DIR / f"{_model_id(linear_name)}.joblib",
        MODEL_DIR / f"{_model_id(ridge_name)}.txt",
        MODEL_DIR / f"{_model_id(ridge_name)}.joblib",
    ]
    if not all(path.exists() for path in expected_files):
        print(f"Variant files for '{base_name}' are incomplete; source retained.")
        return False

    source_model = MODEL_DIR / metadata.get(
        "model_file", f"{metadata_path.stem}.joblib"
    )
    variant_paths = set(expected_files)
    archive_dir = MODEL_DIR / "legacy"
    archive_dir.mkdir(parents=True, exist_ok=True)
    for source_path in (metadata_path, source_model):
        if source_path.exists() and source_path not in variant_paths:
            destination = archive_dir / source_path.name
            if destination.exists():
                destination = archive_dir / f"{source_path.stem}_original{source_path.suffix}"
            source_path.replace(destination)
    print(f"Created Linear and Ridge variants for '{base_name}'.")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        dest="model_names",
        action="append",
        metavar="NAME_OR_ID",
        help="Update only this model; repeat the option to select multiple models",
    )
    parser.add_argument(
        "--create-variants",
        action="store_true",
        help="Replace each selected legacy model with Linear and Ridge variants",
    )
    args = parser.parse_args()

    metadata_files = sorted(MODEL_DIR.glob("*.txt"))
    if not metadata_files:
        print(f"No model metadata found in {MODEL_DIR}.")
        return 1

    if args.model_names:
        requested = {name.casefold() for name in args.model_names}
        selected_files = []
        matched_names = set()
        for path in metadata_files:
            try:
                metadata = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                print(f"Could not read model metadata '{path.name}': {exc}")
                continue
            candidates = {
                path.stem.casefold(),
                str(metadata.get("model_id", "")).casefold(),
                str(metadata.get("model_name", "")).casefold(),
            }
            if requested.intersection(candidates):
                selected_files.append(path)
                matched_names.update(requested.intersection(candidates))

        missing = requested - matched_names
        if missing:
            print(f"Model(s) not found: {', '.join(sorted(missing))}")
            return 1
        metadata_files = selected_files

    if args.create_variants:
        metadata_files = [
            path
            for path in metadata_files
            if not re.search(r"_(?:linear|ridge)$", path.stem, re.IGNORECASE)
        ]
        print(f"Creating Linear and Ridge variants for {len(metadata_files)} models...")
        updated = sum(create_model_variants(path) for path in metadata_files)
    else:
        print(f"Retraining {len(metadata_files)} saved models...")
        updated = sum(update_model(path) for path in metadata_files)
    print(f"Updated {updated} of {len(metadata_files)} models.")
    return 0 if updated else 1


if __name__ == "__main__":
    raise SystemExit(main())
