"""Inference for planned tours."""

import json
from typing import Any

import joblib
import pandas as pd

from src.config import MODEL_DIR


def list_models(model_dir=None) -> list[dict[str, str]]:
    """Return the named models registered in the models directory."""
    model_dir = MODEL_DIR if model_dir is None else model_dir
    models = []
    for metadata_path in sorted(model_dir.glob("*.txt")):
        try:
            with metadata_path.open(encoding="utf-8") as metadata_file:
                metadata = json.load(metadata_file)
            metadata["metadata_file"] = str(metadata_path)
            models.append(metadata)
        except (OSError, json.JSONDecodeError):
            continue
    return models


def predict_tour(
    model_name: str,
    distance_km: float,
    elevation_m: float,
    descent_m: float | None = None,
    stopped_time_s: float | None = None,
) -> dict[str, Any]:
    """Predict a tour using only the selected named model."""
    if (
        distance_km <= 0
        or elevation_m < 0
        or (descent_m is not None and descent_m < 0)
        or (stopped_time_s is not None and stopped_time_s < 0)
    ):
        raise ValueError(
            "distance_km must be positive; elevation and stopped time cannot be negative"
        )

    selected = next(
        (
            model
            for model in list_models()
            if model.get("model_id") == model_name or model.get("model_name") == model_name
        ),
        None,
    )
    if selected is None:
        raise FileNotFoundError(f"Model '{model_name}' not found in {MODEL_DIR}")
    model_path = MODEL_DIR / selected["model_file"]
    if not model_path.exists():
        raise FileNotFoundError(f"Model file for '{model_name}' not found: {model_path}")

    feature_values = {
        "distance_km": distance_km,
        "elevation_m": elevation_m,
        "gradient_pct": elevation_m / (distance_km * 10.0),
        "elevation_per_km": elevation_m / distance_km,
        "elevation_up_m": elevation_m,
        "elevation_down_m": elevation_m if descent_m is None else descent_m,
    }
    model_features = selected.get(
        "features",
        ["distance_km", "elevation_m", "gradient_pct", "elevation_per_km"],
    )
    features = pd.DataFrame([{feature: feature_values[feature] for feature in model_features}])
    warnings = _range_warnings(selected.get("training_ranges", {}), feature_values)
    trained_model = joblib.load(model_path)
    regression_models = trained_model.get("models", {})
    # Legacy artifacts only contain a moving-time model.
    elapsed_model = regression_models.get("elapsed_time")
    moving_model = regression_models.get("moving_time", trained_model.get("time_model"))
    energy_source = selected.get("energy", {}).get("source")
    kcal_model = (
        regression_models.get("kcal_clean", trained_model.get("kcal_model"))
        if energy_source == "calories"
        else None
    )
    if moving_model is None:
        raise ValueError(f"Model artifact for '{model_name}' is incomplete")
    if kcal_model is None:
        warnings.append("Energy unavailable: model was not trained from calories")
    if elapsed_model is None:
        elapsed_model = moving_model
    predicted_moving = max(0.0, float(moving_model.predict(features)[0]))
    stopped_model = regression_models.get("stopped_time")
    if stopped_time_s is not None:
        predicted_stopped = stopped_time_s
        predicted_elapsed = predicted_moving + predicted_stopped
    elif stopped_model is not None:
        predicted_stopped = max(0.0, float(stopped_model.predict(features)[0]))
        predicted_elapsed = predicted_moving + predicted_stopped
    else:
        predicted_elapsed = max(0.0, float(elapsed_model.predict(features)[0]))
        predicted_moving = min(predicted_elapsed, predicted_moving)
        predicted_stopped = predicted_elapsed - predicted_moving
    predicted_kcal = (
        max(0.0, float(kcal_model.predict(features)[0]))
        if kcal_model is not None
        else None
    )
    predicted_average_power = _predict_optional(regression_models.get("average_power_watts"), features)
    predicted_weighted_power = _predict_optional(
        regression_models.get("weighted_average_power_watts"), features
    )
    predicted_avg_hr = _predict_optional(regression_models.get("average_hr_bpm"), features)
    hours, remainder = divmod(round(predicted_elapsed), 3600)
    minutes = remainder // 60
    moving_hours = predicted_moving / 3600
    speed = distance_km / moving_hours if moving_hours else 0.0
    regressions = selected.get("regressions", {})
    moving_error = _error_bound(regressions.get("moving_time"))
    if stopped_time_s is not None:
        elapsed_error = moving_error
    elif stopped_model is not None:
        elapsed_error = moving_error + _error_bound(regressions.get("stopped_time"))
    else:
        elapsed_error = _error_bound(regressions.get("elapsed_time")) or moving_error
    energy_error = _error_bound(regressions.get("kcal_clean"))
    return {
        "model_name": selected["model_name"],
        "sport_type": selected["sport_type"],
        "distance_km": distance_km,
        "elevation_m": elevation_m,
        "elevation_up_m": elevation_m,
        "elevation_down_m": feature_values["elevation_down_m"],
        "predicted_time": f"{hours}h {minutes:02d}m",
        "predicted_time_sec": round(predicted_elapsed),
        "predicted_elapsed_time_sec": round(predicted_elapsed),
        "predicted_moving_time_sec": round(predicted_moving),
        "predicted_stopped_time_sec": round(predicted_stopped),
        "predicted_kcal": round(predicted_kcal) if predicted_kcal is not None else None,
        "predicted_elapsed_time_interval_sec": [
            round(max(0.0, predicted_elapsed - elapsed_error)),
            round(predicted_elapsed + elapsed_error),
        ],
        "predicted_kcal_interval": (
            [
                round(max(0.0, predicted_kcal - energy_error)),
                round(predicted_kcal + energy_error),
            ]
            if predicted_kcal is not None
            else None
        ),
        "avg_speed_kmh": round(speed, 1),
        "warnings": warnings,
        "average_power_watts": _rounded_or_none(predicted_average_power),
        "weighted_average_power_watts": _rounded_or_none(predicted_weighted_power),
        "average_hr_bpm": _rounded_or_none(predicted_avg_hr),
        "predicted_average_power_watts": _rounded_or_none(predicted_average_power),
        "predicted_weighted_average_power_watts": _rounded_or_none(predicted_weighted_power),
        "predicted_average_hr_bpm": _rounded_or_none(predicted_avg_hr),
    }


def _predict_optional(model: Any, features: pd.DataFrame) -> float | None:
    """Predict an optional metric, returning None when its model was not trained."""
    if model is None:
        return None
    return max(0.0, float(model.predict(features)[0]))


def _rounded_or_none(value: float | None) -> int | None:
    return round(value) if value is not None else None


def _error_bound(regression: dict[str, Any] | None) -> float:
    if not regression:
        return 0.0
    value = regression.get("error_p90", regression.get("mae", 0.0))
    return max(0.0, float(value or 0.0))


def _range_warnings(
    training_ranges: dict[str, dict[str, float]],
    feature_values: dict[str, float],
) -> list[str]:
    warnings = []
    for feature, bounds in training_ranges.items():
        value = feature_values.get(feature)
        if value is None:
            continue
        minimum = bounds.get("min")
        maximum = bounds.get("max")
        if minimum is not None and value < minimum:
            warnings.append(
                f"{feature}={value:g} is below the training range ({minimum:g}–{maximum:g})"
            )
        elif maximum is not None and value > maximum:
            warnings.append(
                f"{feature}={value:g} is above the training range ({minimum:g}–{maximum:g})"
            )
    return warnings
