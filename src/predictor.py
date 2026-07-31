"""Inference for planned tours."""

import json
from typing import Any

import joblib
import pandas as pd

from src.config import MODEL_DIR


def list_models(model_dir=MODEL_DIR) -> list[dict[str, str]]:
    """Return the named models registered in the models directory."""
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


def predict_tour(model_name: str, distance_km: float, elevation_m: float) -> dict[str, Any]:
    """Predict a tour using only the selected named model."""
    if distance_km <= 0 or elevation_m < 0:
        raise ValueError("distance_km must be positive and elevation_m cannot be negative")

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
    }
    model_features = selected.get(
        "features",
        ["distance_km", "elevation_m", "gradient_pct", "elevation_per_km"],
    )
    features = pd.DataFrame([{feature: feature_values[feature] for feature in model_features}])
    trained_model = joblib.load(model_path)
    predicted_seconds = max(0.0, float(trained_model["time_model"].predict(features)[0]))
    predicted_kcal = max(0.0, float(trained_model["kcal_model"].predict(features)[0]))
    hours, remainder = divmod(round(predicted_seconds), 3600)
    minutes = remainder // 60
    speed = distance_km / (predicted_seconds / 3600) if predicted_seconds else 0.0
    return {
        "model_name": selected["model_name"],
        "sport_type": selected["sport_type"],
        "distance_km": distance_km,
        "elevation_m": elevation_m,
        "predicted_time": f"{hours}h {minutes:02d}m",
        "predicted_time_sec": round(predicted_seconds),
        "predicted_kcal": round(predicted_kcal),
        "avg_speed_kmh": round(speed, 1),
    }
