"""Inference for planned tours."""

from typing import Any

import joblib
import pandas as pd

from src.config import MODEL_DIR
from src.model_trainer import FEATURES


def predict_tour(sport_type: str, distance_km: float, elevation_m: float) -> dict[str, Any]:
    """Predict duration, calories, and average speed for a planned tour."""
    sport_key = sport_type.strip().lower()
    if sport_key not in {"ride", "run"}:
        raise ValueError("sport_type must be 'ride' or 'run'")
    if distance_km <= 0 or elevation_m < 0:
        raise ValueError("distance_km must be positive and elevation_m cannot be negative")

    time_path = MODEL_DIR / f"{sport_key}_time.joblib"
    kcal_path = MODEL_DIR / f"{sport_key}_kcal.joblib"
    if not time_path.exists() or not kcal_path.exists():
        raise FileNotFoundError(f"Trained models for '{sport_key}' not found in {MODEL_DIR}")

    features = pd.DataFrame([{
        "distance_km": distance_km,
        "elevation_m": elevation_m,
        "gradient_pct": elevation_m / (distance_km * 10.0),
        "elevation_per_km": elevation_m / distance_km,
    }], columns=FEATURES)
    predicted_seconds = max(0.0, float(joblib.load(time_path).predict(features)[0]))
    predicted_kcal = max(0.0, float(joblib.load(kcal_path).predict(features)[0]))
    hours, remainder = divmod(round(predicted_seconds), 3600)
    minutes = remainder // 60
    speed = distance_km / (predicted_seconds / 3600) if predicted_seconds else 0.0
    return {
        "sport_type": sport_key,
        "distance_km": distance_km,
        "elevation_m": elevation_m,
        "predicted_time": f"{hours}h {minutes:02d}m",
        "predicted_time_sec": round(predicted_seconds),
        "predicted_kcal": round(predicted_kcal),
        "avg_speed_kmh": round(speed, 1),
    }
