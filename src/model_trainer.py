"""Training and persistence of tour prediction models."""

import json
import re
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from src.config import MODEL_DIR

BASE_FEATURES = ["distance_km", "elevation_m"]
FEATURES = BASE_FEATURES + ["elevation_per_km"]


def train_models_for_sport(
    df: pd.DataFrame,
    sport_name: str,
    model_name: str,
    model_dir: Path = MODEL_DIR,
    distance_elevation_only: bool = False,
) -> bool:
    """Train and save one named model containing duration and energy regressors."""
    model_features = BASE_FEATURES if distance_elevation_only else FEATURES
    if len(df) < 5:
        print(f"Insufficient data ({len(df)} samples) to train models for '{sport_name}'.")
        return False
    if not set(model_features + ["moving_time", "kcal_clean"]).issubset(df.columns):
        raise ValueError("DataFrame is missing required model columns")

    X = df[model_features]
    y_time, y_kcal = df["moving_time"], df["kcal_clean"]
    X_train, X_test, time_train, time_test, kcal_train, kcal_test = train_test_split(
        X, y_time, y_kcal, test_size=0.2, random_state=42
    )
    model_time = LinearRegression()
    model_kcal = LinearRegression()
    model_time.fit(X_train, time_train)
    model_kcal.fit(X_train, kcal_train)
    print(
        f"[{sport_name.upper()}] MAE -> Duration: "
        f"{mean_absolute_error(time_test, model_time.predict(X_test)) / 60:.1f} min | "
        f"Energy: {mean_absolute_error(kcal_test, model_kcal.predict(X_test)):.0f} kcal"
    )
    model_id = re.sub(r"[^a-zA-Z0-9_-]+", "_", model_name.strip()).strip("_-").lower()
    if not model_id:
        raise ValueError("model_name must contain at least one letter or number")

    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model_name": model_name.strip(),
            "sport_type": sport_name.lower(),
            "time_model": model_time,
            "kcal_model": model_kcal,
        },
        model_dir / f"{model_id}.joblib",
    )
    duration_coefficients = dict(zip(model_features, model_time.coef_))
    kcal_coefficients = dict(zip(model_features, model_kcal.coef_))
    with (model_dir / f"{model_id}.txt").open("w", encoding="utf-8") as metadata_file:
        json.dump(
            {
                "model_id": model_id,
                "model_name": model_name.strip(),
                "sport_type": sport_name.lower(),
                "model_file": f"{model_id}.joblib",
                "features": model_features,
                "duration_intercept_seconds": float(model_time.intercept_),
                "duration_coefficients": {
                    feature: float(coefficient)
                    for feature, coefficient in duration_coefficients.items()
                },
                "kcal_intercept": float(model_kcal.intercept_),
                "kcal_coefficients": {
                    feature: float(coefficient)
                    for feature, coefficient in kcal_coefficients.items()
                },
            },
            metadata_file,
            indent=2,
        )
        metadata_file.write("\n")
    print(f"Saved model '{model_name.strip()}' ({sport_name}) to {model_dir / (model_id + '.txt')}")
    return True
