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
SEPARATE_ELEVATION_FEATURES = ["distance_km", "elevation_up_m", "elevation_down_m"]


def train_models_for_sport(
    df: pd.DataFrame,
    sport_name: str,
    model_name: str,
    model_dir: Path = MODEL_DIR,
    distance_elevation_only: bool = False,
    filters: dict | None = None,
    separate_elevation: bool = False,
) -> bool:
    """Train and save regressors for time, energy, power, and heart rate."""
    if separate_elevation:
        model_features = SEPARATE_ELEVATION_FEATURES
    else:
        model_features = BASE_FEATURES if distance_elevation_only else FEATURES
    if len(df) < 5:
        print(f"Insufficient data ({len(df)} samples) to train models for '{sport_name}'.")
        return False
    if not set(model_features + ["moving_time", "elapsed_time", "kcal_clean"]).issubset(df.columns):
        raise ValueError("DataFrame is missing required model columns")

    X = df[model_features]
    targets = {
        "elapsed_time": "Elapsed time",
        "moving_time": "Moving time",
        "kcal_clean": "Energy",
        "average_power_watts": "Average power",
        "weighted_average_power_watts": "Weighted average power",
        "average_hr_bpm": "Average heart rate",
    }
    models = {}
    metrics = {}
    sample_counts = {}
    for target, label in targets.items():
        if target not in df or not df[target].notna().any():
            continue
        target_frame = pd.concat([X, df[target]], axis=1).dropna()
        if len(target_frame) < 5:
            continue
        target_X = target_frame[model_features]
        target_y = target_frame[target]
        X_train, X_test, y_train, y_test = train_test_split(
            target_X, target_y, test_size=0.2, random_state=42
        )
        model = LinearRegression().fit(X_train, y_train)
        models[target] = model
        metrics[target] = mean_absolute_error(y_test, model.predict(X_test))
        sample_counts[target] = len(target_frame)

    required_targets = {"elapsed_time", "moving_time", "kcal_clean"}
    if not required_targets.issubset(models):
        print(f"Insufficient complete data to train time and energy models for '{sport_name}'.")
        return False
    print(f"[{sport_name.upper()}] Activities used: {len(df)}")
    print(
        f"[{sport_name.upper()}] MAE -> "
        f"Elapsed: {metrics['elapsed_time'] / 60:.1f} min | "
        f"Moving: {metrics['moving_time'] / 60:.1f} min | "
        f"Energy: {metrics['kcal_clean']:.0f} kcal"
    )
    optional_labels = {
        "average_power_watts": "Avg power",
        "weighted_average_power_watts": "Weighted avg power",
        "average_hr_bpm": "Avg HR",
    }
    for target, label in optional_labels.items():
        if target in metrics:
            print(f"[{sport_name.upper()}] MAE -> {label}: {metrics[target]:.1f}")
    print(f"[{sport_name.upper()}] Coefficients:")
    for target, model in models.items():
        coefficients = ", ".join(
            f"{feature}={coefficient:.6g}"
            for feature, coefficient in zip(model_features, model.coef_)
        )
        print(
            f"  {target} (n={sample_counts[target]}): "
            f"intercept={model.intercept_:.6g}, {coefficients}"
        )
    model_id = re.sub(r"[^a-zA-Z0-9_-]+", "_", model_name.strip()).strip("_-").lower()
    if not model_id:
        raise ValueError("model_name must contain at least one letter or number")

    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model_name": model_name.strip(),
            "sport_type": sport_name.lower(),
            "models": models,
            # Keep these aliases for callers of older model artifacts.
            "time_model": models["moving_time"],
            "kcal_model": models["kcal_clean"],
        },
        model_dir / f"{model_id}.joblib",
    )
    with (model_dir / f"{model_id}.txt").open("w", encoding="utf-8") as metadata_file:
        regressions = {}
        for target, model in models.items():
            regressions[target] = {
                "intercept": float(model.intercept_),
                "coefficients": {
                    feature: float(coefficient)
                    for feature, coefficient in zip(model_features, model.coef_)
                },
                "mae": float(metrics[target]),
            }
        json.dump(
            {
                "model_id": model_id,
                "model_name": model_name.strip(),
                "sport_type": sport_name.lower(),
                "model_file": f"{model_id}.joblib",
                "features": model_features,
                "elevation_mode": "separate" if separate_elevation else "legacy",
                "sample_count": len(df),
                "sample_counts": sample_counts,
                "filters": filters or {},
                "regressions": regressions,
                # Preserve the original metadata names for existing tooling.
                "duration_intercept_seconds": regressions["moving_time"]["intercept"],
                "duration_coefficients": regressions["moving_time"]["coefficients"],
                "kcal_intercept": regressions["kcal_clean"]["intercept"],
                "kcal_coefficients": regressions["kcal_clean"]["coefficients"],
            },
            metadata_file,
            indent=2,
        )
        metadata_file.write("\n")
    print(f"Saved model '{model_name.strip()}' ({sport_name}) to {model_dir / (model_id + '.txt')}")
    return True
