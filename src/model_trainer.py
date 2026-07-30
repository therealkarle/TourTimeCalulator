"""Training and persistence of tour prediction models."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from src.config import MODEL_DIR

FEATURES = ["distance_km", "elevation_m", "gradient_pct", "elevation_per_km"]


def train_models_for_sport(
    df: pd.DataFrame, sport_name: str = "ride", model_dir: Path = MODEL_DIR
) -> bool:
    """Train and save duration and energy regressors; return whether training occurred."""
    if len(df) < 5:
        print(f"Insufficient data ({len(df)} samples) to train models for '{sport_name}'.")
        return False
    if not set(FEATURES + ["moving_time", "kcal_clean"]).issubset(df.columns):
        raise ValueError("DataFrame is missing required model columns")

    X = df[FEATURES]
    y_time, y_kcal = df["moving_time"], df["kcal_clean"]
    X_train, X_test, time_train, time_test, kcal_train, kcal_test = train_test_split(
        X, y_time, y_kcal, test_size=0.2, random_state=42
    )
    model_time = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model_kcal = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model_time.fit(X_train, time_train)
    model_kcal.fit(X_train, kcal_train)
    print(
        f"[{sport_name.upper()}] MAE -> Duration: "
        f"{mean_absolute_error(time_test, model_time.predict(X_test)) / 60:.1f} min | "
        f"Energy: {mean_absolute_error(kcal_test, model_kcal.predict(X_test)):.0f} kcal"
    )
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_time, model_dir / f"{sport_name.lower()}_time.joblib")
    joblib.dump(model_kcal, model_dir / f"{sport_name.lower()}_kcal.joblib")
    return True
