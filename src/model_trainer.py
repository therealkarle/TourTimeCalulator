"""Training and persistence of tour prediction models."""

import json
import re
from pathlib import Path
from typing import Any, Iterator

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import MODEL_DIR
BASE_FEATURES = ["distance_km", "elevation_m"]
FEATURES = BASE_FEATURES
SEPARATE_ELEVATION_FEATURES = ["distance_km", "elevation_up_m", "elevation_down_m"]
MIN_TRAINING_SAMPLES = 5
REGRESSION_TYPES = {"linear", "ridge"}


class NonNegativeRegressor(RegressorMixin, BaseEstimator):
    """Wrap a regressor and guarantee nonnegative predictions."""

    def __init__(self, estimator: Any) -> None:
        self.estimator = estimator

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "NonNegativeRegressor":
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.maximum(0.0, self.estimator_.predict(X))


def _ridge_model(alpha: float) -> NonNegativeRegressor:
    return NonNegativeRegressor(
        Pipeline(
            [
                ("scale", StandardScaler()),
                ("regression", Ridge(alpha=alpha, positive=True)),
            ]
        )
    )


def _linear_model() -> NonNegativeRegressor:
    return NonNegativeRegressor(
        Pipeline(
            [
                ("scale", StandardScaler()),
                ("regression", LinearRegression(positive=True)),
            ]
        )
    )


def _chronological_splits(sample_count: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield expanding-window splits with the oldest half used first."""
    first_test = max(2, sample_count // 2)
    test_size = max(1, (sample_count - first_test) // 5)
    test_start = first_test
    while test_start < sample_count:
        test_end = min(sample_count, test_start + test_size)
        yield np.arange(test_start), np.arange(test_start, test_end)
        test_start = test_end


def _cross_validate(
    estimator: NonNegativeRegressor,
    X: pd.DataFrame,
    y: pd.Series,
) -> dict[str, float | None]:
    errors: list[float] = []
    percentage_errors: list[float] = []
    for train_indices, test_indices in _chronological_splits(len(X)):
        candidate = clone(estimator).fit(X.iloc[train_indices], y.iloc[train_indices])
        actual = y.iloc[test_indices].to_numpy(dtype=float)
        predicted = candidate.predict(X.iloc[test_indices])
        absolute_errors = np.abs(actual - predicted)
        errors.extend(absolute_errors.tolist())
        nonzero = actual > 0
        percentage_errors.extend(
            (absolute_errors[nonzero] / actual[nonzero] * 100.0).tolist()
        )
    if not errors:
        raise ValueError("Not enough chronologically ordered samples for validation")
    return {
        "mae": float(np.mean(errors)),
        "median_absolute_error": float(np.median(errors)),
        "mape_pct": float(np.median(percentage_errors)) if percentage_errors else None,
        "error_p90": float(np.quantile(errors, 0.9)),
    }


def _select_model(
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[NonNegativeRegressor, dict[str, float | None], float]:
    """Select Ridge regularization by chronological error."""
    baseline_metrics = _cross_validate(NonNegativeRegressor(LinearRegression()), X, y)
    candidates = [(_ridge_model(alpha), alpha) for alpha in (0.1, 1.0, 10.0, 100.0)]
    scored = [(_cross_validate(model, X, y), model, alpha) for model, alpha in candidates]
    metrics, selected, alpha = min(
        scored, key=lambda item: float(item[0]["median_absolute_error"])
    )
    metrics["legacy_ols_mae"] = baseline_metrics["mae"]
    metrics["legacy_ols_median_absolute_error"] = baseline_metrics[
        "median_absolute_error"
    ]
    return selected.fit(X, y), metrics, alpha


def _original_scale_parameters(model: NonNegativeRegressor) -> tuple[float, np.ndarray]:
    pipeline = model.estimator_
    scaler = pipeline.named_steps["scale"]
    regression = pipeline.named_steps["regression"]
    coefficients = regression.coef_ / scaler.scale_
    intercept = regression.intercept_ - np.sum(
        regression.coef_ * scaler.mean_ / scaler.scale_
    )
    return float(intercept), coefficients


def train_models_for_sport(
    df: pd.DataFrame,
    sport_name: str,
    model_name: str,
    model_dir: Path = MODEL_DIR,
    distance_elevation_only: bool = False,
    filters: dict | None = None,
    separate_elevation: bool = False,
    regression_type: str = "ridge",
) -> bool:
    """Train validated regressors for moving time, stopped time, and energy."""
    del distance_elevation_only  # Retained for compatibility with existing callers.
    if regression_type not in REGRESSION_TYPES:
        raise ValueError(
            f"regression_type must be one of: {', '.join(sorted(REGRESSION_TYPES))}"
        )
    model_features = SEPARATE_ELEVATION_FEATURES if separate_elevation else BASE_FEATURES
    required_columns = model_features + ["moving_time", "stopped_time", "kcal_clean"]
    if len(df) < MIN_TRAINING_SAMPLES:
        print(
            f"Insufficient data ({len(df)} samples); at least "
            f"{MIN_TRAINING_SAMPLES} are required for '{sport_name}'."
        )
        return False
    if not set(required_columns).issubset(df.columns):
        raise ValueError("DataFrame is missing required model columns")

    targets = {
        "moving_time": "Moving time",
        "stopped_time": "Stopped time",
        "kcal_clean": "Energy",
    }
    models: dict[str, NonNegativeRegressor] = {}
    metrics: dict[str, dict[str, float | None]] = {}
    sample_counts: dict[str, int] = {}
    alphas: dict[str, float | None] = {}

    for target, label in targets.items():
        columns = model_features + [target]
        if "start_date" in df:
            columns.append("start_date")
        target_frame = df[columns].dropna()
        if "start_date" in target_frame:
            target_frame = target_frame.sort_values("start_date", kind="stable")
        if len(target_frame) < MIN_TRAINING_SAMPLES:
            print(f"Insufficient complete data to train {label.lower()} for '{sport_name}'.")
            return False
        target_X = target_frame[model_features].reset_index(drop=True)
        target_y = target_frame[target].reset_index(drop=True)
        if regression_type == "ridge":
            model, target_metrics, alpha = _select_model(target_X, target_y)
        else:
            model = _linear_model()
            target_metrics = _cross_validate(model, target_X, target_y)
            baseline_metrics = _cross_validate(
                NonNegativeRegressor(LinearRegression()), target_X, target_y
            )
            target_metrics["legacy_ols_mae"] = baseline_metrics["mae"]
            target_metrics["legacy_ols_median_absolute_error"] = baseline_metrics[
                "median_absolute_error"
            ]
            model = model.fit(target_X, target_y)
            alpha = None
        models[target] = model
        metrics[target] = target_metrics
        sample_counts[target] = len(target_frame)
        alphas[target] = alpha

    print(f"[{sport_name.upper()}] Activities used: {len(df)}")
    for target, label in targets.items():
        target_metrics = metrics[target]
        mape = target_metrics["mape_pct"]
        mape_label = f"{mape:.1f}%" if mape is not None else "n/a"
        print(
            f"[{sport_name.upper()}] {label}: chronological MAE "
            f"{target_metrics['mae']:.1f}, median error "
            f"{target_metrics['median_absolute_error']:.1f}, MAPE {mape_label}, "
            f"p90 ±{target_metrics['error_p90']:.1f}"
        )

    model_id = re.sub(r"[^a-zA-Z0-9_-]+", "_", model_name.strip()).strip("_-").lower()
    if not model_id:
        raise ValueError("model_name must contain at least one letter or number")

    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "schema_version": 2,
            "regression_type": regression_type,
            "model_name": model_name.strip(),
            "sport_type": sport_name.lower(),
            "models": models,
            "time_model": models["moving_time"],
            "kcal_model": models["kcal_clean"],
        },
        model_dir / f"{model_id}.joblib",
    )

    regressions = {}
    for target, model in models.items():
        intercept, coefficients = _original_scale_parameters(model)
        regressions[target] = {
            "intercept": intercept,
            "coefficients": {
                feature: float(coefficient)
                for feature, coefficient in zip(model_features, coefficients)
            },
            "mae": metrics[target]["mae"],
            "median_absolute_error": metrics[target]["median_absolute_error"],
            "mape_pct": metrics[target]["mape_pct"],
            "error_p90": metrics[target]["error_p90"],
            "legacy_ols_mae": metrics[target]["legacy_ols_mae"],
            "alpha": alphas[target],
        }
    training_ranges = {
        feature: {"min": float(df[feature].min()), "max": float(df[feature].max())}
        for feature in model_features
    }
    metadata = {
        "schema_version": 2,
        "regression_type": regression_type,
        "model_id": model_id,
        "model_name": model_name.strip(),
        "sport_type": sport_name.lower(),
        "model_file": f"{model_id}.joblib",
        "features": model_features,
        "elevation_mode": "separate" if separate_elevation else "up",
        "sample_count": len(df),
        "sample_counts": sample_counts,
        "filters": filters or {},
        "training_ranges": training_ranges,
        "regressions": regressions,
        "energy": {
            "unit": "kcal",
            "source": "calories",
        },
        "duration_intercept_seconds": regressions["moving_time"]["intercept"],
        "duration_coefficients": regressions["moving_time"]["coefficients"],
        "kcal_intercept": regressions["kcal_clean"]["intercept"],
        "kcal_coefficients": regressions["kcal_clean"]["coefficients"],
    }
    with (model_dir / f"{model_id}.txt").open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)
        metadata_file.write("\n")
    print(f"Saved model '{model_name.strip()}' ({sport_name}) to {model_dir / (model_id + '.txt')}")
    return True
