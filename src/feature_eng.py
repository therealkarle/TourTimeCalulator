"""Cleaning and feature engineering for cached activities."""

import sqlite3

import pandas as pd

from src.config import DB_PATH


def load_cleaned_data(sport_type: str = "Ride") -> pd.DataFrame:
    """Load one sport from SQLite, remove invalid activities, and add features."""
    with sqlite3.connect(DB_PATH) as conn:
        frame = pd.read_sql_query(
            "SELECT * FROM activities WHERE type IN (?, ?)",
            conn,
            params=(sport_type, f"Virtual{sport_type}"),
        )
    if frame.empty:
        return frame

    frame["distance_km"] = frame["distance"] / 1000.0
    frame["elevation_m"] = frame["total_elevation_gain"].fillna(0.0)
    frame = frame[(frame["distance_km"] >= 3.0) & (frame["moving_time"] > 0)]
    frame = frame[frame["elapsed_time"] > 0]
    frame = frame[(frame["moving_time"] / frame["elapsed_time"]) > 0.70]
    frame["gradient_pct"] = frame["elevation_m"] / (frame["distance_km"] * 10.0)
    frame["elevation_per_km"] = frame["elevation_m"] / frame["distance_km"]
    kilojoules = pd.to_numeric(frame["kilojoules"], errors="coerce")
    calories = pd.to_numeric(frame["calories"], errors="coerce")
    frame["kcal_clean"] = kilojoules.fillna(calories)
    return frame.dropna(subset=["moving_time", "kcal_clean"])
