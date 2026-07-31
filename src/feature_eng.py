"""Cleaning and feature engineering for cached activities."""

import sqlite3

import pandas as pd

from src.config import DB_PATH
from src.strava_client import ensure_db_schema


SPORT_ALIASES = {
    "ride": ("Ride", "VirtualRide"),
    "mtb-ride": ("MountainBikeRide", "EMountainBikeRide"),
    "gravel": ("GravelBikeRide",),
}


def available_sport_profiles() -> dict[str, tuple[str, ...]]:
    """Return English profiles with at least one matching cached Strava activity."""
    ensure_db_schema(DB_PATH)
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute("SELECT DISTINCT type, sport_type FROM activities").fetchall()
    available_types = {value for row in rows for value in row if value}
    return {
        profile: types
        for profile, types in SPORT_ALIASES.items()
        if available_types.intersection(types)
    }


def load_cleaned_data(
    sport_type: str = "Ride",
    activity_types: list[str] | None = None,
    commute: bool | None = None,
    equipment: list[str] | None = None,
    power_data: bool | None = None,
) -> pd.DataFrame:
    """Load Strava activities and apply sport, activity, commute, gear and power filters."""
    ensure_db_schema(DB_PATH)
    requested = SPORT_ALIASES.get(sport_type.lower(), (sport_type, f"Virtual{sport_type}"))
    type_clause = ""
    params: tuple[str, ...] = tuple(requested)
    if activity_types:
        type_clause = f" AND type IN ({','.join('?' for _ in activity_types)})"
        params += tuple(activity_types)
    with sqlite3.connect(DB_PATH) as conn:
        frame = pd.read_sql_query(
            f"SELECT * FROM activities WHERE COALESCE(sport_type, type) IN "
            f"({','.join('?' for _ in requested)}){type_clause}",
            conn,
            params=params,
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
    frame["commute_clean"] = frame["commute"].fillna(frame["workout_type"].eq(2)).fillna(False).astype(bool)
    power_columns = ["average_watts", "weighted_average_watts"]
    present_power_columns = [column for column in power_columns if column in frame]
    frame["power_data_available"] = frame[present_power_columns].notna().any(axis=1)
    if "device_watts" in frame:
        frame["power_data_available"] |= frame["device_watts"].fillna(0).astype(bool)
    if commute is not None:
        frame = frame[frame["commute_clean"] == commute]
    if equipment:
        frame = frame[frame["gear_id"].fillna("").isin(equipment)]
    if power_data is not None:
        frame = frame[frame["power_data_available"] == power_data]
    kilojoules = pd.to_numeric(frame["kilojoules"], errors="coerce")
    calories = pd.to_numeric(frame["calories"], errors="coerce")
    frame["kcal_clean"] = kilojoules.fillna(calories)
    return frame.dropna(subset=["moving_time", "kcal_clean"])
