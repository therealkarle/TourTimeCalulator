"""Cleaning and feature engineering for cached activities."""

import sqlite3
from datetime import date, datetime, time, timedelta, timezone

import pandas as pd

from src.config import DB_PATH
from src.strava_client import ensure_db_schema


SPORT_ALIASES = {
    "ride": ("Ride", "VirtualRide"),
    "mtb-ride": ("MountainBikeRide", "EMountainBikeRide"),
    "gravel": ("GravelBikeRide",),
    "hike": ("Hike",),
    "run": ("Run", "VirtualRun"),
    "trail_run": ("TrailRun",),
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
    min_distance_km: float | None = 3.0,
    max_distance_km: float | None = None,
    min_elevation_m: float | None = None,
    max_elevation_m: float | None = None,
    min_moving_time_s: int | None = None,
    max_moving_time_s: int | None = None,
    min_elapsed_time_s: int | None = None,
    max_elapsed_time_s: int | None = None,
    heart_rate_data: bool | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
) -> pd.DataFrame:
    """Load Strava activities and apply training filters."""
    _validate_filter_ranges(
        min_distance_km, max_distance_km, min_elevation_m, max_elevation_m,
        min_moving_time_s, max_moving_time_s, min_elapsed_time_s,
        max_elapsed_time_s, heart_rate_data, start_date, end_date,
    )
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

    frame["distance_km"] = pd.to_numeric(frame["distance"], errors="coerce") / 1000.0
    frame["elevation_m"] = frame["total_elevation_gain"].fillna(0.0)
    frame["elevation_m"] = pd.to_numeric(frame["elevation_m"], errors="coerce")
    frame["elevation_up_m"] = frame["elevation_m"]
    frame["elevation_down_m"] = pd.to_numeric(frame["descent_elevation_m"], errors="coerce")
    if min_distance_km is not None:
        frame = frame[frame["distance_km"] >= min_distance_km]
    if max_distance_km is not None:
        frame = frame[frame["distance_km"] <= max_distance_km]
    if min_elevation_m is not None:
        frame = frame[frame["elevation_m"] >= min_elevation_m]
    if max_elevation_m is not None:
        frame = frame[frame["elevation_m"] <= max_elevation_m]
    if min_moving_time_s is not None:
        frame = frame[frame["moving_time"] >= min_moving_time_s]
    if max_moving_time_s is not None:
        frame = frame[frame["moving_time"] <= max_moving_time_s]
    if min_elapsed_time_s is not None:
        frame = frame[frame["elapsed_time"] >= min_elapsed_time_s]
    if max_elapsed_time_s is not None:
        frame = frame[frame["elapsed_time"] <= max_elapsed_time_s]
    if start_date is not None:
        frame = frame[frame["start_date"] >= _utc_timestamp(start_date)]
    if end_date is not None:
        frame = frame[frame["start_date"] < _utc_timestamp(end_date + timedelta(days=1))]
    frame = frame[frame["moving_time"] > 0]
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
    if heart_rate_data is not None:
        heart_rate_available = frame["average_heartrate"].notna()
        frame = frame[heart_rate_available == heart_rate_data]
    kilojoules = pd.to_numeric(frame["kilojoules"], errors="coerce")
    calories = pd.to_numeric(frame["calories"], errors="coerce")
    frame["kcal_clean"] = kilojoules.fillna(calories)
    frame["average_power_watts"] = pd.to_numeric(frame["average_watts"], errors="coerce")
    frame["weighted_average_power_watts"] = pd.to_numeric(
        frame["weighted_average_watts"], errors="coerce"
    )
    frame["average_hr_bpm"] = pd.to_numeric(frame["average_heartrate"], errors="coerce")
    return frame.dropna(subset=["moving_time", "kcal_clean"])


def _validate_filter_ranges(
    min_distance_km: float | None,
    max_distance_km: float | None,
    min_elevation_m: float | None,
    max_elevation_m: float | None,
    min_moving_time_s: int | None,
    max_moving_time_s: int | None,
    min_elapsed_time_s: int | None,
    max_elapsed_time_s: int | None,
    heart_rate_data: bool | None,
    start_date: date | None,
    end_date: date | None,
) -> None:
    for name, value in (
        ("min_distance_km", min_distance_km),
        ("max_distance_km", max_distance_km),
        ("min_elevation_m", min_elevation_m),
        ("max_elevation_m", max_elevation_m),
        ("min_moving_time_s", min_moving_time_s),
        ("max_moving_time_s", max_moving_time_s),
        ("min_elapsed_time_s", min_elapsed_time_s),
        ("max_elapsed_time_s", max_elapsed_time_s),
    ):
        if value is not None and value < 0:
            raise ValueError(f"{name} cannot be negative")
    if min_distance_km is not None and max_distance_km is not None and min_distance_km > max_distance_km:
        raise ValueError("min_distance_km cannot be greater than max_distance_km")
    if min_elevation_m is not None and max_elevation_m is not None and min_elevation_m > max_elevation_m:
        raise ValueError("min_elevation_m cannot be greater than max_elevation_m")
    if min_moving_time_s is not None and max_moving_time_s is not None and min_moving_time_s > max_moving_time_s:
        raise ValueError("min_moving_time_s cannot be greater than max_moving_time_s")
    if min_elapsed_time_s is not None and max_elapsed_time_s is not None and min_elapsed_time_s > max_elapsed_time_s:
        raise ValueError("min_elapsed_time_s cannot be greater than max_elapsed_time_s")
    if start_date is not None and end_date is not None and start_date > end_date:
        raise ValueError("start_date cannot be after end_date")


def _utc_timestamp(value: date) -> int:
    return int(datetime.combine(value, time.min, tzinfo=timezone.utc).timestamp())
