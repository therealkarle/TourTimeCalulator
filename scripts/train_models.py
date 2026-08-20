"""Train and save regression models from the locally cached Strava data."""

import argparse
import sys
from datetime import date
from pathlib import Path

# Also support: python scripts/train_models.py
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.feature_eng import SPORT_ALIASES, available_sport_profiles, load_cleaned_data
from src.model_trainer import train_models_for_sport

SPORTS = {key: key for key in SPORT_ALIASES}


def train_sport(
    sport_name: str,
    model_name: str,
    distance_elevation_only: bool = False,
    activity_types: list[str] | None = None,
    commute: bool | None = None,
    equipment: list[str] | None = None,
    power_data: bool | None = None,
    min_distance_km: float | None = 3.0,
    max_distance_km: float | None = None,
    min_elevation_m: float | None = None,
    max_elevation_m: float | None = None,
    min_elevation_up_m: float | None = None,
    max_elevation_up_m: float | None = None,
    min_elevation_down_m: float | None = None,
    max_elevation_down_m: float | None = None,
    min_moving_time_s: int | None = None,
    max_moving_time_s: int | None = None,
    min_elapsed_time_s: int | None = None,
    max_elapsed_time_s: int | None = None,
    heart_rate_data: bool | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    separate_elevation: bool = False,
    elevation_mode: str = "up",
    regression_type: str = "ridge",
) -> bool:
    """Load cleaned data and train one named model for one sport.
    
    elevation_mode: "up" for uphill only, "separate" for uphill and downhill separately.
    """
    # Keep the model feature selection in sync with the selected elevation mode.
    # Explicit ``separate_elevation=True`` remains supported for direct callers.
    separate_elevation = separate_elevation or elevation_mode == "separate"

    data = load_cleaned_data(
        sport_name, activity_types, commute, equipment, power_data,
        min_distance_km, max_distance_km, min_elevation_m, max_elevation_m,
        min_elevation_up_m, max_elevation_up_m, min_elevation_down_m, max_elevation_down_m,
        min_moving_time_s, max_moving_time_s, min_elapsed_time_s,
        max_elapsed_time_s, heart_rate_data, start_date, end_date,
        elevation_mode=elevation_mode,
    )
    if data.empty:
        print(f"No valid Strava activities found for '{sport_name}'. Skipping.")
        return False
    return train_models_for_sport(
        data,
        sport_name,
        model_name,
        distance_elevation_only=distance_elevation_only,
        filters={"activity_types": activity_types, "commute": commute, "equipment": equipment,
                 "power_data": power_data, "min_distance_km": min_distance_km,
                 "max_distance_km": max_distance_km, "min_elevation_m": min_elevation_m,
                 "max_elevation_m": max_elevation_m, "min_elevation_up_m": min_elevation_up_m,
                 "max_elevation_up_m": max_elevation_up_m, "min_elevation_down_m": min_elevation_down_m,
                 "max_elevation_down_m": max_elevation_down_m, "min_moving_time_s": min_moving_time_s,
                 "max_moving_time_s": max_moving_time_s, "min_elapsed_time_s": min_elapsed_time_s,
                 "max_elapsed_time_s": max_elapsed_time_s, "heart_rate_data": heart_rate_data,
                 "start_date": start_date.isoformat() if start_date else None,
                 "end_date": end_date.isoformat() if end_date else None,
                 "elevation_mode": elevation_mode, "regression_type": regression_type},
        separate_elevation=separate_elevation,
        regression_type=regression_type,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sport",
        choices=sorted(SPORTS),
        help="Sport for the model",
    )
    parser.add_argument("--name", help="Name of the model")
    parser.add_argument(
        "--distance-elevation-only",
        action="store_true",
        help="Deprecated compatibility option; new models always use these features",
    )
    parser.add_argument(
        "--separate-elevation",
        action="store_true",
        help="Train using separate ascent and descent elevation features",
    )
    parser.add_argument(
        "--regression",
        choices=["linear", "ridge"],
        default="ridge",
        help="Regression method (default: ridge)",
    )
    parser.add_argument("--activity-type", action="append", dest="activity_types",
                        help="Exact Strava sport_type; may be specified multiple times")
    commute_group = parser.add_mutually_exclusive_group()
    commute_group.add_argument("--commute", action="store_true", help="Include commute activities only")
    commute_group.add_argument("--no-commute", action="store_true", help="Exclude commute activities")
    parser.add_argument("--equipment", action="append", help="Strava gear_id; may be specified multiple times")
    parser.add_argument("--power-data", choices=["any", "available", "missing"], default="any")
    parser.add_argument("--min-distance-km", type=non_negative_float, default=3.0)
    parser.add_argument("--max-distance-km", type=non_negative_float)
    parser.add_argument("--min-elevation-m", type=non_negative_float)
    parser.add_argument("--max-elevation-m", type=non_negative_float)
    parser.add_argument("--start-date", type=iso_date, metavar="YYYY-MM-DD")
    parser.add_argument("--end-date", type=iso_date, metavar="YYYY-MM-DD")
    args = parser.parse_args()

    detected = available_sport_profiles()
    choices = sorted(detected or SPORTS)
    if detected:
        print(f"Detected Strava sport profiles: {', '.join(choices)}")
    sport = args.sport or input("Sport profile (ride/mtb-ride/gravel): ").strip().lower()
    while sport not in choices:
        print(f"Please enter one of: {', '.join(choices)}.")
        sport = input("Sport profile (ride/mtb-ride/gravel): ").strip().lower()

    model_name = args.name or input("Model name: ").strip()
    while not model_name:
        print("The model name cannot be empty.")
        model_name = input("Model name: ").strip()

    power_data = {"any": None, "available": True, "missing": False}[args.power_data]
    commute = True if args.commute else False if args.no_commute else None
    if (args.max_distance_km is not None and
            args.min_distance_km is not None and
            args.min_distance_km > args.max_distance_km):
        parser.error("--min-distance-km cannot be greater than --max-distance-km")
    if (args.max_elevation_m is not None and
            args.min_elevation_m is not None and
            args.min_elevation_m > args.max_elevation_m):
        parser.error("--min-elevation-m cannot be greater than --max-elevation-m")
    if (args.start_date is not None and args.end_date is not None and
            args.start_date > args.end_date):
        parser.error("--start-date cannot be after --end-date")

    train_sport(
        sport, model_name, args.distance_elevation_only, args.activity_types,
        commute, args.equipment, power_data,
        min_distance_km=args.min_distance_km,
        max_distance_km=args.max_distance_km,
        min_elevation_m=args.min_elevation_m,
        max_elevation_m=args.max_elevation_m,
        start_date=args.start_date,
        end_date=args.end_date,
        separate_elevation=args.separate_elevation,
        regression_type=args.regression,
    )


def non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value cannot be negative")
    return parsed


def iso_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("date must use YYYY-MM-DD format") from exc


if __name__ == "__main__":
    main()
