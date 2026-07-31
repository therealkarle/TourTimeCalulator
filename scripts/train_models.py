"""Train and save regression models from the locally cached Strava data."""

import argparse
import sys
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
) -> bool:
    """Load cleaned data and train one named model for one sport."""
    data = load_cleaned_data(sport_name, activity_types, commute, equipment, power_data)
    if data.empty:
        print(f"No valid Strava activities found for '{sport_name}'. Skipping.")
        return False
    return train_models_for_sport(
        data,
        sport_name,
        model_name,
        distance_elevation_only=distance_elevation_only,
        filters={"activity_types": activity_types, "commute": commute, "equipment": equipment,
                 "power_data": power_data},
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
        help="Train using only distance and elevation (omit elevation per km)",
    )
    parser.add_argument("--activity-type", action="append", dest="activity_types",
                        help="Exakter Strava sport_type; mehrfach verwendbar")
    commute_group = parser.add_mutually_exclusive_group()
    commute_group.add_argument("--commute", action="store_true", help="Nur Pendelaktivitäten")
    commute_group.add_argument("--no-commute", action="store_true", help="Pendelfahrten ausschließen")
    parser.add_argument("--equipment", action="append", help="Strava gear_id; mehrfach verwendbar")
    parser.add_argument("--power-data", choices=["any", "available", "missing"], default="any")
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
    train_sport(sport, model_name, args.distance_elevation_only, args.activity_types,
                commute, args.equipment, power_data)


if __name__ == "__main__":
    main()
