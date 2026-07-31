"""Train and save regression models from the locally cached Strava data."""

import argparse

from src.feature_eng import load_cleaned_data
from src.model_trainer import train_models_for_sport

SPORTS = {"ride": "Ride", "run": "Run"}


def train_sport(sport_name: str) -> bool:
    """Load cleaned data and train both models for one sport."""
    sport_type = SPORTS[sport_name]
    data = load_cleaned_data(sport_type)
    if data.empty:
        print(f"No valid {sport_name} activities found. Skipping.")
        return False
    return train_models_for_sport(data, sport_name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sport",
        choices=["ride", "run", "all"],
        default="all",
        help="Sport to train (default: all)",
    )
    args = parser.parse_args()

    sports = SPORTS if args.sport == "all" else {args.sport: SPORTS[args.sport]}
    for sport_name in sports:
        train_sport(sport_name)


if __name__ == "__main__":
    main()
