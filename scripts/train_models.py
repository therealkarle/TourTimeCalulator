"""Train and save regression models from the locally cached Strava data."""

import argparse
import sys
from pathlib import Path

# Also support: python scripts/train_models.py
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.feature_eng import load_cleaned_data
from src.model_trainer import train_models_for_sport

SPORTS = {"ride": "Ride", "run": "Run"}


def train_sport(sport_name: str, model_name: str) -> bool:
    """Load cleaned data and train one named model for one sport."""
    sport_type = SPORTS[sport_name]
    data = load_cleaned_data(sport_type)
    if data.empty:
        print(f"No valid {sport_name} activities found. Skipping.")
        return False
    return train_models_for_sport(data, sport_name, model_name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sport",
        choices=["ride", "run"],
        help="Sport for the model",
    )
    parser.add_argument("--name", help="Name of the model")
    args = parser.parse_args()

    sport = args.sport or input("Sport type (ride/run): ").strip().lower()
    while sport not in SPORTS:
        print("Please enter either 'ride' or 'run'.")
        sport = input("Sport type (ride/run): ").strip().lower()

    model_name = args.name or input("Model name: ").strip()
    while not model_name:
        print("The model name cannot be empty.")
        model_name = input("Model name: ").strip()

    train_sport(sport, model_name)


if __name__ == "__main__":
    main()
