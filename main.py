"""Interactive command-line entry point."""

import sys

from src.feature_eng import load_cleaned_data
from src.model_trainer import train_models_for_sport
from src.predictor import predict_tour
from src.strava_client import StravaClient


def main() -> None:
    """Synchronize data, train available models, and predict a planned tour."""
    print("========================================")
    print("       Strava Tour Predictor")
    print("========================================\n")
    StravaClient().sync_activities()

    print("\nTraining regression models...")
    for sport_type, sport_name in (("Ride", "ride"), ("Run", "run")):
        data = load_cleaned_data(sport_type)
        if not data.empty:
            train_models_for_sport(data, sport_name)

    sport = input("Enter sport type (ride/run) [default: ride]: ").strip().lower() or "ride"
    try:
        distance = float(input("Enter route distance in km: "))
        elevation = float(input("Enter elevation gain in m: "))
        result = predict_tour(sport, distance, elevation)
    except (ValueError, FileNotFoundError) as error:
        print(f"Error: {error}")
        sys.exit(1)

    print("\n================ RESULT ================")
    print(f"Sport:             {result['sport_type'].capitalize()}")
    print(f"Distance / Elev:   {result['distance_km']} km | {result['elevation_m']} m")
    print(f"Estimated Time:    {result['predicted_time']}")
    print(f"Estimated Energy:  {result['predicted_kcal']} kcal")
    print(f"Avg Speed:         {result['avg_speed_kmh']} km/h")


if __name__ == "__main__":
    main()
