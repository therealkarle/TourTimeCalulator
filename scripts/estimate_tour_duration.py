"""Estimate duration, energy consumption, and speed for a planned tour."""

import argparse
import sys
from pathlib import Path

# Also support: python scripts/estimate_tour_duration.py
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.predictor import list_models, predict_tour


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="Name or ID of the trained model")
    parser.add_argument("--distance-km", type=float, help="Route distance in kilometres")
    parser.add_argument("--elevation-m", type=float, help="Elevation gain in metres")
    args = parser.parse_args()

    try:
        models = list_models()
        if not models:
            raise FileNotFoundError("No trained models found. Run train_models.py first.")
        model_name = args.model
        if not model_name:
            print("Available models:")
            for index, model in enumerate(models, start=1):
                print(f"  {index}. {model['model_name']} ({model['sport_type']})")
            selected_index = int(input("Select model number: ")) - 1
            model_name = models[selected_index]["model_id"]
        distance = args.distance_km
        if distance is None:
            distance = float(input("Route distance in km: "))
        elevation = args.elevation_m
        if elevation is None:
            elevation = float(input("Elevation gain in m: "))
        result = predict_tour(model_name, distance, elevation)
    except (ValueError, FileNotFoundError, IndexError, KeyError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error

    print("\n================ RESULT ================")
    print(f"Model:             {result['model_name']}")
    print(f"Sport:             {result['sport_type'].capitalize()}")
    print(f"Distance / Elev:   {result['distance_km']} km | {result['elevation_m']} m")
    print(f"Estimated Time:    {result['predicted_time']}")
    print(f"Estimated Energy:  {result['predicted_kcal']} kcal")
    print(f"Avg Speed:         {result['avg_speed_kmh']} km/h")


if __name__ == "__main__":
    main()
